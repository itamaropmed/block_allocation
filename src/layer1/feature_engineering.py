"""
Layer 1 — Feature Engineering
=============================

This module builds the phase-aligned feature matrix described in the Layer 1
plan. It is intentionally standalone: files can live directly in src/layer1 and
be executed by run_layer1.py without package imports.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from attention_pooling import DEFAULT_WQ, attention_summary, train_global_wq


TARGET_COLUMNS = {'casetime_min', 'turnover_min'}
ID_COLUMNS = {
    'provider_id', 'service_line', 'week_start', 'source_mode',
    'block_historical_id', 'dominant_provider_id',
}
LEAKAGE_COLUMNS = {
    'casetime_util', 'turnover_util', 'total_util',
}


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def _get_t_star(prelayer: dict) -> int:
    for key in ('T_star', 'T*', 'T', 't_star', 'best_T', 'T_best', 'detected_T', 'rotation_period_weeks'):
        if key in prelayer:
            try:
                return max(1, int(prelayer[key]))
            except Exception:
                pass
    labels = prelayer.get('phase_labels') or {}
    vals = [int(v) for v in labels.values()] if isinstance(labels, dict) else []
    return max(vals) + 1 if vals else 1


def _phase_for_week(prelayer: dict, week_index: int) -> int:
    labels = prelayer.get('phase_labels') or {}
    if isinstance(labels, dict) and str(week_index) in labels:
        return int(labels[str(week_index)])
    t_star = _get_t_star(prelayer)
    offset = _safe_int(prelayer.get('phase_offset', 0), 0)
    return int((week_index + offset) % max(1, t_star))


def _build_attention_training_pairs(obs: pd.DataFrame, value_col: str) -> tuple[list[list[float]], list[float]]:
    sequences: list[list[float]] = []
    targets: list[float] = []
    for (_, _), g0 in obs.groupby(['provider_id', 'day_of_week'], dropna=False):
        for _, g in g0.sort_values('week_index').groupby('rotation_phase', dropna=False):
            hist: list[float] = []
            for _, row in g.sort_values('week_index').iterrows():
                y = row.get(value_col, np.nan)
                if len(hist) >= 2 and y == y and np.isfinite(y):
                    sequences.append(hist[-12:].copy())
                    targets.append(float(y))
                if y == y and np.isfinite(y):
                    hist.append(float(y))
    return sequences, targets


def _service_line_lag_context(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Build a leakage-safe cross-provider context feature.

    The planning document calls this service_line_wk_mean. The naive same-week
    mean leaks the validation target, so this implementation uses the previous
    observed week for the same service line and day of week. It keeps the same
    column name for downstream compatibility.
    """
    base = (
        obs.groupby(['service_line', 'day_of_week', 'week_index'], dropna=False)
        .agg(_sl_current_mean=('casetime_util', 'mean'))
        .reset_index()
        .sort_values(['service_line', 'day_of_week', 'week_index'])
    )
    base['service_line_wk_mean'] = (
        base.groupby(['service_line', 'day_of_week'], dropna=False)['_sl_current_mean']
        .shift(1)
    )
    base['_sl_expanding_mean'] = (
        base.groupby(['service_line', 'day_of_week'], dropna=False)['_sl_current_mean']
        .expanding()
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    base['service_line_wk_mean'] = (
        base['service_line_wk_mean']
        .fillna(base['_sl_expanding_mean'])
        .fillna(base['_sl_current_mean'])
        .fillna(0.0)
    )
    return base[['service_line', 'day_of_week', 'week_index', 'service_line_wk_mean']]


def _latest_service_line_context(obs: pd.DataFrame) -> pd.DataFrame:
    base = (
        obs.groupby(['service_line', 'day_of_week', 'week_index'], dropna=False)
        .agg(service_line_wk_mean=('casetime_util', 'mean'))
        .reset_index()
        .sort_values(['service_line', 'day_of_week', 'week_index'])
    )
    if base.empty:
        return pd.DataFrame(columns=['service_line', 'day_of_week', 'service_line_wk_mean'])
    return (
        base.groupby(['service_line', 'day_of_week'], as_index=False, dropna=False)
        .tail(1)[['service_line', 'day_of_week', 'service_line_wk_mean']]
        .reset_index(drop=True)
    )


def _exception_rate_by_provider(prelayer: dict) -> pd.DataFrame:
    """Best-effort extraction of exception/contestedness feature from Pre-Layer."""
    frames: list[pd.DataFrame] = []

    for key in ('exception_rate_per_series', 'block_template'):
        obj = prelayer.get(key)
        if obj is None:
            continue
        if isinstance(obj, dict):
            rows = []
            for _, v in obj.items():
                if isinstance(v, dict):
                    rows.append(v)
            df = pd.DataFrame(rows)
        elif isinstance(obj, list):
            df = pd.DataFrame(obj)
        else:
            continue
        if df.empty:
            continue

        provider_candidates = [
            'dominant_provider_id', 'provider_id', 'modal_provider_id',
            'dominant_provider', 'blockholder_provider_id',
        ]
        rate_candidates = [
            'exception_rate', 'exception_rate_series', 'contestedness',
            'deviation_rate', 'non_modal_rate',
        ]
        provider_col = next((c for c in provider_candidates if c in df.columns), None)
        rate_col = next((c for c in rate_candidates if c in df.columns), None)
        if provider_col and rate_col:
            tmp = df[[provider_col, rate_col]].copy()
            tmp = tmp.rename(columns={provider_col: 'provider_id', rate_col: 'exception_rate_series'})
            tmp['provider_id'] = tmp['provider_id'].fillna('OPEN').astype(str).str.strip()
            tmp['exception_rate_series'] = pd.to_numeric(tmp['exception_rate_series'], errors='coerce').fillna(0.0)
            frames.append(tmp[tmp['provider_id'] != 'OPEN'])

    if not frames:
        return pd.DataFrame(columns=['provider_id', 'exception_rate_series'])

    out = pd.concat(frames, ignore_index=True)
    return (
        out.groupby('provider_id', as_index=False)['exception_rate_series']
        .mean()
    )


def _add_calendar_features(f: dict, week_index: int, day_of_week: int, phase: int, t_star: int) -> None:
    week_num = int(week_index) + 1
    f['sin_woy'] = float(np.sin(2.0 * np.pi * week_num / 52.0))
    f['cos_woy'] = float(np.cos(2.0 * np.pi * week_num / 52.0))
    f['sin_woy_2'] = float(np.sin(4.0 * np.pi * week_num / 52.0))
    f['cos_woy_2'] = float(np.cos(4.0 * np.pi * week_num / 52.0))
    for d in range(7):
        f[f'dow_{d}'] = 1 if int(day_of_week) == d else 0
    for ph in range(max(1, int(t_star))):
        f[f'phase_{ph}'] = 1 if int(phase) == ph else 0


def _add_history_features(f: dict, case_hist: pd.Series, turn_hist: pd.Series, wq_case: float, wq_turn: float) -> None:
    for window in [2, 4, 8, 12]:
        ch = case_hist.tail(window)
        th = turn_hist.tail(window)
        f[f'trailing_{window}w_mean_case'] = float(ch.mean()) if len(ch) else np.nan
        f[f'trailing_{window}w_std_case'] = float(ch.std(ddof=0)) if len(ch) else np.nan
        f[f'trailing_{window}w_min_case'] = float(ch.min()) if len(ch) else np.nan
        f[f'trailing_{window}w_max_case'] = float(ch.max()) if len(ch) else np.nan
        f[f'trailing_{window}w_mean_turn'] = float(th.mean()) if len(th) else np.nan
        f[f'trailing_{window}w_std_turn'] = float(th.std(ddof=0)) if len(th) else np.nan
        f[f'trailing_{window}w_min_turn'] = float(th.min()) if len(th) else np.nan
        f[f'trailing_{window}w_max_turn'] = float(th.max()) if len(th) else np.nan

    ac = attention_summary(case_hist.tail(12).values, wq_case)
    at = attention_summary(turn_hist.tail(12).values, wq_turn)
    f['attn_weighted_util'] = ac['weighted']
    f['attn_entropy'] = ac['entropy']
    f['attn_recency_bias'] = ac['recency_bias']
    f['attn_turn_weighted'] = at['weighted']
    f['attn_turn_entropy'] = at['entropy']
    f['attn_turn_recency'] = at['recency_bias']
    f['weeks_observed'] = int(len(case_hist))


def build_features(obs: pd.DataFrame, prelayer: dict, config: dict) -> tuple[pd.DataFrame, dict]:
    obs = obs.copy()
    t_star = _get_t_star(prelayer)

    # Total utilization is useful for validation and possible later explanations.
    if 'total_util' not in obs.columns:
        obs['total_util'] = np.where(
            obs['allocated_min'] > 0,
            (obs['casetime_min'] + obs['turnover_min']) / obs['allocated_min'],
            0.0,
        )

    layer_cfg = config.get('layer1', {}) if isinstance(config, dict) else {}
    train_attn = bool(layer_cfg.get('train_attention', True))
    fallback = float(layer_cfg.get('attention_default_wq', DEFAULT_WQ))

    if train_attn:
        seq_case, y_case = _build_attention_training_pairs(obs, 'casetime_util')
        seq_turn, y_turn = _build_attention_training_pairs(obs, 'turnover_util')
        wq_case = train_global_wq(seq_case, y_case, fallback=fallback)
        wq_turn = train_global_wq(seq_turn, y_turn, fallback=fallback)
    else:
        wq_case = fallback
        wq_turn = fallback

    sl_context = _service_line_lag_context(obs)
    exc_by_provider = _exception_rate_by_provider(prelayer)

    rows: list[dict] = []
    for (pid, dow), g0 in obs.groupby(['provider_id', 'day_of_week'], dropna=False):
        g0 = g0.sort_values('week_index').reset_index(drop=True)
        histories: dict[int, dict[str, list[float]]] = defaultdict(lambda: {'case': [], 'turn': []})

        for _, r in g0.iterrows():
            phase = int(r['rotation_phase'])
            h = histories[phase]
            case_hist = pd.Series(h['case'], dtype=float)
            turn_hist = pd.Series(h['turn'], dtype=float)

            f = r.to_dict()
            _add_history_features(f, case_hist, turn_hist, wq_case, wq_turn)
            _add_calendar_features(f, int(r['week_index']), int(dow), phase, t_star)
            rows.append(f)

            h['case'].append(float(r.get('casetime_util', 0.0)))
            h['turn'].append(float(r.get('turnover_util', 0.0)))

    features = pd.DataFrame(rows)
    if not features.empty:
        features = features.merge(sl_context, on=['service_line', 'day_of_week', 'week_index'], how='left')
        features = features.merge(exc_by_provider, on='provider_id', how='left')
    else:
        features['service_line_wk_mean'] = []
        features['exception_rate_series'] = []

    if 'exception_rate_series' not in features.columns:
        features['exception_rate_series'] = 0.0
    features['exception_rate_series'] = features['exception_rate_series'].fillna(0.0)
    features['provider_exception_rate'] = features['exception_rate_series']

    if 'service_line_wk_mean' not in features.columns:
        features['service_line_wk_mean'] = 0.0
    features['service_line_wk_mean'] = features['service_line_wk_mean'].fillna(0.0)

    return (
        features.sort_values(['week_index', 'provider_id', 'day_of_week']).reset_index(drop=True),
        {
            'wq_case': float(wq_case),
            'wq_turn': float(wq_turn),
            'attention_trained': bool(train_attn),
            'T_star': int(t_star),
        },
    )


def build_forecast_features(obs: pd.DataFrame, prelayer: dict, config: dict, attn_meta: dict | None = None) -> pd.DataFrame:
    """Build one next-week feature row per observed provider×day."""
    obs = obs.copy()
    if obs.empty:
        return pd.DataFrame()

    t_star = _get_t_star(prelayer)
    next_week = int(obs['week_index'].max()) + 1
    next_phase = _phase_for_week(prelayer, next_week)
    wq_case = float((attn_meta or {}).get('wq_case', DEFAULT_WQ))
    wq_turn = float((attn_meta or {}).get('wq_turn', DEFAULT_WQ))

    sl_latest = _latest_service_line_context(obs)
    exc_by_provider = _exception_rate_by_provider(prelayer)
    rows: list[dict] = []

    for (pid, dow), g in obs.groupby(['provider_id', 'day_of_week'], dropna=False):
        g = g.sort_values('week_index').reset_index(drop=True)
        same_phase = g[g['rotation_phase'].astype(int) == int(next_phase)]
        hist = same_phase if not same_phase.empty else g

        case_hist = pd.Series(hist['casetime_util'].values, dtype=float)
        turn_hist = pd.Series(hist['turnover_util'].values, dtype=float)
        latest = g.iloc[-1]

        f = {
            'provider_id': str(pid),
            'service_line': latest.get('service_line', 'Unknown'),
            'week_start': 'FORECAST_NEXT_WEEK',
            'week_index': next_week,
            'forecast_week_index': next_week,
            'rotation_phase': int(next_phase),
            'day_of_week': int(dow),
            'allocated_min': float(hist['allocated_min'].mean()) if 'allocated_min' in hist.columns else 0.0,
            'early_release_min': float(hist.get('early_release_min', pd.Series([0.0])).mean()),
            'n_blocks': int(round(float(hist.get('n_blocks', pd.Series([0])).mean()))),
            'casetime_min': np.nan,
            'turnover_min': np.nan,
            'casetime_util': np.nan,
            'turnover_util': np.nan,
            'total_util': np.nan,
            'source_mode': 'forecast_next_week',
            'is_included': True,
        }
        _add_history_features(f, case_hist, turn_hist, wq_case, wq_turn)
        _add_calendar_features(f, next_week, int(dow), int(next_phase), t_star)
        rows.append(f)

    forecast = pd.DataFrame(rows)
    if forecast.empty:
        return forecast

    forecast = forecast.merge(sl_latest, on=['service_line', 'day_of_week'], how='left')
    forecast = forecast.merge(exc_by_provider, on='provider_id', how='left')
    forecast['service_line_wk_mean'] = forecast['service_line_wk_mean'].fillna(0.0)
    forecast['exception_rate_series'] = forecast['exception_rate_series'].fillna(0.0)
    forecast['provider_exception_rate'] = forecast['exception_rate_series']
    return forecast.sort_values(['provider_id', 'day_of_week']).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(ID_COLUMNS) | set(TARGET_COLUMNS) | set(LEAKAGE_COLUMNS)
    exclude |= {'is_included', 'forecast_week_index'}
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

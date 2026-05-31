from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


DAY_TO_INT = {
    'monday': 0,
    'mon': 0,
    'tuesday': 1,
    'tue': 1,
    'tues': 1,
    'wednesday': 2,
    'wed': 2,
    'thursday': 3,
    'thu': 3,
    'thur': 3,
    'thurs': 3,
    'friday': 4,
    'fri': 4,
    'saturday': 5,
    'sat': 5,
    'sunday': 6,
    'sun': 6,
}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column by exact name, case-insensitive name, or json_normalize suffix."""
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    lower_to_real = {str(c).lower(): c for c in cols}

    for cand in candidates:
        key = cand.lower()
        if key in lower_to_real:
            return lower_to_real[key]

    for cand in candidates:
        key = cand.lower()
        for c in cols:
            name = str(c).lower()
            if name.endswith('.' + key) or name.endswith('_' + key):
                return c

    return None


def _normalize_day_value(x: Any) -> int | float:
    """Convert Monday/Tuesday/... or 0..6 or 1..7 into Python weekday 0..6."""
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, np.integer)):
        v = int(x)
        if 0 <= v <= 6:
            return v
        if 1 <= v <= 7:
            return v - 1
        return np.nan

    s = str(x).strip()
    if s == '':
        return np.nan

    try:
        v = int(float(s))
        if 0 <= v <= 6:
            return v
        if 1 <= v <= 7:
            return v - 1
    except ValueError:
        pass

    return DAY_TO_INT.get(s.lower(), np.nan)


def _provider_service_map(providers_df: pd.DataFrame) -> dict:
    if providers_df is None or providers_df.empty or 'provider_id' not in providers_df.columns:
        return {}

    if 'primary_service_line' in providers_df.columns:
        service_col = 'primary_service_line'
    elif 'service_line' in providers_df.columns:
        service_col = 'service_line'
    elif 'service_lines' in providers_df.columns:
        service_col = 'service_lines'
    else:
        return {}

    out = {}
    for _, r in providers_df.iterrows():
        pid = str(r.get('provider_id', '')).strip()
        val = r.get(service_col, 'Unknown')
        if isinstance(val, list):
            val = val[0] if val else 'Unknown'
        if not pid:
            continue
        out[pid] = str(val) if pd.notna(val) else 'Unknown'
    return out


def _standardize_cases(cases_df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """
    Standardize the cases table into:
      provider_id, block_case_minutes, turnover_time, day_of_week(optional), plus original cols.

    Current cases JSON:
      case_id, provider_id, turnover_time, block_case_minutes, day_of_week
    """

    if cases_df is None or cases_df.empty:
        return pd.DataFrame()

    cases = cases_df.copy()

    provider_col = _find_col(cases, [
        'provider_id',
        'provider.id',
        'surgeon_id',
        'primary_surgeon_id',
        'case_provider_id',
        'performing_provider_id',
        'current_blockholder.provider_id',
    ])
    case_col = _find_col(cases, [
        'block_case_minutes',
        'case_minutes',
        'casetime_min',
        'case_time_minutes',
        'surgery_minutes',
        'procedure_minutes',
        'duration_minutes',
        'duration_min',
    ])
    turn_col = _find_col(cases, [
        'turnover_time',
        'turnover_minutes',
        'turnover_min',
        'room_turnover_minutes',
    ])
    day_col = _find_col(cases, [
        'day_of_week',
        'dow',
        'weekday',
    ])

    missing = []
    if provider_col is None:
        missing.append('provider_id')
    if case_col is None:
        missing.append('block_case_minutes')
    if turn_col is None:
        missing.append('turnover_time')

    if missing:
        warnings.append(
            'Cases JSON is missing required case-demand columns: '
            + ', '.join(missing)
            + '. Layer 1 will fall back to proxy demand from allocated block minutes.'
        )
        return pd.DataFrame()

    rename = {
        provider_col: 'provider_id',
        case_col: 'block_case_minutes',
        turn_col: 'turnover_time',
    }
    if day_col is not None and day_col != 'day_of_week':
        rename[day_col] = 'day_of_week'

    cases = cases.rename(columns=rename)
    cases['provider_id'] = cases['provider_id'].astype(str).str.strip()
    cases['block_case_minutes'] = pd.to_numeric(cases['block_case_minutes'], errors='coerce').fillna(0.0)
    cases['turnover_time'] = pd.to_numeric(cases['turnover_time'], errors='coerce').fillna(0.0)

    blank_mask = cases['provider_id'].isin(['', 'None', 'nan', 'NaN', 'NULL', 'null'])
    n_blank = int(blank_mask.sum())

    if n_blank > 0:
        blank_case_min = float(cases.loc[blank_mask, 'block_case_minutes'].sum())
        blank_turn_min = float(cases.loc[blank_mask, 'turnover_time'].sum())
        warnings.append(
            f'Dropped {n_blank} case rows with blank provider_id '
            f'({blank_case_min:.1f} case minutes, {blank_turn_min:.1f} turnover minutes). '
            'They cannot be assigned to a provider-day forecast without a provider ID.'
        )
        cases = cases.loc[~blank_mask].copy()

    if 'day_of_week' in cases.columns:
        cases['day_of_week'] = cases['day_of_week'].map(_normalize_day_value)
        bad_day = cases['day_of_week'].isna()

        if bad_day.any():
            warnings.append(
                f'Dropped {int(bad_day.sum())} case rows with missing/unreadable day_of_week.'
            )
            cases = cases.loc[~bad_day].copy()

        cases['day_of_week'] = cases['day_of_week'].astype(int)

    return cases.reset_index(drop=True)


def _prepare_blocks(
    blocks_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    prelayer: Dict[str, Any],
) -> pd.DataFrame:
    service_map = _provider_service_map(providers_df)
    phase_labels = prelayer.get('phase_labels', {}) or {}

    b = blocks_df.copy()

    if 'manual_early_release_min' not in b.columns:
        b['manual_early_release_min'] = 0.0
    if 'duration_min' not in b.columns:
        b['duration_min'] = 0.0
    if 'provider_id' not in b.columns:
        b['provider_id'] = 'OPEN'
    if 'block_historical_id' not in b.columns:
        b['block_historical_id'] = [f'BLOCK_ROW_{i}' for i in range(len(b))]

    b['provider_id'] = b['provider_id'].astype(str).str.strip()
    b.loc[b['provider_id'].isin(['', 'None', 'nan', 'NaN', 'NULL', 'null']), 'provider_id'] = 'OPEN'

    b['rotation_phase'] = (
        b['week_start'].map(lambda w: phase_labels.get(str(w), 0))
        .fillna(0)
        .astype(int)
    )

    block_service = b['service_line_from_block'] if 'service_line_from_block' in b.columns else 'Unknown'

    b['service_line'] = (
        b['provider_id'].map(service_map)
        .fillna(block_service)
        .fillna('Unknown')
    )

    b['duration_min'] = pd.to_numeric(b['duration_min'], errors='coerce').fillna(0.0)
    b['manual_early_release_min'] = pd.to_numeric(b['manual_early_release_min'], errors='coerce').fillna(0.0)
    b['allocated_min'] = (b['duration_min'] - b['manual_early_release_min']).clip(lower=0.0)

    return b


def _build_block_agg(b: pd.DataFrame) -> pd.DataFrame:
    block_agg = (
        b.groupby(
            ['provider_id', 'service_line', 'week_start', 'week_index', 'rotation_phase', 'day_of_week'],
            dropna=False,
        )
        .agg(
            allocated_min=('allocated_min', 'sum'),
            early_release_min=('manual_early_release_min', 'sum'),
            n_blocks=('block_historical_id', 'count'),
        )
        .reset_index()
    )

    block_agg = block_agg[block_agg['provider_id'] != 'OPEN'].copy()
    return block_agg


def _distribute_case_totals_over_rows(
    target_rows: pd.DataFrame,
    total_case_col: str = 'total_casetime_min',
    total_turn_col: str = 'total_turnover_min',
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Spread provider/day totals over target rows using allocated minutes as weights."""

    if group_cols is None:
        group_cols = ['provider_id', 'day_of_week']

    out = target_rows.copy()

    group_alloc = out.groupby(group_cols, dropna=False)['allocated_min'].transform('sum')
    group_size = out.groupby(group_cols, dropna=False)['provider_id'].transform('size').replace(0, 1)

    weights = np.where(
        group_alloc > 0,
        out['allocated_min'] / group_alloc,
        1.0 / group_size,
    )

    out['casetime_min'] = out[total_case_col].fillna(0.0) * weights
    out['turnover_min'] = out[total_turn_col].fillna(0.0) * weights

    return out


def _provider_day_distribution_fallback(
    cases: pd.DataFrame,
    b: pd.DataFrame,
    providers_df: pd.DataFrame,
    warnings: list[str],
    proxy_util: float,
) -> pd.DataFrame:
    """
    Main fix for your current cases JSON.

    Cases have provider_id + day_of_week + block_case_minutes + turnover_time,
    but no block_historical_id and no timestamp.

    Therefore exact case-to-block matching is impossible. The approximation is:
      1. aggregate cases by provider_id + day_of_week;
      2. find the provider's historical blocks on the same day_of_week;
      3. distribute the total case minutes across those historical block weeks using
         allocated block minutes as weights.
    """

    if 'day_of_week' not in cases.columns:
        warnings.append(
            'Cases JSON has provider_id/case minutes/turnover but no day_of_week, block id, or timestamp. '
            'Layer 1 distributed provider-level totals across all historical blocks for each provider.'
        )

        provider_totals = (
            cases.groupby('provider_id', dropna=False)
            .agg(
                total_casetime_min=('block_case_minutes', 'sum'),
                total_turnover_min=('turnover_time', 'sum'),
                n_cases=('provider_id', 'size'),
            )
            .reset_index()
        )

        block_agg = _build_block_agg(b)
        obs = block_agg.merge(provider_totals, on='provider_id', how='inner')
        obs = _distribute_case_totals_over_rows(
            obs,
            group_cols=['provider_id'],
        )
        obs['source_mode'] = 'provider_totals_distributed_to_provider_blocks'

        return obs.drop(columns=['total_casetime_min', 'total_turnover_min', 'n_cases'], errors='ignore')

    warnings.append(
        'Cases JSON has provider_id + day_of_week + case minutes + turnover, but no block id or case timestamp. '
        'Layer 1 is using provider-day proxy mode: case demand is aggregated by provider/day and distributed '
        'across matching historical provider/day block weeks by allocated block minutes. '
        'For exact phase-aligned demand, add block_historical_id or case_start/surgery_start to cases JSON.'
    )

    case_day_totals = (
        cases.groupby(['provider_id', 'day_of_week'], dropna=False)
        .agg(
            total_casetime_min=('block_case_minutes', 'sum'),
            total_turnover_min=('turnover_time', 'sum'),
            n_cases=('provider_id', 'size'),
        )
        .reset_index()
    )

    block_agg = _build_block_agg(b)

    matched = block_agg.merge(
        case_day_totals,
        on=['provider_id', 'day_of_week'],
        how='inner',
    )

    matched = _distribute_case_totals_over_rows(
        matched,
        group_cols=['provider_id', 'day_of_week'],
    )
    matched['source_mode'] = 'provider_day_cases_distributed_to_matching_blocks'

    matched_keys = set(zip(matched['provider_id'].astype(str), matched['day_of_week'].astype(int))) if not matched.empty else set()
    all_keys = set(zip(case_day_totals['provider_id'].astype(str), case_day_totals['day_of_week'].astype(int)))
    missing_keys = all_keys - matched_keys

    extras = []

    if missing_keys:
        service_map = _provider_service_map(providers_df)

        missing = case_day_totals[
            case_day_totals.apply(
                lambda r: (str(r['provider_id']), int(r['day_of_week'])) in missing_keys,
                axis=1,
            )
        ].copy()

        provider_weeks = (
            block_agg.groupby(['provider_id', 'week_start', 'week_index', 'rotation_phase'], dropna=False)
            .agg(
                allocated_min=('allocated_min', 'sum'),
                early_release_min=('early_release_min', 'sum'),
                n_blocks=('n_blocks', 'sum'),
                service_line=('service_line', 'first'),
            )
            .reset_index()
        )

        for _, r in missing.iterrows():
            pid = str(r['provider_id'])
            dow = int(r['day_of_week'])
            total_case = float(r['total_casetime_min'])
            total_turn = float(r['total_turnover_min'])
            n_cases = int(r['n_cases'])

            pw = provider_weeks[provider_weeks['provider_id'].astype(str) == pid].copy()

            if not pw.empty:
                tmp = pw.copy()
                tmp['day_of_week'] = dow
                tmp['total_casetime_min'] = total_case
                tmp['total_turnover_min'] = total_turn
                tmp['n_cases'] = n_cases

                tmp = _distribute_case_totals_over_rows(
                    tmp,
                    group_cols=['provider_id', 'day_of_week'],
                )
                tmp['source_mode'] = 'provider_day_cases_distributed_to_provider_weeks_no_same_day_block'
                extras.append(tmp)

            else:
                alloc = max(total_case / max(proxy_util, 1e-6), total_case + total_turn, 1.0)

                extras.append(pd.DataFrame([{
                    'provider_id': pid,
                    'service_line': service_map.get(pid, 'Unknown'),
                    'week_start': 'CASE_ONLY',
                    'week_index': 0,
                    'rotation_phase': 0,
                    'day_of_week': dow,
                    'allocated_min': alloc,
                    'early_release_min': 0.0,
                    'n_blocks': 0,
                    'total_casetime_min': total_case,
                    'total_turnover_min': total_turn,
                    'n_cases': n_cases,
                    'casetime_min': total_case,
                    'turnover_min': total_turn,
                    'source_mode': 'case_only_provider_not_found_in_blocks',
                }]))

        warnings.append(
            f'{len(missing_keys)} provider/day case groups had no exact matching historical block. '
            'They were kept using provider-week or synthetic fallback rows.'
        )

    pieces = [matched]

    if extras:
        pieces.extend(extras)

    obs = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    obs = obs.drop(columns=['total_casetime_min', 'total_turnover_min', 'n_cases'], errors='ignore')

    return obs


def _case_join_by_block_id(
    cases: pd.DataFrame,
    b: pd.DataFrame,
    block_id_col: str,
    warnings: list[str],
) -> pd.DataFrame:
    case_join = cases.merge(
        b[[
            'block_historical_id',
            'week_start',
            'week_index',
            'day_of_week',
            'rotation_phase',
            'service_line',
        ]],
        left_on=block_id_col,
        right_on='block_historical_id',
        how='left',
    )

    n_missing = int(case_join['week_start'].isna().sum())

    if n_missing > 0:
        warnings.append(
            f'{n_missing} case rows had a block id that did not match geisinger-users_blocks.json; '
            'they were dropped from exact block-id aggregation.'
        )
        case_join = case_join.dropna(subset=['week_start']).copy()

    if case_join.empty:
        return pd.DataFrame()

    return (
        case_join.groupby(
            ['provider_id', 'service_line', 'week_start', 'week_index', 'rotation_phase', 'day_of_week'],
            dropna=False,
        )
        .agg(
            casetime_min=('block_case_minutes', 'sum'),
            turnover_min=('turnover_time', 'sum'),
        )
        .reset_index()
    )


def _case_join_by_timestamp(
    cases: pd.DataFrame,
    b: pd.DataFrame,
    providers_df: pd.DataFrame,
    dt_col: str,
    prelayer: Dict[str, Any],
    warnings: list[str],
) -> pd.DataFrame:
    service_map = _provider_service_map(providers_df)
    phase_labels = prelayer.get('phase_labels', {}) or {}

    cj = cases.copy()
    cj[dt_col] = pd.to_datetime(cj[dt_col], utc=True, errors='coerce')

    bad = cj[dt_col].isna()

    if bad.any():
        warnings.append(f'Dropped {int(bad.sum())} cases with unreadable timestamp column {dt_col}.')
        cj = cj.loc[~bad].copy()

    if cj.empty:
        return pd.DataFrame()

    cj['week_start'] = (cj[dt_col] - pd.to_timedelta(cj[dt_col].dt.weekday, unit='D')).dt.date.astype(str)

    week_order = {w: i for i, w in enumerate(sorted(b['week_start'].dropna().unique()))}
    cj['week_index'] = cj['week_start'].map(week_order)

    missing_week_idx = cj['week_index'].isna()

    if missing_week_idx.any():
        start = int(max(week_order.values())) + 1 if week_order else 0
        new_weeks = sorted(cj.loc[missing_week_idx, 'week_start'].dropna().unique())
        new_map = {w: start + i for i, w in enumerate(new_weeks)}
        cj.loc[missing_week_idx, 'week_index'] = cj.loc[missing_week_idx, 'week_start'].map(new_map)

    cj['week_index'] = cj['week_index'].fillna(0).astype(int)
    cj['day_of_week'] = cj[dt_col].dt.weekday.astype(int)
    cj['rotation_phase'] = cj['week_start'].map(lambda w: phase_labels.get(str(w), 0)).fillna(0).astype(int)
    cj['service_line'] = cj['provider_id'].map(service_map).fillna('Unknown')

    return (
        cj.groupby(
            ['provider_id', 'service_line', 'week_start', 'week_index', 'rotation_phase', 'day_of_week'],
            dropna=False,
        )
        .agg(
            casetime_min=('block_case_minutes', 'sum'),
            turnover_min=('turnover_time', 'sum'),
        )
        .reset_index()
    )


def build_weekly_observations(
    blocks_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    prelayer: Dict[str, Any],
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    warnings: list[str] = []

    proxy_util = float(config.get('layer1', {}).get('proxy_case_utilization', 0.62))
    proxy_turn = float(config.get('layer1', {}).get('proxy_turnover_fraction', 0.08))

    b = _prepare_blocks(blocks_df, providers_df, prelayer)
    block_agg = _build_block_agg(b)

    cases = _standardize_cases(cases_df, warnings)

    if cases.empty:
        warnings.append(
            'No usable provider-level cases remained after standardization. '
            'Layer 1 is running in proxy mode using allocated block minutes.'
        )

        obs = block_agg.copy()
        obs['casetime_min'] = obs['allocated_min'] * proxy_util
        obs['turnover_min'] = obs['casetime_min'] * proxy_turn
        obs['source_mode'] = 'proxy_from_allocated_blocks'

    else:
        block_id_col = _find_col(cases, ['block_historical_id', 'block_id', 'historical_block_id'])
        dt_col = _find_col(cases, [
            'case_start',
            'surgery_start',
            'procedure_start',
            'scheduled_start',
            'actual_start',
            'start',
            'occurrence.start',
        ])

        case_agg = pd.DataFrame()
        source_mode = ''

        if block_id_col is not None:
            case_agg = _case_join_by_block_id(cases, b, block_id_col, warnings)

            if not case_agg.empty:
                source_mode = 'cases_joined_by_block_id'
            elif dt_col is not None:
                warnings.append('Block-id case join produced no rows; falling back to timestamp mapping.')

        if case_agg.empty and dt_col is not None:
            case_agg = _case_join_by_timestamp(cases, b, providers_df, dt_col, prelayer, warnings)

            if not case_agg.empty:
                source_mode = 'cases_mapped_by_timestamp'

        if case_agg.empty:
            obs = _provider_day_distribution_fallback(
                cases=cases,
                b=b,
                providers_df=providers_df,
                warnings=warnings,
                proxy_util=proxy_util,
            )

        else:
            obs = case_agg.merge(
                block_agg,
                on=['provider_id', 'service_line', 'week_start', 'week_index', 'rotation_phase', 'day_of_week'],
                how='left',
            )
            obs['allocated_min'] = obs['allocated_min'].fillna(0.0)
            obs['early_release_min'] = obs['early_release_min'].fillna(0.0)
            obs['n_blocks'] = obs['n_blocks'].fillna(0).astype(int)
            obs['source_mode'] = source_mode

    if obs.empty:
        raise ValueError(
            'Layer 1 could not build any weekly observations. Check that blocks and cases contain provider_id values.'
        )

    for col in ['allocated_min', 'early_release_min', 'casetime_min', 'turnover_min']:
        if col not in obs.columns:
            obs[col] = 0.0
        obs[col] = pd.to_numeric(obs[col], errors='coerce').fillna(0.0)

    if 'n_blocks' not in obs.columns:
        obs['n_blocks'] = 0

    obs['n_blocks'] = pd.to_numeric(obs['n_blocks'], errors='coerce').fillna(0).astype(int)

    if 'service_line' not in obs.columns:
        obs['service_line'] = 'Unknown'

    obs['service_line'] = obs['service_line'].fillna('Unknown').astype(str)

    obs['provider_id'] = obs['provider_id'].astype(str)
    obs['week_index'] = pd.to_numeric(obs['week_index'], errors='coerce').fillna(0).astype(int)
    obs['rotation_phase'] = pd.to_numeric(obs['rotation_phase'], errors='coerce').fillna(0).astype(int)
    obs['day_of_week'] = pd.to_numeric(obs['day_of_week'], errors='coerce').fillna(-1).astype(int)

    obs['casetime_util'] = np.where(
        obs['allocated_min'] > 0,
        obs['casetime_min'] / obs['allocated_min'],
        0.0,
    )
    obs['turnover_util'] = np.where(
        obs['allocated_min'] > 0,
        obs['turnover_min'] / obs['allocated_min'],
        0.0,
    )

    obs['casetime_util'] = obs['casetime_util'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    obs['turnover_util'] = obs['turnover_util'].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    obs['is_included'] = True

    return obs.sort_values(['provider_id', 'day_of_week', 'week_index']).reset_index(drop=True), {'warnings': warnings}


def early_release_projected(blocks_df: pd.DataFrame, optimization_weeks: int = 13) -> pd.DataFrame:
    hist_weeks = max(1, blocks_df['week_index'].nunique()) if 'week_index' in blocks_df.columns else 1
    alpha = optimization_weeks / hist_weeks

    b = blocks_df.copy()

    if 'manual_early_release_min' not in b.columns:
        b['manual_early_release_min'] = 0.0
    if 'provider_id' not in b.columns:
        b['provider_id'] = 'OPEN'

    out = b.groupby('provider_id', dropna=False)['manual_early_release_min'].sum().reset_index()
    out['early_release_projected_min'] = out['manual_early_release_min'] * alpha

    return out[['provider_id', 'early_release_projected_min']]
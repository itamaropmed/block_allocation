from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.layer1.feature_engineering import feature_columns

REASON_CODES = {
    'attn_weighted_util': ('utilization_history', 'Provider characteristic operating level'),
    'attn_entropy': ('pattern_consistency', 'How stable the historical pattern is'),
    'attn_recency_bias': ('trend_direction', 'Whether demand is improving or declining'),
    'attn_turn_weighted': ('turnover_history', 'Characteristic turnover load'),
    'service_line_wk_mean': ('service_line_context', 'Peer service-line context'),
    'provider_exception_rate': ('series_contention', 'How contested the provider block series is'),
    'rotation_phase': ('rotation_cycle_position', 'Position in the detected rotation cycle'),
    'weeks_observed': ('data_maturity', 'Amount of history available'),
    'sin_woy': ('seasonality', 'Time-of-year effect'),
    'cos_woy': ('seasonality', 'Time-of-year effect'),
}


def _model_importances(model, cols):
    base = model
    if hasattr(model, 'named_steps'):
        base = model.named_steps.get('model', model)
    if hasattr(base, 'feature_importances_'):
        vals = np.asarray(base.feature_importances_, dtype=float)
    else:
        vals = np.ones(len(cols), dtype=float) / max(1, len(cols))
    vals = vals / max(1e-9, vals.sum())
    return pd.DataFrame({'feature': cols, 'importance': vals}).sort_values('importance', ascending=False)


def feature_attribution(layer1_dir: str | Path, out_dir: str | Path, top_k: int = 10) -> pd.DataFrame:
    l1 = Path(layer1_dir)
    out = Path(out_dir)
    features = pd.read_csv(l1 / 'features.csv')
    cols = feature_columns(features)
    rows = []
    for target in ['casetime_min', 'turnover_min']:
        model = joblib.load(l1 / f'{target}_model.joblib')
        imp = _model_importances(model, cols).head(top_k)
        for _, r in imp.iterrows():
            reason, plain = REASON_CODES.get(r['feature'], ('model_feature', r['feature']))
            rows.append({'target': target, 'feature': r['feature'], 'importance': float(r['importance']), 'reason_code': reason, 'plain_language': plain})
    df = pd.DataFrame(rows)
    df.to_csv(out / 'feature_attribution.csv', index=False)
    return df


def confidence_bands(layer1_dir: str | Path, out_dir: str | Path) -> pd.DataFrame:
    scenarios = pd.read_csv(Path(layer1_dir) / 'scenarios_long.csv')
    bands = scenarios.groupby(['provider_id', 'service_line', 'day_of_week']).agg(
        casetime_p10=('demand_casetime_min', lambda x: float(np.quantile(x, 0.10))),
        casetime_p50=('demand_casetime_min', lambda x: float(np.quantile(x, 0.50))),
        casetime_p90=('demand_casetime_min', lambda x: float(np.quantile(x, 0.90))),
        turnover_p10=('demand_turnover_min', lambda x: float(np.quantile(x, 0.10))),
        turnover_p50=('demand_turnover_min', lambda x: float(np.quantile(x, 0.50))),
        turnover_p90=('demand_turnover_min', lambda x: float(np.quantile(x, 0.90))),
    ).reset_index()
    bands.to_csv(Path(out_dir) / 'confidence_bands.csv', index=False)
    return bands


def goal_attribution(layer2_dir: str | Path, out_dir: str | Path) -> pd.DataFrame:
    l2 = Path(layer2_dir)
    rec = json.loads((l2 / 'recommended_candidate.json').read_text(encoding='utf-8'))
    cov_path = l2 / f"coverage_{rec['theme']}.csv"
    coverage = pd.read_csv(cov_path)
    coverage['binding_factor'] = np.where(coverage['meets_alpha'], 'covered', 'insufficient_allocated_minutes')
    coverage['gap_min'] = (coverage['required_min_alpha'] - coverage['allocated_min']).clip(lower=0)
    coverage.to_csv(Path(out_dir) / 'goal_attribution.csv', index=False)
    return coverage


def counterfactual_targets(layer1_dir: str | Path, out_dir: str | Path, targets=(0.6, 0.7, 0.8), optimization_weeks: int = 13) -> pd.DataFrame:
    scenarios = pd.read_csv(Path(layer1_dir) / 'scenarios_long.csv')
    rows = []
    q = scenarios.groupby(['provider_id', 'scenario_id']).agg(total=('demand_casetime_min', 'sum'), turn=('demand_turnover_min', 'sum')).reset_index()
    q['quarterly_total_min'] = (q['total'] + q['turn']) * optimization_weeks
    for target in targets:
        req = q.copy()
        req['required_min'] = req['quarterly_total_min'] / max(1e-6, target)
        s = req.groupby('provider_id')['required_min'].quantile(0.85).reset_index()
        s['target_utilization'] = target
        rows.extend(s.to_dict(orient='records'))
    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / 'counterfactual_target_curve.csv', index=False)
    return out


def write_narrative(out_dir: str | Path, attribution: pd.DataFrame, bands: pd.DataFrame, goals: pd.DataFrame) -> Path:
    out = Path(out_dir)
    top = attribution.head(8)
    worst = goals.sort_values('gap_min', ascending=False).head(8) if not goals.empty else pd.DataFrame()
    lines = []
    lines.append('# Layer 3 Explanation Report')
    lines.append('')
    lines.append('## Executive summary')
    lines.append('This report connects the demand forecast, scenario uncertainty, allocation candidate, and goal coverage into an auditable explanation package.')
    lines.append('')
    lines.append('## Top model drivers')
    for _, r in top.iterrows():
        lines.append(f"- **{r['target']} / {r['feature']}** -> `{r['reason_code']}`: {r['plain_language']} (importance={r['importance']:.3f}).")
    lines.append('')
    lines.append('## Largest allocation gaps')
    if worst.empty:
        lines.append('No uncovered goal providers were found.')
    else:
        for _, r in worst.iterrows():
            lines.append(f"- Provider `{r['provider_id']}`: allocated {r['allocated_min']:.1f} min vs required {r['required_min_alpha']:.1f} min; gap {r['gap_min']:.1f} min; factor `{r['binding_factor']}`.")
    lines.append('')
    lines.append('## Confidence bands')
    lines.append(f'Generated scenario bands for {bands["provider_id"].nunique() if not bands.empty else 0} providers.')
    path = out / 'explanation_report.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path

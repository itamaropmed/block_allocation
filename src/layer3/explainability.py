"""
Layer 3 — Explainability  (standalone, no src imports)
=======================================================
Fixes vs original:
  1. Removed  from src.layer1.feature_engineering import feature_columns
     Inlined _feature_columns() directly (Layer 3 lives in a different
     folder than Layer 1, so a relative import would not work).
  2. goal_attribution() — binding_factor now has 4 meaningful categories
     (covered / case_volume / target_level / supply_shortage) instead
     of just 'covered' vs 'insufficient_allocated_minutes'.
  3. write_narrative() — guarded all column accesses so it never crashes
     when an upstream file has an unexpected shape.
  4. counterfactual_targets() — added optional early_release_projected_min
     term so the curve matches the Layer 2 R[p,s] formula exactly.
  5. provider_risk_profiles() — new function that classifies providers into
     the 2×2 risk quadrant (demand uncertainty × outcome uncertainty).
  6. drift_detection() — new function: flags providers whose recent
     trailing_4w_std diverged from their 12w baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Inlined from feature_engineering.py (Layer 1 lives in a different folder)
# ─────────────────────────────────────────────────────────────────────────────

def _feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        'provider_id', 'service_line', 'week_start', 'source_mode',
        'casetime_min', 'turnover_min', 'casetime_util', 'turnover_util',
    }
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Reason code registry
# ─────────────────────────────────────────────────────────────────────────────

REASON_CODES: dict[str, tuple[str, str]] = {
    'attn_weighted_util':    ('utilization_history',     'Provider characteristic operating level'),
    'attn_entropy':          ('pattern_consistency',     'How stable the historical pattern is'),
    'attn_recency_bias':     ('trend_direction',         'Whether demand is improving or declining'),
    'attn_turn_weighted':    ('turnover_history',        'Characteristic turnover load'),
    'attn_turn_entropy':     ('turnover_consistency',    'How stable the turnover pattern is'),
    'attn_turn_recency':     ('turnover_trend',          'Whether turnover is growing or shrinking'),
    'service_line_wk_mean':  ('service_line_context',   'Peer service-line context this week'),
    'provider_exception_rate':('series_contention',     'How contested the provider block series is'),
    'rotation_phase':        ('rotation_cycle_position','Position in the detected rotation cycle'),
    'weeks_observed':        ('data_maturity',           'Amount of history available for this provider'),
    'sin_woy':               ('seasonality',             'Time-of-year seasonal effect'),
    'cos_woy':               ('seasonality',             'Time-of-year seasonal effect'),
    'sin_woy_2':             ('seasonality',             'Semi-annual seasonal effect'),
    'cos_woy_2':             ('seasonality',             'Semi-annual seasonal effect'),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_importances(model, cols: list[str]) -> pd.DataFrame:
    """Extract feature importance array from any sklearn-compatible model."""
    base = model
    if hasattr(model, 'named_steps'):
        base = model.named_steps.get('model', model)
    if hasattr(base, 'feature_importances_'):
        vals = np.asarray(base.feature_importances_, dtype=float)
    elif hasattr(base, 'coef_'):
        vals = np.abs(np.asarray(base.coef_, dtype=float).ravel())
    else:
        vals = np.ones(len(cols), dtype=float)
    s = vals.sum()
    vals = vals / max(s, 1e-9)
    return (
        pd.DataFrame({'feature': cols, 'importance': vals})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )


def _binding_factor(row: pd.Series) -> str:
    """
    Four-category binding factor diagnosis.

    covered               — α ≥ 0.85, goal met
    supply_shortage       — slack_min > 0, no feasible block freed
    case_volume           — allocated < 50% of required (demand too low)
    target_level          — 50–100% of required covered, but threshold not met
    """
    if bool(row.get('meets_alpha', False)):
        return 'covered'
    slack = float(row.get('slack_min', 0.0))
    if slack > 0.01:
        return 'supply_shortage'
    ratio = float(row.get('coverage_ratio', 0.0))
    if pd.isna(ratio) or ratio < 0.50:
        return 'case_volume'
    return 'target_level'


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature attribution (SHAP-style importance from model)
# ─────────────────────────────────────────────────────────────────────────────

def feature_attribution(
    layer1_dir, out_dir, top_k: int = 10
) -> pd.DataFrame:
    """
    Load the trained joblib models from Layer 1, extract feature importances,
    map to plain-language reason codes, save feature_attribution.csv.
    """
    l1  = Path(layer1_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    features_path = l1 / 'features.csv'
    if not features_path.exists():
        raise FileNotFoundError(f'features.csv not found in Layer 1 output: {l1}')

    features = pd.read_csv(features_path)
    cols     = _feature_columns(features)
    if not cols:
        raise ValueError('No numeric feature columns found in features.csv')

    rows = []
    for target in ['casetime_min', 'turnover_min']:
        model_path = l1 / f'{target}_model.joblib'
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        imp   = _model_importances(model, cols).head(top_k)
        for _, r in imp.iterrows():
            reason, plain = REASON_CODES.get(r['feature'], ('model_feature', r['feature']))
            rows.append({
                'target':        target,
                'feature':       r['feature'],
                'importance':    float(r['importance']),
                'reason_code':   reason,
                'plain_language': plain,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / 'feature_attribution.csv', index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confidence bands
# ─────────────────────────────────────────────────────────────────────────────

def confidence_bands(layer1_dir, out_dir) -> pd.DataFrame:
    """
    Compute P10/P50/P90 scenario bands per provider × day.
    Adds band_width_pp and risk flags.
    """
    out       = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = pd.read_csv(Path(layer1_dir) / 'scenarios_long.csv')

    group_cols = ['provider_id', 'day_of_week']
    if 'service_line' in scenarios.columns:
        group_cols = ['provider_id', 'service_line', 'day_of_week']

    bands = (
        scenarios.groupby(group_cols)
        .agg(
            casetime_p10=('demand_casetime_min', lambda x: float(np.quantile(x, 0.10))),
            casetime_p25=('demand_casetime_min', lambda x: float(np.quantile(x, 0.25))),
            casetime_p50=('demand_casetime_min', lambda x: float(np.quantile(x, 0.50))),
            casetime_p75=('demand_casetime_min', lambda x: float(np.quantile(x, 0.75))),
            casetime_p90=('demand_casetime_min', lambda x: float(np.quantile(x, 0.90))),
            turnover_p10=('demand_turnover_min', lambda x: float(np.quantile(x, 0.10))),
            turnover_p50=('demand_turnover_min', lambda x: float(np.quantile(x, 0.50))),
            turnover_p90=('demand_turnover_min', lambda x: float(np.quantile(x, 0.90))),
            sigma_case=('demand_casetime_min', lambda x: float(np.std(x, ddof=0))),
            n_scenarios=('demand_casetime_min', 'count'),
        )
        .reset_index()
    )

    bands['band_width_pp'] = (
        (bands['casetime_p90'] - bands['casetime_p10'])
        / bands['casetime_p50'].clip(lower=1.0) * 100.0
    ).round(1)
    bands['underuse_risk_flag'] = bands['casetime_p50'] < 0.5
    bands['capacity_risk_flag'] = bands['casetime_p90'] > (
        bands['casetime_p50'].clip(lower=1.0) * 2.0
    )

    bands.to_csv(out / 'confidence_bands.csv', index=False)
    return bands


# ─────────────────────────────────────────────────────────────────────────────
# 3. Goal attribution
# ─────────────────────────────────────────────────────────────────────────────

def goal_attribution(layer2_dir, out_dir) -> pd.DataFrame:
    """
    Read the recommended candidate coverage CSV, compute binding factors
    and allocation gaps for every goal provider.
    """
    l2  = Path(layer2_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rec_path = l2 / 'recommended_candidate.json'
    if not rec_path.exists():
        raise FileNotFoundError(f'recommended_candidate.json not found in: {l2}')

    rec   = json.loads(rec_path.read_text(encoding='utf-8'))
    theme = rec.get('theme', 'balanced')

    # Try the recommended coverage file first, then the theme-specific one
    for cov_name in ['coverage_recommended.csv', f'coverage_{theme}.csv']:
        cov_path = l2 / cov_name
        if cov_path.exists():
            break
    else:
        raise FileNotFoundError(
            f'No coverage CSV found in {l2} for theme={theme}'
        )

    coverage = pd.read_csv(cov_path)

    # Normalise boolean column (may be stored as True/False or 1/0)
    coverage['meets_alpha'] = coverage['meets_alpha'].astype(str).str.lower().isin(
        ['true', '1', 'yes']
    )

    # Binding factor — 4 categories
    coverage['binding_factor'] = coverage.apply(_binding_factor, axis=1)

    # Gap in minutes
    coverage['gap_min'] = (
        coverage['required_min_alpha'] - coverage['allocated_min']
    ).clip(lower=0.0).round(1)

    # Useful derived metrics
    coverage['gap_hrs']              = (coverage['gap_min'] / 60.0).round(2)
    coverage['allocated_hrs']        = (coverage['allocated_min'] / 60.0).round(2)
    coverage['required_alpha_hrs']   = (coverage['required_min_alpha'] / 60.0).round(2)
    coverage['recommended_theme']    = theme

    # Enriched binding description
    factor_desc = {
        'covered':          'Goal fully met (α ≥ 85%)',
        'case_volume':      'Demand too low — fewer than 50% of required hours covered even with maximum feasible allocation',
        'target_level':     'Target utilization too high for current demand — consider lowering target',
        'supply_shortage':  'No compatible freed blocks available — requires unfreezing another provider',
    }
    coverage['binding_description'] = coverage['binding_factor'].map(factor_desc).fillna('unknown')

    coverage.to_csv(out / 'goal_attribution.csv', index=False)
    return coverage


# ─────────────────────────────────────────────────────────────────────────────
# 4. Counterfactual target curve
# ─────────────────────────────────────────────────────────────────────────────

def counterfactual_targets(
    layer1_dir,
    out_dir,
    targets=(0.6, 0.7, 0.8),
    optimization_weeks: int = 13,
    early_release_path=None,
) -> pd.DataFrame:
    """
    For each alternative target utilization, compute the α=0.85 percentile of
    BlockTimeRequired = Q_demand / target + EarlyRelease_projected.
    This matches the Layer 2 R[p,s] formula exactly.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenarios = pd.read_csv(Path(layer1_dir) / 'scenarios_long.csv')

    # Early release (optional — zero if file missing)
    er_map: dict[str, float] = {}
    er_file = early_release_path or Path(layer1_dir) / 'early_release_projected.csv'
    if Path(er_file).exists():
        er_df = pd.read_csv(er_file)
        if 'provider_id' in er_df.columns and 'early_release_projected_min' in er_df.columns:
            er_map = dict(zip(er_df['provider_id'].astype(str),
                              er_df['early_release_projected_min']))

    # Quarterly demand per scenario
    q = (
        scenarios.groupby(['provider_id', 'scenario_id'])
        .agg(q_case=('demand_casetime_min', 'sum'),
             q_turn=('demand_turnover_min', 'sum'))
        .reset_index()
    )
    q['quarterly_total_min'] = (q['q_case'] + q['q_turn']) * optimization_weeks

    rows = []
    for target in targets:
        req = q.copy()
        req['early_release_min']    = req['provider_id'].astype(str).map(er_map).fillna(0.0)
        req['required_min_per_scen'] = req['quarterly_total_min'] / max(target, 1e-6) + req['early_release_min']

        s = (
            req.groupby('provider_id')['required_min_per_scen']
            .quantile(0.85)
            .reset_index()
            .rename(columns={'required_min_per_scen': 'required_min_alpha85'})
        )
        s['target_utilization'] = target
        s['required_hrs_alpha85'] = (s['required_min_alpha85'] / 60.0).round(2)

        # Minimum block count needed (assuming 8-hr blocks)
        s['min_blocks_needed'] = np.ceil(s['required_min_alpha85'] / 480.0).astype(int)
        rows.extend(s.to_dict(orient='records'))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out / 'counterfactual_target_curve.csv', index=False)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Provider risk profiles  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def provider_risk_profiles(layer1_dir, out_dir) -> pd.DataFrame:
    """
    Classify each provider into the 2×2 risk quadrant:
      - demand uncertainty  : σ_case from hierarchical_sigma_estimates.csv
      - outcome uncertainty : band_width_pp from confidence_bands.csv

    Quadrants:
      predictable_performer   — low σ, narrow band
      variable_performer      — low σ, wide band
      uncertain_but_stable    — high σ, narrow band
      high_attention          — high σ, wide band  ← flag these to committee
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sigma_path = Path(layer1_dir) / 'hierarchical_sigma_estimates.csv'
    bands_path = out / 'confidence_bands.csv'

    if not sigma_path.exists() or not bands_path.exists():
        return pd.DataFrame()

    sigma = pd.read_csv(sigma_path)
    bands = pd.read_csv(bands_path)

    sigma_agg = (
        sigma.groupby('provider_id')
        .agg(sigma_case_mean=('sigma_case', 'mean'),
             sigma_turn_mean=('sigma_turn', 'mean'))
        .reset_index()
    )
    bands_agg = (
        bands.groupby('provider_id')
        .agg(band_width_pp_mean=('band_width_pp', 'mean'),
             underuse_flag=('underuse_risk_flag', 'any'),
             capacity_flag=('capacity_risk_flag', 'any'))
        .reset_index()
    )

    profiles = sigma_agg.merge(bands_agg, on='provider_id', how='outer')

    HIGH_SIGMA = 15.0
    WIDE_BAND  = 20.0

    def _quadrant(row):
        high_sigma = float(row.get('sigma_case_mean', 0)) >= HIGH_SIGMA
        wide_band  = float(row.get('band_width_pp_mean', 0)) >= WIDE_BAND
        if high_sigma and wide_band:
            return 'high_attention'
        if high_sigma:
            return 'uncertain_but_stable'
        if wide_band:
            return 'variable_performer'
        return 'predictable_performer'

    profiles['risk_quadrant'] = profiles.apply(_quadrant, axis=1)
    profiles.to_csv(out / 'provider_risk_profiles.csv', index=False)
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# 6. Drift detection  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def drift_detection(layer1_dir, out_dir, drift_threshold: float = 0.50) -> pd.DataFrame:
    """
    Flag providers where trailing_4w_std_case diverged more than
    `drift_threshold` (50%) from their trailing_12w_std_case.
    These are candidates for model recalibration next quarter.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    features_path = Path(layer1_dir) / 'features.csv'
    if not features_path.exists():
        return pd.DataFrame()

    features = pd.read_csv(features_path)
    needed   = {'trailing_4w_std_case', 'trailing_12w_std_case'}
    if not needed.issubset(set(features.columns)):
        return pd.DataFrame()

    agg = (
        features.groupby('provider_id')
        .agg(
            t4_std=('trailing_4w_std_case',  'mean'),
            t12_std=('trailing_12w_std_case', 'mean'),
            n_obs=('provider_id', 'size'),
        )
        .reset_index()
    )
    agg['drift_ratio'] = np.where(
        agg['t12_std'] > 0,
        (agg['t4_std'] - agg['t12_std']).abs() / agg['t12_std'],
        0.0,
    )
    agg['drift_flag']    = agg['drift_ratio'] > drift_threshold
    agg['drift_direction'] = np.where(
        agg['t4_std'] > agg['t12_std'], 'increasing', 'decreasing'
    )
    flagged = agg[agg['drift_flag']].sort_values('drift_ratio', ascending=False)
    flagged.to_csv(out / 'drift_flags.csv', index=False)
    return flagged


# ─────────────────────────────────────────────────────────────────────────────
# 7. Narrative report
# ─────────────────────────────────────────────────────────────────────────────

def write_narrative(
    out_dir,
    attribution: pd.DataFrame,
    bands: pd.DataFrame,
    goals: pd.DataFrame,
    risk_profiles: pd.DataFrame | None = None,
    drift_flags: pd.DataFrame | None = None,
) -> Path:
    out   = Path(out_dir)
    lines = ['# Layer 3 — Explanation Report', '']

    # ── Executive summary ────────────────────────────────────────────────────
    lines.append('## Executive Summary')
    n_providers = int(bands['provider_id'].nunique()) if not bands.empty else 0
    n_goals     = int(len(goals)) if not goals.empty else 0
    n_met       = int(goals['meets_alpha'].sum()) if (not goals.empty and 'meets_alpha' in goals.columns) else 0
    lines.append(
        f'Plan covers {n_providers} providers with scenario uncertainty bands. '
        f'{n_met} of {n_goals} goal providers met the 85% coverage threshold.'
    )
    lines.append('')

    # ── Top model drivers ─────────────────────────────────────────────────────
    lines.append('## Top Model Drivers')
    if not attribution.empty and 'feature' in attribution.columns:
        for _, r in attribution.head(8).iterrows():
            lines.append(
                f"- **{r.get('target','?')} / {r['feature']}** "
                f"→ `{r.get('reason_code','?')}`: "
                f"{r.get('plain_language', r['feature'])} "
                f"(importance={r.get('importance',0):.3f})"
            )
    else:
        lines.append('No attribution data available.')
    lines.append('')

    # ── Goal coverage ─────────────────────────────────────────────────────────
    lines.append('## Goal Coverage')
    if not goals.empty:
        needed = {'provider_id', 'allocated_min', 'required_min_alpha', 'gap_min',
                  'binding_factor', 'meets_alpha'}
        if needed.issubset(set(goals.columns)):
            worst = goals.sort_values('gap_min', ascending=False)
            for _, r in worst.iterrows():
                status = '✓' if bool(r['meets_alpha']) else '✗'
                lines.append(
                    f"- {status} Provider `{r['provider_id']}`: "
                    f"allocated {r['allocated_min']:.0f} min "
                    f"vs required {r['required_min_alpha']:.0f} min; "
                    f"gap {r['gap_min']:.0f} min; "
                    f"factor `{r['binding_factor']}`"
                    + (f" — {r['binding_description']}"
                       if 'binding_description' in r else '')
                )
        else:
            lines.append('Goal attribution columns not fully available.')
    else:
        lines.append('No uncovered goal providers.')
    lines.append('')

    # ── Risk profiles ─────────────────────────────────────────────────────────
    if risk_profiles is not None and not risk_profiles.empty and 'risk_quadrant' in risk_profiles.columns:
        lines.append('## Provider Risk Profiles')
        for quadrant in ['high_attention', 'uncertain_but_stable',
                         'variable_performer', 'predictable_performer']:
            grp = risk_profiles[risk_profiles['risk_quadrant'] == quadrant]
            if not grp.empty:
                pids = ', '.join(f'`{p}`' for p in grp['provider_id'].tolist()[:6])
                lines.append(f"- **{quadrant}** ({len(grp)}): {pids}")
        lines.append('')

    # ── Drift flags ───────────────────────────────────────────────────────────
    if drift_flags is not None and not drift_flags.empty:
        lines.append('## Drift Flags (monitor next quarter)')
        for _, r in drift_flags.head(10).iterrows():
            lines.append(
                f"- Provider `{r['provider_id']}`: "
                f"trailing_4w_std={r.get('t4_std',0):.3f} vs "
                f"12w_std={r.get('t12_std',0):.3f} "
                f"(drift={r.get('drift_ratio',0):.0%}, {r.get('drift_direction','')})"
            )
        lines.append('')

    # ── Confidence bands summary ──────────────────────────────────────────────
    lines.append('## Confidence Bands Summary')
    if not bands.empty:
        n_underuse = int(bands.get('underuse_risk_flag', pd.Series([], dtype=bool)).sum())
        n_capacity = int(bands.get('capacity_risk_flag', pd.Series([], dtype=bool)).sum())
        lines.append(f'- Providers with scenario bands: {n_providers}')
        lines.append(f'- Underuse risk flags  (P50 < threshold): {n_underuse}')
        lines.append(f'- Capacity risk flags  (P90 > 2× P50):   {n_capacity}')
    else:
        lines.append('No confidence band data available.')

    path = out / 'explanation_report.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ['provider_id', 'service_line', 'day_of_week']


def _rms(x) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.sqrt(np.mean(arr ** 2)))


def _std_nonzero(x) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(np.std(arr, ddof=0))


def _coalesce_numeric(df: pd.DataFrame, cols: list[str], default: float) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype=float)
    for c in cols:
        if c in df.columns:
            out = out.mask(out.isna() | (out == default), pd.to_numeric(df[c], errors='coerce'))
    return out.fillna(default)


def estimate_sigma(features: pd.DataFrame, validation_predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Empirical-Bayes hierarchical shrinkage approximation for sigma_pd.

    This is a fast production-safe substitute for the full NUTS model. It uses
    out-of-fold residual RMS when available and shrinks sparse provider×day
    estimates toward the service-line and site-level scales. The output columns
    are intentionally named as posterior means/sds so Layer 2 can later swap in
    real MCMC samples without changing the interface.
    """
    df = features.copy()
    for c in ['casetime_min', 'turnover_min', 'provider_exception_rate', 'exception_rate_series']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    hist = (
        df.groupby(KEYS, dropna=False)
        .agg(
            sigma_case_hist=('casetime_min', _std_nonzero),
            sigma_turn_hist=('turnover_min', _std_nonzero),
            n_hist=('casetime_min', 'size'),
            mean_case=('casetime_min', 'mean'),
            mean_turn=('turnover_min', 'mean'),
            zero_week_fraction=('casetime_min', lambda x: float(np.mean(np.asarray(x, dtype=float) <= 1e-6))),
            exception_rate=('provider_exception_rate', 'mean') if 'provider_exception_rate' in df.columns else ('casetime_min', lambda x: 0.0),
        )
        .reset_index()
    )

    # Provider-day residual scale from temporal validation predictions.
    resid = pd.DataFrame(columns=KEYS + ['sigma_case_resid', 'sigma_turn_resid', 'n_resid'])
    if validation_predictions is not None and not validation_predictions.empty:
        vp = validation_predictions.copy()
        if 'pred_casetime_min' in vp.columns and 'pred_turnover_min' in vp.columns:
            vp['resid_case'] = pd.to_numeric(vp['casetime_min'], errors='coerce') - pd.to_numeric(vp['pred_casetime_min'], errors='coerce')
            vp['resid_turn'] = pd.to_numeric(vp['turnover_min'], errors='coerce') - pd.to_numeric(vp['pred_turnover_min'], errors='coerce')
            vp = vp[vp['resid_case'].notna() & vp['resid_turn'].notna()].copy()
            if not vp.empty:
                resid = (
                    vp.groupby(KEYS, dropna=False)
                    .agg(
                        sigma_case_resid=('resid_case', _rms),
                        sigma_turn_resid=('resid_turn', _rms),
                        n_resid=('resid_case', 'size'),
                    )
                    .reset_index()
                )

    base = hist.merge(resid, on=KEYS, how='left')
    base['n_resid'] = base['n_resid'].fillna(0).astype(int)

    sl_hist = (
        df.groupby('service_line', dropna=False)
        .agg(
            sl_sigma_case_hist=('casetime_min', _std_nonzero),
            sl_sigma_turn_hist=('turnover_min', _std_nonzero),
        )
        .reset_index()
    )
    base = base.merge(sl_hist, on='service_line', how='left')

    # Service-line residual fallback if OOF residuals exist.
    if validation_predictions is not None and not validation_predictions.empty and 'pred_casetime_min' in validation_predictions.columns:
        vp = validation_predictions.copy()
        vp['resid_case'] = pd.to_numeric(vp['casetime_min'], errors='coerce') - pd.to_numeric(vp['pred_casetime_min'], errors='coerce')
        vp['resid_turn'] = pd.to_numeric(vp['turnover_min'], errors='coerce') - pd.to_numeric(vp['pred_turnover_min'], errors='coerce')
        sl_res = (
            vp.dropna(subset=['resid_case', 'resid_turn'])
            .groupby('service_line', dropna=False)
            .agg(
                sl_sigma_case_resid=('resid_case', _rms),
                sl_sigma_turn_resid=('resid_turn', _rms),
            )
            .reset_index()
        )
        base = base.merge(sl_res, on='service_line', how='left')
    else:
        base['sl_sigma_case_resid'] = np.nan
        base['sl_sigma_turn_resid'] = np.nan

    site_case = np.nanmedian(pd.concat([
        base['sigma_case_hist'], base.get('sigma_case_resid', pd.Series(dtype=float)),
        base['sl_sigma_case_hist'], base.get('sl_sigma_case_resid', pd.Series(dtype=float)),
    ], ignore_index=True))
    site_turn = np.nanmedian(pd.concat([
        base['sigma_turn_hist'], base.get('sigma_turn_resid', pd.Series(dtype=float)),
        base['sl_sigma_turn_hist'], base.get('sl_sigma_turn_resid', pd.Series(dtype=float)),
    ], ignore_index=True))
    site_case = float(site_case) if np.isfinite(site_case) and site_case > 0 else 30.0
    site_turn = float(site_turn) if np.isfinite(site_turn) and site_turn > 0 else 5.0

    base['sl_sigma_case'] = base['sl_sigma_case_resid'].fillna(base['sl_sigma_case_hist']).fillna(site_case).clip(lower=1.0)
    base['sl_sigma_turn'] = base['sl_sigma_turn_resid'].fillna(base['sl_sigma_turn_hist']).fillna(site_turn).clip(lower=0.5)

    base['sigma_case_hist'] = base['sigma_case_hist'].fillna(base['sl_sigma_case']).fillna(site_case)
    base['sigma_turn_hist'] = base['sigma_turn_hist'].fillna(base['sl_sigma_turn']).fillna(site_turn)
    base['sigma_case_resid'] = base['sigma_case_resid'].fillna(base['sigma_case_hist'])
    base['sigma_turn_resid'] = base['sigma_turn_resid'].fillna(base['sigma_turn_hist'])

    shrink = 5.0
    n_hist = base['n_hist'].fillna(0).astype(float)
    n_resid = base['n_resid'].fillna(0).astype(float)

    base['sigma_case'] = np.sqrt(
        (n_resid * base['sigma_case_resid'] ** 2 + n_hist * base['sigma_case_hist'] ** 2 + shrink * base['sl_sigma_case'] ** 2)
        / (n_resid + n_hist + shrink)
    )
    base['sigma_turn'] = np.sqrt(
        (n_resid * base['sigma_turn_resid'] ** 2 + n_hist * base['sigma_turn_hist'] ** 2 + shrink * base['sl_sigma_turn'] ** 2)
        / (n_resid + n_hist + shrink)
    )

    widen = 1.0 + base['exception_rate'].fillna(0.0).clip(lower=0.0, upper=1.0)
    base['sigma_case'] = (base['sigma_case'] * widen).clip(lower=1.0)
    base['sigma_turn'] = (base['sigma_turn'] * widen).clip(lower=0.5)

    # Approximate posterior uncertainty over sigma itself. Sparse rows get wider sigma draws.
    effective_n = (n_resid + 0.5 * n_hist).clip(lower=1.0)
    cv_sigma = (0.35 / np.sqrt(effective_n) + 0.05).clip(lower=0.05, upper=0.45)
    base['sigma_case_sd'] = (base['sigma_case'] * cv_sigma).clip(lower=0.1)
    base['sigma_turn_sd'] = (base['sigma_turn'] * cv_sigma).clip(lower=0.05)

    mean_case_safe = base['mean_case'].replace(0, np.nan).abs()
    base['coeff_of_variation_case'] = (base['sigma_case_hist'] / mean_case_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def family(row):
        if row['zero_week_fraction'] > 0.15 or row['exception_rate'] > 0.15:
            return 'zero_inflated_gamma_recommended_normal_fallback'
        if row['coeff_of_variation_case'] > 0.50:
            return 'lognormal_or_weibull_recommended_normal_fallback'
        if row['coeff_of_variation_case'] > 0.20:
            return 'gamma_recommended_normal_fallback'
        return 'normal_residual'

    base['likelihood_family_flag'] = base.apply(family, axis=1)

    return base[[
        'provider_id', 'service_line', 'day_of_week',
        'sigma_case', 'sigma_turn', 'sigma_case_sd', 'sigma_turn_sd',
        'n_hist', 'n_resid', 'exception_rate', 'zero_week_fraction',
        'coeff_of_variation_case', 'likelihood_family_flag',
    ]].rename(columns={'n_hist': 'n'})


def _positive_sigma_draws(rng: np.random.Generator, mean: float, sd: float, size: int) -> np.ndarray:
    mean = max(float(mean), 1e-6)
    sd = max(float(sd), 1e-6)
    cv = min(max(sd / mean, 1e-6), 1.0)
    sigma_log = np.sqrt(np.log(1.0 + cv ** 2))
    mu_log = np.log(mean) - 0.5 * sigma_log ** 2
    return rng.lognormal(mu_log, sigma_log, size=size)


def generate_scenarios(point_forecasts: pd.DataFrame, sigma_df: pd.DataFrame, out_dir: str | Path, config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(config.get('layer1', {}).get('random_seed', 42)))
    S = int(config.get('layer1', {}).get('n_scenarios', 200))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    f = point_forecasts.merge(sigma_df, on=KEYS, how='left')
    f['sigma_case'] = pd.to_numeric(f['sigma_case'], errors='coerce').fillna(30.0).clip(lower=1.0)
    f['sigma_turn'] = pd.to_numeric(f['sigma_turn'], errors='coerce').fillna(5.0).clip(lower=0.5)
    f['sigma_case_sd'] = pd.to_numeric(f.get('sigma_case_sd', f['sigma_case'] * 0.15), errors='coerce').fillna(f['sigma_case'] * 0.15)
    f['sigma_turn_sd'] = pd.to_numeric(f.get('sigma_turn_sd', f['sigma_turn'] * 0.15), errors='coerce').fillna(f['sigma_turn'] * 0.15)

    rows = []
    for _, r in f.iterrows():
        sigma_case_draw = _positive_sigma_draws(rng, float(r['sigma_case']), float(r['sigma_case_sd']), S)
        sigma_turn_draw = _positive_sigma_draws(rng, float(r['sigma_turn']), float(r['sigma_turn_sd']), S)
        eps_case = rng.normal(0.0, sigma_case_draw, size=S)
        eps_turn = rng.normal(0.0, sigma_turn_draw, size=S)
        case = np.maximum(0.0, float(r['mu_casetime_min']) + eps_case)
        turn = np.maximum(0.0, float(r['mu_turnover_min']) + eps_turn)

        for s in range(S):
            rows.append({
                'scenario_id': s,
                'provider_id': r['provider_id'],
                'service_line': r['service_line'],
                'day_of_week': int(r['day_of_week']),
                'forecast_week_index': int(r.get('forecast_week_index', r.get('week_index', 0))),
                'demand_casetime_min': float(case[s]),
                'demand_turnover_min': float(turn[s]),
                'demand_total_min': float(case[s] + turn[s]),
                'mu_casetime_min': float(r['mu_casetime_min']),
                'mu_turnover_min': float(r['mu_turnover_min']),
                'sigma_case_draw': float(sigma_case_draw[s]),
                'sigma_turn_draw': float(sigma_turn_draw[s]),
                'sigma_case': float(r['sigma_case']),
                'sigma_turn': float(r['sigma_turn']),
            })

    scenarios = pd.DataFrame(rows)
    scenarios.to_csv(out / 'scenarios_long.csv', index=False)

    summary = scenarios.groupby(['provider_id', 'service_line', 'day_of_week'], dropna=False).agg(
        case_p10=('demand_casetime_min', lambda x: float(np.quantile(x, 0.10))),
        case_p50=('demand_casetime_min', lambda x: float(np.quantile(x, 0.50))),
        case_p90=('demand_casetime_min', lambda x: float(np.quantile(x, 0.90))),
        turn_p10=('demand_turnover_min', lambda x: float(np.quantile(x, 0.10))),
        turn_p50=('demand_turnover_min', lambda x: float(np.quantile(x, 0.50))),
        turn_p90=('demand_turnover_min', lambda x: float(np.quantile(x, 0.90))),
        total_p10=('demand_total_min', lambda x: float(np.quantile(x, 0.10))),
        total_p50=('demand_total_min', lambda x: float(np.quantile(x, 0.50))),
        total_p90=('demand_total_min', lambda x: float(np.quantile(x, 0.90))),
    ).reset_index()
    summary.to_csv(out / 'scenario_summary.csv', index=False)

    # Compact JSON artifact for Layer 2 consumers that prefer records.
    json_records = summary.to_dict(orient='records')
    with open(out / 'demand_scenarios_summary.json', 'w', encoding='utf-8') as fp:
        json.dump(json_records, fp, indent=2, default=str)

    return scenarios

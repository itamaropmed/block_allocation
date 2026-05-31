from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def estimate_sigma(features: pd.DataFrame, validation_predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    df = features.copy()
    # Use historical variability, shrunk by service line. If validation residuals exist, use them too.
    base = df.groupby(['provider_id', 'service_line', 'day_of_week']).agg(
        sigma_case_hist=('casetime_min', lambda x: float(np.std(x, ddof=0))),
        sigma_turn_hist=('turnover_min', lambda x: float(np.std(x, ddof=0))),
        n=('casetime_min', 'size'),
        exception_rate=('provider_exception_rate', 'mean'),
    ).reset_index()
    sl = df.groupby('service_line').agg(
        sl_sigma_case=('casetime_min', lambda x: float(np.std(x, ddof=0))),
        sl_sigma_turn=('turnover_min', lambda x: float(np.std(x, ddof=0))),
    ).reset_index()
    base = base.merge(sl, on='service_line', how='left')
    shrink = 5.0
    base['sigma_case'] = np.sqrt((base['n'] * base['sigma_case_hist'] ** 2 + shrink * base['sl_sigma_case'] ** 2) / (base['n'] + shrink))
    base['sigma_turn'] = np.sqrt((base['n'] * base['sigma_turn_hist'] ** 2 + shrink * base['sl_sigma_turn'] ** 2) / (base['n'] + shrink))
    # Contested series widen the posterior.
    base['sigma_case'] = base['sigma_case'].fillna(base['sl_sigma_case']).fillna(30.0) * (1.0 + base['exception_rate'].fillna(0.0))
    base['sigma_turn'] = base['sigma_turn'].fillna(base['sl_sigma_turn']).fillna(5.0) * (1.0 + base['exception_rate'].fillna(0.0))
    base['sigma_case'] = base['sigma_case'].clip(lower=1.0)
    base['sigma_turn'] = base['sigma_turn'].clip(lower=0.5)
    return base[['provider_id', 'service_line', 'day_of_week', 'sigma_case', 'sigma_turn', 'n', 'exception_rate']]


def generate_scenarios(point_forecasts: pd.DataFrame, sigma_df: pd.DataFrame, out_dir: str | Path, config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(config['layer1'].get('random_seed', 42)))
    S = int(config['layer1'].get('n_scenarios', 200))
    f = point_forecasts.merge(sigma_df, on=['provider_id', 'service_line', 'day_of_week'], how='left')
    f['sigma_case'] = f['sigma_case'].fillna(30.0)
    f['sigma_turn'] = f['sigma_turn'].fillna(5.0)
    rows = []
    for _, r in f.iterrows():
        case = np.maximum(0.0, rng.normal(float(r['mu_casetime_min']), float(r['sigma_case']), size=S))
        turn = np.maximum(0.0, rng.normal(float(r['mu_turnover_min']), float(r['sigma_turn']), size=S))
        for s in range(S):
            rows.append({
                'scenario_id': s,
                'provider_id': r['provider_id'],
                'service_line': r['service_line'],
                'day_of_week': int(r['day_of_week']),
                'demand_casetime_min': float(case[s]),
                'demand_turnover_min': float(turn[s]),
                'mu_casetime_min': float(r['mu_casetime_min']),
                'mu_turnover_min': float(r['mu_turnover_min']),
                'sigma_case': float(r['sigma_case']),
                'sigma_turn': float(r['sigma_turn']),
            })
    scenarios = pd.DataFrame(rows)
    p = Path(out_dir) / 'scenarios_long.csv'
    scenarios.to_csv(p, index=False)
    # Useful wide summary.
    summary = scenarios.groupby(['provider_id', 'service_line', 'day_of_week']).agg(
        case_p10=('demand_casetime_min', lambda x: float(np.quantile(x, 0.10))),
        case_p50=('demand_casetime_min', lambda x: float(np.quantile(x, 0.50))),
        case_p90=('demand_casetime_min', lambda x: float(np.quantile(x, 0.90))),
        turn_p10=('demand_turnover_min', lambda x: float(np.quantile(x, 0.10))),
        turn_p50=('demand_turnover_min', lambda x: float(np.quantile(x, 0.50))),
        turn_p90=('demand_turnover_min', lambda x: float(np.quantile(x, 0.90))),
    ).reset_index()
    summary.to_csv(Path(out_dir) / 'scenario_summary.csv', index=False)
    return scenarios

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import math

import numpy as np
import pandas as pd

from src.common.time_utils import overlap


def provider_site_ok(provider_row: pd.Series, physical_site: str) -> bool:
    sites = provider_row.get('exclusive_sites', [])
    if isinstance(sites, str):
        # CSV roundtrip fallback.
        return physical_site in sites
    return physical_site in (sites or [])


def make_goals_from_scenarios(scenarios: pd.DataFrame, config: dict, max_goals: int = 12) -> pd.DataFrame:
    demand = scenarios.groupby('provider_id').agg(total_mu=('mu_casetime_min', 'sum')).reset_index()
    demand = demand.sort_values('total_mu', ascending=False).head(max_goals)
    demand['goal_type'] = 'target_utilization'
    demand['target_utilization'] = float(config['layer2'].get('default_target_utilization', 0.70))
    return demand[['provider_id', 'goal_type', 'target_utilization']]


def required_minutes_by_provider(scenarios: pd.DataFrame, early_release: pd.DataFrame, goals: pd.DataFrame, config: dict) -> pd.DataFrame:
    alpha = float(config['layer2'].get('coverage_alpha', 0.85))
    nweeks = int(config['layer2'].get('optimization_weeks', 13))
    weekly = scenarios.groupby(['provider_id', 'scenario_id']).agg(
        q_case=('demand_casetime_min', 'sum'),
        q_turn=('demand_turnover_min', 'sum'),
    ).reset_index()
    weekly['quarterly_demand_min'] = (weekly['q_case'] + weekly['q_turn']) * nweeks
    g = goals[['provider_id', 'target_utilization']].copy()
    req_rows = []
    er = dict(zip(early_release['provider_id'].astype(str), early_release['early_release_projected_min'])) if not early_release.empty else {}
    for _, goal in g.iterrows():
        p = str(goal['provider_id'])
        target = float(goal['target_utilization'])
        vals = weekly.loc[weekly['provider_id'].astype(str) == p, 'quarterly_demand_min']
        if vals.empty:
            continue
        required_by_s = vals / max(1e-6, target) + float(er.get(p, 0.0))
        req_rows.append({
            'provider_id': p,
            'target_utilization': target,
            'required_min_p10': float(np.quantile(required_by_s, 0.10)),
            'required_min_p50': float(np.quantile(required_by_s, 0.50)),
            'required_min_p90': float(np.quantile(required_by_s, 0.90)),
            'required_min_alpha': float(np.quantile(required_by_s, alpha)),
            'coverage_alpha': alpha,
        })
    return pd.DataFrame(req_rows)


def _prepare_template(prelayer: dict, config: dict) -> pd.DataFrame:
    template = pd.DataFrame(prelayer.get('block_template', []))
    if template.empty:
        raise ValueError('Prelayer block_template is empty.')
    max_blocks = int(config['layer2'].get('max_template_blocks_for_mip', 900))
    # Prefer non-open and high-observation template slots, then cap for fast local tests.
    template = template.sort_values(['observed_instances', 'duration_min'], ascending=False).head(max_blocks).copy()
    return template


def greedy_allocate(prelayer: dict, providers: pd.DataFrame, scenarios: pd.DataFrame, early_release: pd.DataFrame, config: dict, theme: str = 'balanced') -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    template = _prepare_template(prelayer, config)
    goals = make_goals_from_scenarios(scenarios, config)
    req = required_minutes_by_provider(scenarios, early_release, goals, config)
    if req.empty:
        req = pd.DataFrame(columns=['provider_id', 'required_min_alpha'])
    req_map = dict(zip(req['provider_id'].astype(str), req['required_min_alpha']))

    # Active providers: goal providers + providers already holding many blocks.
    active = set(req_map.keys())
    current_counts = template['dominant_provider_id'].astype(str).value_counts().head(int(config['layer2'].get('max_active_providers', 35))).index.astype(str)
    active.update(current_counts)
    providers = providers[providers['provider_id'].astype(str).isin(active)].copy()
    provider_rows = {str(r['provider_id']): r for _, r in providers.iterrows()}

    assignments = []
    allocated = {p: 0.0 for p in active}
    changed = 0
    # Start from warm-start when eligible; otherwise leave open.
    for _, b in template.iterrows():
        cur = str(b['dominant_provider_id'])
        assigned = cur if cur in active else 'OPEN'
        if assigned != 'OPEN':
            allocated[assigned] = allocated.get(assigned, 0.0) + float(b['duration_min'])
        assignments.append({**b.to_dict(), 'assigned_provider_id': assigned, 'warm_start_provider_id': cur, 'changed': assigned != cur})
    # Reallocate a small number of low-impact blocks from over-covered to under-covered providers.
    max_moves = {'conservative': 10, 'balanced': 30, 'utilization_first': 80, 'continuity_first': 15}.get(theme, 30)
    for _ in range(max_moves):
        under = sorted([(p, req_map.get(p, 0) - allocated.get(p, 0)) for p in req_map], key=lambda x: x[1], reverse=True)
        if not under or under[0][1] <= 0:
            break
        target_p = under[0][0]
        prow = provider_rows.get(target_p)
        if prow is None:
            break
        # Find a donor block from over-covered provider or OPEN, compatible by site.
        best_i = None
        best_score = None
        for i, a in enumerate(assignments):
            if a['assigned_provider_id'] == target_p:
                continue
            if not provider_site_ok(prow, a['physical_site']):
                continue
            donor = a['assigned_provider_id']
            donor_surplus = allocated.get(donor, 0) - req_map.get(donor, 0) if donor != 'OPEN' else 1e9
            if donor != 'OPEN' and donor_surplus < float(a['duration_min']):
                continue
            # low exception and shorter blocks are safer to move.
            score = float(a.get('exception_rate', 0.0)) * 1000 + float(a['duration_min'])
            if best_score is None or score < best_score:
                best_i, best_score = i, score
        if best_i is None:
            break
        old = assignments[best_i]['assigned_provider_id']
        dur = float(assignments[best_i]['duration_min'])
        if old != 'OPEN':
            allocated[old] = allocated.get(old, 0.0) - dur
        assignments[best_i]['assigned_provider_id'] = target_p
        assignments[best_i]['changed'] = assignments[best_i]['warm_start_provider_id'] != target_p
        allocated[target_p] = allocated.get(target_p, 0.0) + dur
        changed += 1

    assign_df = pd.DataFrame(assignments)
    coverage = []
    for p in sorted(active):
        req_alpha = req_map.get(p, 0.0)
        alloc = allocated.get(p, 0.0)
        coverage.append({
            'provider_id': p,
            'allocated_min': float(alloc),
            'required_min_alpha': float(req_alpha),
            'coverage_ratio': float(alloc / req_alpha) if req_alpha > 0 else np.nan,
            'meets_alpha': bool(req_alpha > 0 and alloc >= req_alpha),
        })
    cov_df = pd.DataFrame(coverage)
    objective = float((cov_df['required_min_alpha'] - cov_df['allocated_min']).clip(lower=0).sum() + 5.0 * assign_df['changed'].sum())
    meta = {'theme': theme, 'changed_blocks': int(assign_df['changed'].sum()), 'objective_value': objective, 'goals': goals.to_dict(orient='records')}
    return assign_df, cov_df, meta

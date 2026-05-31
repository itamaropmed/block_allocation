from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.common.time_utils import round_to_grid, fmt_minutes


def _cluster_times(group: pd.DataFrame, margin: int, grid: int) -> pd.DataFrame:
    """Cluster start/end pairs within a margin and snap to the mode pair in each cluster."""
    if group.empty:
        return group
    g = group.copy()
    pairs = g[['start_min', 'end_min']].dropna().round().astype(int).values.tolist()
    if not pairs:
        g['canon_start_min'] = g['start_min'].apply(lambda x: round_to_grid(x, grid))
        g['canon_end_min'] = g['end_min'].apply(lambda x: round_to_grid(x, grid))
        return g
    # Simple deterministic greedy clusters by sorted start/end.
    unique_pairs = sorted(set(map(tuple, pairs)))
    clusters = []
    for pair in unique_pairs:
        placed = False
        for cl in clusters:
            ref = cl[0]
            if abs(pair[0] - ref[0]) <= margin and abs(pair[1] - ref[1]) <= margin:
                cl.append(pair)
                placed = True
                break
        if not placed:
            clusters.append([pair])
    pair_to_canon = {}
    pair_counts = Counter(map(tuple, pairs))
    for cl in clusters:
        mode_pair = max(cl, key=lambda z: (pair_counts[z], -z[0], -z[1]))
        canon = (round_to_grid(mode_pair[0], grid), round_to_grid(mode_pair[1], grid))
        for pair in cl:
            pair_to_canon[pair] = canon
    canons = [pair_to_canon.get((int(round(s)), int(round(e))), (round_to_grid(s, grid), round_to_grid(e, grid))) for s, e in zip(g['start_min'], g['end_min'])]
    g['canon_start_min'] = [c[0] for c in canons]
    g['canon_end_min'] = [c[1] for c in canons]
    return g


def canonicalize_blocks(blocks_df: pd.DataFrame, margin_minutes: int = 120, grid_minutes: int = 5) -> pd.DataFrame:
    keys = ['physical_site', 'room_type', 'day_of_week']
    out = []
    for _, group in blocks_df.groupby(keys, dropna=False):
        out.append(_cluster_times(group, margin_minutes, grid_minutes))
    df = pd.concat(out, ignore_index=True) if out else blocks_df.copy()
    df['cell_id'] = (
        df['physical_site'].astype(str) + '|' +
        df['room_type'].astype(str) + '|dow=' + df['day_of_week'].astype(str) +
        '|start=' + df['canon_start_min'].astype(int).astype(str) +
        '|end=' + df['canon_end_min'].astype(int).astype(str)
    )
    df['series_key'] = df['cell_id']
    return df


def _modal(values: List[str], weeks: List[int]) -> str:
    counts = Counter(values)
    max_count = max(counts.values())
    tied = {v for v, c in counts.items() if c == max_count}
    # tie-break by most recent week
    for _, v in sorted(zip(weeks, values), reverse=True):
        if v in tied:
            return v
    return values[-1]


def deviation_count(df: pd.DataFrame, T: int, offset: int = 0) -> Tuple[int, Dict[Tuple[str, int], str]]:
    templates = {}
    total_dev = 0
    for (cell, phase), g in df.groupby(['cell_id', ((df['week_index'].astype(int) + offset) % T)]):
        vals = g['provider_id'].astype(str).tolist()
        weeks = g['week_index'].astype(int).tolist()
        mode = _modal(vals, weeks)
        templates[(cell, int(phase))] = mode
        total_dev += int((g['provider_id'].astype(str) != mode).sum())
    return total_dev, templates


def bic_scores(df: pd.DataFrame, max_period: int = 13) -> pd.DataFrame:
    H = int(df['week_index'].nunique())
    C = int(df['cell_id'].nunique())
    V = max(2, int(df['provider_id'].nunique()) + 2)
    rows = []
    for T in range(1, min(max_period, H) + 1):
        D, _ = deviation_count(df, T, offset=0)
        bic = T * C * math.log(V) + D * math.log(max(2, H * C * V))
        rows.append({'T': T, 'deviations': D, 'n_cells': C, 'alphabet_size': V, 'n_weeks': H, 'bic': bic})
    return pd.DataFrame(rows)


def choose_period(df: pd.DataFrame, forced_period: int = 0, max_period: int = 13):
    scores = bic_scores(df, max_period=max_period)
    if forced_period and forced_period > 0:
        T_star = int(forced_period)
    else:
        T_star = int(scores.loc[scores['bic'].idxmin(), 'T'])
    offset_rows = []
    best_offset = 0
    best_dev = None
    for k in range(T_star):
        D, _ = deviation_count(df, T_star, offset=k)
        offset_rows.append({'offset': k, 'deviations': D})
        if best_dev is None or D < best_dev:
            best_dev = D
            best_offset = k
    return T_star, best_offset, scores, pd.DataFrame(offset_rows)


def reconstruct_template(blocks_df: pd.DataFrame, providers_df: pd.DataFrame, config: dict) -> dict:
    margin = int(config['pre_layer'].get('canonical_time_margin_minutes', 120))
    grid = int(config['pre_layer'].get('round_time_minutes', 5))
    forced = int(config['pre_layer'].get('template_weeks', 0))
    max_period = int(config['pre_layer'].get('max_candidate_period', 13))

    df = canonicalize_blocks(blocks_df, margin, grid)
    T_star, phase_offset, scores, offset_scores = choose_period(df, forced, max_period)
    df['rotation_phase'] = ((df['week_index'].astype(int) + phase_offset) % T_star).astype(int)

    provider_name_map = dict(zip(providers_df['provider_id'].astype(str), providers_df['provider_name'])) if not providers_df.empty else {}
    template_rows = []
    exceptions = []
    exception_rate = {}

    for (cell, phase), g in df.groupby(['cell_id', 'rotation_phase']):
        vals = g['provider_id'].astype(str).tolist()
        weeks = g['week_index'].astype(int).tolist()
        dominant = _modal(vals, weeks)
        dev_mask = g['provider_id'].astype(str) != dominant
        devs = g[dev_mask]
        exception_rate[f'{cell}|phase={phase}'] = float(len(devs) / max(1, len(g)))
        exemplar = g.sort_values('week_index').iloc[-1]
        template_rows.append({
            'template_block_id': f'{cell}|phase={phase}',
            'series_key': cell,
            'rotation_phase': int(phase),
            'dominant_provider_id': dominant,
            'dominant_provider_name': provider_name_map.get(dominant, exemplar.get('provider_name', dominant)),
            'physical_site': exemplar['physical_site'],
            'service_line': exemplar['service_line_from_block'],
            'room_type': exemplar['room_type'],
            'day_of_week': int(exemplar['day_of_week']),
            'canon_start_min': int(exemplar['canon_start_min']),
            'canon_end_min': int(exemplar['canon_end_min']),
            'canon_start': fmt_minutes(exemplar['canon_start_min']),
            'canon_end': fmt_minutes(exemplar['canon_end_min']),
            'duration_min': float(max(0, exemplar['canon_end_min'] - exemplar['canon_start_min'])),
            'observed_instances': int(len(g)),
            'exception_count': int(len(devs)),
            'exception_rate': float(len(devs) / max(1, len(g))),
        })
        for _, r in devs.iterrows():
            exceptions.append({
                'template_block_id': f'{cell}|phase={phase}',
                'series_key': cell,
                'week_index': int(r['week_index']),
                'date': r['date'],
                'actual_provider_id': r['provider_id'],
                'dominant_provider_id': dominant,
                'room_type': r['room_type'],
                'physical_site': r['physical_site'],
            })

    template_df = pd.DataFrame(template_rows).sort_values(['physical_site', 'room_type', 'day_of_week', 'canon_start_min', 'rotation_phase'])
    utilization = []
    by_p = df[df['provider_id'] != 'OPEN'].groupby('provider_id')
    for p, g in by_p:
        allocated = float((g['duration_min'] - g['manual_early_release_min']).clip(lower=0).sum())
        utilization.append({
            'provider_id': p,
            'provider_name': provider_name_map.get(p, g['provider_name'].iloc[0]),
            'allocated_block_min': allocated,
            'allocated_block_hours': allocated / 60.0,
            # Real utilization needs case and turnover files. This is a structural allocation summary.
            'historical_utilization_proxy': None,
            'n_blocks': int(len(g)),
        })

    phase_labels = {str(w): int(((int(i) + phase_offset) % T_star)) for w, i in df[['week_start', 'week_index']].drop_duplicates().values}
    warnings = []
    missing_providers = sorted(set(df['provider_id'].astype(str)) - set(providers_df['provider_id'].astype(str)) - {'OPEN'})
    if missing_providers:
        warnings.append(f'{len(missing_providers)} provider ids appear in blocks but not in providers JSON. Example: {missing_providers[:10]}')

    return {
        'T_star': T_star,
        'phase_offset': int(phase_offset),
        'phase_labels': phase_labels,
        'block_template': template_df.to_dict(orient='records'),
        'exceptions_list': exceptions,
        'exception_rate_per_series': exception_rate,
        'utilization_per_provider': utilization,
        'bic_scores': scores.to_dict(orient='records'),
        'offset_scores': offset_scores.to_dict(orient='records'),
        'warnings': warnings,
        '_canonical_blocks_df': df,
        '_template_df': template_df,
        '_bic_df': scores,
        '_offset_df': offset_scores,
    }

"""
Pre-Layer — Template Reconstruction  (standalone, no src imports)
=================================================================
Fixes vs original:
  1. Inlined round_to_grid() and fmt_minutes() from src.common.time_utils.
  2. phase_labels now uses week_index integers as string keys ("0","1","2"...)
     instead of week_start date strings.  This matches what Layer 1
     weekly_observations.py expects when doing:
         b['week_index'].astype(str).map(phase_labels)
  3. Guarded all optional column accesses:
       service_line_from_block → falls back to 'Unknown'
       provider_name → tries 'provider_name' then 'name' then provider_id
       date         → falls back to '' if column absent
  4. Provider name map now tries both 'provider_name' and 'name' columns
     (Geisinger uses 'name'; other schemas use 'provider_name').
  5. No src imports — completely standalone.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Inlined from src.common.time_utils
# ─────────────────────────────────────────────────────────────────────────────

def round_to_grid(value: float, grid: int) -> int:
    """Round `value` to the nearest multiple of `grid`."""
    if grid <= 0:
        return int(round(value))
    return int(round(value / grid) * grid)


def fmt_minutes(minutes) -> str:
    """Convert minutes-since-midnight to HH:MM string."""
    try:
        m = int(round(float(minutes)))
        return f'{m // 60:02d}:{m % 60:02d}'
    except Exception:
        return '??:??'


# ─────────────────────────────────────────────────────────────────────────────
# Canonical-time clustering  (±2h window)
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_times(group: pd.DataFrame, margin: int, grid: int) -> pd.DataFrame:
    """
    Cluster (start_min, end_min) pairs within ±margin minutes.
    Snap each pair to the mode pair of its cluster.
    """
    if group.empty:
        return group
    g = group.copy()
    pairs = (
        g[['start_min', 'end_min']].dropna()
        .round().astype(int).values.tolist()
    )
    if not pairs:
        g['canon_start_min'] = g['start_min'].apply(lambda x: round_to_grid(x, grid))
        g['canon_end_min']   = g['end_min'].apply(lambda x:   round_to_grid(x, grid))
        return g

    unique_pairs = sorted(set(map(tuple, pairs)))
    clusters: List[List[tuple]] = []
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

    pair_counts = Counter(map(tuple, pairs))
    pair_to_canon: Dict[tuple, tuple] = {}
    for cl in clusters:
        mode_pair = max(cl, key=lambda z: (pair_counts[z], -z[0], -z[1]))
        canon = (round_to_grid(mode_pair[0], grid), round_to_grid(mode_pair[1], grid))
        for p in cl:
            pair_to_canon[p] = canon

    canons = [
        pair_to_canon.get(
            (int(round(s)), int(round(e))),
            (round_to_grid(s, grid), round_to_grid(e, grid)),
        )
        for s, e in zip(g['start_min'], g['end_min'])
    ]
    g['canon_start_min'] = [c[0] for c in canons]
    g['canon_end_min']   = [c[1] for c in canons]
    return g


def canonicalize_blocks(
    blocks_df: pd.DataFrame,
    margin_minutes: int = 120,
    grid_minutes: int = 5,
) -> pd.DataFrame:
    """
    Group by (physical_site, room_type, day_of_week), cluster times,
    and assign a canonical cell_id.
    """
    keys = ['physical_site', 'room_type', 'day_of_week']
    out  = []
    for _, group in blocks_df.groupby(keys, dropna=False):
        out.append(_cluster_times(group, margin_minutes, grid_minutes))

    df = pd.concat(out, ignore_index=True) if out else blocks_df.copy()

    df['cell_id'] = (
        df['physical_site'].astype(str)  + '|'
        + df['room_type'].astype(str)    + '|dow='
        + df['day_of_week'].astype(str)  + '|start='
        + df['canon_start_min'].astype(int).astype(str) + '|end='
        + df['canon_end_min'].astype(int).astype(str)
    )
    df['series_key'] = df['cell_id']
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Period (T*) detection via BIC
# ─────────────────────────────────────────────────────────────────────────────

def _modal(values: List[str], weeks: List[int]) -> str:
    """Most-frequent value; tie-broken by most recent week."""
    counts    = Counter(values)
    max_count = max(counts.values())
    tied      = {v for v, c in counts.items() if c == max_count}
    for _, v in sorted(zip(weeks, values), reverse=True):
        if v in tied:
            return v
    return values[-1]


def deviation_count(
    df: pd.DataFrame, T: int, offset: int = 0
) -> Tuple[int, Dict[Tuple[str, int], str]]:
    templates: Dict[Tuple[str, int], str] = {}
    total_dev = 0
    phases = (df['week_index'].astype(int) + offset) % T
    for (cell, phase), g in df.groupby(['cell_id', phases]):
        vals  = g['provider_id'].astype(str).tolist()
        weeks = g['week_index'].astype(int).tolist()
        mode  = _modal(vals, weeks)
        templates[(cell, int(phase))] = mode
        total_dev += int((g['provider_id'].astype(str) != mode).sum())
    return total_dev, templates


def bic_scores(df: pd.DataFrame, max_period: int = 13) -> pd.DataFrame:
    H   = int(df['week_index'].nunique())
    C   = int(df['cell_id'].nunique())
    V   = max(2, int(df['provider_id'].nunique()) + 2)
    rows = []
    for T in range(1, min(max_period, H) + 1):
        D, _ = deviation_count(df, T, offset=0)
        bic  = T * C * math.log(V) + D * math.log(max(2, H * C * V))
        rows.append({
            'T': T, 'deviations': D,
            'n_cells': C, 'alphabet_size': V,
            'n_weeks': H, 'bic': round(bic, 2),
        })
    return pd.DataFrame(rows)


def choose_period(
    df: pd.DataFrame,
    forced_period: int = 0,
    max_period: int = 13,
) -> Tuple[int, int, pd.DataFrame, pd.DataFrame]:
    scores = bic_scores(df, max_period=max_period)
    if forced_period and forced_period > 0:
        T_star = int(forced_period)
    else:
        T_star = int(scores.loc[scores['bic'].idxmin(), 'T'])

    best_offset, best_dev = 0, None
    offset_rows = []
    for k in range(T_star):
        D, _ = deviation_count(df, T_star, offset=k)
        offset_rows.append({'offset': k, 'deviations': D})
        if best_dev is None or D < best_dev:
            best_dev, best_offset = D, k

    return T_star, best_offset, scores, pd.DataFrame(offset_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Provider name helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_provider_name_map(providers_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build provider_id → display name map.
    Tries 'provider_name' first (old schema), then 'name' (Geisinger schema).
    """
    if providers_df is None or providers_df.empty:
        return {}
    name_col = None
    for col in ('provider_name', 'name', 'display_name', 'full_name'):
        if col in providers_df.columns:
            name_col = col
            break
    if name_col is None or 'provider_id' not in providers_df.columns:
        return {}
    return dict(
        zip(
            providers_df['provider_id'].astype(str),
            providers_df[name_col].fillna('').astype(str),
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_template(
    blocks_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    config: dict,
) -> dict:
    cfg        = config.get('pre_layer', {})
    margin     = int(cfg.get('canonical_time_margin_minutes', 120))
    grid       = int(cfg.get('round_time_minutes', 5))
    forced     = int(cfg.get('template_weeks', 0))
    max_period = int(cfg.get('max_candidate_period', 13))

    # ── Step 1: canonical-time clustering ────────────────────────────────────
    df = canonicalize_blocks(blocks_df, margin, grid)

    # ── Step 2: BIC period + offset detection ─────────────────────────────────
    T_star, phase_offset, scores, offset_scores = choose_period(df, forced, max_period)
    df['rotation_phase'] = (
        (df['week_index'].astype(int) + phase_offset) % T_star
    ).astype(int)

    # ── Step 3: Provider name map ──────────────────────────────────────────────
    provider_name_map = _build_provider_name_map(providers_df)

    # ── Step 4: Build template + exceptions ───────────────────────────────────
    template_rows = []
    exceptions    = []
    exception_rate: Dict[str, float] = {}

    # Safe column getters
    has_service_line = 'service_line_from_block' in df.columns
    has_date         = 'date' in df.columns

    for (cell, phase), g in df.groupby(['cell_id', 'rotation_phase']):
        vals    = g['provider_id'].astype(str).tolist()
        weeks   = g['week_index'].astype(int).tolist()
        dominant = _modal(vals, weeks)

        dev_mask = g['provider_id'].astype(str) != dominant
        devs     = g[dev_mask]
        er_key   = f'{cell}|phase={phase}'
        exception_rate[er_key] = float(len(devs) / max(1, len(g)))

        exemplar = g.sort_values('week_index').iloc[-1]

        template_rows.append({
            'template_block_id':        er_key,
            'series_key':               str(cell),
            'rotation_phase':           int(phase),
            'dominant_provider_id':     dominant,
            'dominant_provider_name':   provider_name_map.get(dominant, dominant),
            'physical_site':            str(exemplar.get('physical_site', 'Unknown')),
            'service_line':             str(exemplar['service_line_from_block'])
                                        if has_service_line
                                        else str(exemplar.get('service_line', 'Unknown')),
            'room_type':                str(exemplar.get('room_type', 'Unknown')),
            'day_of_week':              int(exemplar.get('day_of_week', -1)),
            'canon_start_min':          int(exemplar.get('canon_start_min', 0)),
            'canon_end_min':            int(exemplar.get('canon_end_min', 0)),
            'canon_start':              fmt_minutes(exemplar.get('canon_start_min', 0)),
            'canon_end':                fmt_minutes(exemplar.get('canon_end_min', 0)),
            'duration_min':             float(max(
                                            0,
                                            exemplar.get('canon_end_min', 0)
                                            - exemplar.get('canon_start_min', 0),
                                        )),
            'observed_instances':       int(len(g)),
            'exception_count':          int(len(devs)),
            'exception_rate':           float(len(devs) / max(1, len(g))),
        })

        for _, r in devs.iterrows():
            exceptions.append({
                'template_block_id':     er_key,
                'series_key':            str(cell),
                'week_index':            int(r.get('week_index', -1)),
                'date':                  str(r['date']) if has_date else '',
                'actual_provider_id':    str(r.get('provider_id', '')),
                'dominant_provider_id':  dominant,
                'room_type':             str(r.get('room_type', 'Unknown')),
                'physical_site':         str(r.get('physical_site', 'Unknown')),
            })

    template_df = (
        pd.DataFrame(template_rows)
        .sort_values([
            'physical_site', 'room_type', 'day_of_week',
            'canon_start_min', 'rotation_phase',
        ])
        .reset_index(drop=True)
    )

    # ── Step 5: Utilization summary ───────────────────────────────────────────
    utilization = []
    non_open    = df[df['provider_id'].astype(str) != 'OPEN']
    for p, g in non_open.groupby('provider_id'):
        allocated = float(
            (g['duration_min'] - g.get('manual_early_release_min', 0))
            .clip(lower=0)
            .sum()
        )
        utilization.append({
            'provider_id':               str(p),
            'provider_name':             provider_name_map.get(str(p), str(p)),
            'allocated_block_min':        round(allocated, 1),
            'allocated_block_hours':      round(allocated / 60.0, 2),
            'historical_utilization_proxy': None,
            'n_blocks':                   int(len(g)),
        })

    # ── Step 6: phase_labels  (FIX: keys are week_index ints, not date strings) ──
    #   Format: {"0": 0, "1": 1, "2": 0, ...}
    #   This matches what Layer 1 weekly_observations.py expects when doing
    #       b['week_index'].astype(str).map(phase_labels)
    unique_week_idx = df['week_index'].astype(int).unique()
    phase_labels = {
        str(int(wi)): int((int(wi) + phase_offset) % T_star)
        for wi in unique_week_idx
    }

    # ── Step 7: Warnings ──────────────────────────────────────────────────────
    warnings_list = []
    block_pids    = set(df['provider_id'].astype(str).unique())
    prov_pids     = set(providers_df['provider_id'].astype(str).unique()) if not providers_df.empty else set()
    missing_pids  = sorted((block_pids - prov_pids) - {'OPEN', '', 'nan'})
    if missing_pids:
        warnings_list.append(
            f'{len(missing_pids)} provider IDs appear in blocks but not in '
            f'providers JSON. Examples: {missing_pids[:10]}'
        )

    return {
        # Scalar outputs consumed by Layer 1 + 2
        'T_star':                   T_star,
        'phase_offset':             int(phase_offset),
        'phase_labels':             phase_labels,   # ← week_index-keyed
        'block_template':           template_df.to_dict(orient='records'),
        'exceptions_list':          exceptions,
        'exception_rate_per_series': exception_rate,
        'utilization_per_provider': utilization,
        'bic_scores_list':          scores.to_dict(orient='records'),
        'offset_scores':            offset_scores.to_dict(orient='records'),
        'warnings':                 warnings_list,
        # Private DataFrames consumed by run_prelayer.py (prefixed with _)
        '_canonical_blocks_df':     df,
        '_template_df':             template_df,
        '_bic_df':                  scores,
        '_offset_df':               offset_scores,
    }

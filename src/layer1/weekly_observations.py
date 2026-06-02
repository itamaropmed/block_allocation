"""
Layer 1 — Weekly Observations  (standalone, no src imports)
============================================================
Key fixes vs original:
  1. _normalize_blocks_from_raw_json() added — parses the Geisinger nested
     JSON fields (occurrence.start/end, current_blockholder.provider_id,
     room.type, manual_early_release) into flat DataFrame columns that the
     rest of the file expects.
  2. phase_labels lookup bug fixed — was looking up week_start date strings
     in a dict keyed by week_index integers → always returned 0.
     Now correctly looks up by week_index.
  3. All src.* imports removed — file is fully standalone.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


DAY_TO_INT = {
    'monday': 0, 'mon': 0,
    'tuesday': 1, 'tue': 1, 'tues': 1,
    'wednesday': 2, 'wed': 2,
    'thursday': 3, 'thu': 3, 'thur': 3, 'thurs': 3,
    'friday': 4, 'fri': 4,
    'saturday': 5, 'sat': 5,
    'sunday': 6, 'sun': 6,
}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    lower_to_real = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_to_real:
            return lower_to_real[key]
    for cand in candidates:
        key = cand.lower()
        for c in df.columns:
            name = str(c).lower()
            if name.endswith('.' + key) or name.endswith('_' + key):
                return c
    return None


def _normalize_day_value(x: Any) -> int | float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, np.integer)):
        v = int(x)
        return v if 0 <= v <= 6 else (v - 1 if 1 <= v <= 7 else np.nan)
    s = str(x).strip()
    if not s:
        return np.nan
    try:
        v = int(float(s))
        return v if 0 <= v <= 6 else (v - 1 if 1 <= v <= 7 else np.nan)
    except ValueError:
        pass
    return DAY_TO_INT.get(s.lower(), np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Geisinger JSON normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_blocks_from_raw_json(blocks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw Geisinger blocks JSON (with nested occurrence, current_blockholder,
    room, manual_early_release) into the flat DataFrame the rest of Layer 1 expects.

    Output columns (guaranteed):
        block_historical_id, provider_id, physical_site, room_type,
        occurrence_start, occurrence_end, duration_min,
        manual_early_release_min, week_start (str YYYY-MM-DD, Monday),
        week_index (int 0-based), day_of_week (int 0=Mon),
        service_line_from_block (str, may be 'Unknown')
    """
    b = blocks_df.copy()

    # ── provider_id ──────────────────────────────────────────────────────────
    if 'provider_id' not in b.columns:
        # IMPORTANT: pd.json_normalize often creates BOTH:
        #   current_blockholder.provider_id  -> the real provider id
        #   current_blockholder              -> NaN when the nested object was expanded
        # The flattened provider_id column must therefore be preferred.  The
        # previous order looked at current_blockholder first, converted the NaNs
        # to OPEN, and made every block look open.
        if 'current_blockholder.provider_id' in b.columns:
            b['provider_id'] = b['current_blockholder.provider_id'].fillna('OPEN').astype(str).str.strip()
        elif 'current_blockholder' in b.columns:
            def _extract_pid(val):
                if isinstance(val, dict):
                    return str(val.get('provider_id', 'OPEN') or 'OPEN').strip()
                return str(val).strip() if pd.notna(val) else 'OPEN'
            b['provider_id'] = b['current_blockholder'].map(_extract_pid)
        else:
            b['provider_id'] = 'OPEN'
    b['provider_id'] = b['provider_id'].fillna('OPEN').astype(str).str.strip()
    b.loc[b['provider_id'].isin(['', 'None', 'nan', 'NaN', 'NULL', 'null']), 'provider_id'] = 'OPEN'

    # ── site ─────────────────────────────────────────────────────────────────
    if 'physical_site' not in b.columns:
        site_col = _find_col(b, ['site', 'physical_site', 'location', 'facility'])
        b['physical_site'] = b[site_col].fillna('Unknown').astype(str) if site_col else 'Unknown'

    # ── room_type ────────────────────────────────────────────────────────────
    if 'room_type' not in b.columns:
        if 'room' in b.columns:
            def _extract_room(val):
                if isinstance(val, dict):
                    return str(val.get('type', 'Unknown') or 'Unknown').strip()
                return str(val).strip() if pd.notna(val) else 'Unknown'
            b['room_type'] = b['room'].map(_extract_room)
        elif 'room.type' in b.columns:
            b['room_type'] = b['room.type'].fillna('Unknown').astype(str)
        else:
            b['room_type'] = 'Unknown'

    # ── occurrence start / end → datetime ────────────────────────────────────
    if 'occurrence_start' not in b.columns:
        if 'occurrence' in b.columns:
            def _extract_start(val):
                if isinstance(val, dict):
                    return val.get('start', None)
                return None
            def _extract_end(val):
                if isinstance(val, dict):
                    return val.get('end', None)
                return None
            b['occurrence_start'] = b['occurrence'].map(_extract_start)
            b['occurrence_end']   = b['occurrence'].map(_extract_end)
        elif 'occurrence.start' in b.columns:
            b['occurrence_start'] = b['occurrence.start']
            b['occurrence_end']   = b.get('occurrence.end', pd.Series([None] * len(b)))
        else:
            # no time info — can't compute duration / week
            b['occurrence_start'] = None
            b['occurrence_end']   = None

    b['occurrence_start'] = pd.to_datetime(b['occurrence_start'], utc=True, errors='coerce')
    b['occurrence_end']   = pd.to_datetime(b['occurrence_end'],   utc=True, errors='coerce')

    # ── duration_min ─────────────────────────────────────────────────────────
    if 'duration_min' not in b.columns:
        valid = b['occurrence_start'].notna() & b['occurrence_end'].notna()
        b['duration_min'] = 0.0
        b.loc[valid, 'duration_min'] = (
            (b.loc[valid, 'occurrence_end'] - b.loc[valid, 'occurrence_start'])
            .dt.total_seconds() / 60.0
        ).clip(lower=0.0)

    # ── manual_early_release_min ──────────────────────────────────────────────
    if 'manual_early_release_min' not in b.columns:
        if 'manual_early_release' in b.columns:
            er = pd.to_datetime(b['manual_early_release'], utc=True, errors='coerce')
            valid = er.notna() & b['occurrence_end'].notna()
            b['manual_early_release_min'] = 0.0
            b.loc[valid, 'manual_early_release_min'] = (
                (b.loc[valid, 'occurrence_end'] - er[valid])
                .dt.total_seconds() / 60.0
            ).clip(lower=0.0)
        else:
            b['manual_early_release_min'] = 0.0

    # ── week_start (Monday of the week), week_index, day_of_week ─────────────
    if 'week_start' not in b.columns or 'week_index' not in b.columns:
        has_ts = b['occurrence_start'].notna()
        b['_date'] = b['occurrence_start'].dt.date
        # Monday of that week
        b['_monday'] = b['occurrence_start'] - pd.to_timedelta(
            b['occurrence_start'].dt.weekday, unit='D'
        )
        b['week_start'] = b['_monday'].dt.date.astype(str).where(has_ts, None)

        # Build week_index: 0-based ordering of unique Mondays
        unique_mondays = sorted(b.loc[has_ts, 'week_start'].dropna().unique())
        monday_to_idx  = {w: i for i, w in enumerate(unique_mondays)}
        b['week_index'] = b['week_start'].map(monday_to_idx).fillna(0).astype(int)

        b['day_of_week'] = b['occurrence_start'].dt.weekday.where(has_ts, -1).fillna(-1).astype(int)

        b.drop(columns=['_date', '_monday'], inplace=True, errors='ignore')

    # ── service_line_from_block (best-effort) ─────────────────────────────────
    if 'service_line_from_block' not in b.columns:
        sl_col = _find_col(b, ['service_line', 'service', 'specialty'])
        if sl_col:
            b['service_line_from_block'] = b[sl_col].fillna('Unknown').astype(str)
        else:
            b['service_line_from_block'] = 'Unknown'

    # ── block_historical_id ───────────────────────────────────────────────────
    if 'block_historical_id' not in b.columns:
        b['block_historical_id'] = [f'BLOCK_ROW_{i}' for i in range(len(b))]

    return b.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Provider → service line map
# ─────────────────────────────────────────────────────────────────────────────

def _provider_service_map(providers_df: pd.DataFrame) -> dict:
    if providers_df is None or providers_df.empty or 'provider_id' not in providers_df.columns:
        return {}
    for col in ('primary_service_line', 'service_line', 'service_lines'):
        if col in providers_df.columns:
            service_col = col
            break
    else:
        return {}
    out = {}
    for _, r in providers_df.iterrows():
        pid = str(r.get('provider_id', '')).strip()
        if not pid:
            continue
        val = r.get(service_col, 'Unknown')
        if isinstance(val, list):
            val = val[0] if val else 'Unknown'
        out[pid] = str(val) if pd.notna(val) else 'Unknown'
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cases standardisation
# ─────────────────────────────────────────────────────────────────────────────

def _standardize_cases(cases_df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    if cases_df is None or cases_df.empty:
        return pd.DataFrame()

    cases = cases_df.copy()
    provider_col = _find_col(cases, ['provider_id','surgeon_id','primary_surgeon_id',
                                      'case_provider_id','performing_provider_id'])
    case_col     = _find_col(cases, ['block_case_minutes','case_minutes','casetime_min',
                                      'case_time_minutes','surgery_minutes','duration_minutes'])
    turn_col     = _find_col(cases, ['turnover_time','turnover_minutes','turnover_min',
                                      'room_turnover_minutes'])
    day_col      = _find_col(cases, ['day_of_week','dow','weekday'])

    missing = [f for f, c in [('provider_id', provider_col),
                                ('block_case_minutes', case_col),
                                ('turnover_time', turn_col)] if c is None]
    if missing:
        warnings.append(
            'Cases JSON is missing required columns: ' + ', '.join(missing)
            + '. Layer 1 will fall back to proxy demand from allocated block minutes.'
        )
        return pd.DataFrame()

    rename = {provider_col: 'provider_id', case_col: 'block_case_minutes',
              turn_col: 'turnover_time'}
    if day_col and day_col != 'day_of_week':
        rename[day_col] = 'day_of_week'
    cases = cases.rename(columns=rename)

    cases['provider_id']        = cases['provider_id'].astype(str).str.strip()
    cases['block_case_minutes'] = pd.to_numeric(cases['block_case_minutes'], errors='coerce').fillna(0.0)
    cases['turnover_time']      = pd.to_numeric(cases['turnover_time'],      errors='coerce').fillna(0.0)

    blank_mask = cases['provider_id'].isin(['', 'None', 'nan', 'NaN', 'NULL', 'null'])
    n_blank = int(blank_mask.sum())
    if n_blank > 0:
        warnings.append(f'Dropped {n_blank} case rows with blank provider_id.')
        cases = cases.loc[~blank_mask].copy()

    if 'day_of_week' in cases.columns:
        cases['day_of_week'] = cases['day_of_week'].map(_normalize_day_value)
        bad = cases['day_of_week'].isna()
        if bad.any():
            warnings.append(f'Dropped {int(bad.sum())} case rows with unreadable day_of_week.')
            cases = cases.loc[~bad].copy()
        cases['day_of_week'] = cases['day_of_week'].astype(int)

    return cases.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Block preparation
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_blocks(
    blocks_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    prelayer: Dict[str, Any],
) -> pd.DataFrame:
    service_map  = _provider_service_map(providers_df)
    # phase_labels keys are week_index integers (stored as strings in JSON)
    phase_labels = {str(k): int(v) for k, v in (prelayer.get('phase_labels') or {}).items()}

    b = blocks_df.copy()

    # Ensure required columns exist (normalization should have done this already)
    for col, default in [
        ('manual_early_release_min', 0.0),
        ('duration_min', 0.0),
        ('provider_id', 'OPEN'),
        ('block_historical_id', None),
        ('week_start', None),
        ('week_index', 0),
        ('day_of_week', -1),
    ]:
        if col not in b.columns:
            b[col] = default

    if b['block_historical_id'].isna().all():
        b['block_historical_id'] = [f'BLOCK_ROW_{i}' for i in range(len(b))]

    b['provider_id'] = b['provider_id'].astype(str).str.strip()
    b.loc[b['provider_id'].isin(['', 'None', 'nan', 'NaN', 'NULL', 'null']), 'provider_id'] = 'OPEN'

    # ── FIX: use week_index (int) to look up phase_labels, not week_start ────
    b['rotation_phase'] = (
        b['week_index'].astype(str)
         .map(phase_labels)
         .fillna(0)
         .astype(int)
    )

    b['service_line'] = (
        b['provider_id'].map(service_map)
        .fillna(b.get('service_line_from_block', 'Unknown'))
        .fillna('Unknown')
    )

    b['duration_min']              = pd.to_numeric(b['duration_min'],              errors='coerce').fillna(0.0)
    b['manual_early_release_min']  = pd.to_numeric(b['manual_early_release_min'],  errors='coerce').fillna(0.0)
    b['allocated_min']             = (b['duration_min'] - b['manual_early_release_min']).clip(lower=0.0)

    return b


def _build_block_agg(b: pd.DataFrame) -> pd.DataFrame:
    agg = (
        b.groupby(
            ['provider_id', 'service_line', 'week_start', 'week_index',
             'rotation_phase', 'day_of_week'],
            dropna=False,
        )
        .agg(
            allocated_min=('allocated_min', 'sum'),
            early_release_min=('manual_early_release_min', 'sum'),
            n_blocks=('block_historical_id', 'count'),
        )
        .reset_index()
    )
    return agg[agg['provider_id'] != 'OPEN'].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Case distribution helpers
# ─────────────────────────────────────────────────────────────────────────────

def _distribute_case_totals_over_rows(
    target_rows: pd.DataFrame,
    total_case_col: str = 'total_casetime_min',
    total_turn_col: str = 'total_turnover_min',
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ['provider_id', 'day_of_week']
    out          = target_rows.copy()
    group_alloc  = out.groupby(group_cols, dropna=False)['allocated_min'].transform('sum')
    group_size   = out.groupby(group_cols, dropna=False)['provider_id'].transform('size').replace(0, 1)
    weights      = np.where(group_alloc > 0, out['allocated_min'] / group_alloc, 1.0 / group_size)
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
    if 'day_of_week' not in cases.columns:
        warnings.append(
            'Cases have no day_of_week — distributing provider totals '
            'across all historical blocks.'
        )
        totals = (
            cases.groupby('provider_id', dropna=False)
            .agg(total_casetime_min=('block_case_minutes', 'sum'),
                 total_turnover_min=('turnover_time', 'sum'),
                 n_cases=('provider_id', 'size'))
            .reset_index()
        )
        block_agg = _build_block_agg(b)
        obs = block_agg.merge(totals, on='provider_id', how='inner')
        obs = _distribute_case_totals_over_rows(obs, group_cols=['provider_id'])
        obs['source_mode'] = 'provider_totals_distributed_to_provider_blocks'
        return obs.drop(columns=['total_casetime_min', 'total_turnover_min', 'n_cases'], errors='ignore')

    warnings.append(
        'Cases have provider_id + day_of_week but no block_id/timestamp. '
        'Distributing case demand by provider/day across matching block weeks.'
    )
    case_day_totals = (
        cases.groupby(['provider_id', 'day_of_week'], dropna=False)
        .agg(total_casetime_min=('block_case_minutes', 'sum'),
             total_turnover_min=('turnover_time', 'sum'),
             n_cases=('provider_id', 'size'))
        .reset_index()
    )
    block_agg = _build_block_agg(b)
    matched = block_agg.merge(case_day_totals, on=['provider_id', 'day_of_week'], how='inner')
    matched = _distribute_case_totals_over_rows(matched, group_cols=['provider_id', 'day_of_week'])
    matched['source_mode'] = 'provider_day_cases_distributed_to_matching_blocks'

    matched_keys = set(zip(matched['provider_id'].astype(str), matched['day_of_week'].astype(int)))
    all_keys     = set(zip(case_day_totals['provider_id'].astype(str), case_day_totals['day_of_week'].astype(int)))
    missing_keys = all_keys - matched_keys
    service_map  = _provider_service_map(providers_df)
    extras = []

    if missing_keys:
        provider_weeks = (
            block_agg.groupby(['provider_id', 'week_start', 'week_index', 'rotation_phase'], dropna=False)
            .agg(allocated_min=('allocated_min', 'sum'),
                 early_release_min=('early_release_min', 'sum'),
                 n_blocks=('n_blocks', 'sum'),
                 service_line=('service_line', 'first'))
            .reset_index()
        )
        missing_df = case_day_totals[
            case_day_totals.apply(
                lambda r: (str(r['provider_id']), int(r['day_of_week'])) in missing_keys, axis=1
            )
        ].copy()

        for _, r in missing_df.iterrows():
            pid  = str(r['provider_id'])
            dow  = int(r['day_of_week'])
            tc   = float(r['total_casetime_min'])
            tt   = float(r['total_turnover_min'])
            nc   = int(r['n_cases'])
            pw   = provider_weeks[provider_weeks['provider_id'].astype(str) == pid].copy()

            if not pw.empty:
                tmp = pw.copy()
                tmp['day_of_week']         = dow
                tmp['total_casetime_min']  = tc
                tmp['total_turnover_min']  = tt
                tmp['n_cases']             = nc
                tmp = _distribute_case_totals_over_rows(tmp, group_cols=['provider_id', 'day_of_week'])
                tmp['source_mode'] = 'provider_day_distributed_to_provider_weeks'
                extras.append(tmp)
            else:
                alloc = max(tc / max(proxy_util, 1e-6), tc + tt, 1.0)
                extras.append(pd.DataFrame([{
                    'provider_id': pid,
                    'service_line': service_map.get(pid, 'Unknown'),
                    'week_start': 'CASE_ONLY', 'week_index': 0,
                    'rotation_phase': 0, 'day_of_week': dow,
                    'allocated_min': alloc, 'early_release_min': 0.0,
                    'n_blocks': 0, 'casetime_min': tc, 'turnover_min': tt,
                    'source_mode': 'case_only_no_matching_block',
                }]))

        warnings.append(
            f'{len(missing_keys)} provider/day groups had no matching historical block.'
        )

    pieces = [matched] + extras
    obs = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return obs.drop(columns=['total_casetime_min', 'total_turnover_min', 'n_cases'], errors='ignore')


def _case_join_by_block_id(cases, b, block_id_col, warnings):
    joined = cases.merge(
        b[['block_historical_id','week_start','week_index','day_of_week',
           'rotation_phase','service_line']],
        left_on=block_id_col, right_on='block_historical_id', how='left',
    )
    n_miss = int(joined['week_start'].isna().sum())
    if n_miss > 0:
        warnings.append(f'{n_miss} cases had block_id not found in blocks — dropped.')
        joined = joined.dropna(subset=['week_start']).copy()
    if joined.empty:
        return pd.DataFrame()
    return (
        joined.groupby(['provider_id','service_line','week_start','week_index',
                        'rotation_phase','day_of_week'], dropna=False)
        .agg(casetime_min=('block_case_minutes','sum'), turnover_min=('turnover_time','sum'))
        .reset_index()
    )


def _case_join_by_timestamp(cases, b, providers_df, dt_col, prelayer, warnings):
    service_map  = _provider_service_map(providers_df)
    phase_labels = {str(k): int(v) for k, v in (prelayer.get('phase_labels') or {}).items()}
    cj = cases.copy()
    cj[dt_col] = pd.to_datetime(cj[dt_col], utc=True, errors='coerce')
    bad = cj[dt_col].isna()
    if bad.any():
        warnings.append(f'Dropped {int(bad.sum())} cases with unreadable timestamp {dt_col}.')
        cj = cj.loc[~bad].copy()
    if cj.empty:
        return pd.DataFrame()

    cj['week_start'] = (cj[dt_col] - pd.to_timedelta(cj[dt_col].dt.weekday, unit='D')).dt.date.astype(str)
    week_order = {w: i for i, w in enumerate(sorted(b['week_start'].dropna().unique()))}
    cj['week_index']     = cj['week_start'].map(week_order).fillna(0).astype(int)
    cj['day_of_week']    = cj[dt_col].dt.weekday.astype(int)
    cj['rotation_phase'] = cj['week_index'].astype(str).map(phase_labels).fillna(0).astype(int)
    cj['service_line']   = cj['provider_id'].map(service_map).fillna('Unknown')

    return (
        cj.groupby(['provider_id','service_line','week_start','week_index',
                    'rotation_phase','day_of_week'], dropna=False)
        .agg(casetime_min=('block_case_minutes','sum'), turnover_min=('turnover_time','sum'))
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_weekly_observations(
    blocks_df: pd.DataFrame,
    providers_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    prelayer: Dict[str, Any],
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    warnings_list: list[str] = []
    proxy_util = float(config.get('layer1', {}).get('proxy_case_utilization', 0.62))
    proxy_turn = float(config.get('layer1', {}).get('proxy_turnover_fraction', 0.08))

    # ── FIX: normalise raw Geisinger JSON → flat DataFrame ───────────────────
    blocks_df = _normalize_blocks_from_raw_json(blocks_df)

    b         = _prepare_blocks(blocks_df, providers_df, prelayer)
    block_agg = _build_block_agg(b)
    cases     = _standardize_cases(cases_df, warnings_list)

    if cases.empty:
        warnings_list.append(
            'No usable cases after standardization. Running in proxy mode.'
        )
        obs = block_agg.copy()
        obs['casetime_min'] = obs['allocated_min'] * proxy_util
        obs['turnover_min'] = obs['casetime_min']  * proxy_turn
        obs['source_mode']  = 'proxy_from_allocated_blocks'
    else:
        block_id_col = _find_col(cases, ['block_historical_id','block_id','historical_block_id'])
        dt_col       = _find_col(cases, ['case_start','surgery_start','procedure_start',
                                          'scheduled_start','actual_start','start',
                                          'occurrence.start'])
        case_agg    = pd.DataFrame()
        source_mode = ''

        if block_id_col is not None:
            case_agg = _case_join_by_block_id(cases, b, block_id_col, warnings_list)
            if not case_agg.empty:
                source_mode = 'cases_joined_by_block_id'

        if case_agg.empty and dt_col is not None:
            case_agg = _case_join_by_timestamp(cases, b, providers_df, dt_col, prelayer, warnings_list)
            if not case_agg.empty:
                source_mode = 'cases_mapped_by_timestamp'

        if case_agg.empty:
            obs = _provider_day_distribution_fallback(
                cases=cases, b=b, providers_df=providers_df,
                warnings=warnings_list, proxy_util=proxy_util,
            )
        else:
            obs = case_agg.merge(
                block_agg,
                on=['provider_id','service_line','week_start','week_index',
                    'rotation_phase','day_of_week'],
                how='left',
            )
            obs['allocated_min']    = obs['allocated_min'].fillna(0.0)
            obs['early_release_min'] = obs['early_release_min'].fillna(0.0)
            obs['n_blocks']          = obs['n_blocks'].fillna(0).astype(int)
            obs['source_mode']       = source_mode

    if obs.empty:
        raise ValueError(
            'Layer 1 could not build any weekly observations. '
            'Check that blocks and cases contain provider_id values.'
        )

    for col in ['allocated_min', 'early_release_min', 'casetime_min', 'turnover_min']:
        if col not in obs.columns:
            obs[col] = 0.0
        obs[col] = pd.to_numeric(obs[col], errors='coerce').fillna(0.0)

    if 'n_blocks' not in obs.columns:
        obs['n_blocks'] = 0
    obs['n_blocks']         = pd.to_numeric(obs['n_blocks'], errors='coerce').fillna(0).astype(int)
    obs['service_line']     = obs.get('service_line', pd.Series(['Unknown'] * len(obs))).fillna('Unknown').astype(str)
    obs['provider_id']      = obs['provider_id'].astype(str)
    obs['week_index']       = pd.to_numeric(obs['week_index'],       errors='coerce').fillna(0).astype(int)
    obs['rotation_phase']   = pd.to_numeric(obs['rotation_phase'],   errors='coerce').fillna(0).astype(int)
    obs['day_of_week']      = pd.to_numeric(obs['day_of_week'],      errors='coerce').fillna(-1).astype(int)

    obs['casetime_util'] = np.where(obs['allocated_min'] > 0,
                                    obs['casetime_min'] / obs['allocated_min'], 0.0)
    obs['turnover_util'] = np.where(obs['allocated_min'] > 0,
                                    obs['turnover_min'] / obs['allocated_min'], 0.0)
    obs['total_util'] = np.where(obs['allocated_min'] > 0,
                                 (obs['casetime_min'] + obs['turnover_min']) / obs['allocated_min'], 0.0)
    obs['casetime_util'] = obs['casetime_util'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    obs['turnover_util'] = obs['turnover_util'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    obs['total_util'] = obs['total_util'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    obs['is_included']   = True

    return obs.sort_values(['provider_id', 'day_of_week', 'week_index']).reset_index(drop=True), \
           {'warnings': warnings_list}


def early_release_projected(blocks_df: pd.DataFrame, optimization_weeks: int = 13) -> pd.DataFrame:
    # Normalise first in case raw JSON was passed
    if 'week_index' not in blocks_df.columns:
        blocks_df = _normalize_blocks_from_raw_json(blocks_df)

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

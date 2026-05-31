from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .io import load_json
from .time_utils import parse_dt, minutes_since_midnight, duration_minutes, infer_physical_site


def flatten_blocks(blocks: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for b in blocks:
        occ = b.get('occurrence') or {}
        holder = b.get('current_blockholder') or {}
        room = b.get('room') or {}
        start = parse_dt(occ.get('start'))
        end = parse_dt(occ.get('end'))
        provider_id = holder.get('provider_id')
        provider_name = holder.get('name')
        if b.get('is_open') or not provider_id:
            provider_id = 'OPEN'
            provider_name = provider_name or 'Open'
        room_type = room.get('type') or 'UNKNOWN_ROOM'
        rows.append({
            'block_historical_id': b.get('block_historical_id'),
            'is_open': bool(b.get('is_open', False)),
            'service_line_from_block': b.get('site') or 'Unknown',
            'physical_site': infer_physical_site(room_type),
            'start_ts': start,
            'end_ts': end,
            'date': start.date().isoformat() if not pd.isna(start) else None,
            'week_start': (start - pd.Timedelta(days=start.weekday())).date().isoformat() if not pd.isna(start) else None,
            'day_of_week': int(start.weekday()) if not pd.isna(start) else -1,
            'start_min': minutes_since_midnight(start),
            'end_min': minutes_since_midnight(end),
            'duration_min': duration_minutes(start, end),
            'provider_id': str(provider_id),
            'provider_name': provider_name or str(provider_id),
            'room_type': room_type,
            'manual_early_release_raw': b.get('manual_early_release'),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    week_order = {w: i for i, w in enumerate(sorted(df['week_start'].dropna().unique()))}
    df['week_index'] = df['week_start'].map(week_order).astype('Int64')
    df['manual_early_release_min'] = 0.0
    return df


def flatten_providers(providers: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in providers:
        service_lines = p.get('service_line') or []
        exclusive_sites = p.get('exclusive_sites') or []
        rows.append({
            'provider_id': str(p.get('provider_id')),
            'provider_name': p.get('name') or str(p.get('provider_id')),
            'service_lines': service_lines,
            'primary_service_line': service_lines[0] if service_lines else 'Unknown',
            'exclusive_sites': exclusive_sites,
        })
    return pd.DataFrame(rows)


def load_raw_tables(blocks_json: str | Path, providers_json: str | Path, cases_json: str | Path | None = None):
    blocks = load_json(blocks_json)
    providers = load_json(providers_json)
    blocks_df = flatten_blocks(blocks)
    providers_df = flatten_providers(providers)
    cases = load_json(cases_json, required=False) if cases_json else None
    cases_df = pd.DataFrame(cases) if isinstance(cases, list) else pd.DataFrame()
    return blocks_df, providers_df, cases_df

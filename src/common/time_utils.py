from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


def parse_dt(x: Any) -> pd.Timestamp:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return pd.NaT
    return pd.to_datetime(x, utc=True, errors='coerce')


def minutes_since_midnight(ts: pd.Timestamp) -> float:
    if pd.isna(ts):
        return float('nan')
    return ts.hour * 60 + ts.minute + ts.second / 60.0


def duration_minutes(start: pd.Timestamp, end: pd.Timestamp) -> float:
    if pd.isna(start) or pd.isna(end):
        return 0.0
    return max(0.0, (end - start).total_seconds() / 60.0)


def round_to_grid(x: float, grid: int = 5) -> int:
    if pd.isna(x):
        return 0
    return int(round(float(x) / grid) * grid)


def fmt_minutes(m: float) -> str:
    m = int(round(m))
    return f'{m//60:02d}:{m%60:02d}'


def infer_physical_site(room_type: str) -> str:
    """Map room names like 'GMC OR 34' or 'GCMC ENDO 01' to provider exclusive_site labels."""
    if not room_type:
        return 'UNKNOWN'
    s = str(room_type).upper().strip()
    tokens = re.split(r'\s+', s)
    known = ['GCMC', 'GMC', 'GWV', 'GSWB', 'OSW', 'OSCP', 'OSSC', 'GLH', 'GBH', 'GECL', 'GMCM', 'GSACH', 'GJSH', 'OSHP']
    site = next((t for t in tokens if t in known), tokens[0] if tokens else 'UNKNOWN')
    if 'ENDO' in s or 'ENDOSCOPY' in s:
        return f'ENDOSCOPY {site}'
    return f'OR {site}'


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)

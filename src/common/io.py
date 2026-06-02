from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path, required: bool = True) -> Any:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f'Missing required JSON file: {p}')
        return None
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj) if not isinstance(obj, (list, dict, tuple)) else False:
        return None
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(json_safe(obj), f, indent=2, ensure_ascii=False)
    return p


def save_df(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == '.parquet':
        try:
            df.to_parquet(p, index=False)
        except Exception:
            alt = p.with_suffix('.csv')
            df.to_csv(alt, index=False)
            return alt
    else:
        df.to_csv(p, index=False)
    return p


def read_df(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == '.parquet':
        return pd.read_parquet(p)
    return pd.read_csv(p)

"""
Layer 2 — Multi-axis Pareto Frontier
====================================

This file is intentionally kept as a separate module because the project had an
older `src/layer2/pareto_frontier.py` that only swept a small number of themes or
only a non-current-holder penalty.  Replace the old file with this one.

The real optimizer lives in `optimizer.py`; this module provides a clean Pareto
API and backward-compatible entry point for older `run_layer2.py` files that call
`build_pareto_candidates(...)`.

Pareto axes produced:
    - useful_utilization
    - continuity_score
    - stability_score
    - preference_score
    - overall_score

Pareto families produced by the current optimizer:
    - eps_utilization_anchor_floor_1p000
    - eps_continuity_floor_*
    - eps_preference_floor_*
    - eps_stability_floor_*
    - eps_balanced_floor_*

The current implementation is an ε-constraint frontier: Layer 2 first finds the
utilization-first optimum A*, then every secondary plan enforces
allocated_minutes >= ε × A*.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:  # same-folder execution: python src/layer2/run_layer2.py
    import optimizer as opt
except ImportError:  # package execution: python -m src.layer2.run_layer2
    from . import optimizer as opt  # type: ignore

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return str(x)


def _write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Public Pareto helpers
# ---------------------------------------------------------------------------

def pareto_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Return the non-dominated rows for the four Layer-2 objective axes.

    All four axes are maximized.  Missing axes are created as zeros so this can
    safely process partial/failed runs without crashing.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    axes = ["useful_utilization", "continuity_score", "stability_score", "preference_score"]
    work = df.copy()
    for c in axes:
        if c not in work.columns:
            work[c] = 0.0
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

    values = work[axes].to_numpy(dtype=float)
    efficient = np.ones(len(work), dtype=bool)
    for i, row in enumerate(values):
        # j dominates i if j >= i in all axes and j > i in at least one axis.
        dominated_by_any = np.any(np.all(values >= row, axis=1) & np.any(values > row, axis=1))
        efficient[i] = not dominated_by_any

    work["pareto_efficient"] = efficient
    return work[work["pareto_efficient"]].copy().reset_index(drop=True)


def run_pareto_grid(
    slots: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the real multi-axis Pareto grid.

    This delegates to optimizer.run_pareto_grid when available, which uses the
    same candidate-pair construction and min-cost-flow solver as the main Layer-2
    utilization-first solve.  The wrapper exists so `pareto_frontier.py` remains
    the visible Pareto module in the project folder.
    """
    if not hasattr(opt, "run_pareto_grid"):
        raise RuntimeError("optimizer.py does not expose run_pareto_grid; replace optimizer.py with the v3 file too.")

    result = opt.run_pareto_grid(slots, targets, cfg)
    # New optimizer.py returns (pareto_all, pareto_frontier, plan_records).
    # Keep this module backward-compatible with older callers that expect two.
    if isinstance(result, tuple) and len(result) == 3:
        pareto_all, pareto_frontier, _plan_records = result
    else:
        pareto_all, pareto_frontier = result  # type: ignore[misc]

    # Be defensive: ensure the columns the downstream notebooks expect exist.
    for df in [pareto_all, pareto_frontier]:
        if df is None or df.empty:
            continue
        for c in ["useful_utilization", "continuity_score", "stability_score", "preference_score", "overall_score"]:
            if c not in df.columns:
                df[c] = 0.0

    # Ensure pareto_all is marked.  Use the same non-dominated calculation, then
    # match rows back by theme/profile columns instead of relying on reset indexes.
    if pareto_all is not None and not pareto_all.empty and "pareto_efficient" not in pareto_all.columns:
        marked = pareto_filter(pareto_all)
        if "theme" in pareto_all.columns and "theme" in marked.columns:
            efficient_themes = set(marked.loc[marked["pareto_efficient"].astype(bool), "theme"].astype(str))
            pareto_all["pareto_efficient"] = pareto_all["theme"].astype(str).isin(efficient_themes)
        else:
            pareto_all["pareto_efficient"] = marked["pareto_efficient"].to_numpy(dtype=bool)

    if pareto_frontier is not None and not pareto_frontier.empty and "pareto_efficient" not in pareto_frontier.columns:
        pareto_frontier["pareto_efficient"] = True

    return pareto_all, pareto_frontier


def _targets_from_scenarios_in_memory(
    scenarios: pd.DataFrame,
    alpha: float,
    providers: pd.DataFrame,
) -> pd.DataFrame:
    """Build provider×day SAA alpha targets from an in-memory scenarios table.

    This mirrors optimizer.load_layer1_targets(...) but does not require writing
    a temporary CSV.  It exists for backward compatibility with older run files
    that loaded scenarios before calling the Pareto module.
    """
    if scenarios is None or scenarios.empty:
        raise ValueError("scenarios table is empty; run Layer 1 first.")

    df = scenarios.copy()
    provider_col = opt.find_col(df, opt.PROVIDER_ALIASES, required=True, name="provider_id")
    day_col = opt.find_col(df, opt.DAY_ALIASES, required=True, name="day_of_week")
    scenario_col = opt.find_col(df, ["scenario_id", "scenario", "saa_scenario", "draw", "sample_id"], required=False)

    total_col = opt.find_col(
        df,
        ["total_demand_min", "demand_total_min", "demand_min", "scenario_total_min", "total_min"],
        required=False,
    )

    if total_col is not None:
        demand = pd.to_numeric(df[total_col], errors="coerce").fillna(0.0)
    else:
        case_candidates = [c for c in df.columns if "case" in str(c).lower() and "min" in str(c).lower()]
        turn_candidates = [c for c in df.columns if ("turn" in str(c).lower() or "turnover" in str(c).lower()) and "min" in str(c).lower()]
        if not case_candidates:
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            numeric = [c for c in numeric if c not in [day_col, scenario_col]]
            if not numeric:
                raise ValueError(f"Could not infer demand columns in scenarios. Columns={list(df.columns)}")
            demand = pd.to_numeric(df[numeric[0]], errors="coerce").fillna(0.0)
        else:
            case_col = opt.prefer_demand_col(case_candidates)
            turn_col = opt.prefer_demand_col(turn_candidates) if turn_candidates else None
            demand = pd.to_numeric(df[case_col], errors="coerce").fillna(0.0)
            if turn_col:
                demand = demand + pd.to_numeric(df[turn_col], errors="coerce").fillna(0.0)

    work = pd.DataFrame(
        {
            "provider_id": df[provider_col].map(opt.clean_provider_id),
            "day_of_week": df[day_col].map(opt.normalize_day_value),
            "scenario_id": df[scenario_col] if scenario_col else np.arange(len(df)),
            "scenario_demand_min": demand.clip(lower=0),
        }
    )
    work = work[(work["provider_id"] != "OPEN") & work["day_of_week"].notna()].copy()
    work["day_of_week"] = work["day_of_week"].astype(int)

    grouped = work.groupby(["provider_id", "day_of_week"], as_index=False).agg(
        target_q_min=("scenario_demand_min", lambda x: float(np.quantile(x.to_numpy(dtype=float), alpha))),
        mean_demand_min=("scenario_demand_min", "mean"),
        p50_demand_min=("scenario_demand_min", "median"),
        p95_demand_min=("scenario_demand_min", lambda x: float(np.quantile(x.to_numpy(dtype=float), 0.95))),
        n_scenarios=("scenario_demand_min", "count"),
    )
    grouped["target_q_int"] = grouped["target_q_min"].round().clip(lower=0).astype(int)
    grouped = grouped[grouped["target_q_int"] > 0].copy()

    providers_norm = opt.normalize_providers(providers) if providers is not None and not providers.empty else pd.DataFrame()
    if not providers_norm.empty:
        meta = providers_norm[["provider_id", "service_line", "site"]].drop_duplicates("provider_id")
        grouped = grouped.merge(meta, on="provider_id", how="left")
    else:
        grouped["service_line"] = "UNKNOWN"
        grouped["site"] = ""

    grouped["service_line"] = grouped["service_line"].fillna("UNKNOWN").astype(str)
    grouped["site"] = grouped["site"].fillna("").astype(str)
    grouped = grouped.sort_values(["day_of_week", "target_q_int", "provider_id"], ascending=[True, False, True]).reset_index(drop=True)
    grouped["target_idx"] = np.arange(len(grouped), dtype=int)
    return grouped


def _config_from_legacy_dict(config: Dict[str, Any], out: str | Path) -> Any:
    """Convert old nested dict config into optimizer.Layer2Config."""
    layer2 = dict(config.get("layer2", {})) if isinstance(config, dict) else {}

    cfg = opt.Layer2Config(output_dir=str(out))

    # Only assign attributes that exist on the current config dataclass.
    alias_map = {
        "alpha": "alpha",
        "saa_alpha": "alpha",
        "time_limit_s": "time_limit_s",
        "workers": "workers",
        "max_slots": "max_slots",
        "candidate_top_per_day": "candidate_top_per_day",
        "pareto_grid": "pareto_grid",
        "continuity_penalty": "continuity_penalty",
        "preference_penalty": "preference_penalty",
        "stability_penalty": "stability_penalty",
    }
    for old_key, new_key in alias_map.items():
        if old_key in layer2 and hasattr(cfg, new_key):
            setattr(cfg, new_key, layer2[old_key])

    # Top-level CLI-like overrides, if present.
    for key in ["alpha", "time_limit_s", "workers", "max_slots", "candidate_top_per_day", "pareto_grid"]:
        if isinstance(config, dict) and key in config and hasattr(cfg, key):
            setattr(cfg, key, config[key])

    return cfg


def build_pareto_candidates(
    prelayer: Dict[str, Any],
    providers: pd.DataFrame,
    scenarios: pd.DataFrame,
    early_release: Optional[pd.DataFrame],
    out: str | Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Backward-compatible Pareto entry point.

    Older `run_layer2.py` files call this function directly.  It now runs the
    same multi-axis Pareto as the new optimizer and writes the new artifacts.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = _config_from_legacy_dict(config, out)
    providers_norm = opt.normalize_providers(providers) if providers is not None else pd.DataFrame()
    template = opt.extract_template_from_prelayer(prelayer, pd.DataFrame())
    slots = opt.normalize_template_to_slots(template, cfg, early_release if early_release is not None else pd.DataFrame())
    targets = _targets_from_scenarios_in_memory(scenarios, float(cfg.alpha), providers_norm)

    pareto_all, frontier = run_pareto_grid(slots, targets, cfg)
    if frontier is None or frontier.empty:
        frontier = pareto_filter(pareto_all)

    recommended = {}
    if frontier is not None and not frontier.empty:
        recommended = frontier.sort_values("overall_score", ascending=False).iloc[0].to_dict()
    elif pareto_all is not None and not pareto_all.empty:
        recommended = pareto_all.sort_values("overall_score", ascending=False).iloc[0].to_dict()

    pareto_all.to_csv(out / "pareto_all.csv", index=False)
    frontier.to_csv(out / "pareto_frontier.csv", index=False)
    _write_json({"candidates": pareto_all.to_dict(orient="records")}, out / "pareto_frontier.json")
    _write_json(recommended, out / "recommended_candidate.json")

    return {
        "frontier": frontier,
        "pareto_all": pareto_all,
        "recommended": recommended,
        "dominated": [],
        "output_dir": str(out),
    }


__all__ = [
    "pareto_filter",
    "run_pareto_grid",
    "build_pareto_candidates",
]

"""
Layer 3 — SHAP Explainability, Counterfactuals, Risk, Pareto Diagnostics
========================================================================

This module is intentionally self-contained so you can place it either at:

    gen3_block_allocation_code_bundle/src/layer3/explainability.py

or run it as a standalone folder next to run_layer3.py.

It expects upstream artifacts from:
  Layer 1: features.csv, scenarios_long.csv, fitted model .joblib files,
           hierarchical_sigma_estimates.csv, early_release_projected.csv
  Layer 2: recommended_candidate.json, candidates_all/json/csv,
           coverage_recommended.csv or coverage_<theme>.csv,
           schedule/recommended schedule files when available.

The code is defensive: it discovers columns and file names where possible and
keeps producing partial outputs instead of crashing when an optional artifact is
missing.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import pickle
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

try:  # plotting is optional but recommended
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

log = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Layer3Config:
    layer1_dir: Path
    layer2_dir: Path
    out_dir: Path
    pre_layer_result: Path | None = None
    selected_theme: str | None = None
    alpha: float = 0.85
    optimization_weeks: int = 4
    block_minutes: int = 480
    default_target_utilization: float = 0.70
    top_k_features: int = 8
    max_shap_rows: int = 5000
    explain_scope: str = "latest"  # latest | all
    compute_interactions: bool = False
    max_interaction_rows: int = 150
    counterfactual_targets: list[float] = field(default_factory=lambda: [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    demand_multipliers: list[float] = field(default_factory=lambda: [0.80, 0.90, 1.00, 1.10, 1.20])
    previous_layer3_dir: Path | None = None
    rhat_threshold: float = 1.05
    drift_threshold: float = 0.50
    utilization_near_optimal_tol: float = 0.01
    write_plots: bool = True
    random_seed: int = 42


# =============================================================================
# Reason code registry
# =============================================================================

REASON_CODE_PATTERNS: list[tuple[str, str, str]] = [
    (r"^attn_weighted_util$|util", "utilization_history", "Provider's characteristic operating level"),
    (r"^attn_entropy$|entropy", "pattern_consistency", "How stable or concentrated the historical pattern is"),
    (r"^attn_recency_bias$|recency|trend", "trend_direction", "Whether demand is improving or declining recently"),
    (r"attn_turn_weighted|turnover.*hist|turn.*weighted", "turnover_history", "Characteristic turnover load"),
    (r"turn.*entropy", "turnover_consistency", "How stable the turnover pattern is"),
    (r"trailing_4w_mean|trailing_2w_mean|trailing_6w_mean", "recent_history", "Recent case demand level"),
    (r"trailing_8w_mean|trailing_12w_mean|rolling.*mean|history_mean", "longer_history", "Longer-run demand history"),
    (r"std|volatility|iqr|mad", "recent_volatility", "Recent demand volatility"),
    (r"exception_rate|contention|manual_early_release|early_release", "series_contention", "How contested or disrupted this block series is"),
    (r"service_line|peer|sl_", "service_line_context", "What peers in the same service line are doing"),
    (r"rotation_phase|phase|T_star|week_mod", "rotation_cycle_position", "Position in the detected rotation cycle"),
    (r"weeks_observed|n_obs|history_len|maturity", "data_maturity", "How much history exists for this provider"),
    (r"sin_woy|cos_woy|season|month|week_of_year|woy", "seasonality", "Time-of-year effect"),
    (r"day_of_week|dow|weekday", "day_pattern", "Day-of-week pattern"),
    (r"allocated|capacity|block_min|slot", "capacity_context", "Historical block capacity context"),
]

REASON_PRIORITY = {
    "utilization_history": 100,
    "recent_history": 90,
    "turnover_history": 85,
    "service_line_context": 80,
    "series_contention": 75,
    "recent_volatility": 70,
    "pattern_consistency": 65,
    "trend_direction": 60,
    "rotation_cycle_position": 50,
    "data_maturity": 40,
    "seasonality": 30,
    "day_pattern": 25,
    "capacity_context": 20,
    "model_feature": 1,
}


def reason_for_feature(feature: str) -> tuple[str, str]:
    f = str(feature)
    for pattern, code, plain in REASON_CODE_PATTERNS:
        if re.search(pattern, f, flags=re.IGNORECASE):
            return code, plain
    return "model_feature", f


# =============================================================================
# Generic helpers
# =============================================================================


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if pd.isna(x) if not isinstance(x, (list, dict, tuple, set)) else False:
        return None
    raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=json_default, ensure_ascii=False), encoding="utf-8")
    return p


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_pickle(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    return p


def find_first_file(base: Path, patterns: Sequence[str]) -> Path | None:
    for pat in patterns:
        matches = sorted(base.glob(pat))
        if matches:
            return matches[0]
    return None


def safe_read_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as exc:
        log.warning("Could not read CSV %s: %s", p, exc)
        return pd.DataFrame()


def first_existing_col(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    cols_lower = {str(c).lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        if a.lower() in cols_lower:
            return cols_lower[a.lower()]
    for c in df.columns:
        lc = str(c).lower()
        for a in aliases:
            if a.lower() in lc:
                return c
    return None


def normalize_provider_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"": "OPEN", "nan": "OPEN", "None": "OPEN"})


def clean_numeric(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
    return out


def candidate_value(row: pd.Series | dict[str, Any], aliases: Sequence[str], default: float | None = None) -> float | None:
    if isinstance(row, pd.Series):
        keys = list(row.index)
        get = row.get
    else:
        keys = list(row.keys())
        get = row.get
    lower_map = {str(k).lower(): k for k in keys}
    for a in aliases:
        if a in keys:
            val = get(a)
            return clean_numeric(val, default if default is not None else np.nan)
        if a.lower() in lower_map:
            val = get(lower_map[a.lower()])
            return clean_numeric(val, default if default is not None else np.nan)
    for k in keys:
        lk = str(k).lower()
        for a in aliases:
            if a.lower() in lk:
                val = get(k)
                return clean_numeric(val, default if default is not None else np.nan)
    return default


def quantile_dict(values: Sequence[float], prefix: str = "") -> dict[str, float | None]:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return {f"{prefix}p10": None, f"{prefix}p25": None, f"{prefix}p50": None, f"{prefix}p75": None, f"{prefix}p90": None}
    return {
        f"{prefix}p10": float(np.quantile(arr, 0.10)),
        f"{prefix}p25": float(np.quantile(arr, 0.25)),
        f"{prefix}p50": float(np.quantile(arr, 0.50)),
        f"{prefix}p75": float(np.quantile(arr, 0.75)),
        f"{prefix}p90": float(np.quantile(arr, 0.90)),
    }


# =============================================================================
# Upstream artifact discovery and loading
# =============================================================================


@dataclass
class Layer1Artifacts:
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    scenarios: pd.DataFrame = field(default_factory=pd.DataFrame)
    point_forecasts: pd.DataFrame = field(default_factory=pd.DataFrame)
    sigma: pd.DataFrame = field(default_factory=pd.DataFrame)
    early_release: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Layer2Artifacts:
    selected_theme: str = "utilization_first"
    selected_candidate: dict[str, Any] = field(default_factory=dict)
    candidates_all: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidates_pareto: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    schedule: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_candidates: Any = None
    paths: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def load_layer1_artifacts(layer1_dir: str | Path) -> Layer1Artifacts:
    l1 = Path(layer1_dir)
    art = Layer1Artifacts()

    paths = {
        "features": find_first_file(l1, ["features.csv", "features_*.csv"]),
        "scenarios": find_first_file(l1, ["scenarios_long.csv", "demand_scenarios_long.csv", "scenarios.csv", "demand_scenarios.csv"]),
        "point_forecasts": find_first_file(l1, ["point_forecasts.csv", "forecasts.csv", "next_week_forecasts.csv", "forecast*.csv"]),
        "sigma": find_first_file(l1, ["hierarchical_sigma_estimates.csv", "sigma_estimates.csv", "posterior_sigma_estimates.csv"]),
        "early_release": find_first_file(l1, ["early_release_projected.csv", "early_release.csv", "projected_early_release.csv"]),
        "metrics_json": find_first_file(l1, ["model_metrics.json", "metrics.json", "point_model_metrics.json", "validation_report.json"]),
    }
    art.paths = {k: str(v) for k, v in paths.items() if v is not None}

    art.features = safe_read_csv(paths["features"])
    art.scenarios = safe_read_csv(paths["scenarios"])
    art.point_forecasts = safe_read_csv(paths["point_forecasts"])
    art.sigma = safe_read_csv(paths["sigma"])
    art.early_release = safe_read_csv(paths["early_release"])
    if paths["metrics_json"]:
        try:
            art.metrics = read_json(paths["metrics_json"])
        except Exception as exc:
            art.warnings.append(f"Could not read metrics JSON: {exc}")

    model_patterns = {
        "casetime_min": ["casetime_min_model.joblib", "casetime_model.joblib", "xgb_casetime*.joblib", "*casetime*model*.joblib"],
        "turnover_min": ["turnover_min_model.joblib", "turnover_model.joblib", "xgb_turnover*.joblib", "*turnover*model*.joblib"],
    }
    for target, pats in model_patterns.items():
        p = find_first_file(l1, pats)
        if p:
            try:
                art.models[target] = joblib.load(p)
                art.paths[f"model_{target}"] = str(p)
            except Exception as exc:
                art.warnings.append(f"Could not load {target} model at {p}: {exc}")

    if art.features.empty:
        art.warnings.append(f"Layer 1 features.csv not found or empty in {l1}")
    if art.scenarios.empty:
        art.warnings.append(f"Layer 1 scenarios_long.csv not found or empty in {l1}")
    return art


def _candidate_records_from_json(obj: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(obj, list):
        for i, cand in enumerate(obj):
            if isinstance(cand, dict):
                flat = flatten_dict(cand)
                flat.setdefault("theme", cand.get("theme", cand.get("name", f"candidate_{i}")))
                records.append(flat)
    elif isinstance(obj, dict):
        if "themes" in obj and isinstance(obj["themes"], dict):
            for theme, cand in obj["themes"].items():
                if isinstance(cand, dict):
                    flat = flatten_dict(cand)
                    flat.setdefault("theme", cand.get("theme", theme))
                    records.append(flat)
        elif "candidates" in obj and isinstance(obj["candidates"], list):
            records.extend(_candidate_records_from_json(obj["candidates"]))
        elif all(isinstance(v, dict) for v in obj.values()):
            for theme, cand in obj.items():
                if isinstance(cand, dict):
                    flat = flatten_dict(cand)
                    flat.setdefault("theme", cand.get("theme", theme))
                    records.append(flat)
        else:
            flat = flatten_dict(obj)
            flat.setdefault("theme", obj.get("theme", obj.get("name", "recommended")))
            records.append(flat)
    return records


def _candidate_object_by_theme(obj: Any, theme: str | None) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        if theme and "themes" in obj and isinstance(obj["themes"], dict) and theme in obj["themes"]:
            cand = obj["themes"][theme]
            if isinstance(cand, dict):
                cand = dict(cand)
                cand.setdefault("theme", theme)
                return cand
        if theme and theme in obj and isinstance(obj[theme], dict):
            cand = dict(obj[theme])
            cand.setdefault("theme", theme)
            return cand
        if obj.get("theme") == theme or theme is None:
            return obj
    if isinstance(obj, list):
        for cand in obj:
            if isinstance(cand, dict) and (cand.get("theme") == theme or cand.get("name") == theme):
                return cand
        for cand in obj:
            if isinstance(cand, dict) and cand.get("recommended"):
                return cand
        if obj and isinstance(obj[0], dict):
            return obj[0]
    return {}


def load_layer2_artifacts(layer2_dir: str | Path, selected_theme: str | None = None) -> Layer2Artifacts:
    l2 = Path(layer2_dir)
    art = Layer2Artifacts(selected_theme=selected_theme or "utilization_first")

    rec_path = find_first_file(l2, ["recommended_candidate.json", "selected_candidate.json", "candidate_recommended.json"])
    all_json_path = find_first_file(l2, ["candidates_all.json", "all_candidates.json", "pareto_candidates_all.json"])
    pareto_json_path = find_first_file(l2, ["candidates_pareto.json", "pareto_candidates.json"])
    all_csv_path = find_first_file(l2, ["candidates_all.csv", "pareto_candidates_all.csv", "pareto_frontier.csv", "pareto_summary.csv"])

    raw_candidates = None
    rec_obj = None
    if rec_path:
        try:
            rec_obj = read_json(rec_path)
            art.paths["recommended_candidate"] = str(rec_path)
        except Exception as exc:
            art.warnings.append(f"Could not read recommended candidate: {exc}")
    if all_json_path:
        try:
            raw_candidates = read_json(all_json_path)
            art.paths["candidates_all_json"] = str(all_json_path)
        except Exception as exc:
            art.warnings.append(f"Could not read candidates_all JSON: {exc}")
    elif pareto_json_path:
        try:
            raw_candidates = read_json(pareto_json_path)
            art.paths["candidates_pareto_json"] = str(pareto_json_path)
        except Exception as exc:
            art.warnings.append(f"Could not read candidates_pareto JSON: {exc}")

    if all_csv_path:
        art.candidates_all = safe_read_csv(all_csv_path)
        art.paths["candidates_all_csv"] = str(all_csv_path)
    elif raw_candidates is not None:
        records = _candidate_records_from_json(raw_candidates)
        art.candidates_all = pd.DataFrame(records)

    if pareto_json_path:
        try:
            po = read_json(pareto_json_path)
            art.candidates_pareto = pd.DataFrame(_candidate_records_from_json(po))
            art.paths["candidates_pareto_json"] = str(pareto_json_path)
        except Exception as exc:
            art.warnings.append(f"Could not parse pareto JSON: {exc}")

    theme = selected_theme
    if not theme and isinstance(rec_obj, dict):
        theme = rec_obj.get("theme") or rec_obj.get("selected_theme") or rec_obj.get("name")
    if not theme and not art.candidates_all.empty:
        theme_col = first_existing_col(art.candidates_all, ["theme", "name", "candidate"])
        rec_col = first_existing_col(art.candidates_all, ["recommended", "selected", "is_recommended"])
        if theme_col and rec_col:
            rec_rows = art.candidates_all[art.candidates_all[rec_col].astype(str).str.lower().isin(["true", "1", "yes"])]
            if not rec_rows.empty:
                theme = str(rec_rows.iloc[0][theme_col])
        if not theme and theme_col:
            themes = art.candidates_all[theme_col].astype(str).tolist()
            theme = "utilization_first" if "utilization_first" in themes else themes[0]
    theme = theme or "utilization_first"
    art.selected_theme = theme

    if rec_obj:
        art.selected_candidate = rec_obj if not raw_candidates else _candidate_object_by_theme(raw_candidates, theme)
        if not art.selected_candidate:
            art.selected_candidate = rec_obj
    elif raw_candidates is not None:
        art.selected_candidate = _candidate_object_by_theme(raw_candidates, theme)
    art.raw_candidates = raw_candidates

    coverage_patterns = [
        "coverage_recommended.csv", f"coverage_{theme}.csv", "selected_coverage.csv",
        "goal_coverage.csv", "coverage.csv", "provider_coverage.csv",
    ]
    cov_path = find_first_file(l2, coverage_patterns)
    art.coverage = safe_read_csv(cov_path)
    if cov_path:
        art.paths["coverage"] = str(cov_path)

    schedule_patterns = [
        "schedule_recommended.csv", "recommended_schedule.csv", f"schedule_{theme}.csv", f"{theme}_schedule.csv",
        "selected_schedule.csv", "assignments_recommended.csv", f"assignments_{theme}.csv", "assignments.csv", "schedule.csv",
    ]
    schedule_path = find_first_file(l2, schedule_patterns)
    if schedule_path:
        art.schedule = safe_read_csv(schedule_path)
        art.paths["schedule"] = str(schedule_path)
    else:
        # Try schedule embedded in selected candidate JSON
        for key in ["block_template_proposed", "schedule", "assignments", "selected_schedule", "allocation", "exceptions_list"]:
            val = art.selected_candidate.get(key) if isinstance(art.selected_candidate, dict) else None
            if isinstance(val, list) and val and isinstance(val[0], dict):
                art.schedule = pd.DataFrame(val)
                art.paths["schedule"] = f"embedded:{key}"
                break

    if art.coverage.empty:
        art.warnings.append(f"No Layer 2 coverage CSV found in {l2}; goal attribution will be partial")
    if art.schedule.empty:
        art.warnings.append(f"No Layer 2 schedule/assignment table found in {l2}; block-change details will be partial")
    return art


# =============================================================================
# Feature columns and model SHAP
# =============================================================================


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that are legitimate Layer-1 model features.

    Layer 3 often joins Layer-1 feature rows with Layer-2 solution columns before
    explaining.  Columns such as ``is_included`` and ``total_util`` are useful for
    schedule diagnostics, but they were not present when the XGBoost/sklearn
    model was fit.  Passing them into a fitted sklearn Pipeline can trigger the
    exact error: "feature names unseen at fit time".  Therefore this function
    intentionally removes known target/metadata/Layer-2 diagnostic columns.
    A second alignment step below uses the fitted model's ``feature_names_in_``
    whenever available, so this list is a safe first pass rather than the only
    guard.
    """
    exclude_exact = {
        # identifiers / metadata
        "provider_id", "provider", "provider_name", "service_line", "week_start", "date", "source_mode",
        "target", "scenario_id", "block_id", "room", "site", "slot_id", "candidate_id", "theme",
        # supervised targets / observed outcomes
        "casetime_min", "turnover_min", "casetime_util", "turnover_util",
        "actual_casetime_min", "actual_turnover_min", "residual_casetime", "residual_turnover",
        # Layer-2 / Layer-3 solution diagnostics, not Layer-1 predictors
        "is_included", "included", "selected", "assigned", "kept", "changed", "removed", "added",
        "total_util", "utilization", "projected_util", "current_util", "recommended_util",
        "allocated_min_recommended", "allocated_min_current", "assigned_min", "capacity_min",
        "shortage_min", "surplus_min", "coverage", "coverage_alpha", "alpha_met",
        "o1", "o2", "o3", "o4", "objective", "score",
    }
    exclude_prefixes = (
        "decision_", "solution_", "pareto_", "frontier_", "coverage_",
        "shortage_", "surplus_", "assigned_", "allocated_recommended_",
    )
    cols: list[str] = []
    for c in df.columns:
        sc = str(c)
        if sc in exclude_exact or any(sc.startswith(pref) for pref in exclude_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(sc)
    return cols


def explain_rows_from_features(features: pd.DataFrame, scope: str = "latest") -> pd.DataFrame:
    if features.empty:
        return features.copy()
    df = features.copy()
    prov_col = first_existing_col(df, ["provider_id", "provider"])
    day_col = first_existing_col(df, ["day_of_week", "dow", "weekday", "day"])
    week_col = first_existing_col(df, ["week_index", "week", "week_start"])
    if scope == "all" or not prov_col:
        return df
    sort_cols = [c for c in [prov_col, day_col, week_col] if c]
    if sort_cols:
        df = df.sort_values(sort_cols)
    group_cols = [c for c in [prov_col, day_col] if c]
    if group_cols:
        return df.groupby(group_cols, as_index=False, dropna=False).tail(1).reset_index(drop=True)
    return df.tail(min(len(df), 2000)).reset_index(drop=True)


def _split_pipeline(model: Any) -> tuple[Any | None, Any]:
    """Return (preprocessor_pipeline_or_None, final_estimator)."""
    if hasattr(model, "steps") and isinstance(getattr(model, "steps"), list) and len(model.steps) >= 1:
        try:
            from sklearn.pipeline import Pipeline
            if len(model.steps) == 1:
                return None, model.steps[-1][1]
            return Pipeline(model.steps[:-1]), model.steps[-1][1]
        except Exception:
            return None, model
    if hasattr(model, "named_steps"):
        try:
            steps = list(model.named_steps.items())
            from sklearn.pipeline import Pipeline
            if len(steps) <= 1:
                return None, steps[-1][1]
            return Pipeline(steps[:-1]), steps[-1][1]
        except Exception:
            return None, model
    return None, model


def _as_string_list(values: Any) -> list[str]:
    try:
        return [str(v) for v in list(values)]
    except Exception:
        return []


def _model_expected_raw_features(model: Any) -> list[str]:
    """Best-effort extraction of the raw feature names used during model fit.

    This is crucial for sklearn Pipelines.  The Layer-1 model was fit on a fixed
    feature matrix.  Layer 3 may carry extra diagnostic numeric columns, and
    sklearn >=1.2 refuses to transform a DataFrame with unseen columns.
    """
    candidates: list[list[str]] = []

    for obj in [model]:
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            candidates.append(_as_string_list(names))

    # Pipeline: earlier transformers usually know the raw input names.
    steps = []
    if hasattr(model, "steps") and isinstance(getattr(model, "steps"), list):
        steps = [step for _, step in model.steps]
    elif hasattr(model, "named_steps"):
        try:
            steps = list(model.named_steps.values())
        except Exception:
            steps = []

    for step in steps:
        names = getattr(step, "feature_names_in_", None)
        if names is not None:
            candidates.append(_as_string_list(names))

    # XGBoost sklearn wrapper may store names on the Booster.
    for obj in ([model] + steps):
        try:
            booster = obj.get_booster()
            names = getattr(booster, "feature_names", None)
            if names:
                candidates.append(_as_string_list(names))
        except Exception:
            pass

    # Prefer the longest named list because it usually corresponds to the raw
    # input before transformations.  Reject auto-generated transformed names.
    candidates = [c for c in candidates if c]
    if not candidates:
        return []
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        if not all(name.startswith("feature_") for name in c):
            return c
    return candidates[0]


def _align_raw_features_to_model(model: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add missing fitted features as NaN, drop unseen extras, and order columns.

    The fitted Pipeline/imputer will handle NaNs.  Dropping extras fixes both:
    1) sklearn ``feature names unseen at fit time``; and
    2) XGBoost ``Feature shape mismatch`` in the fallback path.
    """
    warnings: list[str] = []
    expected = _model_expected_raw_features(model)
    if not expected:
        return X.copy(), warnings

    X2 = X.copy()
    for col in expected:
        if col not in X2.columns:
            X2[col] = np.nan
    extra = [c for c in X2.columns if str(c) not in set(expected)]
    missing = [c for c in expected if c not in X.columns]
    if extra:
        shown = ", ".join(map(str, extra[:12]))
        more = "..." if len(extra) > 12 else ""
        warnings.append(f"Dropped {len(extra)} columns not seen during model fit: {shown}{more}")
    if missing:
        shown = ", ".join(map(str, missing[:12]))
        more = "..." if len(missing) > 12 else ""
        warnings.append(f"Added {len(missing)} missing fitted features as NaN: {shown}{more}")
    return X2[expected].copy(), warnings


def _transform_for_model(model: Any, X: pd.DataFrame) -> tuple[np.ndarray | pd.DataFrame, Any, list[str], list[str]]:
    warnings: list[str] = []
    X_aligned, align_warnings = _align_raw_features_to_model(model, X)
    warnings.extend(align_warnings)

    pre, final = _split_pipeline(model)
    X_proc: Any = X_aligned
    names = list(X_aligned.columns)
    if pre is not None:
        try:
            X_proc = pre.transform(X_aligned)
            if hasattr(pre, "get_feature_names_out"):
                try:
                    names = [str(x) for x in pre.get_feature_names_out()]
                except Exception:
                    pass
        except Exception as exc:
            warnings.append(f"Pipeline preprocessing failed after feature alignment; trying aligned raw features: {exc}")
            X_proc = X_aligned
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    if not isinstance(X_proc, pd.DataFrame):
        X_proc = np.asarray(X_proc)
        if X_proc.ndim == 1:
            X_proc = X_proc.reshape(-1, 1)
        if X_proc.shape[1] != len(names):
            names = [f"feature_{i}" for i in range(X_proc.shape[1])]
    return X_proc, final, names, warnings


def _predict_model(model: Any, X_raw: pd.DataFrame) -> np.ndarray:
    X_aligned, _ = _align_raw_features_to_model(model, X_raw)
    try:
        pred = model.predict(X_aligned)
    except Exception:
        X_proc, final, _, _ = _transform_for_model(model, X_aligned)
        try:
            pred = final.predict(X_proc)
        except Exception:
            # XGBoost may still validate stale feature metadata if fed a DataFrame
            # in fallback mode.  Use ndarray as a final escape hatch.
            pred = final.predict(np.asarray(X_proc, dtype=float))
    pred = np.asarray(pred, dtype=float).ravel()
    return pred


def _is_tree_model(estimator: Any) -> bool:
    name = estimator.__class__.__name__.lower()
    module = estimator.__class__.__module__.lower()
    if "xgb" in name or "xgboost" in module:
        return True
    if any(tok in name for tok in ["forest", "tree", "gradientboost", "histgradient", "lgbm", "catboost"]):
        return True
    return hasattr(estimator, "feature_importances_")


def _fallback_shap_values(model: Any, X_raw: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, float, list[str]]:
    """Coefficient/importance based additive approximation if real SHAP is unavailable."""
    warnings = ["Real SHAP unavailable; used additive importance/coef approximation."]
    pre, final = _split_pipeline(model)
    X_proc, final, proc_names, pp_warnings = _transform_for_model(model, X_raw)
    warnings.extend(pp_warnings)
    arr = np.asarray(X_proc, dtype=float)
    pred = _predict_model(model, X_raw)
    base = float(np.nanmean(pred)) if len(pred) else 0.0
    try:
        if hasattr(final, "coef_"):
            weights = np.asarray(final.coef_, dtype=float).ravel()
            if weights.size == arr.shape[1]:
                centered = arr - np.nanmean(arr, axis=0, keepdims=True)
                vals = centered * weights.reshape(1, -1)
                return vals, base, warnings
        if hasattr(final, "feature_importances_"):
            imp = np.asarray(final.feature_importances_, dtype=float).ravel()
            if imp.size == arr.shape[1] and imp.sum() > 0:
                imp = imp / imp.sum()
                centered = arr - np.nanmean(arr, axis=0, keepdims=True)
                scale = np.nanstd(pred) / max(np.nanmean(np.abs(centered @ imp)), 1e-9)
                vals = centered * imp.reshape(1, -1) * scale
                return vals, base, warnings
    except Exception as exc:
        warnings.append(f"Fallback attribution failed partly: {exc}")
    vals = np.zeros((len(X_raw), len(feature_names)), dtype=float)
    return vals, base, warnings


def compute_shap_for_target(
    target: str,
    model: Any,
    features: pd.DataFrame,
    cfg: Layer3Config,
) -> dict[str, Any]:
    cols = feature_columns(features)
    if not cols:
        raise ValueError("No numeric feature columns available for SHAP.")
    X_raw = features[cols].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    # Align immediately to the raw feature schema used during Layer-1 model fit.
    # This prevents Layer-2 diagnostic columns from leaking into the SHAP call.
    X_raw, early_align_warnings = _align_raw_features_to_model(model, X_raw)
    n_total = len(X_raw)
    if cfg.max_shap_rows and len(X_raw) > cfg.max_shap_rows:
        # deterministic sample with preference for latest rows after caller filtered
        X_raw = X_raw.sample(n=cfg.max_shap_rows, random_state=cfg.random_seed).sort_index()
        row_meta = features.loc[X_raw.index].reset_index(drop=True)
        X_raw = X_raw.reset_index(drop=True)
    else:
        row_meta = features.reset_index(drop=True)
        X_raw = X_raw.reset_index(drop=True)

    X_proc, final, proc_names, pp_warnings = _transform_for_model(model, X_raw)
    warnings: list[str] = list(early_align_warnings) + list(pp_warnings)
    shap_values: np.ndarray
    base_value: float
    explainer_kind = "fallback"

    try:
        import shap  # type: ignore
        if _is_tree_model(final):
            explainer = shap.TreeExplainer(final)
            raw_vals = explainer.shap_values(X_proc)
            if isinstance(raw_vals, list):
                raw_vals = raw_vals[0]
            if hasattr(raw_vals, "values"):
                raw_vals = raw_vals.values
            shap_values = np.asarray(raw_vals, dtype=float)
            expected = getattr(explainer, "expected_value", 0.0)
            if isinstance(expected, (list, np.ndarray)):
                expected = np.asarray(expected).ravel()[0]
            base_value = float(expected)
            explainer_kind = "shap.TreeExplainer"
        else:
            # LinearExplainer works for linear estimators, otherwise fallback.
            try:
                explainer = shap.LinearExplainer(final, X_proc)
                exp = explainer(X_proc)
                shap_values = np.asarray(exp.values, dtype=float)
                base_value = float(np.asarray(exp.base_values).ravel()[0])
                explainer_kind = "shap.LinearExplainer"
            except Exception:
                shap_values, base_value, fallback_warnings = _fallback_shap_values(model, X_raw, proc_names)
                warnings.extend(fallback_warnings)
    except Exception as exc:
        warnings.append(f"SHAP package/model explanation failed: {exc}")
        shap_values, base_value, fallback_warnings = _fallback_shap_values(model, X_raw, proc_names)
        warnings.extend(fallback_warnings)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)
    if shap_values.shape[1] != len(proc_names):
        proc_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

    try:
        pred = _predict_model(model, X_raw)
    except Exception:
        pred = np.repeat(np.nan, shap_values.shape[0])
        warnings.append("Could not compute predictions for explained rows.")

    return {
        "target": target,
        "feature_names": proc_names,
        "shap_values": shap_values,
        "base_value": base_value,
        "predictions": pred,
        "row_meta": row_meta.reset_index(drop=True),
        "explainer_kind": explainer_kind,
        "warnings": warnings,
        "n_total_rows_before_sampling": n_total,
    }


def shap_attribution(art1: Layer1Artifacts, out: Path, cfg: Layer3Config) -> dict[str, Any]:
    out = ensure_dir(out)
    shap_dir = ensure_dir(out / "shap")
    features = explain_rows_from_features(art1.features, cfg.explain_scope)

    all_results: dict[str, Any] = {}
    long_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    provider_rows: list[dict[str, Any]] = []
    peer_rows: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []

    if features.empty:
        return {"warnings": ["No features available; SHAP skipped."], "paths": {}}

    prov_col = first_existing_col(features, ["provider_id", "provider"])
    day_col = first_existing_col(features, ["day_of_week", "dow", "weekday", "day"])
    sl_col = first_existing_col(features, ["service_line", "specialty", "service"])

    for target, model in art1.models.items():
        log.info("Computing SHAP for %s …", target)
        res = compute_shap_for_target(target, model, features, cfg)
        all_results[target] = {
            "feature_names": res["feature_names"],
            "shap_values": res["shap_values"],
            "base_value": res["base_value"],
            "predictions": res["predictions"],
            "explainer_kind": res["explainer_kind"],
            "warnings": res["warnings"],
        }
        feature_names = list(res["feature_names"])
        vals = np.asarray(res["shap_values"], dtype=float)
        meta = res["row_meta"].reset_index(drop=True)
        preds = np.asarray(res["predictions"], dtype=float).ravel()
        base = float(res["base_value"])

        eff_error = np.nanmax(np.abs((vals.sum(axis=1) + base) - preds)) if len(preds) and np.isfinite(preds).any() else np.nan
        validation.append({
            "target": target,
            "explainer_kind": res["explainer_kind"],
            "n_rows_explained": int(vals.shape[0]),
            "n_features": int(vals.shape[1]),
            "base_value": base,
            "max_efficiency_error_abs_min": None if pd.isna(eff_error) else float(eff_error),
            "warnings": "; ".join(res["warnings"]),
        })

        # raw matrix as CSV for UI and pkl for Python
        matrix_df = pd.DataFrame(vals, columns=feature_names)
        id_cols = []
        for c in [prov_col, sl_col, day_col, "week_index", "rotation_phase"]:
            if c and c in meta.columns and c not in id_cols:
                id_cols.append(c)
        matrix_out = pd.concat([meta[id_cols].reset_index(drop=True) if id_cols else pd.DataFrame(index=range(len(matrix_df))), matrix_df], axis=1)
        matrix_out.to_csv(shap_dir / f"shap_matrix_{target}.csv", index=False)

        # long + top tables
        for i in range(vals.shape[0]):
            provider_id = str(meta.iloc[i][prov_col]) if prov_col else str(i)
            service_line = str(meta.iloc[i][sl_col]) if sl_col else "UNKNOWN"
            day_value = meta.iloc[i][day_col] if day_col else None
            order = np.argsort(np.abs(vals[i]))[::-1]
            for rank, j in enumerate(order[: max(cfg.top_k_features, 1)], start=1):
                f = feature_names[j]
                reason, plain = reason_for_feature(f)
                feature_value = None
                if f in meta.columns:
                    feature_value = meta.iloc[i][f]
                elif f in art1.features.columns:
                    try:
                        feature_value = art1.features.iloc[i][f]
                    except Exception:
                        feature_value = None
                row = {
                    "target": target,
                    "row_id": i,
                    "provider_id": provider_id,
                    "service_line": service_line,
                    "day_of_week": day_value,
                    "prediction_min": preds[i] if i < len(preds) else None,
                    "base_value_min": base,
                    "rank": rank,
                    "feature": f,
                    "feature_value": feature_value,
                    "shap_value_min": float(vals[i, j]),
                    "abs_shap_value_min": float(abs(vals[i, j])),
                    "direction": "up" if vals[i, j] >= 0 else "down",
                    "reason_code": reason,
                    "plain_language": plain,
                }
                long_rows.append(row)
                if rank <= cfg.top_k_features:
                    top_rows.append(row)

        # provider aggregates by reason code
        tmp = []
        for i in range(vals.shape[0]):
            provider_id = str(meta.iloc[i][prov_col]) if prov_col else str(i)
            service_line = str(meta.iloc[i][sl_col]) if sl_col else "UNKNOWN"
            for j, f in enumerate(feature_names):
                reason, plain = reason_for_feature(f)
                tmp.append({
                    "target": target,
                    "provider_id": provider_id,
                    "service_line": service_line,
                    "feature": f,
                    "reason_code": reason,
                    "plain_language": plain,
                    "shap_value_min": float(vals[i, j]),
                    "abs_shap_value_min": float(abs(vals[i, j])),
                })
        tmp_df = pd.DataFrame(tmp)
        if not tmp_df.empty:
            agg = (
                tmp_df.groupby(["target", "provider_id", "service_line", "reason_code", "plain_language"], dropna=False)
                .agg(
                    shap_sum_min=("shap_value_min", "sum"),
                    shap_abs_sum_min=("abs_shap_value_min", "sum"),
                    shap_mean_min=("shap_value_min", "mean"),
                    n_terms=("shap_value_min", "size"),
                )
                .reset_index()
                .sort_values(["target", "provider_id", "shap_abs_sum_min"], ascending=[True, True, False])
            )
            provider_rows.extend(agg.to_dict(orient="records"))

            # peer comparison: provider reason vs service-line average
            peer = (
                agg.groupby(["target", "service_line", "reason_code"], dropna=False)["shap_sum_min"]
                .mean()
                .reset_index()
                .rename(columns={"shap_sum_min": "service_line_avg_shap_sum_min"})
            )
            comp = agg.merge(peer, on=["target", "service_line", "reason_code"], how="left")
            comp["provider_vs_peer_delta_min"] = comp["shap_sum_min"] - comp["service_line_avg_shap_sum_min"]
            peer_rows.extend(comp.to_dict(orient="records"))

    shap_long = pd.DataFrame(long_rows)
    shap_top = pd.DataFrame(top_rows)
    shap_provider = pd.DataFrame(provider_rows)
    shap_peer = pd.DataFrame(peer_rows)
    shap_validation = pd.DataFrame(validation)

    if not shap_long.empty:
        shap_long.to_csv(out / "shap_values_long.csv", index=False)
    if not shap_top.empty:
        shap_top.to_csv(out / "shap_top_reasons.csv", index=False)
    if not shap_provider.empty:
        shap_provider.to_csv(out / "shap_provider_summary.csv", index=False)
    if not shap_peer.empty:
        shap_peer.to_csv(out / "shap_peer_comparison.csv", index=False)
    if not shap_validation.empty:
        shap_validation.to_csv(out / "shap_validation.csv", index=False)
    write_pickle(out / "shap_values.pkl", all_results)

    interactions = pd.DataFrame()
    if cfg.compute_interactions:
        interactions = shap_interactions(art1, features, out, cfg)

    if cfg.write_plots:
        plot_shap_summary(shap_provider, out / "plots")

    return {
        "shap_long": shap_long,
        "shap_top": shap_top,
        "shap_provider": shap_provider,
        "shap_peer": shap_peer,
        "shap_validation": shap_validation,
        "interactions": interactions,
        "paths": {
            "shap_values_long": str(out / "shap_values_long.csv"),
            "shap_top_reasons": str(out / "shap_top_reasons.csv"),
            "shap_provider_summary": str(out / "shap_provider_summary.csv"),
            "shap_peer_comparison": str(out / "shap_peer_comparison.csv"),
            "shap_values_pkl": str(out / "shap_values.pkl"),
        },
    }


def shap_interactions(art1: Layer1Artifacts, features: pd.DataFrame, out: Path, cfg: Layer3Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if features.empty:
        return pd.DataFrame()
    cols = feature_columns(features)
    if not cols:
        return pd.DataFrame()
    X_raw = features[cols].copy().replace([np.inf, -np.inf], np.nan)
    if len(X_raw) > cfg.max_interaction_rows:
        X_raw = X_raw.sample(n=cfg.max_interaction_rows, random_state=cfg.random_seed).sort_index().reset_index(drop=True)
    for target, model in art1.models.items():
        try:
            import shap  # type: ignore
            X_proc, final, names, _ = _transform_for_model(model, X_raw)
            if not _is_tree_model(final):
                continue
            explainer = shap.TreeExplainer(final)
            vals = explainer.shap_interaction_values(X_proc)
            if isinstance(vals, list):
                vals = vals[0]
            vals = np.asarray(vals, dtype=float)
            if vals.ndim != 3:
                continue
            mean_abs = np.nanmean(np.abs(vals), axis=0)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    f1, f2 = names[i], names[j]
                    r1, _ = reason_for_feature(f1)
                    r2, _ = reason_for_feature(f2)
                    rows.append({
                        "target": target,
                        "feature_i": f1,
                        "feature_j": f2,
                        "reason_i": r1,
                        "reason_j": r2,
                        "mean_abs_interaction_min": float(mean_abs[i, j]),
                    })
            write_pickle(out / f"shap_interactions_{target}.pkl", {"names": names, "values": vals})
        except Exception as exc:
            log.warning("SHAP interactions skipped for %s: %s", target, exc)
    df = pd.DataFrame(rows).sort_values("mean_abs_interaction_min", ascending=False) if rows else pd.DataFrame()
    if not df.empty:
        df.to_csv(out / "shap_interactions_top.csv", index=False)
    return df


# =============================================================================
# Allocation, goal attribution, counterfactuals
# =============================================================================


def standardize_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return coverage.copy()
    df = coverage.copy()
    rename = {}
    aliases = {
        "provider_id": ["provider_id", "provider", "blockholder", "surgeon_id"],
        "allocated_min": ["allocated_min", "assigned_min", "allocation_min", "proposed_allocated_min", "minutes_allocated", "capacity_allocated_min"],
        "required_min_alpha": ["required_min_alpha", "target_q_min", "alpha_target_min", "required_min", "demand_target_min", "required_alpha_min"],
        "coverage_ratio": ["coverage_ratio", "coverage", "alpha_achieved", "scenario_coverage", "achieved_alpha"],
        "meets_alpha": ["meets_alpha", "goal_met", "covered", "meets_goal", "alpha_met"],
        "shortage_min": ["shortage_min", "gap_min", "unmet_min", "required_gap_min"],
        "slack_min": ["slack_min", "slack", "capacity_slack_min"],
        "target_utilization": ["target_utilization", "target_util", "target", "util_target"],
    }
    for canon, als in aliases.items():
        c = first_existing_col(df, als)
        if c and c != canon:
            rename[c] = canon
    df = df.rename(columns=rename)
    if "provider_id" in df.columns:
        df["provider_id"] = normalize_provider_id_series(df["provider_id"])
    for c in ["allocated_min", "required_min_alpha", "coverage_ratio", "shortage_min", "slack_min", "target_utilization"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "meets_alpha" in df.columns:
        if df["meets_alpha"].dtype != bool:
            df["meets_alpha"] = df["meets_alpha"].astype(str).str.lower().isin(["true", "1", "yes", "y", "met"])
    elif "coverage_ratio" in df.columns:
        df["meets_alpha"] = df["coverage_ratio"] >= 0.85
    if "shortage_min" not in df.columns and {"required_min_alpha", "allocated_min"}.issubset(df.columns):
        df["shortage_min"] = (df["required_min_alpha"] - df["allocated_min"]).clip(lower=0)
    if "coverage_ratio" not in df.columns and {"required_min_alpha", "allocated_min"}.issubset(df.columns):
        df["coverage_ratio"] = df["allocated_min"] / df["required_min_alpha"].clip(lower=1.0)
    return df


def allocation_from_schedule(schedule: pd.DataFrame, block_minutes_default: int = 480) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame(columns=["provider_id", "allocated_min", "n_blocks"])
    df = schedule.copy()
    provider_col = first_existing_col(df, ["provider_id", "assigned_provider", "provider_after", "new_provider", "blockholder", "current_blockholder", "assigned_to"])
    if provider_col is None:
        return pd.DataFrame(columns=["provider_id", "allocated_min", "n_blocks"])
    min_col = first_existing_col(df, ["assigned_min", "allocated_min", "capacity_min", "duration_min", "slot_minutes", "block_minutes", "minutes"])
    df["provider_id"] = normalize_provider_id_series(df[provider_col])
    df = df[df["provider_id"].str.upper() != "OPEN"].copy()
    if min_col:
        df["allocated_min"] = pd.to_numeric(df[min_col], errors="coerce").fillna(block_minutes_default)
    else:
        start_col = first_existing_col(df, ["start_min", "start_minutes"])
        end_col = first_existing_col(df, ["end_min", "end_minutes"])
        if start_col and end_col:
            df["allocated_min"] = (pd.to_numeric(df[end_col], errors="coerce") - pd.to_numeric(df[start_col], errors="coerce")).fillna(block_minutes_default).clip(lower=0)
        else:
            df["allocated_min"] = block_minutes_default
    out = df.groupby("provider_id").agg(allocated_min=("allocated_min", "sum"), n_blocks=("allocated_min", "size")).reset_index()
    return out


def target_map_from_coverage_or_default(coverage: pd.DataFrame, default_target: float) -> dict[str, float]:
    if coverage.empty or "provider_id" not in coverage.columns:
        return {}
    df = coverage.copy()
    target_col = first_existing_col(df, ["target_utilization", "target_util", "target"])
    out: dict[str, float] = {}
    if target_col:
        for _, r in df.iterrows():
            val = clean_numeric(r[target_col], default_target)
            if val > 1.5:  # maybe percent
                val /= 100.0
            out[str(r["provider_id"])] = val
    return out


def early_release_map(er: pd.DataFrame) -> dict[str, float]:
    if er.empty:
        return {}
    df = er.copy()
    prov_col = first_existing_col(df, ["provider_id", "provider"])
    min_col = first_existing_col(df, ["early_release_projected_min", "early_release_min", "manual_early_release", "projected_early_release_min"])
    if not prov_col or not min_col:
        return {}
    df[prov_col] = normalize_provider_id_series(df[prov_col])
    df[min_col] = pd.to_numeric(df[min_col], errors="coerce").fillna(0.0)
    return df.groupby(prov_col)[min_col].sum().to_dict()


def aggregate_scenarios_by_provider(scenarios: pd.DataFrame, optimization_weeks: int) -> pd.DataFrame:
    if scenarios.empty:
        return pd.DataFrame()
    df = scenarios.copy()
    prov_col = first_existing_col(df, ["provider_id", "provider"])
    scen_col = first_existing_col(df, ["scenario_id", "scenario", "s"])
    case_col = first_existing_col(df, ["demand_casetime_min", "casetime_min", "case_min", "mu_casetime_min"])
    turn_col = first_existing_col(df, ["demand_turnover_min", "turnover_min", "turn_min", "mu_turnover_min"])
    if not prov_col or not scen_col:
        return pd.DataFrame()
    if not case_col:
        df["_case"] = 0.0
        case_col = "_case"
    if not turn_col:
        df["_turn"] = 0.0
        turn_col = "_turn"
    df[prov_col] = normalize_provider_id_series(df[prov_col])
    df[case_col] = pd.to_numeric(df[case_col], errors="coerce").fillna(0.0)
    df[turn_col] = pd.to_numeric(df[turn_col], errors="coerce").fillna(0.0)
    q = (
        df.groupby([prov_col, scen_col], dropna=False)
        .agg(case_min_week=(case_col, "sum"), turnover_min_week=(turn_col, "sum"))
        .reset_index()
        .rename(columns={prov_col: "provider_id", scen_col: "scenario_id"})
    )
    q["case_min_horizon"] = q["case_min_week"] * optimization_weeks
    q["turnover_min_horizon"] = q["turnover_min_week"] * optimization_weeks
    q["demand_min_horizon"] = q["case_min_horizon"] + q["turnover_min_horizon"]
    return q


def required_minutes_by_provider_scenario(
    q: pd.DataFrame,
    target_map: dict[str, float],
    early_map: dict[str, float],
    default_target: float,
) -> pd.DataFrame:
    if q.empty:
        return q.copy()
    df = q.copy()
    df["target_utilization"] = df["provider_id"].astype(str).map(target_map).fillna(default_target).astype(float)
    df["target_utilization"] = df["target_utilization"].clip(lower=0.05, upper=2.0)
    df["early_release_min"] = df["provider_id"].astype(str).map(early_map).fillna(0.0)
    df["required_min"] = df["demand_min_horizon"] / df["target_utilization"] + df["early_release_min"]
    return df


def coverage_curve_for_provider(required: pd.Series, block_minutes: int, max_blocks: int, alpha: float) -> pd.DataFrame:
    req = pd.to_numeric(required, errors="coerce").dropna().to_numpy(dtype=float)
    rows = []
    if req.size == 0:
        return pd.DataFrame()
    for n in range(0, max_blocks + 1):
        alloc = n * block_minutes
        cov = float(np.mean(alloc >= req))
        rows.append({
            "n_blocks": n,
            "allocated_min": alloc,
            "allocated_hrs": alloc / 60.0,
            "covered_scenarios": int(np.sum(alloc >= req)),
            "n_scenarios": int(req.size),
            "coverage": cov,
            "meets_alpha": cov >= alpha,
        })
    return pd.DataFrame(rows)


def binding_factor_from_row(row: pd.Series, alpha: float) -> str:
    if bool(row.get("meets_alpha", False)) or clean_numeric(row.get("alpha_achieved", 0.0)) >= alpha:
        return "covered"
    allocated = clean_numeric(row.get("allocated_min", 0.0))
    req = clean_numeric(row.get("required_min_alpha", row.get("required_p85_min", 0.0)))
    shortage = clean_numeric(row.get("shortage_min", max(req - allocated, 0.0)))
    if allocated <= 1e-6:
        return "supply_shortage"
    ratio = allocated / max(req, 1.0)
    if ratio < 0.50:
        return "supply_shortage" if shortage > 0 else "case_volume"
    if clean_numeric(row.get("target_utilization", 0.7)) >= 0.85:
        return "target_level"
    return "block_granularity_or_supply"


def block_changes_detail(schedule: pd.DataFrame, out: Path) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()
    df = schedule.copy()
    before_col = first_existing_col(df, ["provider_before", "previous_provider", "original_provider", "current_blockholder", "before_provider", "old_provider"])
    after_col = first_existing_col(df, ["provider_after", "new_provider", "assigned_provider", "provider_id", "after_provider", "assigned_to"])
    min_col = first_existing_col(df, ["capacity_min", "allocated_min", "assigned_min", "duration_min", "slot_minutes", "block_minutes"])
    block_col = first_existing_col(df, ["block_id", "slot_id", "block_historical_id", "id"])
    room_col = first_existing_col(df, ["room", "or_room", "room_id"])
    day_col = first_existing_col(df, ["day_of_week", "dow", "weekday", "day"])
    week_col = first_existing_col(df, ["week", "week_index", "rotation_phase", "phase"])
    if after_col is None:
        return pd.DataFrame()
    if before_col is None:
        before_col = after_col
    rows = []
    for i, r in df.iterrows():
        before = str(r.get(before_col, "OPEN")).strip() or "OPEN"
        after = str(r.get(after_col, "OPEN")).strip() or "OPEN"
        if before.lower() in ["nan", "none", ""]:
            before = "OPEN"
        if after.lower() in ["nan", "none", ""]:
            after = "OPEN"
        action = "UNCHANGED"
        if before != after:
            if before != "OPEN" and after == "OPEN":
                action = "REMOVED"
            elif before == "OPEN" and after != "OPEN":
                action = "ADDED"
            else:
                action = "MOVED"
        rows.append({
            "row_id": int(i),
            "block_id": r.get(block_col, i) if block_col else i,
            "week": r.get(week_col, None) if week_col else None,
            "day_of_week": r.get(day_col, None) if day_col else None,
            "room": r.get(room_col, None) if room_col else None,
            "provider_before": before,
            "provider_after": after,
            "minutes": clean_numeric(r.get(min_col, 480.0), 480.0) if min_col else 480.0,
            "action": action,
            "changed": action != "UNCHANGED",
        })
    changes = pd.DataFrame(rows)
    changes.to_csv(out / "block_changes_detail.csv", index=False)
    frag = (
        changes[changes["provider_after"] != "OPEN"].groupby("provider_after")
        .agg(n_blocks=("block_id", "count"), n_rooms=("room", pd.Series.nunique), n_days=("day_of_week", pd.Series.nunique), n_changed=("changed", "sum"), minutes=("minutes", "sum"))
        .reset_index()
        .rename(columns={"provider_after": "provider_id"})
    )
    frag["fragmentation_index"] = (frag["n_rooms"].fillna(0) + frag["n_days"].fillna(0) + frag["n_changed"].fillna(0)).astype(float)
    frag.to_csv(out / "fragmentation_detail.csv", index=False)
    return changes


def goal_attribution_and_counterfactuals(
    art1: Layer1Artifacts,
    art2: Layer2Artifacts,
    out: Path,
    cfg: Layer3Config,
) -> dict[str, Any]:
    out = ensure_dir(out)
    coverage = standardize_coverage(art2.coverage)
    allocation = allocation_from_schedule(art2.schedule, cfg.block_minutes)
    if not coverage.empty and "allocated_min" in coverage.columns:
        allocated = coverage[["provider_id", "allocated_min"]].copy()
        if "n_blocks" not in allocated.columns:
            allocated["n_blocks"] = np.ceil(allocated["allocated_min"].clip(lower=0) / cfg.block_minutes).astype(int)
    else:
        allocated = allocation.copy()
    if allocated.empty:
        allocated = pd.DataFrame(columns=["provider_id", "allocated_min", "n_blocks"])

    target_map = target_map_from_coverage_or_default(coverage, cfg.default_target_utilization)
    er_map = early_release_map(art1.early_release)
    q = aggregate_scenarios_by_provider(art1.scenarios, cfg.optimization_weeks)
    req = required_minutes_by_provider_scenario(q, target_map, er_map, cfg.default_target_utilization)
    if not req.empty:
        req.to_csv(out / "required_minutes_by_scenario.csv", index=False)

    rows: list[dict[str, Any]] = []
    cf_rows: list[dict[str, Any]] = []
    demand_mult_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    providers = sorted(set(req["provider_id"].astype(str).unique()) if not req.empty else [])
    if not coverage.empty and "provider_id" in coverage.columns:
        providers = sorted(set(providers).union(set(coverage["provider_id"].astype(str))))
    if not allocated.empty and "provider_id" in allocated.columns:
        providers = sorted(set(providers).union(set(allocated["provider_id"].astype(str))))

    alloc_map = dict(zip(allocated.get("provider_id", pd.Series(dtype=str)).astype(str), pd.to_numeric(allocated.get("allocated_min", pd.Series(dtype=float)), errors="coerce").fillna(0.0)))
    nblock_map = dict(zip(allocated.get("provider_id", pd.Series(dtype=str)).astype(str), pd.to_numeric(allocated.get("n_blocks", pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)))

    for p in providers:
        rp = req[req["provider_id"].astype(str) == str(p)] if not req.empty else pd.DataFrame()
        allocated_min = float(alloc_map.get(str(p), 0.0))
        n_blocks = int(nblock_map.get(str(p), math.ceil(allocated_min / cfg.block_minutes) if allocated_min > 0 else 0))
        target_util = float(target_map.get(str(p), cfg.default_target_utilization))
        er_min = float(er_map.get(str(p), 0.0))
        if rp.empty:
            required_alpha = np.nan
            alpha_achieved = np.nan
            demand_band = quantile_dict([], "demand_")
            required_band = quantile_dict([], "required_")
            case_band = quantile_dict([], "case_")
            turn_band = quantile_dict([], "turnover_")
        else:
            required_alpha = float(np.quantile(rp["required_min"], cfg.alpha))
            alpha_achieved = float(np.mean(allocated_min >= rp["required_min"].to_numpy(dtype=float)))
            demand_band = quantile_dict(rp["demand_min_horizon"], "demand_")
            required_band = quantile_dict(rp["required_min"], "required_")
            case_band = quantile_dict(rp["case_min_horizon"], "case_")
            turn_band = quantile_dict(rp["turnover_min_horizon"], "turnover_")
            max_blocks = max(n_blocks + 5, int(np.ceil(np.nanmax(rp["required_min"]) / cfg.block_minutes)) + 1, 1)
            curve = coverage_curve_for_provider(rp["required_min"], cfg.block_minutes, min(max_blocks, 60), cfg.alpha)
            if not curve.empty:
                curve["provider_id"] = p
                curve_rows.extend(curve.to_dict(orient="records"))

            for t in cfg.counterfactual_targets:
                alt_req = rp["demand_min_horizon"] / max(float(t), 1e-6) + er_min
                cf_rows.append({
                    "provider_id": p,
                    "target_utilization": float(t),
                    "current_allocated_min": allocated_min,
                    "current_allocated_hrs": allocated_min / 60.0,
                    "required_min_p50": float(np.quantile(alt_req, 0.50)),
                    "required_min_alpha": float(np.quantile(alt_req, cfg.alpha)),
                    "coverage_at_current_allocation": float(np.mean(allocated_min >= alt_req)),
                    "min_blocks_to_meet_alpha": int(np.ceil(float(np.quantile(alt_req, cfg.alpha)) / cfg.block_minutes)),
                    "min_hours_to_meet_alpha": float(np.ceil(float(np.quantile(alt_req, cfg.alpha)) / cfg.block_minutes) * cfg.block_minutes / 60.0),
                })
            for lam in cfg.demand_multipliers:
                scaled_req = (rp["demand_min_horizon"] * float(lam)) / max(target_util, 1e-6) + er_min
                demand_mult_rows.append({
                    "provider_id": p,
                    "demand_multiplier": float(lam),
                    "current_allocated_min": allocated_min,
                    "coverage_at_current_allocation": float(np.mean(allocated_min >= scaled_req)),
                    "required_min_alpha": float(np.quantile(scaled_req, cfg.alpha)),
                    "min_blocks_to_meet_alpha": int(np.ceil(float(np.quantile(scaled_req, cfg.alpha)) / cfg.block_minutes)),
                })

        row = {
            "provider_id": p,
            "allocated_min": allocated_min,
            "allocated_hrs": allocated_min / 60.0,
            "n_blocks": n_blocks,
            "target_utilization": target_util,
            "early_release_min": er_min,
            "required_min_alpha": required_alpha,
            "required_hrs_alpha": required_alpha / 60.0 if pd.notna(required_alpha) else np.nan,
            "alpha_achieved": alpha_achieved,
            "meets_alpha": bool(alpha_achieved >= cfg.alpha) if pd.notna(alpha_achieved) else False,
            "shortage_min": max((required_alpha if pd.notna(required_alpha) else 0.0) - allocated_min, 0.0),
            "coverage_ratio": allocated_min / max(required_alpha if pd.notna(required_alpha) else 1.0, 1.0),
            **demand_band,
            **required_band,
            **case_band,
            **turn_band,
        }
        row["binding_factor"] = binding_factor_from_row(pd.Series(row), cfg.alpha)
        rows.append(row)

    goal_df = pd.DataFrame(rows).sort_values(["meets_alpha", "shortage_min"], ascending=[True, False]) if rows else pd.DataFrame()
    cf_df = pd.DataFrame(cf_rows)
    demand_mult_df = pd.DataFrame(demand_mult_rows)
    curve_df = pd.DataFrame(curve_rows)

    # enrich from coverage file if present
    if not goal_df.empty and not coverage.empty and "provider_id" in coverage.columns:
        keep = [c for c in ["provider_id", "coverage_ratio", "meets_alpha", "shortage_min", "slack_min"] if c in coverage.columns]
        cov_small = coverage[keep].copy()
        goal_df = goal_df.merge(cov_small.add_prefix("layer2_"), left_on="provider_id", right_on="layer2_provider_id", how="left")

    if not goal_df.empty:
        goal_df.to_csv(out / "goal_attribution.csv", index=False)
    if not cf_df.empty:
        cf_df.to_csv(out / "counterfactuals.csv", index=False)
    if not demand_mult_df.empty:
        demand_mult_df.to_csv(out / "demand_sensitivity_counterfactuals.csv", index=False)
    if not curve_df.empty:
        curve_df.to_csv(out / "coverage_curves.csv", index=False)

    changes = block_changes_detail(art2.schedule, out)
    if cfg.write_plots:
        plot_goal_coverage(goal_df, out / "plots", cfg)
        plot_counterfactuals(cf_df, out / "plots")

    return {
        "goal_attribution": goal_df,
        "counterfactuals": cf_df,
        "demand_sensitivity": demand_mult_df,
        "coverage_curves": curve_df,
        "block_changes": changes,
        "required_by_scenario": req,
    }


# =============================================================================
# Confidence bands and risk profiles
# =============================================================================


def confidence_bands(
    art1: Layer1Artifacts,
    art2: Layer2Artifacts,
    goal_outputs: dict[str, Any],
    out: Path,
    cfg: Layer3Config,
) -> pd.DataFrame:
    out = ensure_dir(out)
    q = goal_outputs.get("required_by_scenario", pd.DataFrame())
    if q.empty:
        q = aggregate_scenarios_by_provider(art1.scenarios, cfg.optimization_weeks)
    allocation = allocation_from_schedule(art2.schedule, cfg.block_minutes)
    coverage = standardize_coverage(art2.coverage)
    if not coverage.empty and "allocated_min" in coverage.columns:
        allocated = coverage[["provider_id", "allocated_min"]].copy()
    else:
        allocated = allocation[["provider_id", "allocated_min"]].copy() if not allocation.empty else pd.DataFrame(columns=["provider_id", "allocated_min"])
    alloc_map = dict(zip(allocated.get("provider_id", pd.Series(dtype=str)).astype(str), pd.to_numeric(allocated.get("allocated_min", pd.Series(dtype=float)), errors="coerce").fillna(0.0)))

    rows = []
    scen_rows = []
    if q.empty:
        bands = pd.DataFrame()
        bands.to_csv(out / "confidence_bands.csv", index=False)
        return bands

    # q may include required_min already or just demand horizon
    if "demand_min_horizon" not in q.columns:
        return pd.DataFrame()
    for p, grp in q.groupby("provider_id"):
        allocated_min = float(alloc_map.get(str(p), 0.0))
        demand = grp["demand_min_horizon"].to_numpy(dtype=float)
        util = demand / allocated_min if allocated_min > 1e-6 else np.repeat(np.nan, len(demand))
        for sid, u, d in zip(grp["scenario_id"].tolist(), util, demand):
            scen_rows.append({"provider_id": p, "scenario_id": sid, "allocated_min": allocated_min, "demand_min_horizon": float(d), "projected_utilization": u})
        bd = quantile_dict(util, "util_")
        demand_bd = quantile_dict(demand, "demand_")
        p10 = bd.get("util_p10")
        p90 = bd.get("util_p90")
        p50 = bd.get("util_p50")
        rows.append({
            "provider_id": p,
            "allocated_min": allocated_min,
            "allocated_hrs": allocated_min / 60.0,
            **bd,
            **demand_bd,
            "band_width_pp": (p90 - p10) * 100.0 if p90 is not None and p10 is not None else np.nan,
            "underuse_risk_flag": bool(p50 < 0.50) if p50 is not None and pd.notna(p50) else False,
            "capacity_risk_flag": bool(p90 > 1.00) if p90 is not None and pd.notna(p90) else False,
        })
    bands = pd.DataFrame(rows).sort_values("band_width_pp", ascending=False)
    util_scen = pd.DataFrame(scen_rows)
    bands.to_csv(out / "confidence_bands.csv", index=False)
    util_scen.to_csv(out / "utilization_by_scenario.csv", index=False)
    if cfg.write_plots:
        plot_confidence_bands(bands, out / "plots")
    return bands


def provider_risk_profiles(
    art1: Layer1Artifacts,
    bands: pd.DataFrame,
    shap_provider: pd.DataFrame,
    goal_df: pd.DataFrame,
    out: Path,
    cfg: Layer3Config,
) -> pd.DataFrame:
    out = ensure_dir(out)
    if bands.empty:
        return pd.DataFrame()
    profiles = bands.copy()
    profiles["provider_id"] = profiles["provider_id"].astype(str)

    if not art1.sigma.empty:
        sig = art1.sigma.copy()
        prov_col = first_existing_col(sig, ["provider_id", "provider"])
        case_col = first_existing_col(sig, ["sigma_case", "sigma_casetime", "sigma_pd_casetime", "sigma_casetime_min"])
        turn_col = first_existing_col(sig, ["sigma_turn", "sigma_turnover", "sigma_pd_turnover"])
        if prov_col and case_col:
            sig[prov_col] = normalize_provider_id_series(sig[prov_col])
            agg_map = {"demand_sigma_case_mean": (case_col, "mean")}
            if turn_col:
                agg_map["demand_sigma_turn_mean"] = (turn_col, "mean")
            sig_agg = sig.groupby(prov_col).agg(**agg_map).reset_index().rename(columns={prov_col: "provider_id"})
            profiles = profiles.merge(sig_agg, on="provider_id", how="left")
    if "demand_sigma_case_mean" not in profiles.columns:
        profiles["demand_sigma_case_mean"] = profiles.get("demand_p90", 0).fillna(0) - profiles.get("demand_p10", 0).fillna(0)

    # dominant SHAP reason per provider
    if not shap_provider.empty and {"provider_id", "reason_code", "shap_abs_sum_min"}.issubset(shap_provider.columns):
        dom = shap_provider.sort_values("shap_abs_sum_min", ascending=False).groupby("provider_id", as_index=False).head(1)
        dom = dom[["provider_id", "reason_code", "plain_language", "shap_sum_min", "shap_abs_sum_min"]].rename(columns={
            "reason_code": "dominant_shap_reason",
            "plain_language": "dominant_shap_plain_language",
            "shap_sum_min": "dominant_shap_signed_min",
            "shap_abs_sum_min": "dominant_shap_abs_min",
        })
        profiles = profiles.merge(dom, on="provider_id", how="left")

    if not goal_df.empty and {"provider_id", "binding_factor"}.issubset(goal_df.columns):
        profiles = profiles.merge(goal_df[["provider_id", "binding_factor", "meets_alpha", "alpha_achieved", "shortage_min"]], on="provider_id", how="left")

    sigma_threshold = max(15.0, float(np.nanmedian(profiles["demand_sigma_case_mean"])) if profiles["demand_sigma_case_mean"].notna().any() else 15.0)
    band_threshold = max(20.0, float(np.nanmedian(profiles["band_width_pp"])) if profiles["band_width_pp"].notna().any() else 20.0)

    def quadrant(r: pd.Series) -> str:
        high_sigma = clean_numeric(r.get("demand_sigma_case_mean", 0.0)) >= sigma_threshold
        wide_band = clean_numeric(r.get("band_width_pp", 0.0)) >= band_threshold
        if high_sigma and wide_band:
            return "high_attention"
        if high_sigma and not wide_band:
            return "uncertain_but_stable"
        if (not high_sigma) and wide_band:
            return "variable_performer"
        return "predictable_performer"

    def recommendation(r: pd.Series) -> str:
        if bool(r.get("capacity_risk_flag", False)):
            return "Capacity risk: consider overflow capacity, extra block, or closer monitoring."
        if bool(r.get("underuse_risk_flag", False)):
            return "Underuse risk: review whether allocated time can be reduced or shared."
        if r.get("risk_quadrant") == "high_attention":
            return "High attention: review in committee; demand and outcome uncertainty are both high."
        if r.get("risk_quadrant") == "predictable_performer":
            return "Predictable: plan is reliable; safe to keep target unless strategic priorities change."
        return "Monitor: uncertainty is present but not an immediate red flag."

    profiles["risk_quadrant"] = profiles.apply(quadrant, axis=1)
    profiles["risk_recommendation"] = profiles.apply(recommendation, axis=1)
    profiles.to_csv(out / "provider_risk_profiles.csv", index=False)
    if cfg.write_plots:
        plot_risk_matrix(profiles, out / "plots")
    return profiles


# =============================================================================
# Pareto diagnostics
# =============================================================================


def normalize_candidate_metrics(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    df = candidates.copy()
    theme_col = first_existing_col(df, ["theme", "name", "candidate", "candidate_id"])
    if theme_col and theme_col != "theme":
        df = df.rename(columns={theme_col: "theme"})
    if "theme" not in df.columns:
        df["theme"] = [f"candidate_{i}" for i in range(len(df))]

    metric_aliases = {
        "utilization_score": ["utilization_score", "util_score", "coverage_score", "target_coverage", "site_summary.util_after_p50", "util_after_p50", "objectives.utilization", "metrics.utilization_score", "score"],
        "stability_score": ["stability_score", "metrics.stability_score", "objectives.stability", "O_stability"],
        "continuity_score": ["continuity_score", "metrics.continuity_score", "objectives.continuity", "O_continuity"],
        "preference_score": ["preference_score", "metrics.preference_score", "objectives.preference", "O_preference"],
        "fragmentation": ["fragmentation", "fragmentation_count", "O2_fragmentation", "objectives.O2_fragmentation", "metrics.fragmentation"],
        "changes": ["changes", "n_changes", "O3_changes", "objectives.O3_changes", "block_changes", "metrics.changes"],
        "shortage_total": ["shortage_total", "total_shortage", "slack_total", "objectives.slack_total", "undershoot_total", "objectives.undershoot_total"],
        "goals_met_count": ["goals_met_count", "goals_fully_met", "site_summary.goals_fully_met", "metrics.goals_met_count"],
        "solve_time_s": ["solve_time_s", "runtime_s", "time_s"],
    }
    for canon, aliases in metric_aliases.items():
        if canon not in df.columns:
            vals = []
            for _, r in df.iterrows():
                vals.append(candidate_value(r, aliases, np.nan))
            df[canon] = vals
        df[canon] = pd.to_numeric(df[canon], errors="coerce")
    # If utilization score looks like a minimization/deviation, derive utility inversely only when no utility present.
    if df["utilization_score"].isna().all():
        dev_col = first_existing_col(df, ["O1_deviation_min", "objectives.O1_deviation_min", "deviation", "deviation_min"])
        if dev_col:
            dev = pd.to_numeric(df[dev_col], errors="coerce")
            df["utilization_score"] = 1.0 - (dev - dev.min()) / max(float(dev.max() - dev.min()), 1e-9)
    return df


def pareto_non_dominated(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    beneficial = [c for c in ["utilization_score", "stability_score", "continuity_score", "preference_score", "goals_met_count"] if c in df.columns and not df[c].isna().all()]
    negative = [c for c in ["fragmentation", "changes", "shortage_total"] if c in df.columns and not df[c].isna().all()]
    if not beneficial and not negative:
        return pd.Series([True] * len(df), index=df.index)
    mat_cols = beneficial + negative
    mat = df[mat_cols].copy()
    for c in negative:
        mat[c] = -pd.to_numeric(mat[c], errors="coerce")
    mat = mat.fillna(mat.min(numeric_only=True) - 1.0)
    arr = mat.to_numpy(dtype=float)
    nd = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not nd[i]:
            continue
        for j in range(len(df)):
            if i == j:
                continue
            if np.all(arr[j] >= arr[i] - 1e-12) and np.any(arr[j] > arr[i] + 1e-12):
                nd[i] = False
                break
    return pd.Series(nd, index=df.index)


def compute_knee(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "utilization_score" not in df.columns:
        return {"knee_theme": None, "reason": "No utilization_score"}
    cost_col = None
    for c in ["changes", "fragmentation", "shortage_total"]:
        if c in df.columns and df[c].notna().sum() >= 3:
            cost_col = c
            break
    if cost_col is None or df["utilization_score"].notna().sum() < 3:
        best = df.sort_values("utilization_score", ascending=False).iloc[0]
        return {"knee_theme": best.get("theme"), "reason": "Too few points; selected best utilization"}
    d = df[["theme", cost_col, "utilization_score"]].dropna().sort_values(cost_col).reset_index(drop=True)
    if len(d) < 3:
        best = d.sort_values("utilization_score", ascending=False).iloc[0]
        return {"knee_theme": best.get("theme"), "reason": "Too few points; selected best utilization"}
    x = d[cost_col].to_numpy(dtype=float)
    y = d["utilization_score"].to_numpy(dtype=float)
    x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-9)
    y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-9)
    # distance from line between first and last point
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line = p2 - p1
    denom = max(np.linalg.norm(line), 1e-9)
    distances = []
    for xi, yi in zip(x_norm, y_norm):
        p = np.array([xi, yi])
        distances.append(float(np.abs(np.cross(line, p1 - p)) / denom))
    idx = int(np.argmax(distances))
    return {
        "knee_theme": str(d.iloc[idx]["theme"]),
        "cost_metric": cost_col,
        "knee_cost": float(d.iloc[idx][cost_col]),
        "knee_utilization_score": float(d.iloc[idx]["utilization_score"]),
        "distance_to_chord": float(distances[idx]),
        "reason": "Maximum distance-to-chord knee on utilization-vs-cost frontier",
    }


def frontier_shape(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or "utilization_score" not in df.columns:
        return {"shape": "unknown", "interpretation": "No candidate metrics found."}
    cost_col = "changes" if "changes" in df.columns and df["changes"].notna().sum() >= 2 else "fragmentation"
    if cost_col not in df.columns or df[cost_col].notna().sum() < 2:
        return {"shape": "unknown", "interpretation": "Not enough cost/change data to characterize frontier."}
    d = df[[cost_col, "utilization_score"]].dropna().sort_values(cost_col)
    if len(d) < 2:
        return {"shape": "unknown", "interpretation": "Not enough frontier points."}
    dx = float(d[cost_col].max() - d[cost_col].min())
    dy = float(d["utilization_score"].max() - d["utilization_score"].min())
    slope = dy / max(dx, 1e-9)
    if dy < 0.005:
        shape = "flat"
        interp = "Moving blocks changes operational structure but buys very little utilization; supply compatibility may dominate."
    elif slope > 0.02:
        shape = "steep"
        interp = "Utilization improves quickly with changes; current allocation is far from demand distribution."
    else:
        shape = "knee_or_gradual"
        interp = "There is a trade-off curve; inspect the knee point for the best operational compromise."
    return {"shape": shape, "cost_metric": cost_col, "delta_cost": dx, "delta_utilization_score": dy, "slope": slope, "interpretation": interp}


def pareto_diagnostics(art2: Layer2Artifacts, out: Path, cfg: Layer3Config) -> dict[str, Any]:
    out = ensure_dir(out)
    cand = normalize_candidate_metrics(art2.candidates_all)
    if cand.empty and art2.selected_candidate:
        cand = normalize_candidate_metrics(pd.DataFrame([flatten_dict(art2.selected_candidate)]))
    if cand.empty:
        return {"candidates": pd.DataFrame(), "diagnostics": {"warning": "No Pareto candidate table found."}, "near_optimal": pd.DataFrame()}

    cand["is_non_dominated"] = pareto_non_dominated(cand)
    knee = compute_knee(cand[cand["is_non_dominated"]].copy() if cand["is_non_dominated"].any() else cand)
    shape = frontier_shape(cand)
    best_util = float(cand["utilization_score"].max()) if cand["utilization_score"].notna().any() else np.nan
    selected_theme = art2.selected_theme

    # near-optimal alternatives: utilization close to best but better secondary score/cost than selected anchor
    selected = cand[cand["theme"].astype(str) == str(selected_theme)]
    if selected.empty:
        selected = cand.sort_values("utilization_score", ascending=False).head(1)
    selected_row = selected.iloc[0] if not selected.empty else pd.Series(dtype=float)
    tol = cfg.utilization_near_optimal_tol
    near = cand[cand["utilization_score"] >= best_util * (1 - tol)].copy() if pd.notna(best_util) else cand.copy()
    if not near.empty:
        near["delta_utilization_vs_best"] = near["utilization_score"] - best_util
        for c in ["stability_score", "continuity_score", "preference_score"]:
            if c in near.columns and c in selected_row.index:
                near[f"delta_{c}_vs_selected"] = near[c] - clean_numeric(selected_row.get(c), 0.0)
        for c in ["fragmentation", "changes", "shortage_total"]:
            if c in near.columns and c in selected_row.index:
                near[f"delta_{c}_vs_selected"] = near[c] - clean_numeric(selected_row.get(c), 0.0)
    cand.to_csv(out / "frontier_diagnostics_table.csv", index=False)
    near.to_csv(out / "near_optimal_pareto_recommendations.csv", index=False)

    diagnostics = {
        "selected_theme": selected_theme,
        "best_utilization_score": best_util,
        "non_dominated_count": int(cand["is_non_dominated"].sum()),
        "candidate_count": int(len(cand)),
        "knee": knee,
        "frontier_shape": shape,
        "near_optimal_tolerance": tol,
        "near_optimal_count": int(len(near)),
    }
    write_json(out / "frontier_diagnostics.json", diagnostics)
    if cfg.write_plots:
        plot_pareto_frontier(cand, out / "plots")
    return {"candidates": cand, "diagnostics": diagnostics, "near_optimal": near}


# =============================================================================
# Drift and validation
# =============================================================================


def drift_detection(
    art1: Layer1Artifacts,
    shap_provider: pd.DataFrame,
    profiles: pd.DataFrame,
    out: Path,
    cfg: Layer3Config,
) -> dict[str, Any]:
    out = ensure_dir(out)
    shap_drift = pd.DataFrame()
    sigma_drift = pd.DataFrame()
    feature_proxy_drift = pd.DataFrame()

    # Compare against previous Layer 3 run if provided
    if cfg.previous_layer3_dir:
        prev = Path(cfg.previous_layer3_dir)
        prev_shap = safe_read_csv(prev / "shap_provider_summary.csv")
        if not prev_shap.empty and not shap_provider.empty:
            keys = ["target", "provider_id", "reason_code"]
            if set(keys + ["shap_sum_min"]).issubset(prev_shap.columns) and set(keys + ["shap_sum_min"]).issubset(shap_provider.columns):
                cur = shap_provider[keys + ["shap_sum_min", "shap_abs_sum_min"]].rename(columns={"shap_sum_min": "shap_current", "shap_abs_sum_min": "abs_current"})
                old = prev_shap[keys + ["shap_sum_min"]].rename(columns={"shap_sum_min": "shap_previous"})
                shap_drift = cur.merge(old, on=keys, how="inner")
                shap_drift["drift_ratio"] = (shap_drift["shap_current"] - shap_drift["shap_previous"]).abs() / shap_drift["shap_previous"].abs().clip(lower=1e-6)
                shap_drift["drift_flag"] = (shap_drift["drift_ratio"] > cfg.drift_threshold) & (shap_drift["abs_current"] > 10.0)
                shap_drift = shap_drift.sort_values("drift_ratio", ascending=False)
                shap_drift.to_csv(out / "shap_temporal_drift.csv", index=False)
        prev_profiles = safe_read_csv(prev / "provider_risk_profiles.csv")
        if not prev_profiles.empty and not profiles.empty and "demand_sigma_case_mean" in profiles.columns and "demand_sigma_case_mean" in prev_profiles.columns:
            sigma_drift = profiles[["provider_id", "demand_sigma_case_mean"]].merge(
                prev_profiles[["provider_id", "demand_sigma_case_mean"]], on="provider_id", suffixes=("_current", "_previous")
            )
            sigma_drift["sigma_drift_ratio"] = (sigma_drift["demand_sigma_case_mean_current"] - sigma_drift["demand_sigma_case_mean_previous"]).abs() / sigma_drift["demand_sigma_case_mean_previous"].abs().clip(lower=1e-6)
            sigma_drift["sigma_drift_direction"] = np.where(
                sigma_drift["demand_sigma_case_mean_current"] >= sigma_drift["demand_sigma_case_mean_previous"], "widening", "narrowing"
            )
            sigma_drift["drift_flag"] = sigma_drift["sigma_drift_ratio"] > cfg.drift_threshold
            sigma_drift.to_csv(out / "sigma_temporal_drift.csv", index=False)

    # Proxy drift from recent vs longer window in features
    features = art1.features.copy()
    prov_col = first_existing_col(features, ["provider_id", "provider"])
    pairs = [
        ("trailing_4w_std_case", "trailing_12w_std_case", "case_volatility"),
        ("trailing_4w_mean_case", "trailing_12w_mean_case", "case_level"),
        ("trailing_4w_std_turn", "trailing_12w_std_turn", "turnover_volatility"),
        ("trailing_4w_mean_turn", "trailing_12w_mean_turn", "turnover_level"),
    ]
    proxy_rows = []
    if prov_col:
        for recent, base, label in pairs:
            if recent in features.columns and base in features.columns:
                temp = features[[prov_col, recent, base]].copy()
                temp[prov_col] = normalize_provider_id_series(temp[prov_col])
                agg = temp.groupby(prov_col).agg(recent_value=(recent, "mean"), baseline_value=(base, "mean")).reset_index().rename(columns={prov_col: "provider_id"})
                agg["signal"] = label
                agg["drift_ratio"] = (agg["recent_value"] - agg["baseline_value"]).abs() / agg["baseline_value"].abs().clip(lower=1e-6)
                agg["drift_direction"] = np.where(agg["recent_value"] >= agg["baseline_value"], "increasing", "decreasing")
                agg["drift_flag"] = agg["drift_ratio"] > cfg.drift_threshold
                proxy_rows.extend(agg.to_dict(orient="records"))
    feature_proxy_drift = pd.DataFrame(proxy_rows).sort_values("drift_ratio", ascending=False) if proxy_rows else pd.DataFrame()
    if not feature_proxy_drift.empty:
        feature_proxy_drift.to_csv(out / "feature_proxy_drift.csv", index=False)

    report = {
        "shap_temporal_drift_flags": int(shap_drift.get("drift_flag", pd.Series(dtype=bool)).sum()) if not shap_drift.empty else 0,
        "sigma_temporal_drift_flags": int(sigma_drift.get("drift_flag", pd.Series(dtype=bool)).sum()) if not sigma_drift.empty else 0,
        "feature_proxy_drift_flags": int(feature_proxy_drift.get("drift_flag", pd.Series(dtype=bool)).sum()) if not feature_proxy_drift.empty else 0,
        "threshold": cfg.drift_threshold,
        "interpretation": "SHAP drift is earliest; sigma widening follows; calibration decline is last. Flagged providers should be monitored or recalibrated.",
    }
    write_json(out / "drift_report.json", report)
    return {"shap_drift": shap_drift, "sigma_drift": sigma_drift, "feature_proxy_drift": feature_proxy_drift, "report": report}


def posterior_diagnostics(art1: Layer1Artifacts, out: Path, cfg: Layer3Config) -> pd.DataFrame:
    rows = []
    # Look for existing diagnostics in metrics JSON first
    def scan(obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{prefix}.{i}")
        else:
            lk = prefix.lower()
            if "rhat" in lk or "r_hat" in lk or "diverg" in lk or "ess" in lk:
                rows.append({"metric_path": prefix, "value": obj})
    scan(art1.metrics)

    # Also search nearby JSON files
    for p in Path(art1.paths.get("features", out)).parent.glob("*diagnostic*.json") if art1.paths.get("features") else []:
        try:
            obj = read_json(p)
            before = len(rows)
            scan(obj, p.name)
            if len(rows) == before:
                rows.append({"metric_path": p.name, "value": "loaded_no_scalar_diagnostics"})
        except Exception:
            pass
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out / "posterior_diagnostics_summary.csv", index=False)
    return df


# =============================================================================
# Recommendations and narratives
# =============================================================================


def build_recommendations(
    profiles: pd.DataFrame,
    goal_df: pd.DataFrame,
    near_pareto: pd.DataFrame,
    drift: dict[str, Any],
    out: Path,
) -> pd.DataFrame:
    rows = []
    if not profiles.empty:
        for _, r in profiles.iterrows():
            p = str(r.get("provider_id"))
            if bool(r.get("capacity_risk_flag", False)):
                rows.append({"level": "provider", "provider_id": p, "priority": "high", "type": "capacity_risk", "recommendation": "P90 projected utilization exceeds 100%; consider overflow block, add capacity, or manual review."})
            if bool(r.get("underuse_risk_flag", False)):
                rows.append({"level": "provider", "provider_id": p, "priority": "medium", "type": "underuse_risk", "recommendation": "P50 projected utilization is below 50%; review whether some time can be reclaimed or shared."})
            if r.get("risk_quadrant") == "high_attention":
                rows.append({"level": "provider", "provider_id": p, "priority": "high", "type": "high_attention", "recommendation": "Both demand uncertainty and outcome uncertainty are high; bring this provider to committee first."})
    if not goal_df.empty:
        for _, r in goal_df.iterrows():
            if not bool(r.get("meets_alpha", False)):
                rows.append({"level": "provider", "provider_id": str(r.get("provider_id")), "priority": "high", "type": "goal_not_met", "recommendation": f"Goal misses alpha threshold; binding factor={r.get('binding_factor')}; shortage={clean_numeric(r.get('shortage_min'))/60:.1f} hours."})
    if not near_pareto.empty:
        for _, r in near_pareto.head(5).iterrows():
            rows.append({"level": "pareto", "provider_id": "ALL", "priority": "medium", "type": "near_optimal_alternative", "recommendation": f"Candidate {r.get('theme')} is within utilization tolerance; compare secondary metrics before final committee choice."})
    proxy = drift.get("feature_proxy_drift", pd.DataFrame()) if isinstance(drift, dict) else pd.DataFrame()
    if not proxy.empty:
        flagged = proxy[proxy.get("drift_flag", False).astype(bool)].head(20)
        for _, r in flagged.iterrows():
            rows.append({"level": "provider", "provider_id": str(r.get("provider_id")), "priority": "medium", "type": "drift_monitor", "recommendation": f"Recent {r.get('signal')} is {r.get('drift_direction')} vs baseline by {clean_numeric(r.get('drift_ratio')):.0%}; monitor next quarter."})
    rec = pd.DataFrame(rows)
    if not rec.empty:
        rec.to_csv(out / "recommendations.csv", index=False)
        write_json(out / "advisor_recommendations.json", rec.to_dict(orient="records"))
    else:
        rec.to_csv(out / "recommendations.csv", index=False)
        write_json(out / "advisor_recommendations.json", [])
    return rec


def provider_narratives(
    shap_provider: pd.DataFrame,
    goal_df: pd.DataFrame,
    bands: pd.DataFrame,
    profiles: pd.DataFrame,
    out: Path,
    selected_theme: str,
    alpha: float,
) -> dict[str, str]:
    providers = set()
    for df in [shap_provider, goal_df, bands, profiles]:
        if not df.empty and "provider_id" in df.columns:
            providers.update(df["provider_id"].astype(str).unique())
    narratives: dict[str, str] = {}
    for p in sorted(providers):
        parts = []
        g = goal_df[goal_df["provider_id"].astype(str) == p].head(1) if not goal_df.empty and "provider_id" in goal_df.columns else pd.DataFrame()
        b = bands[bands["provider_id"].astype(str) == p].head(1) if not bands.empty and "provider_id" in bands.columns else pd.DataFrame()
        pr = profiles[profiles["provider_id"].astype(str) == p].head(1) if not profiles.empty and "provider_id" in profiles.columns else pd.DataFrame()
        sp = shap_provider[shap_provider["provider_id"].astype(str) == p] if not shap_provider.empty and "provider_id" in shap_provider.columns else pd.DataFrame()
        if not g.empty:
            gr = g.iloc[0]
            status = "met" if bool(gr.get("meets_alpha", False)) else "did not meet"
            parts.append(
                f"Provider {p} {status} the α={alpha:.0%} coverage threshold under the selected `{selected_theme}` plan. "
                f"Allocated time is {clean_numeric(gr.get('allocated_hrs')):.1f} hours, with α-required time {clean_numeric(gr.get('required_hrs_alpha')):.1f} hours and achieved scenario coverage {clean_numeric(gr.get('alpha_achieved')):.0%}."
            )
            bf = gr.get("binding_factor", None)
            if bf and bf != "covered":
                parts.append(f"The binding factor is `{bf}`, so the committee should treat this as an operational constraint rather than a pure model-score issue.")
        if not sp.empty and "shap_abs_sum_min" in sp.columns:
            top = sp.sort_values("shap_abs_sum_min", ascending=False).head(1).iloc[0]
            direction = "raises" if clean_numeric(top.get("shap_sum_min")) >= 0 else "lowers"
            parts.append(
                f"The dominant forecast reason is `{top.get('reason_code')}`, which {direction} predicted demand by about {abs(clean_numeric(top.get('shap_sum_min'))):.1f} model-minutes across the explained provider-day rows."
            )
        if not b.empty:
            br = b.iloc[0]
            parts.append(
                f"Projected utilization band is P10={clean_numeric(br.get('util_p10')):.0%}, P50={clean_numeric(br.get('util_p50')):.0%}, P90={clean_numeric(br.get('util_p90')):.0%}; band width is {clean_numeric(br.get('band_width_pp')):.1f} percentage points."
            )
            flags = []
            if bool(br.get("underuse_risk_flag", False)):
                flags.append("underuse risk")
            if bool(br.get("capacity_risk_flag", False)):
                flags.append("capacity overflow risk")
            if flags:
                parts.append("Automatic flags: " + ", ".join(flags) + ".")
        if not pr.empty:
            rr = pr.iloc[0]
            parts.append(f"Risk profile: `{rr.get('risk_quadrant')}`. Recommended action: {rr.get('risk_recommendation', 'monitor')}")
        if not parts:
            parts.append(f"Provider {p} has limited available explainability artifacts; check upstream Layer 1/Layer 2 outputs.")
        narratives[p] = " ".join(parts)
    write_json(out / "narratives.json", {"per_provider": narratives})
    return narratives


def site_summary_narrative(
    goal_df: pd.DataFrame,
    bands: pd.DataFrame,
    profiles: pd.DataFrame,
    pareto: dict[str, Any],
    recs: pd.DataFrame,
    selected_theme: str,
    alpha: float,
) -> str:
    n_providers = int(bands["provider_id"].nunique()) if not bands.empty and "provider_id" in bands.columns else 0
    n_goals = int(len(goal_df)) if not goal_df.empty else 0
    n_met = int(goal_df["meets_alpha"].sum()) if not goal_df.empty and "meets_alpha" in goal_df.columns else 0
    n_under = int(bands["underuse_risk_flag"].sum()) if not bands.empty and "underuse_risk_flag" in bands.columns else 0
    n_cap = int(bands["capacity_risk_flag"].sum()) if not bands.empty and "capacity_risk_flag" in bands.columns else 0
    n_high = int((profiles["risk_quadrant"] == "high_attention").sum()) if not profiles.empty and "risk_quadrant" in profiles.columns else 0
    knee = pareto.get("diagnostics", {}).get("knee", {}) if isinstance(pareto, dict) else {}
    shape = pareto.get("diagnostics", {}).get("frontier_shape", {}) if isinstance(pareto, dict) else {}
    return (
        f"The selected Layer 2 plan is `{selected_theme}`. Layer 3 generated explanations for {n_providers} providers. "
        f"{n_met}/{n_goals} provider goals meet the α={alpha:.0%} scenario-coverage threshold. "
        f"There are {n_under} underuse-risk flags, {n_cap} capacity-risk flags, and {n_high} high-attention providers. "
        f"Pareto frontier shape is `{shape.get('shape', 'unknown')}`; suggested knee candidate is `{knee.get('knee_theme', 'unknown')}`. "
        f"The recommendation table contains {len(recs)} actionable items."
    )


def write_summary_md(
    out: Path,
    site_narrative: str,
    provider_narratives_map: dict[str, str],
    shap_validation: pd.DataFrame,
    goal_df: pd.DataFrame,
    profiles: pd.DataFrame,
    pareto: dict[str, Any],
    recs: pd.DataFrame,
) -> Path:
    lines: list[str] = []
    lines.append("# Layer 3 — Explainability Summary")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(site_narrative)
    lines.append("")
    lines.append("## SHAP Validation")
    if not shap_validation.empty:
        for _, r in shap_validation.iterrows():
            err = r.get("max_efficiency_error_abs_min")
            lines.append(f"- `{r.get('target')}` used `{r.get('explainer_kind')}` on {int(r.get('n_rows_explained', 0))} rows × {int(r.get('n_features', 0))} features. Max efficiency error: {err if pd.notna(err) else 'N/A'}.")
            if str(r.get("warnings", "")).strip():
                lines.append(f"  - Warning: {r.get('warnings')}")
    else:
        lines.append("No SHAP validation was produced.")
    lines.append("")
    lines.append("## Goals")
    if not goal_df.empty:
        met = int(goal_df["meets_alpha"].sum()) if "meets_alpha" in goal_df.columns else 0
        lines.append(f"- Goals met: {met}/{len(goal_df)}")
        worst = goal_df.sort_values("shortage_min", ascending=False).head(10) if "shortage_min" in goal_df.columns else goal_df.head(10)
        for _, r in worst.iterrows():
            status = "✓" if bool(r.get("meets_alpha", False)) else "✗"
            lines.append(f"- {status} `{r.get('provider_id')}`: allocated={clean_numeric(r.get('allocated_hrs')):.1f}h, required α={clean_numeric(r.get('required_hrs_alpha')):.1f}h, coverage={clean_numeric(r.get('alpha_achieved')):.0%}, factor=`{r.get('binding_factor')}`")
    else:
        lines.append("No goal attribution table available.")
    lines.append("")
    lines.append("## Provider Risk Profiles")
    if not profiles.empty and "risk_quadrant" in profiles.columns:
        counts = profiles["risk_quadrant"].value_counts().to_dict()
        for k, v in counts.items():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("No provider risk profiles available.")
    lines.append("")
    lines.append("## Pareto Frontier")
    diag = pareto.get("diagnostics", {}) if isinstance(pareto, dict) else {}
    lines.append(f"- Candidate count: {diag.get('candidate_count', 'N/A')}")
    lines.append(f"- Non-dominated count: {diag.get('non_dominated_count', 'N/A')}")
    lines.append(f"- Knee: {diag.get('knee', {}).get('knee_theme', 'N/A')}")
    lines.append(f"- Shape: {diag.get('frontier_shape', {}).get('shape', 'N/A')} — {diag.get('frontier_shape', {}).get('interpretation', '')}")
    lines.append("")
    lines.append("## Top Actionable Recommendations")
    if not recs.empty:
        for _, r in recs.head(15).iterrows():
            lines.append(f"- **{r.get('priority')} / {r.get('type')} / {r.get('provider_id')}** — {r.get('recommendation')}")
    else:
        lines.append("No actionable flags were generated.")
    lines.append("")
    lines.append("## Provider Narratives (sample)")
    for p, text in list(provider_narratives_map.items())[:12]:
        lines.append(f"### {p}")
        lines.append(text)
        lines.append("")
    lines.append("## Artifact Inventory")
    for rel in [
        "explanations.json", "shap_values.pkl", "shap_values_long.csv", "shap_top_reasons.csv",
        "shap_provider_summary.csv", "shap_peer_comparison.csv", "counterfactuals.csv",
        "coverage_curves.csv", "confidence_bands.csv", "provider_risk_profiles.csv",
        "frontier_diagnostics.json", "near_optimal_pareto_recommendations.csv", "drift_report.json",
        "recommendations.csv", "narratives.json", "provenance.json", "validation_report.json",
    ]:
        if (out / rel).exists():
            lines.append(f"- `{rel}`")
    path = out / "SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# =============================================================================
# Plots
# =============================================================================


def _savefig(path: Path):
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_shap_summary(shap_provider: pd.DataFrame, plot_dir: Path) -> None:
    if plt is None or shap_provider.empty or "reason_code" not in shap_provider.columns:
        return
    d = shap_provider.groupby("reason_code")["shap_abs_sum_min"].sum().sort_values(ascending=True).tail(12)
    if d.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.barh(d.index.astype(str), d.values)
    plt.xlabel("Total |SHAP| minutes")
    plt.title("Top SHAP Reason Codes")
    _savefig(plot_dir / "shap_reason_codes.png")


def plot_goal_coverage(goal_df: pd.DataFrame, plot_dir: Path, cfg: Layer3Config) -> None:
    if plt is None or goal_df.empty or "alpha_achieved" not in goal_df.columns:
        return
    d = goal_df.sort_values("alpha_achieved").tail(25)
    plt.figure(figsize=(10, 6))
    plt.barh(d["provider_id"].astype(str), d["alpha_achieved"].astype(float))
    plt.axvline(cfg.alpha, linestyle="--")
    plt.xlabel("Scenario coverage")
    plt.title("Goal Coverage by Provider")
    _savefig(plot_dir / "goal_coverage.png")


def plot_counterfactuals(cf_df: pd.DataFrame, plot_dir: Path) -> None:
    if plt is None or cf_df.empty or "target_utilization" not in cf_df.columns:
        return
    top_providers = cf_df.groupby("provider_id")["coverage_at_current_allocation"].mean().sort_values().head(8).index.tolist()
    d = cf_df[cf_df["provider_id"].isin(top_providers)]
    if d.empty:
        return
    plt.figure(figsize=(10, 6))
    for p, grp in d.groupby("provider_id"):
        grp = grp.sort_values("target_utilization")
        plt.plot(grp["target_utilization"], grp["coverage_at_current_allocation"], marker="o", label=str(p))
    plt.xlabel("Alternative target utilization")
    plt.ylabel("Coverage at current allocation")
    plt.title("What-if: Target Utilization Sensitivity")
    plt.legend(fontsize=8)
    _savefig(plot_dir / "counterfactual_target_sensitivity.png")


def plot_confidence_bands(bands: pd.DataFrame, plot_dir: Path) -> None:
    if plt is None or bands.empty or not {"util_p10", "util_p50", "util_p90"}.issubset(bands.columns):
        return
    d = bands.sort_values("band_width_pp", ascending=False).head(25).sort_values("util_p50")
    y = np.arange(len(d))
    lower = d["util_p50"].astype(float) - d["util_p10"].astype(float)
    upper = d["util_p90"].astype(float) - d["util_p50"].astype(float)
    plt.figure(figsize=(10, 7))
    plt.errorbar(d["util_p50"].astype(float), y, xerr=[lower, upper], fmt="o")
    plt.yticks(y, d["provider_id"].astype(str))
    plt.axvline(1.0, linestyle="--")
    plt.xlabel("Projected utilization")
    plt.title("Projected Utilization Bands (P10–P90)")
    _savefig(plot_dir / "utilization_confidence_bands.png")


def plot_risk_matrix(profiles: pd.DataFrame, plot_dir: Path) -> None:
    if plt is None or profiles.empty or not {"demand_sigma_case_mean", "band_width_pp"}.issubset(profiles.columns):
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(profiles["demand_sigma_case_mean"], profiles["band_width_pp"])
    if profiles["demand_sigma_case_mean"].notna().any():
        plt.axvline(max(15.0, np.nanmedian(profiles["demand_sigma_case_mean"])), linestyle="--")
    if profiles["band_width_pp"].notna().any():
        plt.axhline(max(20.0, np.nanmedian(profiles["band_width_pp"])), linestyle="--")
    plt.xlabel("Demand uncertainty σ")
    plt.ylabel("Outcome band width (percentage points)")
    plt.title("Provider Risk Matrix")
    _savefig(plot_dir / "provider_risk_matrix.png")


def plot_pareto_frontier(cand: pd.DataFrame, plot_dir: Path) -> None:
    if plt is None or cand.empty or "utilization_score" not in cand.columns:
        return
    xcol = "changes" if "changes" in cand.columns and cand["changes"].notna().any() else "fragmentation"
    if xcol not in cand.columns:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(cand[xcol], cand["utilization_score"])
    for _, r in cand.iterrows():
        plt.annotate(str(r.get("theme", ""))[:18], (r[xcol], r["utilization_score"]), fontsize=8)
    plt.xlabel(xcol)
    plt.ylabel("Utilization score")
    plt.title("Pareto Frontier Diagnostics")
    _savefig(plot_dir / "pareto_frontier.png")


# =============================================================================
# Unified payload and validation
# =============================================================================


def build_explanations_payload(
    provider_narratives_map: dict[str, str],
    shap_top: pd.DataFrame,
    goal_df: pd.DataFrame,
    cf_df: pd.DataFrame,
    bands: pd.DataFrame,
    profiles: pd.DataFrame,
    recs: pd.DataFrame,
    pareto: dict[str, Any],
    cfg: Layer3Config,
    out: Path,
) -> dict[str, Any]:
    providers = sorted(provider_narratives_map.keys())
    payload: dict[str, Any] = {
        "selected_theme": pareto.get("diagnostics", {}).get("selected_theme", cfg.selected_theme),
        "generated_at": datetime.now().isoformat(),
        "alpha": cfg.alpha,
        "optimization_weeks": cfg.optimization_weeks,
        "per_provider": {},
        "site": {
            "pareto": pareto.get("diagnostics", {}),
            "n_recommendations": int(len(recs)),
        },
    }
    for p in providers:
        item: dict[str, Any] = {"narrative": provider_narratives_map[p]}
        if not shap_top.empty and "provider_id" in shap_top.columns:
            item["shap_reason_codes"] = shap_top[shap_top["provider_id"].astype(str) == p].head(10).to_dict(orient="records")
        if not goal_df.empty and "provider_id" in goal_df.columns:
            item["goal_attribution"] = goal_df[goal_df["provider_id"].astype(str) == p].head(1).to_dict(orient="records")
        if not cf_df.empty and "provider_id" in cf_df.columns:
            item["counterfactuals"] = cf_df[cf_df["provider_id"].astype(str) == p].to_dict(orient="records")
        if not bands.empty and "provider_id" in bands.columns:
            item["confidence_band"] = bands[bands["provider_id"].astype(str) == p].head(1).to_dict(orient="records")
        if not profiles.empty and "provider_id" in profiles.columns:
            item["risk_profile"] = profiles[profiles["provider_id"].astype(str) == p].head(1).to_dict(orient="records")
        if not recs.empty and "provider_id" in recs.columns:
            item["recommendations"] = recs[recs["provider_id"].astype(str) == p].to_dict(orient="records")
        payload["per_provider"][p] = item
    write_json(out / "explanations.json", payload)
    return payload


def artifact_validation(out: Path, art1: Layer1Artifacts, art2: Layer2Artifacts, results: dict[str, Any]) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "", severity: str = "PASS"):
        checks.append({"check": name, "ok": bool(ok), "severity": severity if ok else "FAIL", "detail": detail})

    def add_warn(name: str, ok: bool, detail: str = ""):
        checks.append({"check": name, "ok": bool(ok), "severity": "PASS" if ok else "WARN", "detail": detail})

    add("Layer 1 features loaded", not art1.features.empty, f"rows={len(art1.features)}")
    add("Layer 1 scenarios loaded", not art1.scenarios.empty, f"rows={len(art1.scenarios)}")
    add_warn("Layer 1 models loaded", len(art1.models) > 0, f"models={list(art1.models.keys())}")
    add_warn("Layer 2 coverage loaded", not art2.coverage.empty, f"rows={len(art2.coverage)}")
    add_warn("Layer 2 schedule loaded", not art2.schedule.empty, f"rows={len(art2.schedule)}")

    shap_validation = results.get("shap", {}).get("shap_validation", pd.DataFrame())
    add_warn("SHAP attribution produced", not results.get("shap", {}).get("shap_provider", pd.DataFrame()).empty, "provider-level SHAP summary")
    if not shap_validation.empty and "max_efficiency_error_abs_min" in shap_validation.columns:
        max_err = pd.to_numeric(shap_validation["max_efficiency_error_abs_min"], errors="coerce").max()
        add_warn("SHAP efficiency approximately valid", pd.isna(max_err) or max_err < 1e-3 or "fallback" in " ".join(shap_validation.get("explainer_kind", pd.Series(dtype=str)).astype(str)), f"max_error={max_err}")

    goal_df = results.get("goals", {}).get("goal_attribution", pd.DataFrame())
    add_warn("Goal attribution produced", not goal_df.empty, f"rows={len(goal_df)}")
    bands = results.get("bands", pd.DataFrame())
    add_warn("Confidence bands produced", not bands.empty, f"rows={len(bands)}")
    profiles = results.get("profiles", pd.DataFrame())
    add_warn("Risk profiles produced", not profiles.empty, f"rows={len(profiles)}")
    pareto_cand = results.get("pareto", {}).get("candidates", pd.DataFrame())
    add_warn("Pareto diagnostics produced", not pareto_cand.empty, f"candidates={len(pareto_cand)}")

    required_files = [
        "explanations.json", "SUMMARY.md", "provenance.json", "recommendations.csv",
        "confidence_bands.csv", "provider_risk_profiles.csv", "frontier_diagnostics.json",
    ]
    for f in required_files:
        add_warn(f"Artifact written: {f}", (out / f).exists(), str(out / f))

    # Include upstream warnings
    for w in art1.warnings + art2.warnings:
        checks.append({"check": "Upstream warning", "ok": False, "severity": "WARN", "detail": w})

    df = pd.DataFrame(checks)
    df.to_csv(out / "validation_report.csv", index=False)
    write_json(out / "validation_report.json", checks)
    return df


def write_provenance(out: Path, cfg: Layer3Config, art1: Layer1Artifacts, art2: Layer2Artifacts) -> None:
    prov = {
        "generated_at": datetime.now().isoformat(),
        "layer1_dir": str(cfg.layer1_dir),
        "layer2_dir": str(cfg.layer2_dir),
        "pre_layer_result": str(cfg.pre_layer_result) if cfg.pre_layer_result else None,
        "layer3_out_dir": str(out),
        "config": dataclasses.asdict(cfg),
        "layer1_paths": art1.paths,
        "layer2_paths": art2.paths,
        "selected_theme": art2.selected_theme,
    }
    # dataclasses asdict leaves Paths; json_default handles them.
    write_json(out / "provenance.json", prov)


# =============================================================================
# Orchestration
# =============================================================================


def run_layer3(cfg: Layer3Config) -> dict[str, Any]:
    np.random.seed(cfg.random_seed)
    out = ensure_dir(cfg.out_dir / f"run_{now_tag()}" if cfg.out_dir.name != "layer3_current_run" else cfg.out_dir)
    ensure_dir(out / "plots")

    log.info("Layer 3 output dir: %s", out)
    art1 = load_layer1_artifacts(cfg.layer1_dir)
    art2 = load_layer2_artifacts(cfg.layer2_dir, cfg.selected_theme)
    cfg.selected_theme = art2.selected_theme
    write_provenance(out, cfg, art1, art2)

    log.info("Stage 0 — SHAP attribution")
    shap_result = shap_attribution(art1, out, cfg)

    log.info("Stage 1+2 — Goal attribution and counterfactuals")
    goal_result = goal_attribution_and_counterfactuals(art1, art2, out, cfg)

    log.info("Stage 3 — Confidence bands")
    bands = confidence_bands(art1, art2, goal_result, out, cfg)

    log.info("Stage 3b — Provider risk profiles")
    profiles = provider_risk_profiles(art1, bands, shap_result.get("shap_provider", pd.DataFrame()), goal_result.get("goal_attribution", pd.DataFrame()), out, cfg)

    log.info("Stage 4 — Pareto frontier diagnostics")
    pareto = pareto_diagnostics(art2, out, cfg)

    log.info("Stage 5 — Drift detection")
    drift = drift_detection(art1, shap_result.get("shap_provider", pd.DataFrame()), profiles, out, cfg)
    posterior = posterior_diagnostics(art1, out, cfg)

    log.info("Stage 6 — Recommendations and narratives")
    recs = build_recommendations(profiles, goal_result.get("goal_attribution", pd.DataFrame()), pareto.get("near_optimal", pd.DataFrame()), drift, out)
    provider_text = provider_narratives(
        shap_result.get("shap_provider", pd.DataFrame()),
        goal_result.get("goal_attribution", pd.DataFrame()),
        bands,
        profiles,
        out,
        art2.selected_theme,
        cfg.alpha,
    )
    site_text = site_summary_narrative(goal_result.get("goal_attribution", pd.DataFrame()), bands, profiles, pareto, recs, art2.selected_theme, cfg.alpha)
    narratives_obj = read_json(out / "narratives.json") if (out / "narratives.json").exists() else {"per_provider": provider_text}
    narratives_obj["site_summary"] = site_text
    write_json(out / "narratives.json", narratives_obj)

    build_explanations_payload(provider_text, shap_result.get("shap_top", pd.DataFrame()), goal_result.get("goal_attribution", pd.DataFrame()), goal_result.get("counterfactuals", pd.DataFrame()), bands, profiles, recs, pareto, cfg, out)

    summary_path = write_summary_md(
        out,
        site_text,
        provider_text,
        shap_result.get("shap_validation", pd.DataFrame()),
        goal_result.get("goal_attribution", pd.DataFrame()),
        profiles,
        pareto,
        recs,
    )

    results = {
        "out_dir": str(out),
        "summary_path": str(summary_path),
        "shap": shap_result,
        "goals": goal_result,
        "bands": bands,
        "profiles": profiles,
        "pareto": pareto,
        "drift": drift,
        "posterior_diagnostics": posterior,
        "recommendations": recs,
    }
    validation = artifact_validation(out, art1, art2, results)
    results["validation"] = validation

    log.info("Layer 3 complete: %s", out)
    return results

#!/usr/bin/env python3
"""
Layer 3 — Explainability, TRUE SHAP edition
===========================================

Drop-in replacement for:
    src/layer3/run_layer3.py

Main fixes compared with the fallback version:
  1. Uses real Tree SHAP for XGBoost models whenever SHAP is installed.
  2. Does NOT silently use permutation fallback by default.
  3. Extracts the exact feature columns used during Layer 1 training from the
     saved sklearn Pipeline / model_metrics.json, so Layer 2 columns such as
     is_included and total_util are never passed into the model.
  4. Computes and saves SHAP tables, reason-code summaries, plots, interaction
     diagnostics, goal attribution, counterfactuals, confidence bands, provider
     risk profiles, Pareto diagnostics, temporal drift flags, recommendations,
     provenance, and validation.

Expected upstream files
-----------------------
Layer 1 directory:
    features.csv
    scenarios_long.csv
    point_forecasts.csv
    model_metrics.json
    casetime_min_model.joblib
    turnover_min_model.joblib
    sigma_estimates.csv          optional but useful

Layer 2 directory:
    provider_day_coverage.csv
    plans/<selected_theme>_selected/schedule_by_week.csv
    pareto/frontier_summary.csv, pareto_frontier.csv, or similar optional

Run example
-----------
python3 src/layer3/run_layer3.py \
  --layer1_dir outputs/layer1 \
  --layer2_dir outputs/layer2 \
  --pre_layer_result outputs/pre_layer/prelayer_result.json \
  --out outputs/layer3 \
  --alpha 0.85 \
  --optimization_weeks 4 \
  --selected_theme utilization_first \
  --top_k_features 12 \
  --max_shap_rows 5000 \
  --make_interactions
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import logging
import math
import os
import pickle
import random
import re
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG = logging.getLogger("layer3_true_shap")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class Layer3Config:
    layer1_dir: str = "outputs/layer1"
    layer2_dir: str = "outputs/layer2"
    pre_layer_result: str = "outputs/pre_layer/prelayer_result.json"
    out: str = "outputs/layer3"
    selected_theme: str = "utilization_first"
    alpha: float = 0.85
    optimization_weeks: int = 4
    block_minutes: float = 480.0
    top_k_features: int = 12
    max_shap_rows: int = 5000
    make_interactions: bool = False
    max_interaction_rows: int = 1000
    random_seed: int = 42
    # SHAP efficiency is measured in minutes. Tree SHAP is additive up to tiny
    # floating-point noise; 0.01 min = 0.6 sec is still essentially exact.
    shap_efficiency_abs_tol_min: float = 1e-2
    shap_efficiency_rel_tol: float = 1e-6
    shap_efficiency_warn_tol_min: float = 1.0
    # default is strict because this layer should honestly report true SHAP.
    allow_permutation_fallback: bool = False
    require_true_shap: bool = True
    # plots
    n_waterfall_examples: int = 3
    n_dependence_plots: int = 4


# -----------------------------------------------------------------------------
# Small IO helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        if not np.isfinite(o):
            return None
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, _dt.datetime, _dt.date)):
        return str(o)
    return str(o)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)


def write_table(df: pd.DataFrame, path_base: str | Path) -> None:
    """Write CSV and Parquet when available."""
    base = Path(path_base)
    ensure_dir(base.parent)
    df.to_csv(base.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(base.with_suffix(".parquet"), index=False)
    except Exception:
        pass


def load_csv(path: str | Path, required: bool = False) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(str(p))
        return pd.DataFrame()
    return pd.read_csv(p)


def first_existing(paths: Sequence[str | Path]) -> Optional[Path]:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None


def safe_num(s: Any, default: float = 0.0) -> pd.Series | float:
    try:
        return pd.to_numeric(s, errors="coerce").fillna(default)
    except Exception:
        try:
            val = float(s)
            return val if np.isfinite(val) else default
        except Exception:
            return default


def find_col(df: pd.DataFrame, aliases: Sequence[str], required: bool = False, label: str = "column") -> Optional[str]:
    if df is None or df.empty:
        if required:
            raise ValueError(f"Could not find {label}; DataFrame is empty")
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for a in aliases:
        if a in df.columns:
            return a
        la = a.lower()
        if la in lower:
            return lower[la]
    if required:
        raise ValueError(f"Could not find {label}. Tried aliases={aliases}. Columns={list(df.columns)}")
    return None


def normalise_id_series(s: Any) -> pd.Series:
    ser = pd.Series(s).astype(str).str.strip()
    ser = ser.replace({"nan": "", "None": "", "NaN": ""})
    return ser


# -----------------------------------------------------------------------------
# Loading Layer 1/2 artifacts
# -----------------------------------------------------------------------------

def load_layer1_artifacts(cfg: Layer3Config) -> Dict[str, Any]:
    l1 = Path(cfg.layer1_dir)
    artifacts: Dict[str, Any] = {}

    artifacts["features"] = load_csv(l1 / "features.csv", required=True)
    artifacts["scenarios"] = load_csv(l1 / "scenarios_long.csv", required=True)
    artifacts["forecasts"] = load_csv(l1 / "point_forecasts.csv")
    # Layer 1 has used a few names across versions. Prefer the real Bayesian
    # posterior sigma summary when present; otherwise fall back to older names.
    sigma_candidates = [
        l1 / "hierarchical_sigma_estimates.csv",
        l1 / "sigma_estimates.csv",
        l1 / "bayesian_sigma_estimates.csv",
        l1 / "sigma_posterior_summary.csv",
        l1 / "residual_sigma_estimates.csv",
    ]
    sigma_path = next((p for p in sigma_candidates if p.exists()), None)
    artifacts["sigma"] = load_csv(sigma_path) if sigma_path is not None else pd.DataFrame()
    if sigma_path is not None:
        LOG.info("Layer 1 sigma estimates loaded: %s rows=%s", sigma_path, len(artifacts["sigma"]))
    else:
        LOG.warning("No Layer 1 sigma estimates CSV found. Provider risk profiles will use bands and goals only.")
    artifacts["metrics"] = read_json(l1 / "model_metrics.json", default={}) or {}

    models: Dict[str, Any] = {}
    model_paths = {
        "casetime_min": l1 / "casetime_min_model.joblib",
        "turnover_min": l1 / "turnover_min_model.joblib",
    }
    if joblib is None:
        raise ImportError("joblib is required to load saved Layer 1 models. Install with: python3 -m pip install joblib")
    for target, path in model_paths.items():
        if path.exists():
            models[target] = joblib.load(path)
            LOG.info("Layer 1 model loaded: %s -> %s", target, path)
        else:
            LOG.warning("Layer 1 model missing: %s", path)
    artifacts["models"] = models

    LOG.info("Layer 1 features loaded: %s rows=%s cols=%s", l1 / "features.csv", len(artifacts["features"]), artifacts["features"].shape[1])
    LOG.info("Layer 1 scenarios loaded: %s rows=%s cols=%s", l1 / "scenarios_long.csv", len(artifacts["scenarios"]), artifacts["scenarios"].shape[1])
    if not artifacts["forecasts"].empty:
        LOG.info("Layer 1 forecasts loaded: rows=%s", len(artifacts["forecasts"]))
    return artifacts


def locate_schedule(layer2_dir: str | Path, selected_theme: str) -> Optional[Path]:
    l2 = Path(layer2_dir)
    candidates = [
        l2 / "plans" / f"{selected_theme}_selected" / "schedule_by_week.csv",
        l2 / "plans" / selected_theme / "schedule_by_week.csv",
        l2 / f"{selected_theme}_selected" / "schedule_by_week.csv",
        l2 / "schedule_by_week.csv",
        l2 / "selected_schedule.csv",
        l2 / "schedule.csv",
    ]
    found = first_existing(candidates)
    if found is not None:
        return found
    matches = sorted(l2.glob("**/schedule_by_week.csv")) + sorted(l2.glob("**/*schedule*.csv"))
    return matches[0] if matches else None


def load_layer2_artifacts(cfg: Layer3Config) -> Dict[str, Any]:
    l2 = Path(cfg.layer2_dir)
    artifacts: Dict[str, Any] = {}

    coverage_path = first_existing([
        l2 / "provider_day_coverage.csv",
        l2 / "coverage_by_provider_day.csv",
        l2 / "selected_provider_day_coverage.csv",
    ])
    if coverage_path is not None:
        coverage = load_csv(coverage_path)
        coverage_source = str(coverage_path)
        LOG.info("Layer 2 coverage loaded: %s rows=%s", coverage_path, len(coverage))
    else:
        coverage = pd.DataFrame()
        coverage_source = "missing"
        LOG.warning("No Layer 2 coverage CSV found in %s", l2)

    schedule_path = locate_schedule(l2, cfg.selected_theme)
    if schedule_path is not None:
        schedule = load_csv(schedule_path)
        LOG.info("Layer 2 schedule loaded: %s rows=%s cols=%s", schedule_path, len(schedule), schedule.shape[1])
    else:
        schedule = pd.DataFrame()
        LOG.warning("No Layer 2 schedule CSV found in %s", l2)

    artifacts["coverage"] = coverage
    artifacts["coverage_source"] = coverage_source
    artifacts["schedule"] = schedule
    artifacts["schedule_path"] = str(schedule_path) if schedule_path else ""
    return artifacts


# -----------------------------------------------------------------------------
# Model introspection and true SHAP
# -----------------------------------------------------------------------------

KNOWN_NON_FEATURE_COLS = {
    "provider_id", "provider", "provider_name", "name", "day_of_week", "dow_name",
    "week_start", "week_end", "date", "target", "target_name",
    "casetime_min", "turnover_min", "case_min", "turn_min", "actual_casetime_min",
    "actual_turnover_min", "mu_casetime_min", "mu_turnover_min",
    "is_included", "total_util", "overall_score", "useful_utilization", "target_coverage",
    "allocated_min", "allocated_total_min", "target_total_min", "shortage_min", "excess_min",
}


def get_feature_columns_from_metrics(metrics: Dict[str, Any], target: str) -> List[str]:
    """Supports both older and newer model_metrics.json shapes."""
    if not metrics:
        return []

    candidates: List[Any] = []
    candidates.append(metrics.get("feature_columns"))
    if target in metrics and isinstance(metrics[target], dict):
        candidates.append(metrics[target].get("feature_columns"))
    # common nested keys
    for key in ("point_models", "models", "model_metrics"):
        obj = metrics.get(key)
        if isinstance(obj, dict):
            if target in obj and isinstance(obj[target], dict):
                candidates.append(obj[target].get("feature_columns"))
            candidates.append(obj.get("feature_columns"))

    for cand in candidates:
        if isinstance(cand, list) and cand:
            return [str(c) for c in cand]
    return []


def get_feature_columns_from_model(model: Any) -> List[str]:
    """Extract the exact input feature columns expected by the sklearn Pipeline.

    This is the key fix for the previous failure where Layer 3 passed extra
    columns (is_included, total_util) into a pipeline trained on 63 features.
    """
    attrs = ["feature_names_in_", "feature_name_", "feature_names"]
    for a in attrs:
        val = getattr(model, a, None)
        if val is not None:
            try:
                return [str(x) for x in list(val)]
            except Exception:
                pass

    # sklearn Pipeline: inspect each step, especially SimpleImputer.
    steps = getattr(model, "steps", None)
    if steps:
        for _, step in steps:
            val = getattr(step, "feature_names_in_", None)
            if val is not None:
                try:
                    return [str(x) for x in list(val)]
                except Exception:
                    pass

    # XGBoost booster names, if present.
    try:
        booster = model.get_booster()
        if booster is not None and booster.feature_names:
            return [str(x) for x in booster.feature_names]
    except Exception:
        pass

    # Pipeline final estimator booster names.
    if steps:
        try:
            final = steps[-1][1]
            booster = final.get_booster()
            if booster is not None and booster.feature_names:
                return [str(x) for x in booster.feature_names]
        except Exception:
            pass

    return []


def infer_feature_columns(features: pd.DataFrame, model: Any, metrics: Dict[str, Any], target: str) -> List[str]:
    cols = get_feature_columns_from_model(model)
    if not cols:
        cols = get_feature_columns_from_metrics(metrics, target)
    if cols:
        missing = [c for c in cols if c not in features.columns]
        if missing:
            raise ValueError(
                f"Saved model expects {len(cols)} feature columns for {target}, but {len(missing)} are missing from features.csv. "
                f"Missing examples={missing[:15]}"
            )
        return cols

    # Last resort: numeric columns excluding known non-feature columns.
    numeric_cols = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    cols = [c for c in numeric_cols if str(c) not in KNOWN_NON_FEATURE_COLS]
    if not cols:
        raise ValueError(f"Could not infer feature columns for {target}.")
    LOG.warning("Could not read feature columns from saved model/metrics for %s. Falling back to %d numeric columns.", target, len(cols))
    return cols


def get_pipeline_steps(model: Any) -> List[Tuple[str, Any]]:
    steps = getattr(model, "steps", None)
    if steps:
        return list(steps)
    return []


def transform_for_final_estimator(model: Any, X_input: pd.DataFrame) -> Tuple[Any, Any, List[str], str]:
    """Return transformed X, final estimator, transformed feature names, source label."""
    steps = get_pipeline_steps(model)
    if not steps:
        feature_names = list(X_input.columns)
        return X_input, model, feature_names, "raw_estimator"

    X_work: Any = X_input.copy()
    feature_names = list(X_input.columns)

    for name, step in steps[:-1]:
        X_work = step.transform(X_work)
        # best effort names after transformer
        try:
            feature_names = [str(x) for x in step.get_feature_names_out(feature_names)]
        except Exception:
            # SimpleImputer/scaler preserve count/order.
            if hasattr(X_work, "shape") and len(feature_names) == int(X_work.shape[1]):
                pass
            else:
                feature_names = [f"f{i}" for i in range(int(X_work.shape[1]))]

    final_name, final_est = steps[-1]
    if hasattr(X_work, "to_numpy"):
        X_arr = X_work.to_numpy()
    else:
        X_arr = np.asarray(X_work)
    if len(feature_names) != X_arr.shape[1]:
        feature_names = [f"f{i}" for i in range(X_arr.shape[1])]
    return X_arr, final_est, feature_names, f"sklearn_pipeline_final={final_name}"


def predict_pipeline_safely(model: Any, X_input: pd.DataFrame) -> np.ndarray:
    pred = model.predict(X_input)
    return np.asarray(pred, dtype=float).reshape(-1)


def predict_final_safely(final_est: Any, X_proc: Any) -> np.ndarray:
    try:
        pred = final_est.predict(X_proc)
    except Exception:
        # XGBoost can sometimes validate feature metadata too strictly.
        try:
            pred = final_est.predict(np.asarray(X_proc), validate_features=False)
        except Exception:
            pred = final_est.predict(np.asarray(X_proc))
    return np.asarray(pred, dtype=float).reshape(-1)


def import_shap_or_raise() -> Any:
    try:
        import shap  # type: ignore
        return shap
    except Exception as e:
        raise ImportError(
            "SHAP is not installed or failed to import. For true SHAP run:\n"
            "  python3 -m pip install shap\n"
            "Then rerun Layer 3. The new code does not silently call the result SHAP if SHAP is missing."
        ) from e


def reason_code_for_feature(feature: str) -> Tuple[str, str]:
    f = str(feature).lower()
    mapping = [
        (("attn_weighted_util", "weighted_util", "util"), "utilization_history", "Historical utilization level drives the forecast."),
        (("attn_turn_weighted", "turn_weighted"), "turnover_history", "Historical turnover pattern drives the forecast."),
        (("attn_entropy", "entropy"), "pattern_consistency", "Historical pattern consistency affects confidence."),
        (("attn_recency", "recency"), "trend_direction", "Recent weeks receive attention in the forecast."),
        (("trailing_4w_mean", "trailing_2w_mean", "recent_mean"), "recent_history", "Recent demand history drives the forecast."),
        (("trailing_12w", "trailing_8w", "trailing_6w"), "long_history", "Longer historical window drives the forecast."),
        (("std", "volatility"), "recent_volatility", "Demand variability affects uncertainty and predicted need."),
        (("exception_rate", "exception"), "series_contention", "Historical exceptions or block disruptions affect this slot/provider."),
        (("service_line", "sl_"), "service_line_context", "Peer service-line context affects the forecast."),
        (("rotation_phase", "phase"), "rotation_cycle_position", "Position in the detected T* rotation cycle affects the forecast."),
        (("week", "woy", "sin", "cos", "month"), "seasonality", "Calendar seasonality affects the forecast."),
        (("allocated", "block", "capacity"), "template_capacity", "Template capacity/history affects the forecast."),
        (("early_release", "release"), "early_release_history", "Early release history affects required allocation."),
        (("weeks_observed", "observed"), "data_maturity", "Amount of observed history affects reliability."),
    ]
    for keys, code, desc in mapping:
        if any(k in f for k in keys):
            return code, desc
    return "other_model_signal", "Other model feature contribution."


def _sample_features(features: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(features) <= max_rows:
        return features.copy().reset_index(drop=True)
    return features.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def compute_true_shap_for_target(
    target: str,
    model: Any,
    features: pd.DataFrame,
    metrics: Dict[str, Any],
    out_dir: Path,
    cfg: Layer3Config,
) -> Dict[str, Any]:
    shap = import_shap_or_raise()

    feature_cols = infer_feature_columns(features, model, metrics, target)
    X_full = features[feature_cols].copy()
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X_sample = _sample_features(X_full, cfg.max_shap_rows, cfg.random_seed)

    # This call intentionally uses only exact training columns.
    pipeline_pred = predict_pipeline_safely(model, X_sample)
    X_proc, final_est, proc_names, transform_source = transform_for_final_estimator(model, X_sample)
    final_pred = predict_final_safely(final_est, X_proc)

    # True Tree SHAP for XGBoost / tree model.
    explainer = shap.TreeExplainer(final_est)
    shap_values = explainer.shap_values(X_proc)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        # multiclass-like; use first output, not expected here.
        shap_values = shap_values[:, :, 0]

    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        expected_arr = np.asarray(expected).reshape(-1)
        base_value = float(expected_arr[0])
    else:
        base_value = float(expected)

    shap_pred = base_value + shap_values.sum(axis=1)
    efficiency_error = np.abs(shap_pred - final_pred)
    pipeline_consistency_error = np.abs(final_pred - pipeline_pred)

    # Tables
    idx_cols = [c for c in ["provider_id", "day_of_week", "week_index", "rotation_phase"] if c in features.columns]
    sampled_meta = features.loc[X_sample.index if len(features) == len(X_sample) else X_sample.index, idx_cols].copy() if idx_cols else pd.DataFrame(index=np.arange(len(X_sample)))
    # The sample reset_index loses original index for random sample, so rebuild meta directly from original sampling routine.
    if len(features) > cfg.max_shap_rows:
        sampled_original = features.sample(n=cfg.max_shap_rows, random_state=cfg.random_seed).reset_index(drop=True)
    else:
        sampled_original = features.reset_index(drop=True)
    sampled_meta = sampled_original[[c for c in idx_cols if c in sampled_original.columns]].copy() if idx_cols else pd.DataFrame(index=np.arange(len(X_sample)))

    # Keep SHAP contribution columns separate from metadata/model feature names.
    # Some training features are also metadata columns (for example rotation_phase
    # and day_of_week).  If we use raw feature names as SHAP columns and then
    # insert metadata, pandas raises: "cannot insert rotation_phase, already exists".
    # Prefixing only the wide SHAP-value columns preserves the real feature names
    # in the importance/top-k tables while making this export collision-proof.
    shap_value_cols = [f"shap__{str(c)}" for c in proc_names]
    shap_wide = pd.DataFrame(shap_values, columns=shap_value_cols)
    shap_wide.insert(0, "target", target)
    shap_wide.insert(1, "base_value", base_value)
    shap_wide.insert(2, "prediction_from_shap", shap_pred)
    shap_wide.insert(3, "model_prediction", final_pred)
    shap_wide.insert(4, "pipeline_prediction", pipeline_pred)
    for c in reversed(sampled_meta.columns):
        out_col = str(c)
        if out_col in shap_wide.columns:
            out_col = f"meta__{out_col}"
        shap_wide.insert(0, out_col, sampled_meta[c].values)
    write_table(shap_wide, out_dir / "shap" / f"shap_values_wide_{target}")

    global_imp = pd.DataFrame({
        "target": target,
        "feature": proc_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
        "std_shap": shap_values.std(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    global_imp[["reason_code", "reason_description"]] = global_imp["feature"].apply(lambda x: pd.Series(reason_code_for_feature(x)))
    write_csv(global_imp, out_dir / "shap" / f"shap_global_importance_{target}.csv")

    # Top-k per row long table.
    top_rows: List[Dict[str, Any]] = []
    k = min(cfg.top_k_features, shap_values.shape[1])
    for i in range(shap_values.shape[0]):
        order = np.argsort(np.abs(shap_values[i]))[::-1][:k]
        base: Dict[str, Any] = {"target": target, "row_id": i, "prediction": float(final_pred[i]), "base_value": base_value}
        for c in sampled_meta.columns:
            base[c] = sampled_meta.iloc[i][c]
        for rank, j in enumerate(order, start=1):
            code, desc = reason_code_for_feature(proc_names[j])
            row = dict(base)
            row.update({
                "rank": rank,
                "feature": proc_names[j],
                "shap_value": float(shap_values[i, j]),
                "abs_shap_value": float(abs(shap_values[i, j])),
                "direction": "pushes_up" if shap_values[i, j] >= 0 else "pushes_down",
                "feature_value": float(np.asarray(X_proc)[i, j]) if np.issubdtype(np.asarray(X_proc).dtype, np.number) else None,
                "reason_code": code,
                "reason_description": desc,
            })
            top_rows.append(row)
    top_df = pd.DataFrame(top_rows)
    write_csv(top_df, out_dir / "shap" / f"shap_top_features_{target}.csv")

    # Provider-level aggregation if provider_id exists.
    provider_reason = pd.DataFrame()
    if "provider_id" in top_df.columns:
        provider_reason = (
            top_df.groupby(["target", "provider_id", "reason_code"], dropna=False)
            .agg(
                total_abs_shap=("abs_shap_value", "sum"),
                mean_abs_shap=("abs_shap_value", "mean"),
                mean_signed_shap=("shap_value", "mean"),
                n_contributions=("shap_value", "size"),
            )
            .reset_index()
            .sort_values(["target", "provider_id", "total_abs_shap"], ascending=[True, True, False])
        )
        write_csv(provider_reason, out_dir / "shap" / f"shap_provider_reason_codes_{target}.csv")

    # Plots
    plot_dir = ensure_dir(out_dir / "plots" / "shap")
    make_shap_plots(
        shap_module=shap,
        target=target,
        shap_values=shap_values,
        X_proc=np.asarray(X_proc),
        feature_names=proc_names,
        base_value=base_value,
        predictions=final_pred,
        global_imp=global_imp,
        plot_dir=plot_dir,
        cfg=cfg,
    )

    interaction_summary = None
    if cfg.make_interactions:
        interaction_summary = compute_shap_interactions(
            shap_module=shap,
            explainer=explainer,
            X_proc=np.asarray(X_proc),
            feature_names=proc_names,
            target=target,
            out_dir=out_dir,
            cfg=cfg,
        )

    result = {
        "target": target,
        "model_source": "shap_tree_explainer_pipeline",
        "transform_source": transform_source,
        "n_rows_explained": int(shap_values.shape[0]),
        "n_training_feature_columns": int(len(feature_cols)),
        "n_transformed_feature_columns": int(len(proc_names)),
        "input_feature_columns": feature_cols,
        "transformed_feature_columns": proc_names,
        "base_value": base_value,
        "mean_efficiency_error": float(np.mean(efficiency_error)),
        "max_efficiency_error": float(np.max(efficiency_error)),
        "p95_efficiency_error": float(np.quantile(efficiency_error, 0.95)),
        "max_efficiency_error_seconds": float(np.max(efficiency_error) * 60.0),
        "max_abs_prediction": float(np.max(np.abs(final_pred))) if len(final_pred) else 0.0,
        "mean_pipeline_consistency_error": float(np.mean(pipeline_consistency_error)),
        "max_pipeline_consistency_error": float(np.max(pipeline_consistency_error)),
        "top_features": global_imp.head(cfg.top_k_features).to_dict(orient="records"),
        "interaction_summary": interaction_summary,
    }
    write_json(result, out_dir / "shap" / f"shap_diagnostics_{target}.json")
    return result


def make_shap_plots(
    shap_module: Any,
    target: str,
    shap_values: np.ndarray,
    X_proc: np.ndarray,
    feature_names: List[str],
    base_value: float,
    predictions: np.ndarray,
    global_imp: pd.DataFrame,
    plot_dir: Path,
    cfg: Layer3Config,
) -> None:
    if plt is None:
        return
    X_display = pd.DataFrame(X_proc, columns=feature_names)

    # Bar summary
    try:
        plt.figure(figsize=(10, 7))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*NumPy global RNG was seeded.*", category=FutureWarning)
            shap_module.summary_plot(shap_values, X_display, feature_names=feature_names, plot_type="bar", show=False, max_display=cfg.top_k_features)
        plt.tight_layout()
        plt.savefig(plot_dir / f"shap_summary_bar_{target}.png", dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as e:
        LOG.warning("Could not create SHAP bar plot for %s: %s", target, e)

    # Beeswarm summary
    try:
        plt.figure(figsize=(10, 7))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*NumPy global RNG was seeded.*", category=FutureWarning)
            shap_module.summary_plot(shap_values, X_display, feature_names=feature_names, show=False, max_display=cfg.top_k_features)
        plt.tight_layout()
        plt.savefig(plot_dir / f"shap_beeswarm_{target}.png", dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as e:
        LOG.warning("Could not create SHAP beeswarm plot for %s: %s", target, e)

    # Dependence plots for top features, using plain matplotlib for robustness.
    try:
        top_features = global_imp["feature"].head(cfg.n_dependence_plots).tolist()
        for feat in top_features:
            if feat not in feature_names:
                continue
            j = feature_names.index(feat)
            plt.figure(figsize=(7, 5))
            plt.scatter(X_display[feat], shap_values[:, j], alpha=0.45, s=14)
            plt.axhline(0, linewidth=1)
            plt.xlabel(feat)
            plt.ylabel(f"SHAP value for {target}")
            plt.title(f"SHAP dependence: {target} — {feat}")
            plt.tight_layout()
            safe_feat = re.sub(r"[^A-Za-z0-9_.-]+", "_", feat)[:80]
            plt.savefig(plot_dir / f"shap_dependence_{target}_{safe_feat}.png", dpi=180, bbox_inches="tight")
            plt.close()
    except Exception as e:
        LOG.warning("Could not create SHAP dependence plots for %s: %s", target, e)

    # Waterfall examples: largest predicted rows.
    try:
        order = np.argsort(predictions)[::-1][: max(0, cfg.n_waterfall_examples)]
        for rank, i in enumerate(order, start=1):
            exp = shap_module.Explanation(
                values=shap_values[i],
                base_values=base_value,
                data=X_display.iloc[i].values,
                feature_names=feature_names,
            )
            plt.figure(figsize=(10, 7))
            shap_module.plots.waterfall(exp, max_display=cfg.top_k_features, show=False)
            plt.tight_layout()
            plt.savefig(plot_dir / f"shap_waterfall_{target}_example_{rank}.png", dpi=180, bbox_inches="tight")
            plt.close()
    except Exception as e:
        LOG.warning("Could not create SHAP waterfall examples for %s: %s", target, e)


def compute_shap_interactions(
    shap_module: Any,
    explainer: Any,
    X_proc: np.ndarray,
    feature_names: List[str],
    target: str,
    out_dir: Path,
    cfg: Layer3Config,
) -> Dict[str, Any]:
    n = min(len(X_proc), cfg.max_interaction_rows)
    if n <= 0:
        return {"n_rows": 0}
    rng = np.random.default_rng(cfg.random_seed)
    if len(X_proc) > n:
        idx = rng.choice(np.arange(len(X_proc)), size=n, replace=False)
        X_int = X_proc[idx]
    else:
        X_int = X_proc

    LOG.info("Computing SHAP interaction values for %s on %d rows", target, len(X_int))
    try:
        inter = explainer.shap_interaction_values(X_int)
        if isinstance(inter, list):
            inter = inter[0]
        inter = np.asarray(inter, dtype=float)
        if inter.ndim == 4:
            inter = inter[:, :, :, 0]
    except Exception as e:
        LOG.warning("SHAP interactions failed for %s: %s", target, e)
        return {"error": str(e), "n_rows": int(len(X_int))}

    mean_abs = np.abs(inter).mean(axis=0)
    rows: List[Dict[str, Any]] = []
    p = mean_abs.shape[0]
    for i in range(p):
        for j in range(i + 1, p):
            rows.append({
                "target": target,
                "feature_i": feature_names[i],
                "feature_j": feature_names[j],
                "mean_abs_interaction": float(mean_abs[i, j]),
            })
    pairs = pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False)
    write_csv(pairs, out_dir / "shap" / f"shap_interaction_pairs_{target}.csv")

    if plt is not None and not pairs.empty:
        try:
            top_feats = list(dict.fromkeys(pairs.head(40)[["feature_i", "feature_j"]].to_numpy().reshape(-1).tolist()))[:15]
            inds = [feature_names.index(f) for f in top_feats if f in feature_names]
            mat = mean_abs[np.ix_(inds, inds)]
            plt.figure(figsize=(9, 7))
            plt.imshow(mat, aspect="auto")
            plt.colorbar(label="Mean |SHAP interaction|")
            plt.xticks(range(len(inds)), [feature_names[i] for i in inds], rotation=75, ha="right", fontsize=8)
            plt.yticks(range(len(inds)), [feature_names[i] for i in inds], fontsize=8)
            plt.title(f"Top SHAP interactions — {target}")
            plt.tight_layout()
            plt.savefig(out_dir / "plots" / "shap" / f"shap_interaction_heatmap_{target}.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as e:
            LOG.warning("Could not plot interaction heatmap for %s: %s", target, e)

    return {
        "n_rows": int(len(X_int)),
        "n_pairs": int(len(pairs)),
        "top_pairs": pairs.head(15).to_dict(orient="records"),
    }


def permutation_fallback_for_target(
    target: str,
    model: Any,
    features: pd.DataFrame,
    metrics: Dict[str, Any],
    out_dir: Path,
    cfg: Layer3Config,
) -> Dict[str, Any]:
    """Only used if --allow_permutation_fallback is explicitly passed.

    This is not called SHAP in diagnostics. It prevents silent thesis-risky behavior.
    """
    feature_cols = infer_feature_columns(features, model, metrics, target)
    X = _sample_features(features[feature_cols].replace([np.inf, -np.inf], np.nan), cfg.max_shap_rows, cfg.random_seed)
    pred = predict_pipeline_safely(model, X)
    baseline = float(np.mean(pred))
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for c in feature_cols:
        Xp = X.copy()
        Xp[c] = rng.permutation(Xp[c].values)
        pp = predict_pipeline_safely(model, Xp)
        rows.append({"target": target, "feature": c, "permutation_delta_mean_abs_pred": float(np.mean(np.abs(pred - pp)))})
    imp = pd.DataFrame(rows).sort_values("permutation_delta_mean_abs_pred", ascending=False)
    imp[["reason_code", "reason_description"]] = imp["feature"].apply(lambda x: pd.Series(reason_code_for_feature(x)))
    write_csv(imp, out_dir / "shap" / f"permutation_fallback_importance_{target}.csv")
    return {
        "target": target,
        "model_source": "permutation_fallback_NOT_SHAP",
        "n_rows_explained": int(len(X)),
        "n_training_feature_columns": int(len(feature_cols)),
        "base_value": baseline,
        "mean_efficiency_error": None,
        "max_efficiency_error": None,
        "top_features": imp.head(cfg.top_k_features).to_dict(orient="records"),
    }


def run_shap_attribution(art1: Dict[str, Any], out_dir: Path, cfg: Layer3Config) -> Dict[str, Any]:
    ensure_dir(out_dir / "shap")
    features = art1["features"]
    models = art1["models"]
    metrics = art1.get("metrics", {}) or {}
    results: Dict[str, Any] = {}

    for target in ["casetime_min", "turnover_min"]:
        if target not in models:
            LOG.warning("Skipping SHAP for %s because model is missing", target)
            continue
        LOG.info("Computing TRUE Tree SHAP for %s …", target)
        try:
            results[target] = compute_true_shap_for_target(target, models[target], features, metrics, out_dir, cfg)
        except Exception as e:
            if cfg.allow_permutation_fallback:
                LOG.exception("True SHAP failed for %s; using explicit permutation fallback because --allow_permutation_fallback was set", target)
                results[target] = permutation_fallback_for_target(target, models[target], features, metrics, out_dir, cfg)
                results[target]["true_shap_error"] = str(e)
            else:
                raise

    write_json(results, out_dir / "shap" / "shap_all_diagnostics.json")
    # combined global importance
    frames = []
    for target in results:
        path = out_dir / "shap" / f"shap_global_importance_{target}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        write_csv(combined, out_dir / "shap_global_importance.csv")
    return results


# -----------------------------------------------------------------------------
# Coverage / schedule normalization
# -----------------------------------------------------------------------------

def reconstruct_coverage_from_schedule(schedule: pd.DataFrame, forecasts: pd.DataFrame, cfg: Layer3Config) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()
    provider_col = find_col(schedule, ["provider_id", "assigned_provider_id", "blockholder", "current_blockholder"], False)
    day_col = find_col(schedule, ["day_of_week", "dow", "weekday", "day"], False)
    dur_col = find_col(schedule, ["allocated_min", "duration_min", "slot_capacity_min", "capacity_min", "block_minutes"], False)
    if provider_col is None or day_col is None:
        return pd.DataFrame()
    s = schedule.copy()
    s["provider_id"] = normalise_id_series(s[provider_col])
    s = s[(s["provider_id"] != "") & (s["provider_id"].str.upper() != "OPEN")]
    s["day_of_week"] = pd.to_numeric(s[day_col], errors="coerce").fillna(-1).astype(int)
    if dur_col is not None:
        s["allocated_total_min"] = pd.to_numeric(s[dur_col], errors="coerce").fillna(cfg.block_minutes)
    else:
        s["allocated_total_min"] = cfg.block_minutes
    cov = s.groupby(["provider_id", "day_of_week"], dropna=False).agg(
        allocated_total_min=("allocated_total_min", "sum"),
        n_slots=("allocated_total_min", "size"),
    ).reset_index()

    if not forecasts.empty:
        f = forecasts.copy()
        pcol = find_col(f, ["provider_id", "provider"], False)
        dcol = find_col(f, ["day_of_week", "dow", "weekday", "day"], False)
        ccol = find_col(f, ["mu_casetime_min", "casetime_mu", "casetime_min", "pred_casetime_min"], False)
        tcol = find_col(f, ["mu_turnover_min", "turnover_mu", "turnover_min", "pred_turnover_min"], False)
        if pcol and dcol and ccol:
            f["provider_id"] = normalise_id_series(f[pcol])
            f["day_of_week"] = pd.to_numeric(f[dcol], errors="coerce").fillna(-1).astype(int)
            f["mu_casetime_min"] = pd.to_numeric(f[ccol], errors="coerce").fillna(0.0)
            f["mu_turnover_min"] = pd.to_numeric(f[tcol], errors="coerce").fillna(0.0) if tcol else 0.0
            f["target_total_min"] = (f["mu_casetime_min"] + f["mu_turnover_min"]) * float(cfg.optimization_weeks)
            cov = cov.merge(f[["provider_id", "day_of_week", "target_total_min"]], on=["provider_id", "day_of_week"], how="left")
    if "target_total_min" not in cov.columns:
        cov["target_total_min"] = np.nan
    return cov


def normalize_coverage(coverage: pd.DataFrame, schedule: pd.DataFrame, forecasts: pd.DataFrame, cfg: Layer3Config) -> Tuple[pd.DataFrame, str]:
    source = "layer2_file"
    if coverage.empty:
        coverage = reconstruct_coverage_from_schedule(schedule, forecasts, cfg)
        source = "reconstructed_from_schedule"
    if coverage.empty:
        return pd.DataFrame(), source

    cov = coverage.copy()
    pcol = find_col(cov, ["provider_id", "provider", "assigned_provider_id"], True, "provider_id")
    dcol = find_col(cov, ["day_of_week", "dow", "weekday", "day"], False, "day_of_week")
    cov["provider_id"] = normalise_id_series(cov[pcol])
    if dcol is not None:
        cov["day_of_week"] = pd.to_numeric(cov[dcol], errors="coerce").fillna(-1).astype(int)
    else:
        cov["day_of_week"] = -1

    allocated_col = find_col(cov, ["allocated_total_min", "allocated_min", "assigned_min", "capacity_assigned_min", "A_min"], False)
    target_col = find_col(cov, ["target_total_min", "target_q_min", "target_min", "required_min", "demand_target_min", "R_q_min"], False)
    coverage_col = find_col(cov, ["coverage_probability", "alpha_achieved", "coverage_prob", "scenario_coverage", "target_coverage"], False)
    shortage_col = find_col(cov, ["shortage_min", "slack_min", "under_min", "gap_min"], False)
    excess_col = find_col(cov, ["excess_min", "over_min", "surplus_min"], False)

    if allocated_col is not None:
        cov["allocated_total_min"] = pd.to_numeric(cov[allocated_col], errors="coerce").fillna(0.0)
    elif "allocated_total_min" not in cov.columns:
        cov["allocated_total_min"] = 0.0

    if target_col is not None:
        cov["target_total_min"] = pd.to_numeric(cov[target_col], errors="coerce").fillna(0.0)
    elif "target_total_min" not in cov.columns:
        cov["target_total_min"] = 0.0

    if coverage_col is not None:
        cov["coverage_probability"] = pd.to_numeric(cov[coverage_col], errors="coerce")
    else:
        cov["coverage_probability"] = np.nan

    denom = cov["target_total_min"].replace(0, np.nan)
    cov["coverage_ratio"] = (cov["allocated_total_min"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if shortage_col is not None:
        cov["shortage_min"] = pd.to_numeric(cov[shortage_col], errors="coerce").fillna(0.0)
    else:
        cov["shortage_min"] = np.maximum(cov["target_total_min"] - cov["allocated_total_min"], 0.0)

    if excess_col is not None:
        cov["excess_min"] = pd.to_numeric(cov[excess_col], errors="coerce").fillna(0.0)
    else:
        cov["excess_min"] = np.maximum(cov["allocated_total_min"] - cov["target_total_min"], 0.0)

    goal_col = find_col(cov, ["goal_met", "meets_goal", "alpha_met", "is_goal_met"], False)
    if goal_col is not None:
        raw = cov[goal_col]
        if raw.dtype == bool:
            cov["goal_met"] = raw
        else:
            cov["goal_met"] = raw.astype(str).str.lower().isin(["1", "true", "yes", "y", "pass"])
    else:
        cov["goal_met"] = np.where(
            cov["coverage_probability"].notna(),
            cov["coverage_probability"] >= cfg.alpha,
            cov["shortage_min"] <= 1e-6,
        )

    cov["binding_factor"] = np.select(
        [
            cov["goal_met"].astype(bool),
            cov["allocated_total_min"] <= 1e-9,
            cov["shortage_min"] > 0,
            cov["coverage_ratio"] < 0.50,
        ],
        ["covered", "supply_shortage", "target_level", "case_volume"],
        default="near_target_or_granularity",
    )
    cov["binding_description"] = cov["binding_factor"].map({
        "covered": "Goal is met under the selected chance-constraint/coverage rule.",
        "supply_shortage": "No meaningful capacity was assigned; supply or compatibility is binding.",
        "target_level": "Assigned capacity is below target; more blocks or a softer target would be needed.",
        "case_volume": "Very low allocation relative to target; demand/volume signal may be too weak or target too high.",
        "near_target_or_granularity": "Close to target; block granularity or epsilon constraints likely explain remaining gap.",
    })
    return cov, source


# -----------------------------------------------------------------------------
# Goal attribution, counterfactuals, confidence bands, risks
# -----------------------------------------------------------------------------

def provider_goal_attribution(cov: pd.DataFrame, schedule: pd.DataFrame, selected_theme: str) -> pd.DataFrame:
    if cov.empty:
        return pd.DataFrame()
    needed = ["excess_min", "shortage_min", "coverage_ratio", "coverage_probability", "goal_met", "binding_factor"]
    for c in needed:
        if c not in cov.columns:
            if c in ["excess_min", "shortage_min", "coverage_ratio", "coverage_probability"]:
                cov[c] = 0.0
            elif c == "goal_met":
                cov[c] = False
            else:
                cov[c] = "unknown"

    provider = cov.groupby("provider_id", dropna=False).agg(
        target_total_min=("target_total_min", "sum"),
        allocated_total_min=("allocated_total_min", "sum"),
        shortage_min=("shortage_min", "sum"),
        excess_min=("excess_min", "sum"),
        mean_coverage_ratio=("coverage_ratio", "mean"),
        min_coverage_ratio=("coverage_ratio", "min"),
        mean_coverage_probability=("coverage_probability", "mean"),
        n_provider_days=("day_of_week", "nunique"),
        n_goal_days_met=("goal_met", "sum"),
    ).reset_index()
    provider["theme"] = selected_theme
    provider["provider_goal_met"] = provider["shortage_min"] <= 1e-6
    provider["gap_hours"] = provider["shortage_min"] / 60.0
    provider["excess_hours"] = provider["excess_min"] / 60.0
    provider["allocated_hours"] = provider["allocated_total_min"] / 60.0
    provider["target_hours"] = provider["target_total_min"] / 60.0

    bf = (
        cov.groupby(["provider_id", "binding_factor"], dropna=False).size().reset_index(name="n")
        .sort_values(["provider_id", "n"], ascending=[True, False])
        .drop_duplicates("provider_id")
        .rename(columns={"binding_factor": "dominant_binding_factor"})[["provider_id", "dominant_binding_factor"]]
    )
    provider = provider.merge(bf, on="provider_id", how="left")

    # Schedule fragmentation stats.
    frag = schedule_fragmentation(schedule)
    if not frag.empty:
        provider = provider.merge(frag, on="provider_id", how="left")
    return provider.sort_values(["shortage_min", "target_total_min"], ascending=[False, False])


def schedule_fragmentation(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame()
    provider_col = find_col(schedule, ["provider_id", "assigned_provider_id", "blockholder", "current_blockholder"], False)
    if provider_col is None:
        return pd.DataFrame()
    s = schedule.copy()
    s["provider_id"] = normalise_id_series(s[provider_col])
    s = s[(s["provider_id"] != "") & (s["provider_id"].str.upper() != "OPEN")]
    if s.empty:
        return pd.DataFrame()
    day_col = find_col(s, ["day_of_week", "dow", "weekday", "day"], False)
    room_col = find_col(s, ["room", "room_id", "or_room"], False)
    week_col = find_col(s, ["optimization_week", "week", "week_index"], False)
    site_col = find_col(s, ["site", "site_id"], False)
    dur_col = find_col(s, ["allocated_min", "duration_min", "slot_capacity_min", "capacity_min", "block_minutes"], False)
    if dur_col:
        s["duration_min"] = pd.to_numeric(s[dur_col], errors="coerce").fillna(480.0)
    else:
        s["duration_min"] = 480.0
    agg_spec: Dict[str, Tuple[str, str]] = {
        "n_assigned_slots": ("duration_min", "size"),
        "assigned_minutes": ("duration_min", "sum"),
    }
    if day_col:
        agg_spec["n_distinct_days"] = (day_col, "nunique")
    if room_col:
        agg_spec["n_distinct_rooms"] = (room_col, "nunique")
    if week_col:
        agg_spec["n_distinct_weeks"] = (week_col, "nunique")
    if site_col:
        agg_spec["n_distinct_sites"] = (site_col, "nunique")
    frag = s.groupby("provider_id", dropna=False).agg(**agg_spec).reset_index()
    frag["assigned_hours"] = frag["assigned_minutes"] / 60.0
    frag["fragmentation_index"] = (
        frag.get("n_distinct_days", 1).fillna(1) + frag.get("n_distinct_rooms", 1).fillna(1) + frag.get("n_distinct_sites", 1).fillna(1)
    ) / np.maximum(frag["n_assigned_slots"], 1)
    return frag


def build_counterfactuals(cov: pd.DataFrame, scenarios: pd.DataFrame, cfg: Layer3Config) -> pd.DataFrame:
    if cov.empty or scenarios.empty:
        return pd.DataFrame()
    sc = normalize_scenarios(scenarios)
    if sc.empty:
        return pd.DataFrame()
    alloc = cov.groupby("provider_id", dropna=False).agg(allocated_total_min=("allocated_total_min", "sum")).reset_index()
    demand = sc.groupby(["provider_id", "scenario_id"], dropna=False).agg(
        demand_min=("total_demand_min", "sum")
    ).reset_index()
    demand["demand_q_min"] = demand["demand_min"] * float(cfg.optimization_weeks)
    merged = demand.merge(alloc, on="provider_id", how="left")
    merged["allocated_total_min"] = merged["allocated_total_min"].fillna(0.0)
    targets = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    rows: List[Dict[str, Any]] = []
    for provider, g in merged.groupby("provider_id", dropna=False):
        allocated = float(g["allocated_total_min"].iloc[0])
        for target_util in targets:
            req = g["demand_q_min"] / max(target_util, 1e-6)
            coverage = float((allocated >= req).mean())
            min_blocks = int(math.ceil(float(np.quantile(req, cfg.alpha)) / max(cfg.block_minutes, 1.0)))
            rows.append({
                "provider_id": provider,
                "target_utilization": target_util,
                "allocated_min": allocated,
                "allocated_hours": allocated / 60.0,
                "required_min_p50": float(np.quantile(req, 0.50)),
                "required_min_p85": float(np.quantile(req, cfg.alpha)),
                "required_hours_p50": float(np.quantile(req, 0.50) / 60.0),
                "required_hours_p85": float(np.quantile(req, cfg.alpha) / 60.0),
                "coverage_at_current_allocation": coverage,
                "min_blocks_needed_for_alpha": min_blocks,
                "min_hours_needed_for_alpha": min_blocks * cfg.block_minutes / 60.0,
                "meets_alpha": coverage >= cfg.alpha,
            })
    return pd.DataFrame(rows)


def normalize_scenarios(scenarios: pd.DataFrame) -> pd.DataFrame:
    if scenarios.empty:
        return pd.DataFrame()
    sc = scenarios.copy()
    pcol = find_col(sc, ["provider_id", "provider"], True, "provider_id")
    dcol = find_col(sc, ["day_of_week", "dow", "weekday", "day"], False)
    scol = find_col(sc, ["scenario_id", "scenario", "s"], False)
    ccol = find_col(sc, ["casetime_min", "demand_casetime_min", "case_min", "casetime"], False)
    tcol = find_col(sc, ["turnover_min", "demand_turnover_min", "turn_min", "turnover"], False)
    total_col = find_col(sc, ["total_demand_min", "demand_total_min", "demand_min"], False)
    sc["provider_id"] = normalise_id_series(sc[pcol])
    sc["day_of_week"] = pd.to_numeric(sc[dcol], errors="coerce").fillna(-1).astype(int) if dcol else -1
    sc["scenario_id"] = pd.to_numeric(sc[scol], errors="coerce").fillna(0).astype(int) if scol else 0
    if total_col:
        sc["total_demand_min"] = pd.to_numeric(sc[total_col], errors="coerce").fillna(0.0)
    else:
        sc["casetime_min"] = pd.to_numeric(sc[ccol], errors="coerce").fillna(0.0) if ccol else 0.0
        sc["turnover_min"] = pd.to_numeric(sc[tcol], errors="coerce").fillna(0.0) if tcol else 0.0
        sc["total_demand_min"] = sc["casetime_min"] + sc["turnover_min"]
    return sc[["provider_id", "day_of_week", "scenario_id", "total_demand_min"]]


def infer_scenario_horizon_multiplier(demand: pd.DataFrame, cov: pd.DataFrame, cfg: Layer3Config) -> Tuple[float, Dict[str, float]]:
    """Infer whether scenarios are already in the same horizon as Layer 2 coverage.

    Across code versions, Layer 1 scenarios have sometimes been saved as weekly
    demand and sometimes already scaled to the optimization horizon. Multiplying
    blindly by optimization_weeks can create fake p90 utilization around 3-4x.
    We choose the multiplier whose alpha-quantile total is closest to Layer 2's
    target total, because the optimizer also used alpha-SAA demand.
    """
    meta = {
        "scenario_horizon_multiplier": 1.0,
        "scenario_alpha_total_raw_min": 0.0,
        "coverage_target_total_min": 0.0,
        "coverage_allocated_total_min": 0.0,
    }
    if demand.empty or cov.empty or "demand_min" not in demand.columns:
        return 1.0, meta

    total_by_s = demand.groupby("scenario_id", dropna=False)["demand_min"].sum().to_numpy(dtype=float)
    total_by_s = total_by_s[np.isfinite(total_by_s)]
    if total_by_s.size == 0:
        return 1.0, meta

    scenario_alpha_total = float(np.quantile(total_by_s, float(cfg.alpha)))
    target_total = float(pd.to_numeric(cov.get("target_total_min", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    allocated_total = float(pd.to_numeric(cov.get("allocated_total_min", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    reference_total = target_total if target_total > 0 else allocated_total

    meta.update({
        "scenario_alpha_total_raw_min": scenario_alpha_total,
        "coverage_target_total_min": target_total,
        "coverage_allocated_total_min": allocated_total,
    })
    if reference_total <= 0 or scenario_alpha_total <= 0:
        return 1.0, meta

    candidates = [1.0]
    weeks = float(cfg.optimization_weeks)
    if weeks > 0 and abs(weeks - 1.0) > 1e-9:
        candidates.append(weeks)

    # Also allow a learned continuous factor as a last-resort diagnostic, but only
    # use it if both canonical choices are very far from the Layer 2 horizon.
    continuous = reference_total / scenario_alpha_total
    if np.isfinite(continuous) and continuous > 0:
        candidates.append(float(continuous))

    def rel_err(mult: float) -> float:
        return abs((scenario_alpha_total * mult) - reference_total) / max(reference_total, 1.0)

    canonical = min(candidates[:2], key=rel_err)
    best = min(candidates, key=rel_err)
    # Prefer clear, interpretable 1x or T* scaling unless both are badly off.
    chosen = canonical if rel_err(canonical) <= 0.20 else best
    meta["scenario_horizon_multiplier"] = float(chosen)
    meta["scenario_alpha_total_scaled_min"] = float(scenario_alpha_total * chosen)
    meta["scenario_horizon_relative_error"] = float(rel_err(chosen))
    return float(chosen), meta


def confidence_bands(cov: pd.DataFrame, scenarios: pd.DataFrame, cfg: Layer3Config) -> pd.DataFrame:
    if cov.empty or scenarios.empty:
        return pd.DataFrame()
    sc = normalize_scenarios(scenarios)
    if sc.empty:
        return pd.DataFrame()
    demand = sc.groupby(["provider_id", "scenario_id"], dropna=False).agg(
        demand_min=("total_demand_min", "sum")
    ).reset_index()

    horizon_multiplier, horizon_meta = infer_scenario_horizon_multiplier(demand, cov, cfg)
    LOG.info(
        "Confidence-band horizon scaling: multiplier=%.4g raw_alpha_total=%.0f scaled_alpha_total=%.0f coverage_target=%.0f rel_error=%.4g",
        horizon_multiplier,
        horizon_meta.get("scenario_alpha_total_raw_min", 0.0),
        horizon_meta.get("scenario_alpha_total_scaled_min", 0.0),
        horizon_meta.get("coverage_target_total_min", 0.0),
        horizon_meta.get("scenario_horizon_relative_error", 0.0),
    )

    demand["demand_q_min"] = demand["demand_min"] * horizon_multiplier
    alloc = cov.groupby("provider_id", dropna=False).agg(
        allocated_total_min=("allocated_total_min", "sum"),
        target_total_min=("target_total_min", "sum"),
    ).reset_index()
    df = demand.merge(alloc, on="provider_id", how="left")
    df["allocated_total_min"] = df["allocated_total_min"].fillna(0.0)
    df["projected_utilization"] = np.where(
        df["allocated_total_min"] > 0,
        df["demand_q_min"] / df["allocated_total_min"],
        np.nan,
    )
    rows = []
    for provider, g in df.groupby("provider_id", dropna=False):
        vals = g["projected_utilization"].replace([np.inf, -np.inf], np.nan).dropna().values
        alloc_min = float(g["allocated_total_min"].iloc[0]) if len(g) else 0.0
        target_min = float(g["target_total_min"].iloc[0]) if len(g) and "target_total_min" in g.columns else 0.0
        if len(vals) == 0:
            rows.append({
                "provider_id": provider, "allocated_min": alloc_min, "target_min": target_min,
                "p10_util": np.nan, "p25_util": np.nan, "p50_util": np.nan, "p75_util": np.nan, "p90_util": np.nan,
                **horizon_meta,
            })
            continue
        q = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])
        rows.append({
            "provider_id": provider,
            "allocated_min": alloc_min,
            "target_min": target_min,
            "allocated_hours": alloc_min / 60.0,
            "target_hours": target_min / 60.0,
            "p10_util": float(q[0]),
            "p25_util": float(q[1]),
            "p50_util": float(q[2]),
            "p75_util": float(q[3]),
            "p90_util": float(q[4]),
            "band_width_pp": float((q[4] - q[0]) * 100.0),
            "underuse_risk_flag": bool(q[2] < 0.50),
            "capacity_risk_flag": bool(q[4] > 1.00),
            **horizon_meta,
        })
    return pd.DataFrame(rows).sort_values("band_width_pp", ascending=False, na_position="last")


def provider_risk_profiles(goal_attr: pd.DataFrame, bands: pd.DataFrame, sigma: pd.DataFrame, shap_reason: pd.DataFrame) -> pd.DataFrame:
    providers = set()
    for df in [goal_attr, bands, sigma, shap_reason]:
        if df is not None and not df.empty and "provider_id" in df.columns:
            providers.update(df["provider_id"].astype(str).tolist())
    if not providers:
        return pd.DataFrame()
    risk = pd.DataFrame({"provider_id": sorted(providers)})
    if not goal_attr.empty:
        cols = [c for c in ["provider_id", "dominant_binding_factor", "shortage_min", "allocated_hours", "target_hours", "provider_goal_met"] if c in goal_attr.columns]
        risk = risk.merge(goal_attr[cols], on="provider_id", how="left")
    if not bands.empty:
        cols = [c for c in ["provider_id", "p50_util", "p90_util", "band_width_pp", "underuse_risk_flag", "capacity_risk_flag"] if c in bands.columns]
        risk = risk.merge(bands[cols], on="provider_id", how="left")
    sig = normalize_sigma(sigma)
    if not sig.empty:
        risk = risk.merge(sig, on="provider_id", how="left")
    if not shap_reason.empty:
        sr = shap_reason.sort_values(["provider_id", "total_abs_shap"], ascending=[True, False]).drop_duplicates("provider_id")
        cols = [c for c in ["provider_id", "reason_code", "total_abs_shap", "mean_signed_shap"] if c in sr.columns]
        risk = risk.merge(sr[cols].rename(columns={"reason_code": "dominant_shap_reason"}), on="provider_id", how="left")

    if "band_width_pp" not in risk.columns:
        risk["band_width_pp"] = np.nan
    if "sigma_case_mean" not in risk.columns:
        risk["sigma_case_mean"] = np.nan

    has_sigma = bool(risk["sigma_case_mean"].notna().any())
    sigma_cut = float(np.nanmedian(risk["sigma_case_mean"].values)) if has_sigma else np.nan
    # Use the observed distribution of uncertainty bands; this avoids labeling every
    # provider high-risk when all providers have some uncertainty.
    band_cut = float(np.nanquantile(risk["band_width_pp"].dropna().values, 0.75)) if risk["band_width_pp"].notna().any() else 20.0
    high_sigma = (risk["sigma_case_mean"] >= sigma_cut).fillna(False) if has_sigma else pd.Series(False, index=risk.index)
    high_band = (risk["band_width_pp"] >= band_cut).fillna(False)
    risk["risk_quadrant"] = np.select(
        [~high_sigma & ~high_band, ~high_sigma & high_band, high_sigma & ~high_band, high_sigma & high_band],
        ["predictable_performer", "variable_performer", "uncertain_but_stable", "high_attention_needed"],
        default="unknown",
    )
    risk["sigma_available"] = has_sigma
    risk["risk_band_cut_pp"] = band_cut
    risk["risk_sigma_cut_min"] = sigma_cut
    capacity_flag = risk["capacity_risk_flag"].fillna(False).astype(bool) if "capacity_risk_flag" in risk.columns else pd.Series(False, index=risk.index)
    underuse_flag = risk["underuse_risk_flag"].fillna(False).astype(bool) if "underuse_risk_flag" in risk.columns else pd.Series(False, index=risk.index)
    risk["monitoring_priority"] = np.select(
        [risk["risk_quadrant"].eq("high_attention_needed"), capacity_flag, underuse_flag],
        ["review_first", "capacity_watch", "underuse_watch"],
        default="standard_review",
    )
    return risk


def normalize_sigma(sigma: pd.DataFrame) -> pd.DataFrame:
    if sigma is None or sigma.empty:
        return pd.DataFrame()
    s = sigma.copy()
    pcol = find_col(s, ["provider_id", "provider"], False)
    if pcol is None:
        return pd.DataFrame()
    s["provider_id"] = normalise_id_series(s[pcol])
    case_col = find_col(s, ["sigma_case", "sigma_casetime", "sigma_casetime_min", "sigma_pd_casetime", "case_sigma"], False)
    turn_col = find_col(s, ["sigma_turn", "sigma_turnover", "sigma_turnover_min", "sigma_pd_turnover", "turn_sigma"], False)
    agg: Dict[str, Tuple[str, str]] = {}
    if case_col:
        s["sigma_case_mean"] = pd.to_numeric(s[case_col], errors="coerce")
        agg["sigma_case_mean"] = ("sigma_case_mean", "mean")
    if turn_col:
        s["sigma_turnover_mean"] = pd.to_numeric(s[turn_col], errors="coerce")
        agg["sigma_turnover_mean"] = ("sigma_turnover_mean", "mean")
    if not agg:
        return pd.DataFrame()
    return s.groupby("provider_id", dropna=False).agg(**agg).reset_index()


def load_combined_shap_reason(out_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((out_dir / "shap").glob("shap_provider_reason_codes_*.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            pass
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# -----------------------------------------------------------------------------
# Pareto, drift, recommendations
# -----------------------------------------------------------------------------

def load_pareto_candidates(layer2_dir: str | Path) -> pd.DataFrame:
    l2 = Path(layer2_dir)
    patterns = [
        "**/pareto_frontier.csv",
        "**/frontier_summary.csv",
        "**/pareto_summary.csv",
        "**/pareto_candidates.csv",
        "**/frontier*.csv",
    ]
    frames = []
    for pat in patterns:
        for path in sorted(l2.glob(pat)):
            try:
                df = pd.read_csv(path)
                if len(df):
                    df["source_file"] = str(path)
                    frames.append(df)
            except Exception:
                pass
    if frames:
        return pd.concat(frames, ignore_index=True).drop_duplicates()

    # JSON fallback.
    rows = []
    for path in sorted(l2.glob("**/frontier*.json")) + sorted(l2.glob("**/pareto*.json")):
        obj = read_json(path, default=None)
        if obj is None:
            continue
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    item = dict(item)
                    item["source_file"] = str(path)
                    rows.append(item)
        elif isinstance(obj, dict):
            cand = obj.get("candidates") or obj.get("solutions") or obj.get("frontier")
            if isinstance(cand, list):
                for item in cand:
                    if isinstance(item, dict):
                        item = dict(item)
                        item["source_file"] = str(path)
                        rows.append(item)
            else:
                obj = dict(obj)
                obj["source_file"] = str(path)
                rows.append(obj)
    return pd.DataFrame(rows)


def pareto_diagnostics(layer2_dir: str | Path, out_dir: Path) -> Dict[str, Any]:
    df = load_pareto_candidates(layer2_dir)
    if df.empty:
        res = {"n_candidates": 0, "message": "No Pareto/frontier candidate file found."}
        write_json(res, out_dir / "frontier_diagnostics.json")
        return res

    # Normalize common objective names.
    colmap = {}
    for canonical, aliases in {
        "utilization_score": ["utilization_score", "useful_utilization", "target_coverage", "utilization", "O1_utilization"],
        "continuity_score": ["continuity_score", "continuity", "O2_continuity"],
        "preference_score": ["preference_score", "preference", "O3_preference"],
        "stability_score": ["stability_score", "stability", "O4_stability"],
        "overall_score": ["overall_score", "score", "objective", "weighted_score"],
    }.items():
        c = find_col(df, aliases, False)
        if c:
            colmap[c] = canonical
    d = df.rename(columns=colmap).copy()
    metric_cols = [c for c in ["utilization_score", "continuity_score", "preference_score", "stability_score", "overall_score"] if c in d.columns]
    for c in metric_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Non-dominated in available metrics, assuming higher is better.
    nd = np.ones(len(d), dtype=bool)
    use_cols = [c for c in ["utilization_score", "continuity_score", "preference_score", "stability_score"] if c in d.columns]
    if use_cols and len(d) > 1:
        vals = d[use_cols].fillna(-np.inf).to_numpy()
        for i in range(len(vals)):
            dominated = np.any(np.all(vals >= vals[i], axis=1) & np.any(vals > vals[i], axis=1))
            nd[i] = not dominated
    d["is_nondominated"] = nd
    write_csv(d, out_dir / "pareto_candidates_normalized.csv")

    if plt is not None and len(d) > 0:
        pdir = ensure_dir(out_dir / "plots" / "pareto")
        try:
            if "utilization_score" in d.columns and "continuity_score" in d.columns:
                plt.figure(figsize=(7, 5))
                plt.scatter(d["utilization_score"], d["continuity_score"], s=35)
                plt.xlabel("Utilization score")
                plt.ylabel("Continuity score")
                plt.title("Pareto frontier: utilization vs continuity")
                plt.tight_layout()
                plt.savefig(pdir / "pareto_utilization_vs_continuity.png", dpi=180, bbox_inches="tight")
                plt.close()
            if "utilization_score" in d.columns and "preference_score" in d.columns:
                plt.figure(figsize=(7, 5))
                plt.scatter(d["utilization_score"], d["preference_score"], s=35)
                plt.xlabel("Utilization score")
                plt.ylabel("Preference score")
                plt.title("Pareto frontier: utilization vs preference")
                plt.tight_layout()
                plt.savefig(pdir / "pareto_utilization_vs_preference.png", dpi=180, bbox_inches="tight")
                plt.close()
            if "utilization_score" in d.columns and "stability_score" in d.columns:
                plt.figure(figsize=(7, 5))
                plt.scatter(d["utilization_score"], d["stability_score"], s=35)
                plt.xlabel("Utilization score")
                plt.ylabel("Stability score")
                plt.title("Pareto frontier: utilization vs stability")
                plt.tight_layout()
                plt.savefig(pdir / "pareto_utilization_vs_stability.png", dpi=180, bbox_inches="tight")
                plt.close()
        except Exception as e:
            LOG.warning("Could not create Pareto plots: %s", e)

    best = {}
    if "overall_score" in d.columns and d["overall_score"].notna().any():
        best = d.loc[d["overall_score"].idxmax()].to_dict()
    elif "utilization_score" in d.columns and d["utilization_score"].notna().any():
        best = d.loc[d["utilization_score"].idxmax()].to_dict()

    res = {
        "n_candidates": int(len(d)),
        "n_nondominated": int(d["is_nondominated"].sum()),
        "metric_columns": metric_cols,
        "best_candidate": best,
        "source_files": sorted(set(d.get("source_file", pd.Series(dtype=str)).dropna().astype(str).tolist())),
    }
    write_json(res, out_dir / "frontier_diagnostics.json")
    return res


def drift_detection(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or "provider_id" not in features.columns:
        return pd.DataFrame()
    f = features.copy()
    f["provider_id"] = normalise_id_series(f["provider_id"])
    candidates = [
        ("trailing_4w_mean_case", "trailing_12w_mean_case", "mean_case"),
        ("trailing_4w_std_case", "trailing_12w_std_case", "std_case"),
        ("trailing_4w_mean_turn", "trailing_12w_mean_turn", "mean_turn"),
        ("trailing_4w_std_turn", "trailing_12w_std_turn", "std_turn"),
    ]
    rows = []
    for provider, g in f.groupby("provider_id", dropna=False):
        row: Dict[str, Any] = {"provider_id": provider}
        max_ratio = 0.0
        flags = []
        for short_col, long_col, name in candidates:
            if short_col in g.columns and long_col in g.columns:
                a = float(pd.to_numeric(g[short_col], errors="coerce").dropna().tail(1).mean()) if pd.to_numeric(g[short_col], errors="coerce").notna().any() else np.nan
                b = float(pd.to_numeric(g[long_col], errors="coerce").dropna().tail(1).mean()) if pd.to_numeric(g[long_col], errors="coerce").notna().any() else np.nan
                ratio = abs(a - b) / max(abs(b), 1.0) if np.isfinite(a) and np.isfinite(b) else np.nan
                row[f"drift_ratio_{name}"] = ratio
                if np.isfinite(ratio):
                    max_ratio = max(max_ratio, float(ratio))
                    if ratio >= 0.50:
                        flags.append(name)
        row["max_drift_ratio"] = max_ratio
        row["drift_flag"] = bool(max_ratio >= 0.50)
        row["drift_features"] = ";".join(flags)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("max_drift_ratio", ascending=False)


def make_recommendations(goal_attr: pd.DataFrame, bands: pd.DataFrame, risk: pd.DataFrame, shap_reason: pd.DataFrame, cfg: Layer3Config) -> pd.DataFrame:
    providers = set()
    for df in [goal_attr, bands, risk, shap_reason]:
        if df is not None and not df.empty and "provider_id" in df.columns:
            providers.update(df["provider_id"].astype(str).tolist())
    rows = []
    for p in sorted(providers):
        ga = goal_attr[goal_attr["provider_id"].astype(str) == p].iloc[0].to_dict() if not goal_attr.empty and (goal_attr["provider_id"].astype(str) == p).any() else {}
        rb = risk[risk["provider_id"].astype(str) == p].iloc[0].to_dict() if not risk.empty and (risk["provider_id"].astype(str) == p).any() else {}
        band = bands[bands["provider_id"].astype(str) == p].iloc[0].to_dict() if not bands.empty and (bands["provider_id"].astype(str) == p).any() else {}
        sr = shap_reason[shap_reason["provider_id"].astype(str) == p].sort_values("total_abs_shap", ascending=False).head(1).to_dict(orient="records") if not shap_reason.empty and (shap_reason["provider_id"].astype(str) == p).any() else []
        dominant_reason = sr[0].get("reason_code") if sr else rb.get("dominant_shap_reason", "unknown")

        shortage = float(ga.get("shortage_min", 0.0) or 0.0)
        excess = float(ga.get("excess_min", 0.0) or 0.0)
        quadrant = str(rb.get("risk_quadrant", "unknown"))
        p50 = band.get("p50_util", np.nan)
        p90 = band.get("p90_util", np.nan)

        if shortage > cfg.block_minutes * 0.5:
            action = "add_or_reassign_capacity"
            priority = "high"
            explanation = f"Provider is short by {shortage/60:.1f} hours versus target. Dominant model reason: {dominant_reason}."
        elif bool(band.get("capacity_risk_flag", False)) or (isinstance(p90, float) and np.isfinite(p90) and p90 > 1.0):
            action = "capacity_watch"
            priority = "medium"
            explanation = f"P90 utilization exceeds capacity; monitor overflow risk. Dominant model reason: {dominant_reason}."
        elif bool(band.get("underuse_risk_flag", False)) or (isinstance(p50, float) and np.isfinite(p50) and p50 < 0.50):
            action = "underuse_review"
            priority = "medium"
            explanation = f"Median projected utilization is below 50%; consider future softening/reclaim. Dominant model reason: {dominant_reason}."
        elif quadrant == "high_attention_needed":
            action = "committee_review"
            priority = "medium"
            explanation = f"Both demand and outcome uncertainty are high. Dominant model reason: {dominant_reason}."
        elif excess > cfg.block_minutes:
            action = "possible_granularity_excess"
            priority = "low"
            explanation = f"Allocation exceeds modeled target by {excess/60:.1f} hours; may be block granularity."
        else:
            action = "no_action_standard_monitoring"
            priority = "low"
            explanation = f"Allocation appears aligned. Dominant model reason: {dominant_reason}."

        rows.append({
            "provider_id": p,
            "priority": priority,
            "recommended_action": action,
            "explanation": explanation,
            "dominant_shap_reason": dominant_reason,
            "risk_quadrant": quadrant,
            "shortage_hours": shortage / 60.0,
            "excess_hours": excess / 60.0,
            "p50_util": p50,
            "p90_util": p90,
        })
    return pd.DataFrame(rows).sort_values(["priority", "shortage_hours"], ascending=[True, False]).replace({np.nan: None})


# -----------------------------------------------------------------------------
# Plots and summary
# -----------------------------------------------------------------------------

def plot_goal_and_risk(goal_attr: pd.DataFrame, bands: pd.DataFrame, risk: pd.DataFrame, out_dir: Path) -> None:
    if plt is None:
        return
    pdir = ensure_dir(out_dir / "plots")
    if not goal_attr.empty:
        try:
            top = goal_attr.sort_values("shortage_min", ascending=False).head(25)
            plt.figure(figsize=(10, 6))
            plt.barh(top["provider_id"].astype(str), top["shortage_min"] / 60.0)
            plt.xlabel("Shortage hours")
            plt.ylabel("Provider")
            plt.title("Top provider allocation shortages")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(pdir / "top_provider_shortages.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as e:
            LOG.warning("Could not plot shortages: %s", e)
    if not bands.empty:
        try:
            top = bands.sort_values("band_width_pp", ascending=False).head(25)
            plt.figure(figsize=(10, 6))
            plt.barh(top["provider_id"].astype(str), top["band_width_pp"])
            plt.xlabel("P90-P10 utilization band width, percentage points")
            plt.ylabel("Provider")
            plt.title("Providers with widest utilization uncertainty bands")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(pdir / "widest_confidence_bands.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as e:
            LOG.warning("Could not plot bands: %s", e)
    if not risk.empty and "risk_quadrant" in risk.columns:
        try:
            counts = risk["risk_quadrant"].value_counts()
            plt.figure(figsize=(8, 5))
            plt.bar(counts.index.astype(str), counts.values)
            plt.xticks(rotation=25, ha="right")
            plt.ylabel("Providers")
            plt.title("Provider risk profile counts")
            plt.tight_layout()
            plt.savefig(pdir / "provider_risk_profile_counts.png", dpi=180, bbox_inches="tight")
            plt.close()
        except Exception as e:
            LOG.warning("Could not plot risk counts: %s", e)


def write_summary(
    out_dir: Path,
    cfg: Layer3Config,
    validation: List[Dict[str, Any]],
    shap_diag: Dict[str, Any],
    goal_attr: pd.DataFrame,
    bands: pd.DataFrame,
    risk: pd.DataFrame,
    recommendations: pd.DataFrame,
    pareto: Dict[str, Any],
) -> None:
    lines = []
    lines.append("# Layer 3 Explainability Summary — TRUE SHAP edition")
    lines.append("")
    lines.append(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Selected theme: `{cfg.selected_theme}`")
    lines.append(f"Alpha: `{cfg.alpha}`")
    lines.append(f"Optimization weeks: `{cfg.optimization_weeks}`")
    lines.append("")

    status = "PASS" if all(v["status"] == "PASS" for v in validation) else ("FAIL" if any(v["status"] == "FAIL" for v in validation) else "WARN")
    lines.append(f"## Validation: {status}")
    lines.append("")
    for v in validation:
        mark = "✓" if v["status"] == "PASS" else ("⚠" if v["status"] == "WARN" else "✗")
        lines.append(f"- {mark} **{v['name']}** — {v.get('details','')}")
    lines.append("")

    lines.append("## SHAP diagnostics")
    for target, diag in shap_diag.items():
        lines.append(f"### {target}")
        lines.append(f"- Model source: `{diag.get('model_source')}`")
        lines.append(f"- Rows explained: `{diag.get('n_rows_explained')}`")
        lines.append(f"- Training feature columns: `{diag.get('n_training_feature_columns')}`")
        lines.append(f"- Mean efficiency error: `{diag.get('mean_efficiency_error')}`")
        lines.append(f"- Max efficiency error: `{diag.get('max_efficiency_error')}`")
        top = diag.get("top_features") or []
        if top:
            lines.append("- Top features:")
            for r in top[:8]:
                lines.append(f"  - `{r.get('feature')}`: mean |SHAP|={r.get('mean_abs_shap')} ({r.get('reason_code')})")
    lines.append("")

    if not goal_attr.empty:
        total_target = goal_attr["target_total_min"].sum() / 60.0 if "target_total_min" in goal_attr.columns else 0.0
        total_alloc = goal_attr["allocated_total_min"].sum() / 60.0 if "allocated_total_min" in goal_attr.columns else 0.0
        met = int(goal_attr.get("provider_goal_met", pd.Series(dtype=bool)).fillna(False).sum()) if "provider_goal_met" in goal_attr.columns else 0
        lines.append("## Goal attribution")
        lines.append(f"- Providers: `{len(goal_attr)}`")
        lines.append(f"- Provider goals met: `{met}/{len(goal_attr)}`")
        lines.append(f"- Total target hours: `{total_target:.1f}`")
        lines.append(f"- Total allocated hours: `{total_alloc:.1f}`")
        lines.append("")

    if not bands.empty:
        lines.append("## Confidence bands")
        lines.append(f"- Providers with bands: `{len(bands)}`")
        if "capacity_risk_flag" in bands.columns:
            lines.append(f"- Capacity risk providers: `{int(bands['capacity_risk_flag'].fillna(False).sum())}`")
        if "underuse_risk_flag" in bands.columns:
            lines.append(f"- Underuse risk providers: `{int(bands['underuse_risk_flag'].fillna(False).sum())}`")
        lines.append("")

    if not risk.empty and "risk_quadrant" in risk.columns:
        lines.append("## Provider risk profiles")
        counts = risk["risk_quadrant"].value_counts().to_dict()
        for k, v in counts.items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")

    lines.append("## Pareto diagnostics")
    lines.append(f"- Candidates found: `{pareto.get('n_candidates', 0)}`")
    lines.append(f"- Non-dominated candidates: `{pareto.get('n_nondominated', 0)}`")
    lines.append("")

    if not recommendations.empty:
        lines.append("## Top recommendations")
        for _, r in recommendations.head(12).iterrows():
            lines.append(f"- **{r['provider_id']}** — `{r['recommended_action']}` ({r['priority']}): {r['explanation']}")
        lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def add_validation(rows: List[Dict[str, Any]], status: str, name: str, details: str = "") -> None:
    rows.append({"status": status, "name": name, "details": details})


def build_validation(
    art1: Dict[str, Any],
    art2: Dict[str, Any],
    cov: pd.DataFrame,
    shap_diag: Dict[str, Any],
    goal_attr: pd.DataFrame,
    bands: pd.DataFrame,
    pareto: Dict[str, Any],
    required_artifacts: Sequence[Path],
    cfg: Layer3Config,
) -> List[Dict[str, Any]]:
    v: List[Dict[str, Any]] = []
    add_validation(v, "PASS" if not art1["features"].empty else "FAIL", "Layer 1 features loaded", f"rows={len(art1['features'])}")
    add_validation(v, "PASS" if not art1["scenarios"].empty else "FAIL", "Layer 1 scenarios loaded", f"rows={len(art1['scenarios'])}")
    add_validation(v, "PASS" if len(art1.get("models", {})) >= 2 else "FAIL", "Layer 1 target models loaded", f"models={list(art1.get('models', {}).keys())}")
    add_validation(v, "PASS" if not art2["schedule"].empty else "WARN", "Layer 2 schedule loaded", f"rows={len(art2['schedule'])}")
    add_validation(v, "PASS" if not cov.empty else "WARN", "Layer 2 coverage loaded", f"rows={len(cov)}")

    for target in ["casetime_min", "turnover_min"]:
        diag = shap_diag.get(target)
        if not diag:
            add_validation(v, "FAIL", f"True SHAP produced for {target}", "missing diagnostics")
            continue
        source = str(diag.get("model_source"))
        ok_source = source == "shap_tree_explainer_pipeline"
        add_validation(v, "PASS" if ok_source else "FAIL", f"True SHAP source for {target}", f"model_source={source}")
        max_err = diag.get("max_efficiency_error")
        if max_err is None:
            add_validation(v, "FAIL", f"SHAP efficiency for {target}", "not available")
        else:
            # TreeExplainer should reconstruct the transformed model prediction up
            # to floating-point noise. Use both absolute minutes and relative error
            # so large-minute predictions do not produce false WARNs.
            max_err_f = float(max_err)
            max_pred = float(diag.get("max_abs_prediction", 0.0) or 0.0)
            rel_err = max_err_f / max(max_pred, 1.0)
            pass_ok = (max_err_f <= cfg.shap_efficiency_abs_tol_min) or (rel_err <= cfg.shap_efficiency_rel_tol)
            status = "PASS" if pass_ok else ("WARN" if max_err_f <= cfg.shap_efficiency_warn_tol_min else "FAIL")
            add_validation(
                v,
                status,
                f"SHAP efficiency for {target}",
                f"max_error={max_err_f:.6g} min ({max_err_f*60:.3g} sec), rel_error={rel_err:.3g}",
            )
        nfeat = int(diag.get("n_training_feature_columns", 0) or 0)
        add_validation(v, "PASS" if nfeat > 0 else "FAIL", f"Training feature columns locked for {target}", f"n={nfeat}")

    add_validation(v, "PASS" if not goal_attr.empty else "WARN", "Goal attribution produced", f"rows={len(goal_attr)}")
    add_validation(v, "PASS" if not bands.empty else "WARN", "Confidence bands produced", f"rows={len(bands)}")
    add_validation(v, "PASS" if int(pareto.get("n_candidates", 0)) > 0 else "WARN", "Pareto diagnostics produced", f"candidates={pareto.get('n_candidates', 0)}")

    for p in required_artifacts:
        add_validation(v, "PASS" if p.exists() else "FAIL", f"Artifact written: {p.name}", str(p))
    return v


def print_validation(validation: List[Dict[str, Any]]) -> str:
    print("\n" + "─" * 62)
    print("  LAYER 3 VALIDATION REPORT")
    print("─" * 62)
    for r in validation:
        mark = "✓ PASS" if r["status"] == "PASS" else ("⚠ WARN" if r["status"] == "WARN" else "✗ FAIL")
        print(f"  {mark:<8} {r['name']:<42} {r.get('details','')}")
    print("─" * 62)
    final = "PASS" if all(r["status"] == "PASS" for r in validation) else ("FAIL" if any(r["status"] == "FAIL" for r in validation) else "WARN")
    print(f"  Validation : {final}")
    print("─" * 62)
    return final


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------

def run(cfg: Layer3Config) -> Path:
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    run_id = _dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path(cfg.out) / run_id)
    ensure_dir(out_dir / "plots")

    LOG.info("Layer 3 output dir: %s", out_dir)
    LOG.info("Effective layer1_dir=%s", cfg.layer1_dir)
    LOG.info("Effective layer2_dir=%s", cfg.layer2_dir)
    LOG.info("Effective out_dir=%s", cfg.out)
    LOG.info("Effective selected_theme=%s", cfg.selected_theme)
    LOG.info("Effective alpha=%.3f optimization_weeks=%s block_minutes=%s", cfg.alpha, cfg.optimization_weeks, cfg.block_minutes)
    LOG.info("Effective TRUE SHAP max_rows=%s top_k=%s interactions=%s", cfg.max_shap_rows, cfg.top_k_features, cfg.make_interactions)
    LOG.info("Effective SHAP efficiency tolerances: abs<=%.4g min rel<=%.4g warn<=%.4g min", cfg.shap_efficiency_abs_tol_min, cfg.shap_efficiency_rel_tol, cfg.shap_efficiency_warn_tol_min)

    provenance = {
        "config": asdict(cfg),
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "cwd": os.getcwd(),
    }
    write_json(provenance, out_dir / "provenance.json")

    art1 = load_layer1_artifacts(cfg)
    art2 = load_layer2_artifacts(cfg)

    LOG.info("Stage 0 — Coverage loading/reconstruction")
    cov, coverage_source = normalize_coverage(art2["coverage"], art2["schedule"], art1.get("forecasts", pd.DataFrame()), cfg)
    cov["coverage_source"] = coverage_source if not cov.empty else "missing"
    write_csv(cov, out_dir / "coverage_by_provider_day.csv")

    LOG.info("Stage 1 — TRUE SHAP attribution")
    shap_diag = run_shap_attribution(art1, out_dir, cfg)
    shap_reason = load_combined_shap_reason(out_dir)

    LOG.info("Stage 2 — Goal attribution, fragmentation, goals met")
    goal_attr = provider_goal_attribution(cov, art2["schedule"], cfg.selected_theme)
    write_csv(goal_attr, out_dir / "goal_attribution_by_provider.csv")
    frag = schedule_fragmentation(art2["schedule"])
    write_csv(frag, out_dir / "schedule_fragmentation_by_provider.csv")

    goals_summary = pd.DataFrame([{
        "n_provider_day_rows": int(len(cov)),
        "n_providers": int(cov["provider_id"].nunique()) if not cov.empty and "provider_id" in cov.columns else 0,
        "provider_day_goals_met": int(cov["goal_met"].fillna(False).sum()) if not cov.empty and "goal_met" in cov.columns else 0,
        "provider_day_goals_total": int(len(cov)),
        "providers_all_days_met": int(goal_attr.get("provider_goal_met", pd.Series(dtype=bool)).fillna(False).sum()) if not goal_attr.empty and "provider_goal_met" in goal_attr.columns else 0,
        "providers_total": int(len(goal_attr)),
        "total_target_hours": float(cov["target_total_min"].sum() / 60.0) if not cov.empty and "target_total_min" in cov.columns else 0.0,
        "total_allocated_hours": float(cov["allocated_total_min"].sum() / 60.0) if not cov.empty and "allocated_total_min" in cov.columns else 0.0,
        "total_shortage_hours": float(cov["shortage_min"].sum() / 60.0) if not cov.empty and "shortage_min" in cov.columns else 0.0,
        "coverage_source": coverage_source,
    }])
    write_csv(goals_summary, out_dir / "goals_met_summary.csv")

    LOG.info("Stage 3 — Counterfactuals and confidence bands")
    counter = build_counterfactuals(cov, art1["scenarios"], cfg)
    write_csv(counter, out_dir / "counterfactual_targets.csv")
    bands = confidence_bands(cov, art1["scenarios"], cfg)
    write_csv(bands, out_dir / "confidence_bands.csv")

    LOG.info("Stage 4 — Provider risk profiles and drift")
    risk = provider_risk_profiles(goal_attr, bands, art1.get("sigma", pd.DataFrame()), shap_reason)
    write_csv(risk, out_dir / "provider_risk_profiles.csv")
    drift = drift_detection(art1["features"])
    write_csv(drift, out_dir / "temporal_drift_flags.csv")

    LOG.info("Stage 5 — Pareto diagnostics")
    pareto = pareto_diagnostics(cfg.layer2_dir, out_dir)

    LOG.info("Stage 6 — Recommendations and narratives")
    recommendations = make_recommendations(goal_attr, bands, risk, shap_reason, cfg)
    write_csv(recommendations, out_dir / "recommendations.csv")
    explanations = {
        "shap": shap_diag,
        "goals_summary": goals_summary.to_dict(orient="records"),
        "pareto": pareto,
        "top_recommendations": recommendations.head(30).to_dict(orient="records"),
    }
    write_json(explanations, out_dir / "explanations.json")
    plot_goal_and_risk(goal_attr, bands, risk, out_dir)

    required = [
        out_dir / "coverage_by_provider_day.csv",
        out_dir / "goal_attribution_by_provider.csv",
        out_dir / "goals_met_summary.csv",
        out_dir / "confidence_bands.csv",
        out_dir / "provider_risk_profiles.csv",
        out_dir / "frontier_diagnostics.json",
        out_dir / "recommendations.csv",
        out_dir / "explanations.json",
        out_dir / "provenance.json",
    ]
    validation = build_validation(art1, art2, cov, shap_diag, goal_attr, bands, pareto, required, cfg)
    write_json(validation, out_dir / "validation.json")
    final_status = print_validation(validation)
    write_summary(out_dir, cfg, validation, shap_diag, goal_attr, bands, risk, recommendations, pareto)

    print("\n" + "─" * 62)
    print("  Layer 3 complete")
    print(f"  Output dir : {out_dir}")
    print(f"  Summary    : {out_dir / 'SUMMARY.md'}")
    print(f"  Validation : {final_status}")
    print("─" * 62)
    return out_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> Layer3Config:
    ap = argparse.ArgumentParser(description="Layer 3 explainability with true Tree SHAP")
    ap.add_argument("--layer1_dir", default="outputs/layer1")
    ap.add_argument("--layer2_dir", default="outputs/layer2")
    ap.add_argument("--pre_layer_result", default="outputs/pre_layer/prelayer_result.json")
    ap.add_argument("--out", default="outputs/layer3")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--optimization_weeks", type=int, default=4)
    ap.add_argument("--block_minutes", type=float, default=480.0)
    ap.add_argument("--selected_theme", default="utilization_first")
    ap.add_argument("--top_k_features", type=int, default=12)
    ap.add_argument("--max_shap_rows", type=int, default=5000)
    ap.add_argument("--make_interactions", action="store_true")
    ap.add_argument("--max_interaction_rows", type=int, default=1000)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--shap_efficiency_abs_tol_min", type=float, default=1e-2)
    ap.add_argument("--shap_efficiency_rel_tol", type=float, default=1e-6)
    ap.add_argument("--shap_efficiency_warn_tol_min", type=float, default=1.0)
    ap.add_argument("--allow_permutation_fallback", action="store_true", help="Use explicit permutation fallback if true SHAP fails. Default is false.")
    ap.add_argument("--n_waterfall_examples", type=int, default=3)
    ap.add_argument("--n_dependence_plots", type=int, default=4)
    args = ap.parse_args(argv)
    return Layer3Config(
        layer1_dir=args.layer1_dir,
        layer2_dir=args.layer2_dir,
        pre_layer_result=args.pre_layer_result,
        out=args.out,
        alpha=args.alpha,
        optimization_weeks=args.optimization_weeks,
        block_minutes=args.block_minutes,
        selected_theme=args.selected_theme,
        top_k_features=args.top_k_features,
        max_shap_rows=args.max_shap_rows,
        make_interactions=args.make_interactions,
        max_interaction_rows=args.max_interaction_rows,
        random_seed=args.random_seed,
        shap_efficiency_abs_tol_min=args.shap_efficiency_abs_tol_min,
        shap_efficiency_rel_tol=args.shap_efficiency_rel_tol,
        shap_efficiency_warn_tol_min=args.shap_efficiency_warn_tol_min,
        allow_permutation_fallback=args.allow_permutation_fallback,
        require_true_shap=not args.allow_permutation_fallback,
        n_waterfall_examples=args.n_waterfall_examples,
        n_dependence_plots=args.n_dependence_plots,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    cfg = parse_args(argv)
    try:
        run(cfg)
        return 0
    except Exception as e:
        LOG.error("Layer 3 failed: %s", e)
        traceback.print_exc()
        print("\nLayer 3 failed.")
        print("Most common causes:")
        print("  1. SHAP missing: python3 -m pip install shap")
        print("  2. Layer 1 models missing: rerun Layer 1 with --point_model xgboost")
        print("  3. Feature mismatch: rerun Layer 1 so model_metrics.json and features.csv match the saved model")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
Real Hierarchical Bayesian residual model for Layer 1.

Place in: src/layer1/bayesian_residual_model.py

Outputs written to outputs/layer1:
- bayesian_training_residuals.csv
- posterior_casetime.nc / posterior_turnover.nc, or .pkl fallback
- posterior_sigma_draws_casetime.npz / posterior_sigma_draws_turnover.npz
- posterior_sigma_summary_casetime.csv / posterior_sigma_summary_turnover.csv
- posterior_parameter_summary_casetime.csv / posterior_parameter_summary_turnover.csv
- bayesian_diagnostics_*.json/csv
- bayesian_loss_metrics*.csv/json
- bayesian_calibration_metrics.csv
- hierarchical_sigma_estimates.csv
- scenarios_long.csv
- scenario_summary.csv
- quarterly_provider_scenarios.csv
- demand_scenarios.json
"""
from __future__ import annotations

import json
import math
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


TARGETS = {
    "casetime": {
        "actual": "casetime_min",
        "pred": "pred_casetime_min",
        "mu": "mu_casetime_min",
        "legacy_sigma": "sigma_case",
        "floor_min": 1.0,
    },
    "turnover": {
        "actual": "turnover_min",
        "pred": "pred_turnover_min",
        "mu": "mu_turnover_min",
        "legacy_sigma": "sigma_turn",
        "floor_min": 0.5,
    },
}


def _out(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_json(obj: dict, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _clean(x: Any) -> str:
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).strip()
    return s if s else "UNKNOWN"


def _pd_key(provider_id: Any, day_of_week: Any) -> str:
    try:
        d = int(day_of_week)
    except Exception:
        d = -1
    return f"{_clean(provider_id)}__D{d}"


def _require_pymc():
    try:
        import pymc as pm
        import arviz as az
        return pm, az
    except Exception as exc:
        raise ImportError(
            "PyMC and ArviZ are required for --sigma_model bayesian.\n"
            "Install inside the venv with:\n"
            "    pip install pymc arviz h5netcdf netcdf4\n"
            f"Original error: {exc}"
        ) from exc


def _posterior_matrix(idata, name: str) -> np.ndarray:
    arr = idata.posterior[name].stack(sample=("chain", "draw")).transpose("sample", ...).values
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _nll(resid: np.ndarray, sigma: np.ndarray) -> float:
    resid = np.asarray(resid, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + resid**2 / (2 * sigma**2)))


def _save_idata_robust(idata, az, path: Path):
    """Save InferenceData across old/new ArviZ/xarray APIs."""
    path = Path(path)
    try:
        if hasattr(idata, "to_netcdf"):
            idata.to_netcdf(path)
            return path
        if hasattr(az, "to_netcdf"):
            az.to_netcdf(idata, path)
            return path
        raise AttributeError("No available to_netcdf writer on idata or arviz")
    except Exception as exc:
        warnings.warn(f"Could not save NetCDF posterior: {exc}; saving pickle fallback")
        pkl_path = path.with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(idata, f)
        return pkl_path


def _summary_robust(idata, az, target_name: str, out_dir: Path) -> pd.DataFrame:
    """ArviZ changed summary kwargs/column names across versions; handle both."""
    var_names = ["sigma_site", "sigma_sl", "sigma_pd"]
    attempts = [
        {"ci_prob": 0.90},     # ArviZ 1.x / arviz-stats split
        {"hdi_prob": 0.90},    # classic ArviZ
        {},
    ]
    last_exc = None
    for kwargs in attempts:
        try:
            s = az.summary(idata, var_names=var_names, **kwargs)
            s = s.reset_index().rename(columns={"index": "parameter"})
            # Normalize diagnostic column names so run_layer1 can read them.
            rename = {}
            if "rhat" in s.columns and "r_hat" not in s.columns:
                rename["rhat"] = "r_hat"
            if "ess_bulk" not in s.columns:
                for c in s.columns:
                    if c.lower().replace("-", "_") in {"bulk_ess", "ess_bulk"}:
                        rename[c] = "ess_bulk"
            if "ess_tail" not in s.columns:
                for c in s.columns:
                    if c.lower().replace("-", "_") in {"tail_ess", "ess_tail"}:
                        rename[c] = "ess_tail"
            if rename:
                s = s.rename(columns=rename)
            return s
        except Exception as exc:
            last_exc = exc
    warnings.warn(f"ArviZ summary failed for {target_name}: {last_exc}")
    return pd.DataFrame()


def _metric_values(obj) -> np.ndarray:
    """Flatten xarray/numpy/pandas metric output to finite floats."""
    try:
        if hasattr(obj, "to_array"):
            arr = obj.to_array().values
        elif hasattr(obj, "values"):
            arr = obj.values
        else:
            arr = np.asarray(obj)
        arr = np.asarray(arr, dtype=float).ravel()
        return arr[np.isfinite(arr)]
    except Exception:
        return np.asarray([], dtype=float)


def _compute_sampling_diagnostics(idata, az, param_summary: pd.DataFrame) -> dict[str, Any]:
    """Get R-hat/ESS/divergences even when az.summary has different APIs."""
    var_names = ["sigma_site", "sigma_sl", "sigma_pd"]

    max_r_hat = None
    min_ess_bulk = None
    min_ess_tail = None

    if not param_summary.empty:
        if "r_hat" in param_summary.columns:
            vals = pd.to_numeric(param_summary["r_hat"], errors="coerce").dropna()
            if len(vals):
                max_r_hat = float(vals.max())
        if "ess_bulk" in param_summary.columns:
            vals = pd.to_numeric(param_summary["ess_bulk"], errors="coerce").dropna()
            if len(vals):
                min_ess_bulk = float(vals.min())
        if "ess_tail" in param_summary.columns:
            vals = pd.to_numeric(param_summary["ess_tail"], errors="coerce").dropna()
            if len(vals):
                min_ess_tail = float(vals.min())

    if max_r_hat is None:
        try:
            vals = _metric_values(az.rhat(idata, var_names=var_names))
            if vals.size:
                max_r_hat = float(vals.max())
        except Exception:
            pass

    if min_ess_bulk is None:
        try:
            vals = _metric_values(az.ess(idata, var_names=var_names, method="bulk"))
            if vals.size:
                min_ess_bulk = float(vals.min())
        except TypeError:
            try:
                vals = _metric_values(az.ess(idata, var_names=var_names))
                if vals.size:
                    min_ess_bulk = float(vals.min())
            except Exception:
                pass
        except Exception:
            pass

    if min_ess_tail is None:
        try:
            vals = _metric_values(az.ess(idata, var_names=var_names, method="tail"))
            if vals.size:
                min_ess_tail = float(vals.min())
        except Exception:
            pass

    divergences = None
    try:
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    except Exception:
        pass

    return {
        "max_r_hat": max_r_hat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "divergences": divergences,
    }


def _forecast_keys(point_forecasts: pd.DataFrame) -> pd.DataFrame:
    need = ["provider_id", "service_line", "day_of_week"]
    miss = [c for c in need if c not in point_forecasts.columns]
    if miss:
        raise ValueError(f"point_forecasts missing columns: {miss}")
    k = point_forecasts[need].copy()
    k["provider_id"] = k["provider_id"].map(_clean)
    k["service_line"] = k["service_line"].map(_clean)
    k["day_of_week"] = pd.to_numeric(k["day_of_week"], errors="coerce").fillna(-1).astype(int)
    k["pd_key"] = [_pd_key(p, d) for p, d in zip(k["provider_id"], k["day_of_week"])]
    return k.drop_duplicates("pd_key").reset_index(drop=True)


def build_residual_training_frame(
    features: pd.DataFrame,
    models: dict[str, Any],
    feature_cols: list[str],
    out_dir: str | Path,
) -> pd.DataFrame:
    """Predict all historical rows and compute residual = actual - prediction."""
    out = _out(out_dir)
    need = ["provider_id", "service_line", "day_of_week", "casetime_min", "turnover_min"]
    miss = [c for c in need if c not in features.columns]
    if miss:
        raise ValueError(f"features missing columns needed for Bayesian residuals: {miss}")

    df = features.copy()
    df["provider_id"] = df["provider_id"].map(_clean)
    df["service_line"] = df["service_line"].map(_clean)
    df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(-1).astype(int)
    df["pd_key"] = [_pd_key(p, d) for p, d in zip(df["provider_id"], df["day_of_week"])]

    for target in ["casetime_min", "turnover_min"]:
        if target not in models:
            raise ValueError(f"models missing fitted model for {target}")
        pred = np.asarray(models[target].predict(df[feature_cols]), dtype=float)
        df[f"pred_{target}"] = np.maximum(0.0, pred)
        df[f"residual_{target}"] = pd.to_numeric(df[target], errors="coerce") - df[f"pred_{target}"]

    keep = [
        "provider_id", "service_line", "day_of_week", "pd_key", "week_index", "week_start",
        "casetime_min", "turnover_min", "pred_casetime_min", "pred_turnover_min",
        "residual_casetime_min", "residual_turnover_min",
    ]
    keep = [c for c in keep if c in df.columns]
    res = df[keep].replace([np.inf, -np.inf], np.nan)
    res.to_csv(out / "bayesian_training_residuals.csv", index=False)
    return res


def _fit_one(
    residuals: pd.DataFrame,
    all_keys: pd.DataFrame,
    target_name: str,
    out_dir: Path,
    config: dict,
) -> dict[str, Any]:
    pm, az = _require_pymc()
    cfg = config.get("layer1", {})
    seed = int(cfg.get("random_seed", 42))
    draws = int(cfg.get("mcmc_draws", cfg.get("mcmc_samples", 1000)))
    tune = int(cfg.get("mcmc_tune", cfg.get("mcmc_warmup", 1000)))
    chains = int(cfg.get("mcmc_chains", 4))
    cores = int(cfg.get("mcmc_cores", min(chains, 4)))
    target_accept = float(cfg.get("mcmc_target_accept", 0.92))
    rhat_threshold = float(cfg.get("rhat_threshold", 1.05))
    beta = float(cfg.get("bayesian_sigma_site_beta", 5.0))
    max_obs = int(cfg.get("bayesian_max_obs", 0) or 0)

    spec = TARGETS[target_name]
    resid_col = f"residual_{spec['actual']}"
    floor_min = float(spec["floor_min"])

    df = residuals[["provider_id", "service_line", "day_of_week", "pd_key", resid_col]].copy()
    df[resid_col] = pd.to_numeric(df[resid_col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[resid_col])
    if df.empty:
        raise ValueError(f"No residual rows for Bayesian target {target_name}")

    # Include forecast provider×day keys too. If a forecast key has no residuals, it
    # has no likelihood term and borrows from its service-line posterior.
    residual_keys = df[["provider_id", "service_line", "day_of_week", "pd_key"]].drop_duplicates("pd_key")
    key_table = pd.concat([all_keys, residual_keys], ignore_index=True).drop_duplicates("pd_key").reset_index(drop=True)

    service_lines = sorted(key_table["service_line"].map(_clean).unique().tolist())
    sl_to_idx = {sl: i for i, sl in enumerate(service_lines)}
    key_to_idx = {k: i for i, k in enumerate(key_table["pd_key"].tolist())}

    key_table["pd_idx"] = key_table["pd_key"].map(key_to_idx).astype(int)
    key_table["sl_idx"] = key_table["service_line"].map(sl_to_idx).astype(int)
    df["pd_idx"] = df["pd_key"].map(key_to_idx).astype(int)

    if max_obs > 0 and len(df) > max_obs:
        rng = np.random.default_rng(seed)
        df = df.loc[rng.choice(df.index.to_numpy(), size=max_obs, replace=False)].sort_index()

    y_raw = df[resid_col].to_numpy(dtype=float)
    scale = float(np.nanstd(y_raw))
    if not np.isfinite(scale) or scale <= 1e-8:
        mad = float(np.nanmedian(np.abs(y_raw - np.nanmedian(y_raw))))
        scale = 1.4826 * mad if mad > 1e-8 else 1.0

    y = y_raw / scale
    obs_pd_idx = df["pd_idx"].to_numpy(dtype=int)
    sl_for_pd = key_table["sl_idx"].to_numpy(dtype=int)

    coords = {
        "provider_day": key_table["pd_key"].astype(str).tolist(),
        "service_line": service_lines,
        "obs": np.arange(len(y)),
    }

    with pm.Model(coords=coords) as model:
        sigma_site = pm.HalfCauchy("sigma_site", beta=beta)
        sigma_sl = pm.HalfNormal("sigma_sl", sigma=sigma_site, dims="service_line")
        sigma_pd_raw = pm.HalfNormal("sigma_pd_raw", sigma=sigma_sl[sl_for_pd], dims="provider_day")
        sigma_pd = pm.Deterministic("sigma_pd", sigma_pd_raw + floor_min / scale, dims="provider_day")
        pm.Normal("residual", mu=0.0, sigma=sigma_pd[obs_pd_idx], observed=y, dims="obs")
        try:
            idata = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores,
                target_accept=target_accept,
                max_treedepth=int(cfg.get("mcmc_max_treedepth", 12)),
                random_seed=seed, return_inferencedata=True, progressbar=True,
            )
        except TypeError:
            idata = pm.sample(
                draws=draws, tune=tune, chains=chains, cores=cores,
                target_accept=target_accept,
                random_seed=seed, return_inferencedata=True, progressbar=True,
            )

    posterior_nc = out_dir / f"posterior_{target_name}.nc"
    posterior_path = _save_idata_robust(idata, az, posterior_nc)

    param_summary = _summary_robust(idata, az, target_name, out_dir)
    param_summary.to_csv(out_dir / f"posterior_parameter_summary_{target_name}.csv", index=False)

    # Diagnostics.  Do not depend only on az.summary, because the ArviZ 1.x
    # split changed kwargs and some column names.
    sampling_diag = _compute_sampling_diagnostics(idata, az, param_summary)
    diag = {"target": target_name, "posterior_path": str(posterior_path)}
    diag.update(sampling_diag)
    diag["rhat_threshold"] = rhat_threshold
    diag["r_hat_pass"] = bool(diag["max_r_hat"] is not None and diag["max_r_hat"] <= rhat_threshold)
    diag["divergence_pass"] = bool(diag["divergences"] == 0)
    diag["overall_pass"] = bool(diag["r_hat_pass"] and diag["divergence_pass"])
    _save_json(diag, out_dir / f"bayesian_diagnostics_{target_name}.json")
    pd.DataFrame([diag]).to_csv(out_dir / f"bayesian_diagnostics_{target_name}.csv", index=False)

    # Convert posterior sigma to minutes.
    sigma_draws = np.maximum(_posterior_matrix(idata, "sigma_pd") * scale, floor_min)
    np.savez_compressed(
        out_dir / f"posterior_sigma_draws_{target_name}.npz",
        sigma_pd=sigma_draws,
        provider_id=key_table["provider_id"].astype(str).to_numpy(),
        service_line=key_table["service_line"].astype(str).to_numpy(),
        day_of_week=key_table["day_of_week"].to_numpy(dtype=int),
        pd_key=key_table["pd_key"].astype(str).to_numpy(),
        posterior_draw_id=np.arange(sigma_draws.shape[0]),
    )

    qs = np.quantile(sigma_draws, [0.05, 0.10, 0.50, 0.90, 0.95], axis=0)
    summary = key_table[["provider_id", "service_line", "day_of_week", "pd_key", "pd_idx", "sl_idx"]].copy()
    summary[f"sigma_{target_name}_mean"] = sigma_draws.mean(axis=0)
    summary[f"sigma_{target_name}_sd"] = sigma_draws.std(axis=0)
    summary[f"sigma_{target_name}_p05"] = qs[0]
    summary[f"sigma_{target_name}_p10"] = qs[1]
    summary[f"sigma_{target_name}_p50"] = qs[2]
    summary[f"sigma_{target_name}_p90"] = qs[3]
    summary[f"sigma_{target_name}_p95"] = qs[4]
    counts = df.groupby("pd_key").size().rename(f"n_residual_rows_{target_name}").reset_index()
    summary = summary.merge(counts, on="pd_key", how="left")
    summary[f"n_residual_rows_{target_name}"] = summary[f"n_residual_rows_{target_name}"].fillna(0).astype(int)
    summary.to_csv(out_dir / f"posterior_sigma_summary_{target_name}.csv", index=False)

    sigma_mean_by_pd = sigma_draws.mean(axis=0)
    obs_sigma = sigma_mean_by_pd[obs_pd_idx]
    loss = {
        "target": target_name,
        "n_residual_rows_used": int(len(y_raw)),
        "n_provider_day_keys": int(len(key_table)),
        "n_service_lines": int(len(service_lines)),
        "residual_rmse_minutes": float(np.sqrt(np.mean(y_raw**2))),
        "residual_mae_minutes": float(np.mean(np.abs(y_raw))),
        "mean_negative_log_likelihood": _nll(y_raw, obs_sigma),
        "standardization_scale_minutes": scale,
    }
    _save_json(loss, out_dir / f"bayesian_loss_metrics_{target_name}.json")
    pd.DataFrame([loss]).to_csv(out_dir / f"bayesian_loss_metrics_{target_name}.csv", index=False)

    return {
        "target": target_name,
        "sigma_draws": sigma_draws,
        "key_table": key_table,
        "sigma_summary": summary,
        "diagnostics": diag,
        "loss_metrics": loss,
        "posterior_path": str(posterior_path),
    }


def _calibration(
    validation_predictions: Optional[pd.DataFrame],
    results: dict[str, dict[str, Any]],
    out_dir: Path,
    config: dict,
) -> dict[str, pd.DataFrame]:
    if validation_predictions is None or validation_predictions.empty:
        rows = pd.DataFrame()
        metrics = pd.DataFrame()
        rows.to_csv(out_dir / "bayesian_calibration_rows.csv", index=False)
        metrics.to_csv(out_dir / "bayesian_calibration_metrics.csv", index=False)
        return {"rows": rows, "metrics": metrics}

    rng = np.random.default_rng(int(config.get("layer1", {}).get("random_seed", 42)) + 444)
    n_draws = int(config.get("layer1", {}).get("calibration_draws", 1000))
    all_rows = []
    all_metrics = []

    for target_name, result in results.items():
        spec = TARGETS[target_name]
        if spec["actual"] not in validation_predictions.columns or spec["pred"] not in validation_predictions.columns:
            continue
        df = validation_predictions.copy()
        df["provider_id"] = df["provider_id"].map(_clean)
        df["service_line"] = df["service_line"].map(_clean)
        df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(-1).astype(int)
        df["pd_key"] = [_pd_key(p, d) for p, d in zip(df["provider_id"], df["day_of_week"])]
        df[spec["actual"]] = pd.to_numeric(df[spec["actual"]], errors="coerce")
        df[spec["pred"]] = pd.to_numeric(df[spec["pred"]], errors="coerce")
        df = df.dropna(subset=[spec["actual"], spec["pred"]])

        key_to_idx = {k: int(i) for k, i in zip(result["key_table"]["pd_key"], result["key_table"]["pd_idx"])}
        sigma_draws = result["sigma_draws"]
        D = sigma_draws.shape[0]
        use = min(n_draws, D)
        draw_ids = rng.choice(np.arange(D), size=use, replace=(use > D))

        recs = []
        for _, r in df.iterrows():
            idx = key_to_idx.get(r["pd_key"])
            if idx is None:
                continue
            sig = sigma_draws[draw_ids, idx]
            sims = np.maximum(0.0, float(r[spec["pred"]]) + rng.normal(0, sig))
            rec = {
                "target": target_name,
                "provider_id": r["provider_id"],
                "service_line": r["service_line"],
                "day_of_week": int(r["day_of_week"]),
                "actual": float(r[spec["actual"]]),
                "point_prediction": float(r[spec["pred"]]),
                "scenario_mean": float(sims.mean()),
                "scenario_sd": float(sims.std()),
                "sigma_mean": float(sig.mean()),
            }
            for level in [0.50, 0.80, 0.95]:
                a = 1.0 - level
                lo = float(np.quantile(sims, a / 2))
                hi = float(np.quantile(sims, 1 - a / 2))
                rec[f"pi{int(level*100)}_lower"] = lo
                rec[f"pi{int(level*100)}_upper"] = hi
                rec[f"pi{int(level*100)}_covered"] = bool(lo <= rec["actual"] <= hi)
            recs.append(rec)

        row_df = pd.DataFrame(recs)
        if row_df.empty:
            continue
        all_rows.append(row_df)
        for level in [0.50, 0.80, 0.95]:
            col = f"pi{int(level*100)}_covered"
            all_metrics.append({
                "target": target_name,
                "nominal_coverage": level,
                "empirical_coverage": float(row_df[col].mean()),
                "coverage_error": float(row_df[col].mean() - level),
                "n_rows": int(len(row_df)),
            })
        resid = row_df["actual"].to_numpy() - row_df["point_prediction"].to_numpy()
        all_metrics.append({
            "target": target_name,
            "nominal_coverage": np.nan,
            "empirical_coverage": np.nan,
            "coverage_error": np.nan,
            "n_rows": int(len(row_df)),
            "rmse": float(np.sqrt(np.mean(resid**2))),
            "mae": float(np.mean(np.abs(resid))),
            "mean_negative_log_likelihood": _nll(resid, row_df["sigma_mean"].to_numpy()),
        })

    rows = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    metrics = pd.DataFrame(all_metrics)
    rows.to_csv(out_dir / "bayesian_calibration_rows.csv", index=False)
    metrics.to_csv(out_dir / "bayesian_calibration_metrics.csv", index=False)
    return {"rows": rows, "metrics": metrics}


def fit_bayesian_residual_models(
    features: pd.DataFrame,
    point_forecasts: pd.DataFrame,
    models: dict[str, Any],
    feature_cols: list[str],
    out_dir: str | Path,
    config: dict,
    validation_predictions: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    out = _out(out_dir)
    residuals = build_residual_training_frame(features, models, feature_cols, out)
    all_keys = _forecast_keys(point_forecasts)

    _save_json(
        {
            "model": "hierarchical_bayesian_residual_sigma",
            "likelihood": "residual_z[p,d,t] ~ Normal(0, sigma_pd[p,d])",
            "hierarchy": [
                "sigma_pd[p,d] ~ HalfNormal(sigma_sl[service_line(p)])",
                "sigma_sl[sl] ~ HalfNormal(sigma_site)",
                "sigma_site ~ HalfCauchy(beta=5.0)",
            ],
            "note": "Residuals are standardized for NUTS; saved sigmas are minutes.",
            "n_forecast_provider_day_keys": int(len(all_keys)),
            "layer1_config": config.get("layer1", {}),
        },
        out / "learned_bayesian_model_spec.json",
    )

    case = _fit_one(residuals, all_keys, "casetime", out, config)
    turn = _fit_one(residuals, all_keys, "turnover", out, config)

    sigma = case["sigma_summary"].merge(
        turn["sigma_summary"],
        on=["provider_id", "service_line", "day_of_week", "pd_key"],
        how="outer",
        suffixes=("_case_table", "_turn_table"),
    )
    sigma["sigma_case"] = sigma["sigma_casetime_mean"]
    sigma["sigma_turn"] = sigma["sigma_turnover_mean"]
    sigma["sigma_case_p10"] = sigma["sigma_casetime_p10"]
    sigma["sigma_case_p50"] = sigma["sigma_casetime_p50"]
    sigma["sigma_case_p90"] = sigma["sigma_casetime_p90"]
    sigma["sigma_turn_p10"] = sigma["sigma_turnover_p10"]
    sigma["sigma_turn_p50"] = sigma["sigma_turnover_p50"]
    sigma["sigma_turn_p90"] = sigma["sigma_turnover_p90"]
    front = [
        "provider_id", "service_line", "day_of_week", "sigma_case", "sigma_turn",
        "sigma_case_p10", "sigma_case_p50", "sigma_case_p90",
        "sigma_turn_p10", "sigma_turn_p50", "sigma_turn_p90",
        "n_residual_rows_casetime", "n_residual_rows_turnover",
    ]
    rest = [c for c in sigma.columns if c not in front]
    sigma = sigma[[c for c in front if c in sigma.columns] + rest]
    sigma.to_csv(out / "hierarchical_sigma_estimates.csv", index=False)

    calibration = _calibration(validation_predictions, {"casetime": case, "turnover": turn}, out, config)
    loss_summary = pd.concat([pd.DataFrame([case["loss_metrics"]]), pd.DataFrame([turn["loss_metrics"]])], ignore_index=True)
    loss_summary.to_csv(out / "bayesian_loss_metrics.csv", index=False)

    diag_summary = {
        "casetime": case["diagnostics"],
        "turnover": turn["diagnostics"],
        "overall_pass": bool(case["diagnostics"].get("overall_pass") and turn["diagnostics"].get("overall_pass")),
    }
    _save_json(diag_summary, out / "bayesian_diagnostics_summary.json")

    return {
        "sigma_estimates": sigma,
        "residuals": residuals,
        "casetime": case,
        "turnover": turn,
        "calibration": calibration,
        "loss_summary": loss_summary,
        "diagnostics_summary": diag_summary,
    }


def _early_release_map(early_release_df: Optional[pd.DataFrame]) -> dict[str, float]:
    if early_release_df is None or early_release_df.empty:
        return {}
    provider_col = next((c for c in ["provider_id", "provider", "blockholder_id"] if c in early_release_df.columns), None)
    value_col = None
    for c in early_release_df.columns:
        lc = c.lower()
        if "early" in lc and pd.api.types.is_numeric_dtype(early_release_df[c]):
            value_col = c
            if "project" in lc:
                break
    if provider_col is None or value_col is None:
        return {}
    tmp = early_release_df[[provider_col, value_col]].copy()
    tmp[provider_col] = tmp[provider_col].map(_clean)
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce").fillna(0.0)
    return tmp.groupby(provider_col)[value_col].sum().to_dict()


def _idx(result: dict[str, Any]) -> dict[str, int]:
    kt = result["key_table"]
    return {str(k): int(i) for k, i in zip(kt["pd_key"], kt["pd_idx"])}


def generate_bayesian_saa_scenarios(
    point_forecasts: pd.DataFrame,
    bayes_result: dict[str, Any],
    out_dir: str | Path,
    config: dict,
    early_release_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out = _out(out_dir)
    cfg = config.get("layer1", {})
    rng = np.random.default_rng(int(cfg.get("random_seed", 42)) + 777)
    S = int(cfg.get("n_scenarios", 200))
    n_weeks = int(config.get("layer2", {}).get("optimization_weeks", cfg.get("scenario_n_weeks", 13)))
    write_json = bool(cfg.get("write_demand_scenarios_json", True))

    f = point_forecasts.copy()
    for c in ["provider_id", "service_line", "day_of_week", "mu_casetime_min", "mu_turnover_min"]:
        if c not in f.columns:
            raise ValueError(f"point_forecasts missing {c}")
    f["provider_id"] = f["provider_id"].map(_clean)
    f["service_line"] = f["service_line"].map(_clean)
    f["day_of_week"] = pd.to_numeric(f["day_of_week"], errors="coerce").fillna(-1).astype(int)
    f["pd_key"] = [_pd_key(p, d) for p, d in zip(f["provider_id"], f["day_of_week"])]
    f["mu_casetime_min"] = pd.to_numeric(f["mu_casetime_min"], errors="coerce").fillna(0).clip(lower=0)
    f["mu_turnover_min"] = pd.to_numeric(f["mu_turnover_min"], errors="coerce").fillna(0).clip(lower=0)

    case_idx = _idx(bayes_result["casetime"])
    turn_idx = _idx(bayes_result["turnover"])
    f["case_i"] = f["pd_key"].map(case_idx)
    f["turn_i"] = f["pd_key"].map(turn_idx)
    if f["case_i"].isna().any() or f["turn_i"].isna().any():
        raise ValueError("Some point forecast keys are missing from Bayesian posterior tables.")
    f["case_i"] = f["case_i"].astype(int)
    f["turn_i"] = f["turn_i"].astype(int)

    case_draws = bayes_result["casetime"]["sigma_draws"]
    turn_draws = bayes_result["turnover"]["sigma_draws"]
    case_draw_ids = rng.choice(np.arange(case_draws.shape[0]), size=S, replace=(S > case_draws.shape[0]))
    turn_draw_ids = rng.choice(np.arange(turn_draws.shape[0]), size=S, replace=(S > turn_draws.shape[0]))

    rows = []
    provider = f["provider_id"].to_numpy()
    service_line = f["service_line"].to_numpy()
    day = f["day_of_week"].to_numpy(dtype=int)
    mu_case = f["mu_casetime_min"].to_numpy(dtype=float)
    mu_turn = f["mu_turnover_min"].to_numpy(dtype=float)
    ci = f["case_i"].to_numpy(dtype=int)
    ti = f["turn_i"].to_numpy(dtype=int)

    for s in range(S):
        sig_case = case_draws[case_draw_ids[s], ci]
        sig_turn = turn_draws[turn_draw_ids[s], ti]
        dem_case = np.maximum(0.0, mu_case + rng.normal(0, sig_case))
        dem_turn = np.maximum(0.0, mu_turn + rng.normal(0, sig_turn))
        rows.append(pd.DataFrame({
            "scenario_id": s,
            "provider_id": provider,
            "service_line": service_line,
            "day_of_week": day,
            "demand_casetime_min": dem_case,
            "demand_turnover_min": dem_turn,
            "demand_total_min": dem_case + dem_turn,
            "mu_casetime_min": mu_case,
            "mu_turnover_min": mu_turn,
            "sigma_case_draw": sig_case,
            "sigma_turn_draw": sig_turn,
            "posterior_draw_id_case": int(case_draw_ids[s]),
            "posterior_draw_id_turn": int(turn_draw_ids[s]),
        }))

    scenarios = pd.concat(rows, ignore_index=True)
    scenarios.to_csv(out / "scenarios_long.csv", index=False)

    summary = scenarios.groupby(["provider_id", "service_line", "day_of_week"]).agg(
        case_p10=("demand_casetime_min", lambda x: float(np.quantile(x, 0.10))),
        case_p50=("demand_casetime_min", lambda x: float(np.quantile(x, 0.50))),
        case_p90=("demand_casetime_min", lambda x: float(np.quantile(x, 0.90))),
        turn_p10=("demand_turnover_min", lambda x: float(np.quantile(x, 0.10))),
        turn_p50=("demand_turnover_min", lambda x: float(np.quantile(x, 0.50))),
        turn_p90=("demand_turnover_min", lambda x: float(np.quantile(x, 0.90))),
        total_p10=("demand_total_min", lambda x: float(np.quantile(x, 0.10))),
        total_p50=("demand_total_min", lambda x: float(np.quantile(x, 0.50))),
        total_p90=("demand_total_min", lambda x: float(np.quantile(x, 0.90))),
        sigma_case_mean=("sigma_case_draw", "mean"),
        sigma_turn_mean=("sigma_turn_draw", "mean"),
        mu_casetime_min=("mu_casetime_min", "first"),
        mu_turnover_min=("mu_turnover_min", "first"),
    ).reset_index()
    summary.to_csv(out / "scenario_summary.csv", index=False)

    er = _early_release_map(early_release_df)
    q = scenarios.groupby(["scenario_id", "provider_id", "service_line"], as_index=False).agg(
        weekly_casetime_min=("demand_casetime_min", "sum"),
        weekly_turnover_min=("demand_turnover_min", "sum"),
        weekly_total_min=("demand_total_min", "sum"),
    )
    q["n_weeks"] = n_weeks
    q["quarterly_casetime_min"] = q["weekly_casetime_min"] * n_weeks
    q["quarterly_turnover_min"] = q["weekly_turnover_min"] * n_weeks
    q["quarterly_demand_min"] = q["weekly_total_min"] * n_weeks
    q["early_release_projected_min"] = q["provider_id"].map(er).fillna(0.0)
    q["quarterly_total_with_early_release_min"] = q["quarterly_demand_min"] + q["early_release_projected_min"]
    q.to_csv(out / "quarterly_provider_scenarios.csv", index=False)

    if write_json:
        nested = []
        for (p, sl, d), g in scenarios.groupby(["provider_id", "service_line", "day_of_week"], sort=False):
            nested.append({
                "provider_id": p,
                "service_line": sl,
                "day_of_week": int(d),
                "point_prediction_casetime_min": float(g["mu_casetime_min"].iloc[0]),
                "point_prediction_turnover_min": float(g["mu_turnover_min"].iloc[0]),
                "sigma_posterior_mean_casetime": float(g["sigma_case_draw"].mean()),
                "sigma_posterior_mean_turnover": float(g["sigma_turn_draw"].mean()),
                "scenarios": [
                    {
                        "scenario_id": int(r.scenario_id),
                        "casetime_min": float(r.demand_casetime_min),
                        "turnover_min": float(r.demand_turnover_min),
                        "total_min": float(r.demand_total_min),
                    }
                    for r in g.itertuples(index=False)
                ],
            })
        with open(out / "demand_scenarios.json", "w", encoding="utf-8") as fjson:
            json.dump(nested, fjson, indent=2)

    _save_json({
        "n_scenarios": S,
        "n_provider_day_rows": int(len(f)),
        "n_scenario_rows": int(len(scenarios)),
        "n_weeks_for_quarterly_scaling": n_weeks,
        "posterior_draws_casetime_available": int(case_draws.shape[0]),
        "posterior_draws_turnover_available": int(turn_draws.shape[0]),
    }, out / "saa_scenario_generation_metadata.json")

    return scenarios

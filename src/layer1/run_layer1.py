"""
Updated Layer 1 runner with real hierarchical Bayesian residual model.

Place in: src/layer1/run_layer1.py
Also add: src/layer1/bayesian_residual_model.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

# same-folder imports only
from weekly_observations import build_weekly_observations, early_release_projected
from feature_engineering import build_features
from forecasting import train_point_forecasts, make_next_week_forecasts
from scenarios import estimate_sigma, generate_scenarios
from bayesian_residual_model import fit_bayesian_residual_models, generate_bayesian_saa_scenarios

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_df(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_default_config() -> Optional[Path]:
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "config" / "default_config.yaml",
        here.parents[1] / "config" / "default_config.yaml" if len(here.parents) > 1 else None,
        here.parents[2] / "config" / "default_config.yaml" if len(here.parents) > 2 else None,
        Path("config") / "default_config.yaml",
        Path("default_config.yaml"),
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return None


def load_raw_tables(blocks_json: str, providers_json: str, cases_json: Optional[str] = None):
    with open(blocks_json, encoding="utf-8") as f:
        blocks_raw = json.load(f)
    try:
        blocks_df = pd.json_normalize(blocks_raw)
    except Exception:
        blocks_df = pd.DataFrame(blocks_raw)

    with open(providers_json, encoding="utf-8") as f:
        providers_raw = json.load(f)
    try:
        providers_df = pd.json_normalize(providers_raw)
    except Exception:
        providers_df = pd.DataFrame(providers_raw)

    cases_df = pd.DataFrame()
    if cases_json and Path(cases_json).exists():
        with open(cases_json, encoding="utf-8") as f:
            cases_raw = json.load(f)
        try:
            cases_df = pd.json_normalize(cases_raw)
        except Exception:
            cases_df = pd.DataFrame(cases_raw)

    log.info("Loaded: blocks=%d  providers=%d  cases=%d", len(blocks_df), len(providers_df), len(cases_df))
    return blocks_df, providers_df, cases_df


def ensure_config_shape(config: Optional[dict]) -> dict:
    cfg = dict(config or {})
    cfg.setdefault("paths", {})
    cfg.setdefault("layer1", {})
    cfg.setdefault("layer2", {})
    return cfg


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    cfg = ensure_config_shape(config)
    if args.blocks:
        cfg["paths"]["blocks_json"] = args.blocks
    if args.providers:
        cfg["paths"]["providers_json"] = args.providers
    if args.cases:
        cfg["paths"]["cases_json"] = args.cases
    if args.out:
        cfg["paths"]["output_dir"] = args.out
    if args.n_scenarios is not None:
        cfg["layer1"]["n_scenarios"] = int(args.n_scenarios)
    if args.optuna_trials is not None:
        cfg["layer1"]["optuna_trials"] = int(args.optuna_trials)
    if args.point_model:
        cfg["layer1"]["point_model"] = args.point_model
    if args.sigma_model:
        cfg["layer1"]["sigma_model"] = args.sigma_model
    if args.mcmc_draws is not None:
        cfg["layer1"]["mcmc_draws"] = int(args.mcmc_draws)
    if args.mcmc_tune is not None:
        cfg["layer1"]["mcmc_tune"] = int(args.mcmc_tune)
    if args.mcmc_chains is not None:
        cfg["layer1"]["mcmc_chains"] = int(args.mcmc_chains)
    if args.mcmc_cores is not None:
        cfg["layer1"]["mcmc_cores"] = int(args.mcmc_cores)
    if args.mcmc_target_accept is not None:
        cfg["layer1"]["mcmc_target_accept"] = float(args.mcmc_target_accept)
    if args.mcmc_max_treedepth is not None:
        cfg["layer1"]["mcmc_max_treedepth"] = int(args.mcmc_max_treedepth)
    if args.rhat_threshold is not None:
        cfg["layer1"]["rhat_threshold"] = float(args.rhat_threshold)
    if args.bayesian_max_obs is not None:
        cfg["layer1"]["bayesian_max_obs"] = int(args.bayesian_max_obs)
    if args.no_demand_scenarios_json:
        cfg["layer1"]["write_demand_scenarios_json"] = False

    cfg["layer1"].setdefault("sigma_model", "shrinkage")
    cfg["layer1"].setdefault("n_scenarios", 200)
    cfg["layer1"].setdefault("random_seed", 42)
    cfg["layer1"].setdefault("mcmc_draws", 1000)
    cfg["layer1"].setdefault("mcmc_tune", 1000)
    cfg["layer1"].setdefault("mcmc_chains", 4)
    cfg["layer1"].setdefault("mcmc_cores", min(int(cfg["layer1"]["mcmc_chains"]), 4))
    cfg["layer1"].setdefault("mcmc_target_accept", 0.92)
    cfg["layer1"].setdefault("mcmc_max_treedepth", 12)
    cfg["layer1"].setdefault("rhat_threshold", 1.05)
    cfg["layer1"].setdefault("bayesian_sigma_site_beta", 5.0)
    return cfg


def log_effective_config(config: dict) -> None:
    p = config.get("paths", {})
    l1 = config.get("layer1", {})
    log.info("Effective paths.blocks_json     = %s", p.get("blocks_json"))
    log.info("Effective paths.providers_json  = %s", p.get("providers_json"))
    log.info("Effective paths.cases_json      = %s", p.get("cases_json"))
    log.info("Effective paths.output_dir      = %s", p.get("output_dir"))
    log.info("Effective layer1.point_model    = %s", l1.get("point_model"))
    log.info("Effective layer1.sigma_model    = %s", l1.get("sigma_model"))
    log.info("Effective layer1.n_scenarios    = %s", l1.get("n_scenarios"))
    log.info("Effective layer1.optuna_trials  = %s", l1.get("optuna_trials"))
    if l1.get("sigma_model") == "bayesian":
        log.info("Effective layer1.mcmc_draws     = %s", l1.get("mcmc_draws"))
        log.info("Effective layer1.mcmc_tune      = %s", l1.get("mcmc_tune"))
        log.info("Effective layer1.mcmc_chains    = %s", l1.get("mcmc_chains"))
        log.info("Effective layer1.mcmc_cores     = %s", l1.get("mcmc_cores"))
        log.info("Effective layer1.rhat_threshold = %s", l1.get("rhat_threshold"))


def sanity_log_packages(config: dict) -> None:
    log.info("Python executable: %s", sys.executable)
    try:
        import xgboost as xgb
        log.info("XGBoost import OK: version=%s", getattr(xgb, "__version__", "unknown"))
    except Exception as exc:
        log.warning("XGBoost import failed: %s", exc)
    if config.get("layer1", {}).get("sigma_model") == "bayesian":
        try:
            import pymc as pm
            import arviz as az
            log.info("PyMC import OK: version=%s", getattr(pm, "__version__", "unknown"))
            log.info("ArviZ import OK: version=%s", getattr(az, "__version__", "unknown"))
        except Exception as exc:
            raise ImportError("Install Bayesian dependencies: pip install pymc arviz h5netcdf netcdf4") from exc


def save_residual_plot(predictions: pd.DataFrame, path: str | Path, target_col: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pred_col = f"pred_{target_col}"
        if pred_col not in predictions.columns or target_col not in predictions.columns:
            return
        actual = pd.to_numeric(predictions[target_col], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(predictions[pred_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(actual) & np.isfinite(pred)
        actual, pred = actual[ok], pred[ok]
        if len(actual) == 0:
            return
        resid = actual - pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(pred, actual, alpha=0.4, s=10)
        mn, mx = min(pred.min(), actual.min()), max(pred.max(), actual.max())
        axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.5)
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_title(f"{target_col} — Predicted vs Actual")
        axes[1].hist(resid, bins=50, alpha=0.7, edgecolor="white")
        axes[1].axvline(0, color="red", lw=1.5, ls="--")
        axes[1].set_xlabel("Residual (actual − predicted)")
        axes[1].set_title(f"{target_col} — Residuals")
        fig.tight_layout()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130)
        plt.close(fig)
        log.info("Residual plot saved → %s", p)
    except Exception as exc:
        log.warning("Residual plot skipped (%s)", exc)


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str = "", warn: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    checks.append({"name": name, "status": status, "detail": detail})


def validate_outputs(prelayer, obs, features, point, sigma, scenarios, model_result, out: Path, config, bayes_result=None) -> dict:
    checks = []
    l1 = config.get("layer1", {})
    sigma_model = l1.get("sigma_model", "shrinkage")
    T = prelayer.get("T_star", prelayer.get("T*", prelayer.get("best_T")))
    add_check(checks, "Pre-Layer T* present", T is not None, f"T*={T}")
    phase_labels = prelayer.get("phase_labels", {})
    add_check(checks, "Pre-Layer phase_labels present", bool(phase_labels), f"n_labels={len(phase_labels)}")
    add_check(checks, "Weekly observations non-empty", not obs.empty, f"{len(obs)} rows")
    add_check(checks, "Features non-empty", not features.empty, f"{features.shape[0]} rows × {features.shape[1]} cols")
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    add_check(checks, "Numeric feature columns generated", len(numeric_cols) > 0, f"{len(numeric_cols)} numeric columns")
    add_check(checks, "No infinite feature values", not np.isinf(features.select_dtypes(include=[np.number]).to_numpy()).any(), "NaN allowed")
    add_check(checks, "Point forecasts non-empty", not point.empty, f"{len(point)} provider×day rows")
    nonneg_point = all(c in point.columns and (pd.to_numeric(point[c], errors="coerce") >= -1e-9).all() for c in ["mu_casetime_min", "mu_turnover_min"])
    add_check(checks, "Point forecasts non-negative", bool(nonneg_point), "mu_casetime_min, mu_turnover_min ≥ 0")
    add_check(checks, "Point model metrics written", bool(model_result.get("metrics")), str(model_result.get("metrics", {}))[:400])
    add_check(checks, "Sigma estimates positive", not sigma.empty, f"{len(sigma)} sigma rows")
    sigma_ok = all(c in sigma.columns and (pd.to_numeric(sigma[c], errors="coerce") > 0).all() for c in ["sigma_case", "sigma_turn"])
    add_check(checks, "Sigma values positive", bool(sigma_ok), "sigma_case and sigma_turn > 0")
    S = int(l1.get("n_scenarios", 200))
    add_check(checks, "Scenario row count correct", len(scenarios) == len(point) * S, f"actual={len(scenarios)}, expected={len(point)*S}")
    scen_ok = all(c in scenarios.columns and (pd.to_numeric(scenarios[c], errors="coerce") >= -1e-9).all() for c in ["demand_casetime_min", "demand_turnover_min"])
    add_check(checks, "Scenario demands non-negative", bool(scen_ok), "all generated demand ≥ 0")

    required = [
        out / "weekly_observations.csv", out / "features.csv", out / "point_forecasts.csv",
        out / "hierarchical_sigma_estimates.csv", out / "scenarios_long.csv",
        out / "scenario_summary.csv", out / "layer1_metadata.json",
    ]
    if sigma_model == "bayesian":
        required += [
            out / "learned_bayesian_model_spec.json",
            out / "bayesian_diagnostics_summary.json",
            out / "bayesian_calibration_metrics.csv",
            out / "bayesian_loss_metrics.csv",
            out / "posterior_sigma_draws_casetime.npz",
            out / "posterior_sigma_draws_turnover.npz",
            out / "quarterly_provider_scenarios.csv",
        ]
        posterior_ok = ((out / "posterior_casetime.nc").exists() or (out / "posterior_casetime.pkl").exists()) and ((out / "posterior_turnover.nc").exists() or (out / "posterior_turnover.pkl").exists())
        add_check(checks, "Bayesian posterior files written", posterior_ok, "posterior .nc or .pkl for both targets")
        if bayes_result:
            for target in ["casetime", "turnover"]:
                d = bayes_result.get("diagnostics_summary", {}).get(target, {})
                add_check(checks, f"Bayesian {target} R-hat acceptable", bool(d.get("r_hat_pass")), f"max_r_hat={d.get('max_r_hat')}", warn=True)
                add_check(checks, f"Bayesian {target} divergences zero", bool(d.get("divergence_pass")), f"divergences={d.get('divergences')}", warn=True)
    missing = [str(p) for p in required if not p.exists()]
    add_check(checks, "Required artifact files written", not missing, "present" if not missing else f"missing={missing}")

    report = pd.DataFrame(checks)
    report.to_csv(out / "layer1_validation_report.csv", index=False)
    save_json({"checks": checks}, out / "layer1_validation_report.json")

    print("\n" + "─" * 62)
    print("  LAYER 1 VALIDATION REPORT")
    print("─" * 62)
    for c in checks:
        icon = "✓" if c["status"] == "PASS" else ("⚠" if c["status"] == "WARN" else "✗")
        print(f"  {icon} {c['status']:<6} {c['name']:<45} {c['detail']}")
    print("─" * 62)
    status = "FAIL" if (report["status"] == "FAIL").any() else ("WARN" if (report["status"] == "WARN").any() else "PASS")
    return {"status": status, "checks": checks}


def run(config: Optional[dict] = None, pre_layer_result_path: Optional[str] = None) -> dict:
    if config is None:
        cfg_path = find_default_config()
        if cfg_path is None:
            raise FileNotFoundError("Could not find default_config.yaml. Pass --config explicitly.")
        config = load_config(cfg_path)
        log.info("Config loaded from %s", cfg_path)
    config = ensure_config_shape(config)
    log_effective_config(config)
    sanity_log_packages(config)

    out = ensure_dir(Path(config["paths"]["output_dir"]) / "layer1")

    pre_path = Path(pre_layer_result_path or Path(config["paths"]["output_dir"]) / "pre_layer" / "prelayer_result.json")
    if not pre_path.exists():
        raise FileNotFoundError(f"Pre-Layer result not found: {pre_path}")
    log.info("Loading Pre-Layer result: %s", pre_path)
    prelayer = load_json(pre_path)

    log.info("Loading raw tables …")
    blocks_df, providers_df, cases_df = load_raw_tables(config["paths"]["blocks_json"], config["paths"]["providers_json"], config["paths"].get("cases_json"))

    log.info("Stage 0 — Building weekly observations …")
    obs, obs_meta = build_weekly_observations(blocks_df, providers_df, cases_df, prelayer, config)
    save_df(obs, out / "weekly_observations.csv")
    if obs_meta.get("warnings"):
        log.warning("Observation warnings:")
        for w in obs_meta["warnings"]:
            log.warning("  • %s", w)

    er = early_release_projected(blocks_df, optimization_weeks=int(config.get("layer2", {}).get("optimization_weeks", 13)))
    save_df(er, out / "early_release_projected.csv")

    log.info("Stage 1+2 — Building phase-aligned features and attention features …")
    features, attn_meta = build_features(obs, prelayer, config)
    save_df(features, out / "features.csv")
    log.info("  Features: %d rows × %d columns", *features.shape)

    log.info("Stage 3 — Training point forecast models …")
    model_result = train_point_forecasts(features, out, config)

    log.info("Stage 3 — Making next-week point forecasts …")
    point = make_next_week_forecasts(features, model_result["models"], model_result["feature_columns"], out)

    sigma_model = str(config.get("layer1", {}).get("sigma_model", "shrinkage")).lower()
    bayes_result = None
    if sigma_model == "bayesian":
        log.info("Stage 4 — Fitting real hierarchical Bayesian residual model with NUTS …")
        bayes_result = fit_bayesian_residual_models(
            features=features,
            point_forecasts=point,
            models=model_result["models"],
            feature_cols=model_result["feature_columns"],
            out_dir=out,
            config=config,
            validation_predictions=model_result.get("predictions"),
        )
        sigma = bayes_result["sigma_estimates"]
    elif sigma_model == "shrinkage":
        log.info("Stage 4 — Estimating σ_pd via hierarchical shrinkage approximation …")
        sigma = estimate_sigma(features, model_result.get("predictions"))
        save_df(sigma, out / "hierarchical_sigma_estimates.csv")
    else:
        raise ValueError("--sigma_model must be either shrinkage or bayesian")

    log.info("Stage 5 — Generating scenarios …")
    if sigma_model == "bayesian":
        scenarios = generate_bayesian_saa_scenarios(point, bayes_result, out, config, early_release_df=er)
    else:
        scenarios = generate_scenarios(point, sigma, out, config)

    metadata = {
        "warnings": obs_meta.get("warnings", []),
        "attention": attn_meta,
        "n_obs_rows": int(len(obs)),
        "n_features": int(len(features)),
        "n_forecast_rows": int(len(point)),
        "n_scenarios": int(config.get("layer1", {}).get("n_scenarios", 200)),
        "n_scenario_rows": int(len(scenarios)),
        "point_model_requested": config.get("layer1", {}).get("point_model"),
        "sigma_model": sigma_model,
        "model_metrics": model_result.get("metrics", {}),
    }
    if bayes_result is not None:
        metadata["bayesian_diagnostics_summary"] = bayes_result.get("diagnostics_summary", {})
        metadata["bayesian_loss_summary"] = bayes_result.get("loss_summary", pd.DataFrame()).to_dict(orient="records")
    save_json(metadata, out / "layer1_metadata.json")

    save_residual_plot(model_result["predictions"], out / "casetime_residuals.png", "casetime_min")
    save_residual_plot(model_result["predictions"], out / "turnover_residuals.png", "turnover_min")

    validation = validate_outputs(prelayer, obs, features, point, sigma, scenarios, model_result, out, config, bayes_result=bayes_result)

    log.info("=" * 60)
    log.info("Layer 1 complete")
    log.info("  Observation rows : %d", len(obs))
    log.info("  Feature rows     : %d", len(features))
    log.info("  Forecast rows    : %d", len(point))
    log.info("  Scenario rows    : %d", len(scenarios))
    log.info("  Validation       : %s", validation["status"])
    log.info("  Point model      : %s", config.get("layer1", {}).get("point_model"))
    log.info("  Sigma model      : %s", sigma_model)
    log.info("  Output dir       : %s", out)
    log.info("=" * 60)
    return {"output_dir": str(out), "validation": validation, "metadata": metadata}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Layer 1 — Stochastic Demand Forecasting")
    p.add_argument("--config", default=None)
    p.add_argument("--blocks", default=None)
    p.add_argument("--providers", default=None)
    p.add_argument("--cases", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--pre_layer_result", default=None)
    p.add_argument("--n_scenarios", type=int, default=None)
    p.add_argument("--optuna_trials", type=int, default=None)
    p.add_argument("--point_model", default=None, choices=["auto", "xgboost", "ridge", "gradient_boosting"])
    p.add_argument("--sigma_model", default=None, choices=["shrinkage", "bayesian"])
    p.add_argument("--mcmc_draws", type=int, default=None)
    p.add_argument("--mcmc_tune", type=int, default=None)
    p.add_argument("--mcmc_chains", type=int, default=None)
    p.add_argument("--mcmc_cores", type=int, default=None)
    p.add_argument("--mcmc_target_accept", type=float, default=None)
    p.add_argument("--mcmc_max_treedepth", type=int, default=None)
    p.add_argument("--rhat_threshold", type=float, default=None)
    p.add_argument("--bayesian_max_obs", type=int, default=None, help="0 = use all residual rows")
    p.add_argument("--no_demand_scenarios_json", action="store_true")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
        log.info("Config loaded from %s", args.config)
    else:
        cfg_path = find_default_config()
        if cfg_path is None:
            raise FileNotFoundError("Could not find default_config.yaml. Pass --config explicitly.")
        cfg = load_config(cfg_path)
        log.info("Config loaded from %s", cfg_path)
    cfg = apply_cli_overrides(cfg, args)
    run(config=cfg, pre_layer_result_path=args.pre_layer_result)
    sys.stdout.flush()
    sys.stderr.flush()

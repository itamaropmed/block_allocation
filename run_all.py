#!/usr/bin/env python3
"""
Run the full Gen3 Block Allocation pipeline:
  Pre-Layer  ->  Layer 1  ->  Layer 2  ->  Layer 2 validation  ->  Layer 3

Put this file in the project root, next to src/, data/, config/, and outputs/.
Then run for example:

python3 run_all.py \
  --blocks data/raw/geisinger-users_blocks.json \
  --providers data/raw/geisinger-users_providers.json \
  --cases data/raw/geisinger-users_cases.json \
  --out outputs \
  --expect_T 4 \
  --point_model xgboost \
  --sigma_model bayesian \
  --n_scenarios 200 \
  --optuna_trials 0 \
  --mcmc_draws 500 \
  --mcmc_tune 500 \
  --mcmc_chains 4 \
  --mcmc_cores 4 \
  --alpha 0.85 \
  --optimization_weeks 4 \
  --time_limit_s 180 \
  --workers 8 \
  --candidate_top_per_day 120 \
  --pareto_grid 6 \
  --make_interactions

Useful faster smoke test:

python3 run_all.py \
  --out outputs_smoke \
  --n_scenarios 20 \
  --point_model xgboost \
  --sigma_model shrinkage \
  --time_limit_s 60 \
  --pareto_grid 3 \
  --max_shap_rows 1000 \
  --no_layer3_interactions
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"


class PipelineError(RuntimeError):
    pass


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def resolve_under_root(root: Path, value: str | Path) -> Path:
    p = as_path(value)
    if p.is_absolute():
        return p
    return root / p


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_unlink_tree(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def read_text(path: Path, max_chars: Optional[int] = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    if max_chars is not None and len(text) > max_chars:
        return text[-max_chars:]
    return text


def write_json(path: Path, obj: object) -> None:
    mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=str))


def file_exists_and_nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def add_env_path(env: Dict[str, str], root: Path) -> Dict[str, str]:
    env = dict(env)
    old = env.get("PYTHONPATH", "")
    root_s = str(root)
    src_s = str(root / "src")
    parts = [root_s, src_s]
    if old:
        parts.append(old)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # Avoid oversubscription when OR-Tools/XGBoost/BLAS are also multithreaded.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def command_to_string(cmd: Sequence[str]) -> str:
    # Simple readable representation. It is not meant to be shell-perfect.
    out = []
    for x in cmd:
        s = str(x)
        if any(ch.isspace() for ch in s):
            out.append(repr(s))
        else:
            out.append(s)
    return " ".join(out)


def parse_status_from_log(log_text: str) -> str:
    """Best-effort parser for the PASS/WARN/FAIL status printed by layers."""
    patterns = [
        r"Validation\s*[:=]\s*(PASS|WARN|FAIL)",
        r"Status\s*[:=]\s*(PASS|WARN|FAIL)",
        r"\bValidation\s+\|?\s*(PASS|WARN|FAIL)\b",
        r"\bstatus\b['\"]?\s*[:=]\s*['\"]?(PASS|WARN|FAIL)",
    ]
    found: List[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, log_text, flags=re.IGNORECASE))
    if not found:
        if "Traceback (most recent call last)" in log_text or " ERROR " in log_text:
            return FAIL
        return "UNKNOWN"
    # Last status printed is usually the final one.
    return found[-1].upper()


def try_load_json_status(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(errors="replace"))
    except Exception:
        return None
    for key in ("status", "validation_status"):
        val = data.get(key)
        if isinstance(val, str) and val.upper() in {PASS, WARN, FAIL}:
            return val.upper()
    val = data.get("validation")
    if isinstance(val, dict):
        s = val.get("status")
        if isinstance(s, str) and s.upper() in {PASS, WARN, FAIL}:
            return s.upper()
    return None


def newest_subdir(parent: Path) -> Optional[Path]:
    if not parent.exists():
        return None
    dirs = [p for p in parent.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class RunAllConfig:
    project_root: Path
    python_executable: str
    config: Optional[Path]

    blocks: Path
    providers: Path
    cases: Path
    out_base: Path

    only: List[str]
    start_at: Optional[str]
    stop_after: Optional[str]
    resume: bool
    clean: bool
    dry_run: bool
    keep_going: bool
    fail_on_warn: bool
    skip_preflight: bool

    # Pre-layer
    expect_T: Optional[int]

    # Layer 1
    point_model: str
    sigma_model: str
    n_scenarios: int
    optuna_trials: int
    mcmc_draws: int
    mcmc_tune: int
    mcmc_chains: int
    mcmc_cores: int
    mcmc_target_accept: float
    mcmc_max_treedepth: int
    rhat_threshold: float
    bayesian_max_obs: int
    no_demand_scenarios_json: bool

    # Layer 2
    alpha: float
    time_limit_s: float
    workers: int
    max_slots: int
    candidate_top_per_day: int
    pareto_grid: int
    pareto_candidate_top_per_day: int
    pareto_min_utilization_fraction: float
    pareto_slack_cost_per_min: int
    noncurrent_penalty_per_min: Optional[int]
    service_line_mismatch_penalty_per_min: Optional[int]
    stability_fit_penalty_per_min: Optional[int]
    open_slot_bonus_per_min: Optional[int]
    no_pareto: bool
    no_layer2_validate: bool

    # Layer 3
    optimization_weeks: int
    selected_theme: str
    top_k_features: int
    max_shap_rows: int
    make_interactions: bool
    max_interaction_rows: int
    allow_permutation_fallback: bool
    n_waterfall_examples: int
    n_dependence_plots: int
    shap_efficiency_abs_tol_min: float
    shap_efficiency_rel_tol: float
    shap_efficiency_warn_tol_min: float
    random_seed: int

    # Runtime directories
    run_stamp: str = field(default_factory=now_stamp)

    @property
    def pre_layer_dir(self) -> Path:
        return self.out_base / "pre_layer"

    @property
    def pre_layer_result(self) -> Path:
        return self.pre_layer_dir / "prelayer_result.json"

    @property
    def layer1_dir(self) -> Path:
        return self.out_base / "layer1"

    @property
    def layer2_dir(self) -> Path:
        return self.out_base / "layer2"

    @property
    def layer3_dir(self) -> Path:
        return self.out_base / "layer3"

    @property
    def run_dir(self) -> Path:
        return self.out_base / "run_all" / self.run_stamp

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"


STAGE_ORDER = ["prelayer", "layer1", "layer2", "layer2_validate", "layer3"]
ALIASES = {
    "pre": "prelayer",
    "pre_layer": "prelayer",
    "prelayer": "prelayer",
    "l1": "layer1",
    "layer1": "layer1",
    "l2": "layer2",
    "layer2": "layer2",
    "validate_layer2": "layer2_validate",
    "layer2_validation": "layer2_validate",
    "layer2_validate": "layer2_validate",
    "l3": "layer3",
    "layer3": "layer3",
    "all": "all",
}


def normalize_stage_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key not in ALIASES:
        raise argparse.ArgumentTypeError(
            f"Unknown stage '{name}'. Use one of: all, prelayer, layer1, layer2, layer2_validate, layer3"
        )
    return ALIASES[key]


def selected_stages(only: List[str], start_at: Optional[str], stop_after: Optional[str], no_layer2_validate: bool) -> List[str]:
    if not only or "all" in only:
        stages = list(STAGE_ORDER)
    else:
        stages = [s for s in STAGE_ORDER if s in set(only)]

    if start_at:
        start_at = normalize_stage_name(start_at)
        if start_at == "all":
            start_at = STAGE_ORDER[0]
        idx = STAGE_ORDER.index(start_at)
        stages = [s for s in stages if STAGE_ORDER.index(s) >= idx]

    if stop_after:
        stop_after = normalize_stage_name(stop_after)
        if stop_after == "all":
            stop_after = STAGE_ORDER[-1]
        idx = STAGE_ORDER.index(stop_after)
        stages = [s for s in stages if STAGE_ORDER.index(s) <= idx]

    if no_layer2_validate and "layer2_validate" in stages:
        stages.remove("layer2_validate")
    return stages


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


@dataclass
class StageResult:
    stage: str
    command: List[str]
    returncode: int
    status: str
    started_at: str
    ended_at: str
    elapsed_s: float
    log_path: str
    artifacts: Dict[str, str]
    skipped: bool = False
    reason: str = ""


class PipelineRunner:
    def __init__(self, cfg: RunAllConfig) -> None:
        self.cfg = cfg
        self.results: List[StageResult] = []
        mkdir(cfg.run_dir)
        mkdir(cfg.log_dir)

    def run(self) -> int:
        cfg = self.cfg
        stages = selected_stages(cfg.only, cfg.start_at, cfg.stop_after, cfg.no_layer2_validate)

        self.banner("GEN3 BLOCK ALLOCATION — RUN ALL")
        print(f"Project root : {cfg.project_root}")
        print(f"Python       : {cfg.python_executable}")
        print(f"Output base  : {cfg.out_base}")
        print(f"Run dir      : {cfg.run_dir}")
        print(f"Stages       : {', '.join(stages)}")
        print()

        if cfg.clean and not cfg.dry_run:
            self.clean_outputs(stages)

        if not cfg.skip_preflight:
            self.preflight(stages)

        exit_code = 0
        for stage in stages:
            try:
                result = self.run_stage(stage)
                self.results.append(result)
                if result.returncode != 0 or result.status == FAIL or (cfg.fail_on_warn and result.status == WARN):
                    exit_code = result.returncode if result.returncode != 0 else 2
                    if not cfg.keep_going:
                        break
            except Exception as exc:
                exit_code = 2
                print(f"\nERROR in stage {stage}: {exc}")
                fail_result = StageResult(
                    stage=stage,
                    command=[],
                    returncode=2,
                    status=FAIL,
                    started_at=datetime.now().isoformat(timespec="seconds"),
                    ended_at=datetime.now().isoformat(timespec="seconds"),
                    elapsed_s=0.0,
                    log_path="",
                    artifacts={},
                    skipped=False,
                    reason=str(exc),
                )
                self.results.append(fail_result)
                if not cfg.keep_going:
                    break

        self.write_manifest(exit_code)
        self.print_summary(exit_code)
        return exit_code

    def banner(self, title: str) -> None:
        print("=" * 78)
        print(title)
        print("=" * 78)

    def clean_outputs(self, stages: Sequence[str]) -> None:
        """Delete layer outputs only for stages that will be run."""
        targets: List[Path] = []
        if "prelayer" in stages:
            targets.append(self.cfg.pre_layer_dir)
        if "layer1" in stages:
            targets.append(self.cfg.layer1_dir)
        if "layer2" in stages:
            targets.append(self.cfg.layer2_dir)
        if "layer3" in stages:
            targets.append(self.cfg.layer3_dir)
        # Do not delete this run directory if it is under outputs/run_all.
        targets = [p for p in targets if p.resolve() != self.cfg.run_dir.resolve()]
        for p in targets:
            print(f"Cleaning: {p}")
            safe_unlink_tree(p)
        mkdir(self.cfg.log_dir)

    def preflight(self, stages: Sequence[str]) -> None:
        cfg = self.cfg
        self.banner("PREFLIGHT")
        errors: List[str] = []

        required_src = [
            cfg.project_root / "src" / "pre_layer" / "run_prelayer.py",
            cfg.project_root / "src" / "layer1" / "run_layer1.py",
            cfg.project_root / "src" / "layer2" / "run_layer2.py",
            cfg.project_root / "src" / "layer3" / "run_layer3.py",
        ]
        for p in required_src:
            if not p.exists():
                errors.append(f"Missing source file: {p}")

        for name, p in [("blocks", cfg.blocks), ("providers", cfg.providers), ("cases", cfg.cases)]:
            if name in {"blocks", "providers", "cases"} and not p.exists():
                errors.append(f"Missing {name} input: {p}")

        if cfg.config is not None and not cfg.config.exists():
            print(f"Config path not found, continuing without --config: {cfg.config}")
            cfg.config = None

        if "layer1" in stages:
            if cfg.point_model == "xgboost":
                self.check_python_import("xgboost", required=False)
            if cfg.sigma_model == "bayesian":
                self.check_python_import("pymc", required=False)
                self.check_python_import("arviz", required=False)

        if "layer2" in stages:
            self.check_python_import("ortools", required=False)

        if "layer3" in stages:
            self.check_python_import("shap", required=False)

        if errors:
            for e in errors:
                print(f"ERROR: {e}")
            raise PipelineError("Preflight failed. Fix the missing paths above, or pass the correct --blocks/--providers/--cases/--project_root.")
        print("Preflight completed.\n")

    def check_python_import(self, package: str, required: bool = False) -> bool:
        cmd = [self.cfg.python_executable, "-c", f"import {package}; print(getattr({package}, '__version__', 'OK'))"]
        try:
            proc = subprocess.run(cmd, cwd=self.cfg.project_root, text=True, capture_output=True, timeout=20)
            if proc.returncode == 0:
                print(f"Import OK: {package} {proc.stdout.strip()}")
                return True
            print(f"Import warning: {package} not importable: {proc.stderr.strip() or proc.stdout.strip()}")
            if required:
                raise PipelineError(f"Required package is missing: {package}")
            return False
        except Exception as exc:
            print(f"Import warning: {package}: {exc}")
            if required:
                raise
            return False

    def run_stage(self, stage: str) -> StageResult:
        if stage == "prelayer":
            return self.run_prelayer()
        if stage == "layer1":
            return self.run_layer1()
        if stage == "layer2":
            return self.run_layer2()
        if stage == "layer2_validate":
            return self.run_layer2_validate()
        if stage == "layer3":
            return self.run_layer3()
        raise PipelineError(f"Unknown stage: {stage}")

    def maybe_skip(self, stage: str, required: Sequence[Path]) -> Optional[StageResult]:
        cfg = self.cfg
        if not cfg.resume:
            return None
        missing = [p for p in required if not file_exists_and_nonempty(p)]
        if missing:
            return None
        print(f"\nSkipping {stage}: --resume and required artifacts already exist.")
        return StageResult(
            stage=stage,
            command=[],
            returncode=0,
            status=SKIP,
            started_at=datetime.now().isoformat(timespec="seconds"),
            ended_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_s=0.0,
            log_path="",
            artifacts={str(i): str(p) for i, p in enumerate(required)},
            skipped=True,
            reason="resume artifacts exist",
        )

    def common_config_args(self) -> List[str]:
        if self.cfg.config is not None and self.cfg.config.exists():
            return ["--config", str(self.cfg.config)]
        return []

    def run_prelayer(self) -> StageResult:
        cfg = self.cfg
        skip = self.maybe_skip("prelayer", [cfg.pre_layer_result])
        if skip:
            return skip
        cmd = [
            cfg.python_executable,
            "src/pre_layer/run_prelayer.py",
            *self.common_config_args(),
            "--blocks", str(cfg.blocks),
            "--providers", str(cfg.providers),
            "--cases", str(cfg.cases),
            "--out", str(cfg.out_base),
        ]
        if cfg.expect_T is not None:
            cmd += ["--expect_T", str(cfg.expect_T)]
        return self.execute("prelayer", cmd, expected_artifacts={"prelayer_result": cfg.pre_layer_result})

    def run_layer1(self) -> StageResult:
        cfg = self.cfg
        required = [cfg.layer1_dir / "point_forecasts.csv", cfg.layer1_dir / "scenarios_long.csv"]
        skip = self.maybe_skip("layer1", required)
        if skip:
            return skip
        self.ensure_upstream("Pre-Layer result", cfg.pre_layer_result)
        cmd = [
            cfg.python_executable,
            "src/layer1/run_layer1.py",
            *self.common_config_args(),
            "--blocks", str(cfg.blocks),
            "--providers", str(cfg.providers),
            "--cases", str(cfg.cases),
            "--out", str(cfg.out_base),
            "--pre_layer_result", str(cfg.pre_layer_result),
            "--n_scenarios", str(cfg.n_scenarios),
            "--optuna_trials", str(cfg.optuna_trials),
            "--point_model", cfg.point_model,
            "--sigma_model", cfg.sigma_model,
            "--rhat_threshold", str(cfg.rhat_threshold),
        ]
        if cfg.sigma_model == "bayesian":
            cmd += [
                "--mcmc_draws", str(cfg.mcmc_draws),
                "--mcmc_tune", str(cfg.mcmc_tune),
                "--mcmc_chains", str(cfg.mcmc_chains),
                "--mcmc_cores", str(cfg.mcmc_cores),
                "--mcmc_target_accept", str(cfg.mcmc_target_accept),
                "--mcmc_max_treedepth", str(cfg.mcmc_max_treedepth),
                "--bayesian_max_obs", str(cfg.bayesian_max_obs),
            ]
        if cfg.no_demand_scenarios_json:
            cmd += ["--no_demand_scenarios_json"]
        return self.execute(
            "layer1",
            cmd,
            expected_artifacts={
                "features": cfg.layer1_dir / "features.csv",
                "point_forecasts": cfg.layer1_dir / "point_forecasts.csv",
                "scenarios_long": cfg.layer1_dir / "scenarios_long.csv",
                "sigma_estimates": cfg.layer1_dir / "hierarchical_sigma_estimates.csv",
                "model_metrics": cfg.layer1_dir / "model_metrics.json",
            },
        )

    def run_layer2(self) -> StageResult:
        cfg = self.cfg
        required = [cfg.layer2_dir / "provider_day_coverage.csv", cfg.layer2_dir / "pareto_frontier.csv"]
        skip = self.maybe_skip("layer2", required)
        if skip:
            return skip
        self.ensure_upstream("Layer 1 scenarios", cfg.layer1_dir / "scenarios_long.csv")
        cmd = [
            cfg.python_executable,
            "src/layer2/run_layer2.py",
            *self.common_config_args(),
            "--pre_layer_result", str(cfg.pre_layer_result),
            "--layer1_dir", str(cfg.layer1_dir),
            "--blocks", str(cfg.blocks),
            "--providers", str(cfg.providers),
            "--out", str(cfg.layer2_dir),
            "--alpha", str(cfg.alpha),
            "--time_limit_s", str(cfg.time_limit_s),
            "--workers", str(cfg.workers),
            "--max_slots", str(cfg.max_slots),
            "--candidate_top_per_day", str(cfg.candidate_top_per_day),
            "--pareto_grid", str(cfg.pareto_grid),
            "--pareto_candidate_top_per_day", str(cfg.pareto_candidate_top_per_day),
            "--pareto_min_utilization_fraction", str(cfg.pareto_min_utilization_fraction),
            "--pareto_slack_cost_per_min", str(cfg.pareto_slack_cost_per_min),
            "--random_seed", str(cfg.random_seed),
        ]
        optional_ints = [
            ("--noncurrent_penalty_per_min", cfg.noncurrent_penalty_per_min),
            ("--service_line_mismatch_penalty_per_min", cfg.service_line_mismatch_penalty_per_min),
            ("--stability_fit_penalty_per_min", cfg.stability_fit_penalty_per_min),
            ("--open_slot_bonus_per_min", cfg.open_slot_bonus_per_min),
        ]
        for flag, val in optional_ints:
            if val is not None:
                cmd += [flag, str(val)]
        if cfg.no_pareto:
            cmd += ["--no_pareto"]
        return self.execute(
            "layer2",
            cmd,
            expected_artifacts={
                "layer2_result": cfg.layer2_dir / "layer2_result.json",
                "schedule_by_week": cfg.layer2_dir / "schedule_by_week.csv",
                "provider_day_coverage": cfg.layer2_dir / "provider_day_coverage.csv",
                "pareto_all": cfg.layer2_dir / "pareto_all.csv",
                "pareto_frontier": cfg.layer2_dir / "pareto_frontier.csv",
                "selected_plan_schedule": cfg.layer2_dir / "plans" / "utilization_first_selected" / "schedule_by_week.csv",
            },
            json_status_path=cfg.layer2_dir / "layer2_result.json",
        )

    def run_layer2_validate(self) -> StageResult:
        cfg = self.cfg
        self.ensure_upstream("Layer 2 output", cfg.layer2_dir / "layer2_result.json")
        cmd = [
            cfg.python_executable,
            "src/layer2/validate_layer2_outputs.py",
            "--layer2_dir", str(cfg.layer2_dir),
        ]
        return self.execute(
            "layer2_validate",
            cmd,
            expected_artifacts={"standalone_validation": cfg.layer2_dir / "validation_report_standalone.json"},
            json_status_path=cfg.layer2_dir / "validation_report_standalone.json",
        )

    def run_layer3(self) -> StageResult:
        cfg = self.cfg
        if cfg.resume:
            latest = newest_subdir(cfg.layer3_dir)
            if latest and file_exists_and_nonempty(latest / "SUMMARY.md"):
                return StageResult(
                    stage="layer3",
                    command=[],
                    returncode=0,
                    status=SKIP,
                    started_at=datetime.now().isoformat(timespec="seconds"),
                    ended_at=datetime.now().isoformat(timespec="seconds"),
                    elapsed_s=0.0,
                    log_path="",
                    artifacts={"latest_layer3_run": str(latest), "summary": str(latest / "SUMMARY.md")},
                    skipped=True,
                    reason="resume artifacts exist",
                )
        self.ensure_upstream("Layer 2 coverage", cfg.layer2_dir / "provider_day_coverage.csv")
        cmd = [
            cfg.python_executable,
            "src/layer3/run_layer3.py",
            "--layer1_dir", str(cfg.layer1_dir),
            "--layer2_dir", str(cfg.layer2_dir),
            "--pre_layer_result", str(cfg.pre_layer_result),
            "--out", str(cfg.layer3_dir),
            "--alpha", str(cfg.alpha),
            "--optimization_weeks", str(cfg.optimization_weeks),
            "--selected_theme", cfg.selected_theme,
            "--top_k_features", str(cfg.top_k_features),
            "--max_shap_rows", str(cfg.max_shap_rows),
            "--max_interaction_rows", str(cfg.max_interaction_rows),
            "--random_seed", str(cfg.random_seed),
            "--shap_efficiency_abs_tol_min", str(cfg.shap_efficiency_abs_tol_min),
            "--shap_efficiency_rel_tol", str(cfg.shap_efficiency_rel_tol),
            "--shap_efficiency_warn_tol_min", str(cfg.shap_efficiency_warn_tol_min),
            "--n_waterfall_examples", str(cfg.n_waterfall_examples),
            "--n_dependence_plots", str(cfg.n_dependence_plots),
        ]
        if cfg.make_interactions:
            cmd += ["--make_interactions"]
        if cfg.allow_permutation_fallback:
            cmd += ["--allow_permutation_fallback"]
        before = newest_subdir(cfg.layer3_dir)
        result = self.execute("layer3", cmd, expected_artifacts={}, json_status_path=None)
        after = newest_subdir(cfg.layer3_dir)
        if after is not None and after != before:
            result.artifacts.update({
                "layer3_run_dir": str(after),
                "summary": str(after / "SUMMARY.md"),
                "recommendations": str(after / "recommendations.csv"),
                "explanations": str(after / "explanations.json"),
            })
            # Layer 3 status is usually only in log, but keep common artifacts visible.
        return result

    def ensure_upstream(self, label: str, path: Path) -> None:
        if not file_exists_and_nonempty(path):
            raise PipelineError(f"Missing required upstream artifact: {label}: {path}")

    def execute(
        self,
        stage: str,
        cmd: List[str],
        expected_artifacts: Dict[str, Path],
        json_status_path: Optional[Path] = None,
    ) -> StageResult:
        cfg = self.cfg
        log_path = cfg.log_dir / f"{stage}.log"
        start_dt = datetime.now()
        started = start_dt.isoformat(timespec="seconds")

        print("\n" + "─" * 78)
        print(f"RUNNING {stage}")
        print("Command:")
        print(command_to_string(cmd))
        print(f"Log: {log_path}")
        print("─" * 78)

        if cfg.dry_run:
            log_path.write_text(command_to_string(cmd) + "\n")
            return StageResult(
                stage=stage,
                command=cmd,
                returncode=0,
                status=SKIP,
                started_at=started,
                ended_at=datetime.now().isoformat(timespec="seconds"),
                elapsed_s=0.0,
                log_path=str(log_path),
                artifacts={k: str(v) for k, v in expected_artifacts.items()},
                skipped=True,
                reason="dry run",
            )

        env = add_env_path(os.environ, cfg.project_root)
        mkdir(log_path.parent)
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"# Stage: {stage}\n")
            log_file.write(f"# Started: {started}\n")
            log_file.write(f"# Command: {command_to_string(cmd)}\n\n")
            log_file.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=cfg.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_file.write(line)
            returncode = proc.wait()
            ended = datetime.now().isoformat(timespec="seconds")
            elapsed = (datetime.now() - start_dt).total_seconds()
            log_file.write(f"\n# Ended: {ended}\n")
            log_file.write(f"# Return code: {returncode}\n")
            log_file.write(f"# Elapsed seconds: {elapsed:.2f}\n")

        log_text = read_text(log_path, max_chars=200_000)
        status = try_load_json_status(json_status_path) if json_status_path is not None else None
        if status is None:
            status = parse_status_from_log(log_text)
        if returncode != 0:
            status = FAIL

        artifacts = {k: str(v) for k, v in expected_artifacts.items() if v.exists()}
        missing = {k: str(v) for k, v in expected_artifacts.items() if not v.exists()}
        if missing:
            print(f"\nMissing expected artifacts for {stage}:")
            for k, v in missing.items():
                print(f"  - {k}: {v}")
            if status == PASS:
                status = WARN

        result = StageResult(
            stage=stage,
            command=cmd,
            returncode=returncode,
            status=status,
            started_at=started,
            ended_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_s=elapsed,
            log_path=str(log_path),
            artifacts=artifacts,
        )
        print(f"\nFinished {stage}: returncode={returncode}, status={status}, elapsed={elapsed:.1f}s")
        return result

    def write_manifest(self, exit_code: int) -> None:
        cfg = self.cfg
        manifest = {
            "run_stamp": cfg.run_stamp,
            "exit_code": exit_code,
            "project_root": str(cfg.project_root),
            "python_executable": cfg.python_executable,
            "platform": platform.platform(),
            "config": str(cfg.config) if cfg.config else None,
            "inputs": {
                "blocks": str(cfg.blocks),
                "providers": str(cfg.providers),
                "cases": str(cfg.cases),
            },
            "outputs": {
                "out_base": str(cfg.out_base),
                "pre_layer_result": str(cfg.pre_layer_result),
                "layer1_dir": str(cfg.layer1_dir),
                "layer2_dir": str(cfg.layer2_dir),
                "layer3_dir": str(cfg.layer3_dir),
                "run_dir": str(cfg.run_dir),
                "log_dir": str(cfg.log_dir),
            },
            "parameters": self.serializable_config(),
            "results": [asdict(r) for r in self.results],
        }
        write_json(cfg.run_dir / "manifest.json", manifest)
        latest = cfg.out_base / "run_all" / "latest_manifest.json"
        write_json(latest, manifest)

    def serializable_config(self) -> Dict[str, object]:
        data = asdict(self.cfg)
        for k, v in list(data.items()):
            if isinstance(v, Path):
                data[k] = str(v)
        return data

    def print_summary(self, exit_code: int) -> None:
        cfg = self.cfg
        self.banner("RUN ALL SUMMARY")
        for r in self.results:
            elapsed = f"{r.elapsed_s:.1f}s" if not r.skipped else "skipped"
            print(f"{r.stage:<16} status={r.status:<7} returncode={r.returncode:<3} elapsed={elapsed:<10} log={r.log_path}")
            if r.reason:
                print(f"  reason: {r.reason}")
        print("─" * 78)
        print(f"Manifest : {cfg.run_dir / 'manifest.json'}")
        print(f"Latest   : {cfg.out_base / 'run_all' / 'latest_manifest.json'}")
        print(f"Exit code: {exit_code}")
        print("=" * 78)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all layers of the Gen3 Block Allocation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General paths
    p.add_argument("--project_root", default=".", help="Project root containing src/, data/, config/.")
    p.add_argument("--python", dest="python_executable", default=sys.executable, help="Python executable to use for child processes.")
    p.add_argument("--config", default="config/default_config.yaml", help="Optional config YAML. If missing, it is not passed.")
    p.add_argument("--blocks", default="data/raw/geisinger-users_blocks.json")
    p.add_argument("--providers", default="data/raw/geisinger-users_providers.json")
    p.add_argument("--cases", default="data/raw/geisinger-users_cases.json")
    p.add_argument("--out", default="outputs", help="Base output directory. Pre-layer/layer1/layer2/layer3 go under this.")

    # Stage control
    p.add_argument("--only", nargs="+", type=normalize_stage_name, default=["all"], help="Stages to run: all, prelayer, layer1, layer2, layer2_validate, layer3.")
    p.add_argument("--start_at", default=None, help="Start from this stage.")
    p.add_argument("--stop_after", default=None, help="Stop after this stage.")
    p.add_argument("--resume", action="store_true", help="Skip a stage if its required artifacts already exist.")
    p.add_argument("--clean", action="store_true", help="Delete output folders for the selected stages before running.")
    p.add_argument("--dry_run", action="store_true", help="Print commands but do not execute them.")
    p.add_argument("--keep_going", action="store_true", help="Continue to later stages after a failure.")
    p.add_argument("--fail_on_warn", action="store_true", help="Exit nonzero when a stage status is WARN.")
    p.add_argument("--skip_preflight", action="store_true", help="Skip path/import checks.")

    # Pre-layer
    p.add_argument("--expect_T", type=int, default=4, help="Expected T*. Use -1 to disable this validation argument.")

    # Layer 1
    p.add_argument("--point_model", choices=["auto", "xgboost", "ridge", "gradient_boosting"], default="xgboost")
    p.add_argument("--sigma_model", choices=["shrinkage", "bayesian"], default="bayesian")
    p.add_argument("--n_scenarios", type=int, default=200)
    p.add_argument("--optuna_trials", type=int, default=0)
    p.add_argument("--mcmc_draws", type=int, default=500)
    p.add_argument("--mcmc_tune", type=int, default=500)
    p.add_argument("--mcmc_chains", type=int, default=4)
    p.add_argument("--mcmc_cores", type=int, default=4)
    p.add_argument("--mcmc_target_accept", type=float, default=0.95)
    p.add_argument("--mcmc_max_treedepth", type=int, default=12)
    p.add_argument("--rhat_threshold", type=float, default=1.05)
    p.add_argument("--bayesian_max_obs", type=int, default=0, help="0 means use all residual rows.")
    p.add_argument("--write_demand_scenarios_json", action="store_true", help="Also write the large demand_scenarios.json if Layer 1 supports it.")

    # Layer 2
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--time_limit_s", type=float, default=180.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_slots", type=int, default=0, help="0 means no slot limit in Layer 2.")
    p.add_argument("--candidate_top_per_day", type=int, default=120)
    p.add_argument("--pareto_grid", type=int, default=6)
    p.add_argument("--pareto_candidate_top_per_day", type=int, default=120, help="0 means inherit Layer 2 adaptive candidate depth.")
    p.add_argument("--pareto_min_utilization_fraction", type=float, default=0.95)
    p.add_argument("--pareto_slack_cost_per_min", type=int, default=1)
    p.add_argument("--noncurrent_penalty_per_min", type=int, default=None)
    p.add_argument("--service_line_mismatch_penalty_per_min", type=int, default=None)
    p.add_argument("--stability_fit_penalty_per_min", type=int, default=None)
    p.add_argument("--open_slot_bonus_per_min", type=int, default=None)
    p.add_argument("--no_pareto", action="store_true")
    p.add_argument("--no_layer2_validate", action="store_true")

    # Layer 3
    p.add_argument("--optimization_weeks", type=int, default=4)
    p.add_argument("--selected_theme", default="utilization_first")
    p.add_argument("--top_k_features", type=int, default=12)
    p.add_argument("--max_shap_rows", type=int, default=5000)
    p.add_argument("--make_interactions", action="store_true", help="Compute SHAP interactions. This can be slower.")
    p.add_argument("--no_layer3_interactions", action="store_true", help="Alias to keep interactions off explicitly.")
    p.add_argument("--max_interaction_rows", type=int, default=1000)
    p.add_argument("--allow_permutation_fallback", action="store_true")
    p.add_argument("--n_waterfall_examples", type=int, default=3)
    p.add_argument("--n_dependence_plots", type=int, default=4)
    p.add_argument("--shap_efficiency_abs_tol_min", type=float, default=1e-2)
    p.add_argument("--shap_efficiency_rel_tol", type=float, default=1e-6)
    p.add_argument("--shap_efficiency_warn_tol_min", type=float, default=1.0)
    p.add_argument("--random_seed", type=int, default=42)

    return p


def make_config(args: argparse.Namespace) -> RunAllConfig:
    root = as_path(args.project_root).resolve()
    out_base = resolve_under_root(root, args.out)
    config = resolve_under_root(root, args.config) if args.config else None
    if args.expect_T is not None and args.expect_T < 0:
        expect_T = None
    else:
        expect_T = args.expect_T

    make_interactions = bool(args.make_interactions) and not bool(args.no_layer3_interactions)

    return RunAllConfig(
        project_root=root,
        python_executable=args.python_executable,
        config=config,
        blocks=resolve_under_root(root, args.blocks),
        providers=resolve_under_root(root, args.providers),
        cases=resolve_under_root(root, args.cases),
        out_base=out_base,
        only=args.only,
        start_at=args.start_at,
        stop_after=args.stop_after,
        resume=args.resume,
        clean=args.clean,
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        fail_on_warn=args.fail_on_warn,
        skip_preflight=args.skip_preflight,
        expect_T=expect_T,
        point_model=args.point_model,
        sigma_model=args.sigma_model,
        n_scenarios=args.n_scenarios,
        optuna_trials=args.optuna_trials,
        mcmc_draws=args.mcmc_draws,
        mcmc_tune=args.mcmc_tune,
        mcmc_chains=args.mcmc_chains,
        mcmc_cores=args.mcmc_cores,
        mcmc_target_accept=args.mcmc_target_accept,
        mcmc_max_treedepth=args.mcmc_max_treedepth,
        rhat_threshold=args.rhat_threshold,
        bayesian_max_obs=args.bayesian_max_obs,
        no_demand_scenarios_json=not args.write_demand_scenarios_json,
        alpha=args.alpha,
        time_limit_s=args.time_limit_s,
        workers=args.workers,
        max_slots=args.max_slots,
        candidate_top_per_day=args.candidate_top_per_day,
        pareto_grid=args.pareto_grid,
        pareto_candidate_top_per_day=args.pareto_candidate_top_per_day,
        pareto_min_utilization_fraction=args.pareto_min_utilization_fraction,
        pareto_slack_cost_per_min=args.pareto_slack_cost_per_min,
        noncurrent_penalty_per_min=args.noncurrent_penalty_per_min,
        service_line_mismatch_penalty_per_min=args.service_line_mismatch_penalty_per_min,
        stability_fit_penalty_per_min=args.stability_fit_penalty_per_min,
        open_slot_bonus_per_min=args.open_slot_bonus_per_min,
        no_pareto=args.no_pareto,
        no_layer2_validate=args.no_layer2_validate,
        optimization_weeks=args.optimization_weeks,
        selected_theme=args.selected_theme,
        top_k_features=args.top_k_features,
        max_shap_rows=args.max_shap_rows,
        make_interactions=make_interactions,
        max_interaction_rows=args.max_interaction_rows,
        allow_permutation_fallback=args.allow_permutation_fallback,
        n_waterfall_examples=args.n_waterfall_examples,
        n_dependence_plots=args.n_dependence_plots,
        shap_efficiency_abs_tol_min=args.shap_efficiency_abs_tol_min,
        shap_efficiency_rel_tol=args.shap_efficiency_rel_tol,
        shap_efficiency_warn_tol_min=args.shap_efficiency_warn_tol_min,
        random_seed=args.random_seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = make_config(args)
    runner = PipelineRunner(cfg)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())

"""
Layer 2 — Pareto Frontier
=========================

Fixed to work with the current optimizer.py WITHOUT changing optimizer.py.

Important:
- optimizer.py exports greedy_allocate(...)
- greedy_allocate internally uses the MIP when PuLP is installed.
- This file does NOT import solve_mip_allocation.
- This file does NOT require src.common.io.
- This file catches CBC/PuLP crashes and retries with smaller MIP block caps.
- Epsilon-bound quick solves are OFF by default, because CBC crashed there.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd


try:
    from optimizer import greedy_allocate
except ImportError:
    from .optimizer import greedy_allocate


log = logging.getLogger(__name__)


DEFAULT_THEMES = [
    "conservative",
    "continuity_first",
    "balanced",
    "utilization_first",
]


def _save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _config_with_block_cap(config: dict, block_cap: int) -> dict:
    cfg = copy.deepcopy(config)
    cfg.setdefault("layer2", {})
    cfg["layer2"]["max_template_blocks_for_mip"] = int(block_cap)
    return cfg


def _get_retry_block_caps(config: dict) -> list[int]:
    """
    CBC on Mac/PuLP is crashing for very large models.

    Your log:
      900 blocks -> 237,600 eligible pairs -> CBC crash
      700 blocks -> 170,800 eligible pairs -> CBC crash
      500 blocks -> 101,500 eligible pairs -> still very large

    So Pareto should start smaller by default.
    This does NOT change optimizer.py.
    """

    layer2 = config.get("layer2", {})

    configured = layer2.get("mip_retry_block_caps")

    if configured:
        caps = [int(x) for x in configured]
    else:
        caps = [
            250,
            150,
            100,
            60,
        ]

    out: list[int] = []
    for c in caps:
        c = int(c)
        if c > 0 and c not in out:
            out.append(c)

    return out


def _call_optimizer_with_recovery(
    *,
    prelayer: dict,
    providers: pd.DataFrame,
    scenarios: pd.DataFrame,
    early_release: pd.DataFrame,
    config: dict,
    theme: str,
    eps_O1: Optional[float],
    eps_O3: Optional[float],
    time_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Call the existing optimizer entry point.

    If CBC crashes from a too-large MIP, retry the same optimizer with smaller
    max_template_blocks_for_mip values. This keeps optimizer.py unchanged.
    """

    errors: list[str] = []

    for block_cap in _get_retry_block_caps(config):
        cfg_try = _config_with_block_cap(config, block_cap)

        try:
            log.info(
                "Calling optimizer: theme=%s block_cap=%d eps_O1=%s eps_O3=%s",
                theme,
                block_cap,
                eps_O1,
                eps_O3,
            )

            assignments, coverage, meta = greedy_allocate(
                prelayer=prelayer,
                providers=providers,
                scenarios=scenarios,
                early_release=early_release,
                config=cfg_try,
                theme=theme,
                eps_O1=eps_O1,
                eps_O3=eps_O3,
                time_limit=time_limit,
            )

            meta = dict(meta)
            meta["mip_block_cap_used"] = block_cap
            meta["recovery_errors_before_success"] = errors

            return assignments, coverage, meta

        except Exception as e:
            err = f"block_cap={block_cap}: {type(e).__name__}: {e}"
            errors.append(err)
            log.exception(
                "Optimizer failed for theme=%s with block_cap=%d. Retrying smaller model if possible.",
                theme,
                block_cap,
            )

    # Do not crash the whole run. Save failed metadata for inspection.
    failed_meta = {
        "theme": theme,
        "status": "Failed",
        "solve_time_s": 0.0,
        "objective_value": 0.0,
        "O1_deviation_min": 0.0,
        "O3_changes": 0.0,
        "changed_blocks": 0,
        "goals_met": 0,
        "goals_total": 0,
        "assigned_blocks": 0,
        "open_blocks": 0,
        "total_blocks": 0,
        "eps_O1": eps_O1,
        "eps_O3": eps_O3,
        "optimizer_errors": errors,
    }

    return pd.DataFrame(), pd.DataFrame(), failed_meta


def _enrich_meta_from_outputs(
    meta: dict[str, Any],
    assignments: pd.DataFrame,
    coverage: pd.DataFrame,
    theme: str,
    eps_O1: Optional[float],
    eps_O3: Optional[float],
) -> dict[str, Any]:
    meta = dict(meta)

    meta["theme"] = meta.get("theme", theme)
    meta["eps_O1_applied"] = eps_O1
    meta["eps_O3_applied"] = eps_O3

    if assignments is not None and not assignments.empty:
        meta["total_blocks"] = int(len(assignments))

        if "assigned_provider_id" in assignments.columns:
            assigned = assignments["assigned_provider_id"].astype(str).ne("OPEN")
            meta["assigned_blocks"] = int(assigned.sum())
            meta["open_blocks"] = int((~assigned).sum())
        else:
            meta["assigned_blocks"] = 0
            meta["open_blocks"] = int(len(assignments))

        if "changed" in assignments.columns:
            meta["changed_blocks"] = int(
                assignments["changed"].fillna(False).astype(bool).sum()
            )
        else:
            meta["changed_blocks"] = _safe_int(meta.get("changed_blocks", 0))
    else:
        meta["total_blocks"] = _safe_int(meta.get("total_blocks", 0))
        meta["assigned_blocks"] = _safe_int(meta.get("assigned_blocks", 0))
        meta["open_blocks"] = _safe_int(meta.get("open_blocks", 0))
        meta["changed_blocks"] = _safe_int(meta.get("changed_blocks", 0))

    if coverage is not None and not coverage.empty:
        if "meets_alpha" in coverage.columns:
            meta["goals_met"] = int(
                coverage["meets_alpha"].fillna(False).astype(bool).sum()
            )

        if "required_min_alpha" in coverage.columns:
            required = pd.to_numeric(
                coverage["required_min_alpha"],
                errors="coerce",
            ).fillna(0.0)
            meta["goals_total"] = int((required > 0).sum())

        if "allocated_min" in coverage.columns:
            allocated = pd.to_numeric(
                coverage["allocated_min"],
                errors="coerce",
            ).fillna(0.0)
            meta["total_allocated_min"] = round(float(allocated.sum()), 2)
            meta["total_allocated_hrs"] = round(meta["total_allocated_min"] / 60.0, 2)

        if "slack_min" in coverage.columns:
            slack = pd.to_numeric(
                coverage["slack_min"],
                errors="coerce",
            ).fillna(0.0)
            meta["max_slack_min"] = round(float(slack.max()), 2)

    meta["goals_met"] = _safe_int(meta.get("goals_met", 0))
    meta["goals_total"] = _safe_int(meta.get("goals_total", 0))
    meta["O1_deviation_min"] = _safe_float(meta.get("O1_deviation_min", 0.0))
    meta["O3_changes"] = _safe_float(meta.get("O3_changes", meta.get("changed_blocks", 0.0)))
    meta["objective_value"] = _safe_float(meta.get("objective_value", 0.0))
    meta["solve_time_s"] = _safe_float(meta.get("solve_time_s", 0.0))

    return meta


def _compute_epsilon_bounds(
    prelayer: dict,
    providers: pd.DataFrame,
    scenarios: pd.DataFrame,
    early_release: pd.DataFrame,
    config: dict,
    time_limit: int,
) -> tuple[Optional[float], Optional[float]]:
    """
    Epsilon quick solves are disabled by default.

    Your crash happened inside this step, while solving the quick
    utilization_first model. So by default we skip it.
    """

    use_epsilon = bool(config.get("layer2", {}).get("use_epsilon_bounds", False))

    if not use_epsilon:
        log.info("Skipping epsilon-bound quick solves: use_epsilon_bounds=False.")
        return None, None

    quick_limit = int(
        config.get("layer2", {}).get("epsilon_time_limit_s", min(time_limit, 60))
    )

    log.info("Computing epsilon bounds with quick solves.")

    try:
        _, _, meta_o1 = _call_optimizer_with_recovery(
            prelayer=prelayer,
            providers=providers,
            scenarios=scenarios,
            early_release=early_release,
            config=config,
            theme="utilization_first",
            eps_O1=None,
            eps_O3=None,
            time_limit=quick_limit,
        )

        _, _, meta_o3 = _call_optimizer_with_recovery(
            prelayer=prelayer,
            providers=providers,
            scenarios=scenarios,
            early_release=early_release,
            config=config,
            theme="conservative",
            eps_O1=None,
            eps_O3=None,
            time_limit=quick_limit,
        )

        if meta_o1.get("status") == "Failed" or meta_o3.get("status") == "Failed":
            log.warning("Epsilon quick solves failed. Continuing without eps constraints.")
            return None, None

        o1_star = _safe_float(meta_o1.get("O1_deviation_min", 0.0))
        o3_star = _safe_float(meta_o3.get("O3_changes", 0.0))

        eps_o1_mult = float(config.get("layer2", {}).get("eps_O1_multiplier", 2.50))
        eps_o3_mult = float(config.get("layer2", {}).get("eps_O3_multiplier", 1.50))

        eps_o1_add = float(config.get("layer2", {}).get("eps_O1_add_min", 60.0))
        eps_o3_add = float(config.get("layer2", {}).get("eps_O3_add", 5.0))

        eps_o1 = max(o1_star * eps_o1_mult, o1_star + eps_o1_add)
        eps_o3 = max(o3_star * eps_o3_mult, o3_star + eps_o3_add)

        log.info(
            "Epsilon bounds: O1*=%.2f, O3*=%.2f, eps_O1=%.2f, eps_O3=%.2f",
            o1_star,
            o3_star,
            eps_o1,
            eps_o3,
        )

        return eps_o1, eps_o3

    except Exception:
        log.exception("Epsilon-bound computation crashed. Continuing without eps constraints.")
        return None, None


def _theme_eps(
    theme: str,
    eps_O1: Optional[float],
    eps_O3: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    if eps_O1 is None and eps_O3 is None:
        return None, None

    if theme == "conservative":
        return eps_O1 * 1.50 if eps_O1 is not None else None, None

    if theme == "continuity_first":
        return eps_O1, None

    if theme == "balanced":
        return None, eps_O3 * 1.50 if eps_O3 is not None else None

    if theme == "utilization_first":
        return None, eps_O3 * 2.50 if eps_O3 is not None else None

    return None, None


def _dominance_flags(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        out = frontier.copy()
        out["dominated"] = []
        return out

    rows = frontier.to_dict("records")
    dominated: list[bool] = []

    for i, a in enumerate(rows):
        is_dom = False

        for j, b in enumerate(rows):
            if i == j:
                continue

            better_or_equal = (
                _safe_float(b.get("O1_deviation_min"), 1e18)
                <= _safe_float(a.get("O1_deviation_min"), 1e18)
                and _safe_float(b.get("O3_changes"), 1e18)
                <= _safe_float(a.get("O3_changes"), 1e18)
                and _safe_int(b.get("goals_met"), -1)
                >= _safe_int(a.get("goals_met"), -1)
                and _safe_int(b.get("assigned_blocks"), -1)
                >= _safe_int(a.get("assigned_blocks"), -1)
            )

            strictly_better = (
                _safe_float(b.get("O1_deviation_min"), 1e18)
                < _safe_float(a.get("O1_deviation_min"), 1e18)
                or _safe_float(b.get("O3_changes"), 1e18)
                < _safe_float(a.get("O3_changes"), 1e18)
                or _safe_int(b.get("goals_met"), -1)
                > _safe_int(a.get("goals_met"), -1)
                or _safe_int(b.get("assigned_blocks"), -1)
                > _safe_int(a.get("assigned_blocks"), -1)
            )

            if better_or_equal and strictly_better:
                is_dom = True
                break

        dominated.append(is_dom)

    out = frontier.copy()
    out["dominated"] = dominated
    return out


def _choose_recommended(frontier: pd.DataFrame) -> dict[str, Any]:
    if frontier.empty:
        return {}

    f = frontier.copy()

    f["status"] = f.get("status", "").astype(str)
    f["solver_ok"] = f["status"].isin(
        [
            "Optimal",
            "Feasible",
            "Greedy-Fallback",
            "WarmStartOnly",
        ]
    )

    f["non_empty"] = (
        pd.to_numeric(f.get("assigned_blocks", 0), errors="coerce")
        .fillna(0)
        .astype(int)
        > 0
    )

    f["all_goals_met"] = (
        pd.to_numeric(f.get("goals_met", 0), errors="coerce").fillna(0)
        >= pd.to_numeric(f.get("goals_total", 0), errors="coerce").fillna(0)
    )

    candidates = f[
        (~f["dominated"].fillna(False).astype(bool))
        & f["solver_ok"]
        & f["non_empty"]
    ].copy()

    if candidates.empty:
        candidates = f[f["solver_ok"] & f["non_empty"]].copy()

    if candidates.empty:
        candidates = f.copy()

    candidates = candidates.sort_values(
        [
            "all_goals_met",
            "goals_met",
            "O1_deviation_min",
            "O3_changes",
            "assigned_blocks",
        ],
        ascending=[False, False, True, True, False],
    )

    rec = candidates.iloc[0].to_dict()
    rec["recommended"] = True

    return rec


def build_pareto_candidates(
    prelayer: dict,
    providers: pd.DataFrame,
    scenarios: pd.DataFrame,
    early_release: pd.DataFrame,
    out_dir: str | Path,
    config: dict,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    themes = config.get("layer2", {}).get("pareto_themes", DEFAULT_THEMES)
    time_limit = int(config.get("layer2", {}).get("mip_time_limit_s", 120))

    eps_O1, eps_O3 = _compute_epsilon_bounds(
        prelayer=prelayer,
        providers=providers,
        scenarios=scenarios,
        early_release=early_release,
        config=config,
        time_limit=time_limit,
    )

    rows: list[dict[str, Any]] = []
    all_assignments: dict[str, pd.DataFrame] = {}
    all_coverage: dict[str, pd.DataFrame] = {}

    for theme in themes:
        print(f"Solving theme: {theme} …")
        log.info("Solving theme: %s", theme)

        theme_eps_O1, theme_eps_O3 = _theme_eps(theme, eps_O1, eps_O3)

        assignments, coverage, meta = _call_optimizer_with_recovery(
            prelayer=prelayer,
            providers=providers,
            scenarios=scenarios,
            early_release=early_release,
            config=config,
            theme=theme,
            eps_O1=theme_eps_O1,
            eps_O3=theme_eps_O3,
            time_limit=time_limit,
        )

        meta = _enrich_meta_from_outputs(
            meta=meta,
            assignments=assignments,
            coverage=coverage,
            theme=theme,
            eps_O1=theme_eps_O1,
            eps_O3=theme_eps_O3,
        )

        assignments.to_csv(out / f"assignments_{theme}.csv", index=False)
        coverage.to_csv(out / f"coverage_{theme}.csv", index=False)

        all_assignments[theme] = assignments
        all_coverage[theme] = coverage
        rows.append(meta)

        log.info(
            "%-20s status=%s assigned=%s open=%s changed=%s goals=%s/%s O1=%.2f O3=%.2f cap=%s",
            theme,
            meta.get("status", "?"),
            meta.get("assigned_blocks", 0),
            meta.get("open_blocks", 0),
            meta.get("changed_blocks", 0),
            meta.get("goals_met", 0),
            meta.get("goals_total", 0),
            _safe_float(meta.get("O1_deviation_min", 0.0)),
            _safe_float(meta.get("O3_changes", 0.0)),
            meta.get("mip_block_cap_used", "?"),
        )

    frontier = pd.DataFrame(rows)

    if frontier.empty:
        frontier.to_csv(out / "pareto_candidates.csv", index=False)
        _save_json({}, out / "recommended_candidate.json")
        return {
            "frontier": frontier,
            "recommended": {},
            "dominated": [],
        }

    frontier = _dominance_flags(frontier)

    frontier = frontier.sort_values(
        [
            "dominated",
            "goals_met",
            "O1_deviation_min",
            "O3_changes",
            "assigned_blocks",
        ],
        ascending=[True, False, True, True, False],
    ).reset_index(drop=True)

    recommended = _choose_recommended(frontier)
    rec_theme = str(recommended.get("theme", ""))

    frontier.to_csv(out / "pareto_candidates.csv", index=False)
    _save_json(recommended, out / "recommended_candidate.json")

    if rec_theme in all_assignments:
        all_assignments[rec_theme].to_csv(out / "assignments_recommended.csv", index=False)

    if rec_theme in all_coverage:
        all_coverage[rec_theme].to_csv(out / "coverage_recommended.csv", index=False)

    dominated_themes = (
        frontier.loc[frontier["dominated"].fillna(False).astype(bool), "theme"]
        .astype(str)
        .tolist()
        if "theme" in frontier.columns
        else []
    )

    _save_json(
        {
            "themes": {
                str(row.get("theme", f"theme_{i}")): row
                for i, row in enumerate(frontier.to_dict("records"))
            },
            "dominated": dominated_themes,
            "recommendation": rec_theme,
            "pareto_config": {
                "eps_O1": eps_O1,
                "eps_O3": eps_O3,
                "coverage_alpha": float(config.get("layer2", {}).get("coverage_alpha", 0.85)),
                "mip_time_limit_s": time_limit,
                "retry_block_caps": _get_retry_block_caps(config),
            },
        },
        out / "pareto_frontier.json",
    )

    print(f"Recommended theme: {recommended.get('theme', 'NONE')}")

    return {
        "frontier": frontier,
        "recommended": recommended,
        "dominated": dominated_themes,
    }
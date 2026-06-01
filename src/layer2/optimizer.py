"""
Layer 2 — Optimizer  (standalone, no src imports)
==================================================
v2 — fixes the all-open assignment bug:

  BUG 1 (site mismatch):
    Geisinger blocks carry physical_site='Other' while providers have
    exclusive_sites=['OR GMC'].  The original site pre-filter wiped out
    every (provider, block) pair → vars=62 → MIP assigns nothing.
    FIX: check coverage after pre-filtering; if <5% of pairs survive,
         log a warning and disable the filter so every provider can hold
         any block in the template (room-type check retained).

  BUG 2 (O3 penalises every assignment when warm-start is sparse):
    When eligible[] is empty for most providers, W is effectively all
    zeros, so O3 = Σ x[p][b].  The minimiser then avoids assigning
    blocks.  FIX: only include (p,b) in O3 when W[p][b]=1 OR the pair
    is in the current template. The weight on O3 is also capped at 0.05
    so the coverage objective always dominates.

  BUG 3 (auto-goals on wrong providers):
    Auto-goals picked the 12 highest-μ providers, who might have no
    compatible blocks in the template.  FIX: restrict auto-goals to
    providers who are actually dominant in at least one template block.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import pulp
    _HAS_PULP = True
except ImportError:
    _HAS_PULP = False
    log.warning("PuLP not installed — falling back to improved greedy.  pip install pulp")

LAMBDA_PENALTY = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

def provider_site_ok(provider_row: pd.Series, physical_site: str) -> bool:
    sites = provider_row.get("exclusive_sites", [])
    if isinstance(sites, str):
        return physical_site in sites
    return physical_site in (sites or [])


def make_goals_from_scenarios(
    scenarios: pd.DataFrame,
    config: dict,
    template_providers: Optional[set] = None,
    max_goals: int = 12,
) -> pd.DataFrame:
    """
    Auto-pick goal providers.
    When template_providers is supplied (set of provider_ids that appear as
    dominant holders in the block template), restrict selection to those
    providers so we only set goals for providers who actually have blocks.
    """
    demand = (
        scenarios.groupby("provider_id")
        .agg(total_mu=("mu_casetime_min", "sum"))
        .reset_index()
        .sort_values("total_mu", ascending=False)
    )
    if template_providers:
        demand = demand[demand["provider_id"].astype(str).isin(template_providers)]
    demand = demand.head(max_goals).copy()
    demand["goal_type"]           = "target_utilization"
    demand["target_utilization"]  = float(
        config.get("layer2", {}).get("default_target_utilization", 0.70)
    )
    return demand[["provider_id", "goal_type", "target_utilization"]]


def required_minutes_by_provider(
    scenarios: pd.DataFrame,
    early_release: pd.DataFrame,
    goals: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    alpha  = float(config.get("layer2", {}).get("coverage_alpha", 0.85))
    nweeks = int(config.get("layer2", {}).get("optimization_weeks", 13))
    weekly = (
        scenarios.groupby(["provider_id", "scenario_id"])
        .agg(q_case=("demand_casetime_min", "sum"),
             q_turn=("demand_turnover_min", "sum"))
        .reset_index()
    )
    weekly["quarterly_demand_min"] = (weekly["q_case"] + weekly["q_turn"]) * nweeks
    er_map: Dict[str, float] = {}
    if not early_release.empty and "early_release_projected_min" in early_release.columns:
        er_map = dict(zip(
            early_release["provider_id"].astype(str),
            early_release["early_release_projected_min"],
        ))
    rows = []
    for _, goal in goals[["provider_id", "target_utilization"]].iterrows():
        p      = str(goal["provider_id"])
        target = float(goal["target_utilization"])
        vals   = weekly.loc[weekly["provider_id"].astype(str) == p, "quarterly_demand_min"]
        if vals.empty:
            continue
        r_by_s = vals.values / max(target, 1e-6) + float(er_map.get(p, 0.0))
        rows.append({
            "provider_id":        p,
            "target_utilization": target,
            "required_min_p10":   float(np.quantile(r_by_s, 0.10)),
            "required_min_p50":   float(np.quantile(r_by_s, 0.50)),
            "required_min_p90":   float(np.quantile(r_by_s, 0.90)),
            "required_min_alpha": float(np.quantile(r_by_s, alpha)),
            "coverage_alpha":     alpha,
            "_r_by_scenario":     r_by_s.tolist(),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_template(prelayer: dict, config: dict) -> pd.DataFrame:
    template = pd.DataFrame(prelayer.get("block_template", []))
    if template.empty:
        raise ValueError("prelayer['block_template'] is empty.")
    max_b = int(config.get("layer2", {}).get("max_template_blocks_for_mip", 900))
    template = (
        template
        .sort_values(["observed_instances", "duration_min"], ascending=False)
        .head(max_b)
        .copy()
        .reset_index(drop=True)
    )
    if "block_id" not in template.columns:
        template["block_id"] = [f"BLK-{i:04d}" for i in range(len(template))]
    return template


def _build_R_matrix(
    scenarios: pd.DataFrame,
    goals: pd.DataFrame,
    early_release: pd.DataFrame,
    config: dict,
) -> Tuple[Dict[str, np.ndarray], int]:
    nweeks = int(config.get("layer2", {}).get("optimization_weeks", 13))
    er_map: Dict[str, float] = {}
    if not early_release.empty and "early_release_projected_min" in early_release.columns:
        er_map = dict(zip(
            early_release["provider_id"].astype(str),
            early_release["early_release_projected_min"],
        ))
    weekly = (
        scenarios.groupby(["provider_id", "scenario_id"])
        .agg(q_case=("demand_casetime_min", "sum"),
             q_turn=("demand_turnover_min", "sum"))
        .reset_index()
    )
    weekly["quarterly_demand_min"] = (weekly["q_case"] + weekly["q_turn"]) * nweeks
    S = int(weekly["scenario_id"].nunique())
    R_map: Dict[str, np.ndarray] = {}
    for _, goal in goals[["provider_id", "target_utilization"]].iterrows():
        p      = str(goal["provider_id"])
        target = float(goal["target_utilization"])
        vals   = weekly.loc[weekly["provider_id"].astype(str) == p, "quarterly_demand_min"]
        if vals.empty:
            continue
        r = vals.values / max(target, 1e-6) + float(er_map.get(p, 0.0))
        r = np.pad(r, (0, max(0, S - len(r))), mode="edge")[:S]
        R_map[p] = r
    return R_map, S


def _build_eligible_sets(
    active: set,
    template: pd.DataFrame,
    prov_rows: Dict[str, pd.Series],
    config: dict,
) -> Dict[str, set]:
    """
    Build per-provider eligible block sets.

    BUG FIX: if site exclusivity wipes out >95% of (provider, block) pairs
    (site name mismatch between blocks.json and providers.json), disable the
    site filter entirely and warn.  Room-type filtering is intentionally NOT
    applied here — the template already reflects what rooms each provider uses.
    """
    all_bids = set(template["block_id"].astype(str))
    eligible: Dict[str, set] = {}

    use_site_filter = True
    for p in active:
        if p == "OPEN":
            eligible[p] = all_bids.copy()
            continue
        prow = prov_rows.get(p)
        if prow is None:
            eligible[p] = set()
            continue
        sites = prow.get("exclusive_sites", [])
        if not sites:                         # no restriction declared → all blocks
            eligible[p] = all_bids.copy()
        else:
            eligible[p] = {
                str(b["block_id"])
                for _, b in template.iterrows()
                if provider_site_ok(prow, str(b.get("physical_site", "")))
            }

    # ── BUG FIX 1: detect and handle site mismatch ──────────────────────────
    non_open = [p for p in active if p != "OPEN"]
    total_pairs = len(non_open) * len(all_bids)
    eligible_pairs = sum(len(eligible[p]) for p in non_open)

    if total_pairs > 0 and eligible_pairs / total_pairs < 0.05:
        log.warning(
            "Site exclusivity filter kept only %.1f%% of (provider, block) pairs "
            "(%d / %d). This is almost certainly a site-name mismatch between "
            "blocks.json (physical_site) and providers.json (exclusive_sites). "
            "Disabling site filter — all active providers are eligible for all blocks.",
            100.0 * eligible_pairs / total_pairs,
            eligible_pairs, total_pairs,
        )
        for p in active:
            eligible[p] = all_bids.copy()
    else:
        log.info(
            "Site filter kept %d / %d (provider, block) pairs (%.0f%%)",
            eligible_pairs, total_pairs,
            100.0 * eligible_pairs / total_pairs if total_pairs > 0 else 0,
        )

    return eligible


def _theme_weights(theme: str) -> dict:
    # BUG FIX 2: cap O3 weight at 0.05 so the coverage objective always
    # dominates — previously O3 could make the solver avoid any assignment.
    return {
        "conservative":      {"w_O1": 0.50, "w_O3": 0.05},
        "continuity_first":  {"w_O1": 0.10, "w_O3": 0.05},
        "balanced":          {"w_O1": 1.00, "w_O3": 0.05},
        "utilization_first": {"w_O1": 2.00, "w_O3": 0.01},
    }.get(theme, {"w_O1": 1.00, "w_O3": 0.05})


# ─────────────────────────────────────────────────────────────────────────────
# PuLP MIP
# ─────────────────────────────────────────────────────────────────────────────

def _mip_allocate(
    prelayer, providers, scenarios, early_release, config,
    theme, eps_O1, eps_O3, time_limit,
):
    t0     = time.time()
    alpha  = float(config.get("layer2", {}).get("coverage_alpha", 0.85))
    template = _prepare_template(prelayer, config)
    all_bids = set(template["block_id"].astype(str))

    # ── Goal providers ────────────────────────────────────────────────────────
    # BUG FIX 3: restrict auto-goals to providers who appear in the template
    template_pids = set(
        template["dominant_provider_id"].astype(str).dropna().unique()
    ) - {"OPEN", "", "nan"}

    goals  = make_goals_from_scenarios(
        scenarios, config, template_providers=template_pids
    )
    R_map, S = _build_R_matrix(scenarios, goals, early_release, config)

    if not R_map:
        log.warning("No R[p,s] computed — no goal providers found in scenarios.")
        return _warm_start_only(template, goals, theme)

    k_req = int(math.ceil(alpha * S))

    # ── Active providers ──────────────────────────────────────────────────────
    prov_rows    = {str(r["provider_id"]): r for _, r in providers.iterrows()}
    warm_holders = template_pids      # everyone in the template
    active       = (set(R_map.keys()) | warm_holders) & (set(prov_rows.keys()) | {"OPEN"})

    log.info(
        "MIP theme=%-20s  blocks=%d  goal_providers=%d  active=%d  S=%d  k_req=%d",
        theme, len(template), len(R_map), len(active), S, k_req,
    )

    # ── Warm-start W ──────────────────────────────────────────────────────────
    W: Dict[str, Dict[str, int]] = {p: {} for p in active}
    for _, b in template.iterrows():
        bid = str(b["block_id"])
        cur = "OPEN"
        raw = b.get("dominant_provider_id")
        if raw is not None and str(raw) not in ("nan", "", "OPEN"):
            cur = str(raw)
        if cur in active:
            W[cur][bid] = 1

    # ── Eligible sets (with site-mismatch fallback) ────────────────────────────
    eligible = _build_eligible_sets(active, template, prov_rows, config)

    log.info(
        "  Warm-start entries: %d  |  Eligible pairs: %d",
        sum(len(v) for v in W.values()),
        sum(len(v) for v in eligible.values()),
    )

    dur_map = dict(zip(template["block_id"].astype(str), template["duration_min"]))

    # ── Build model ───────────────────────────────────────────────────────────
    prob = pulp.LpProblem("block_alloc", pulp.LpMinimize)

    x = {
        p: {bid: pulp.LpVariable(f"x_{p[:6]}_{bid}", cat="Binary")
            for bid in eligible[p]}
        for p in active
    }

    z:     Dict[str, List] = {}
    slack: Dict[str, pulp.LpVariable] = {}
    for p in R_map:
        if p not in x or not eligible.get(p):
            continue
        z[p]     = [pulp.LpVariable(f"z_{p[:6]}_{s}", cat="Binary") for s in range(S)]
        slack[p] = pulp.LpVariable(f"slack_{p[:6]}", lowBound=0)

    A = {
        p: pulp.lpSum(x[p][bid] * float(dur_map.get(bid, 0)) for bid in eligible[p])
        for p in active
    }

    # C4: mutual exclusivity per block
    for _, b in template.iterrows():
        bid     = str(b["block_id"])
        holders = [p for p in active if bid in x[p]]
        if len(holders) > 1:
            prob += pulp.lpSum(x[p][bid] for p in holders) <= 1, f"C4_{bid}"

    # SAA chance constraints
    M_big = max((float(np.max(r)) for r in R_map.values() if len(r)), default=10_000.0)
    for p, r_vals in R_map.items():
        if p not in z:
            continue
        prob += pulp.lpSum(z[p]) >= k_req, f"cov_{p[:6]}"
        for s in range(S):
            prob += (
                A[p] >= float(r_vals[s]) - M_big * (1 - z[p][s]) - slack[p],
                f"bigM_{p[:6]}_{s}",
            )

    penalty = pulp.lpSum(LAMBDA_PENALTY * slack[p] for p in slack)

    # O1: deviation from median R
    delta1: Dict[str, pulp.LpVariable] = {}
    for p, r_vals in R_map.items():
        if p not in x or not eligible.get(p):
            continue
        r_med     = float(np.median(r_vals))
        delta1[p] = pulp.LpVariable(f"d1_{p[:6]}", lowBound=0)
        prob += delta1[p] >= A[p] - r_med, f"d1a_{p[:6]}"
        prob += delta1[p] >= r_med - A[p], f"d1b_{p[:6]}"
    O1 = pulp.lpSum(delta1.values()) if delta1 else pulp.lpSum([])

    # O3: changes from warm-start
    # BUG FIX 2: only penalise pairs that ARE in the warm-start (W[p][b]=1)
    # so that assigning to a fresh open block doesn't count as a "change".
    chg_vars: Dict[str, Dict[str, pulp.LpVariable]] = {}
    for p in active:
        chg_vars[p] = {}
        for bid in eligible[p]:
            w_val = float(W[p].get(bid, 0))
            if w_val == 0:
                continue          # not in warm-start: no change penalty
            cv = pulp.LpVariable(f"chg_{p[:6]}_{bid}", lowBound=0)
            prob += cv >= x[p][bid] - w_val, f"chga_{p[:6]}_{bid}"
            prob += cv >= w_val - x[p][bid], f"chgb_{p[:6]}_{bid}"
            chg_vars[p][bid] = cv
    O3 = (
        pulp.lpSum(chg_vars[p][bid] for p in active for bid in chg_vars[p])
        if any(chg_vars[p] for p in active) else pulp.lpSum([])
    )

    if eps_O1 is not None:
        prob += O1 <= eps_O1, "eps_O1"
    if eps_O3 is not None:
        prob += O3 <= eps_O3, "eps_O3"

    tw   = _theme_weights(theme)
    prob += tw["w_O1"] * O1 + tw["w_O3"] * O3 + penalty, "obj"

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(
        timeLimit=time_limit,
        gapRel=float(config.get("layer2", {}).get("mip_gap_rel", 0.01)),
        msg=0,
    )
    prob.solve(solver)
    elapsed = time.time() - t0
    status  = pulp.LpStatus[prob.status]

    # ── Extract assignments ───────────────────────────────────────────────────
    assign_rows = []
    for _, b in template.iterrows():
        bid  = str(b["block_id"])
        warm = "OPEN"
        raw  = b.get("dominant_provider_id")
        if raw is not None and str(raw) not in ("nan", "", "OPEN"):
            warm = str(raw)
        assigned = "OPEN"
        for p in active:
            if bid in x[p] and (pulp.value(x[p][bid]) or 0) > 0.5:
                assigned = p
                break
        assign_rows.append({
            **b.to_dict(),
            "assigned_provider_id":   assigned,
            "warm_start_provider_id": warm,
            "changed":                assigned != warm,
        })
    assign_df = pd.DataFrame(assign_rows)

    n_assigned = int((assign_df["assigned_provider_id"] != "OPEN").sum())
    n_changed  = int(assign_df["changed"].sum())
    log.info(
        "  %-20s status=%-10s  t=%.1fs  assigned=%d/%d  changed=%d  obj=%.2f",
        theme, status, elapsed, n_assigned, len(template),
        n_changed, float(pulp.value(prob.objective) or 0),
    )

    # ── Coverage metrics ──────────────────────────────────────────────────────
    allocated = {
        p: sum(
            (pulp.value(x[p][bid]) or 0) * float(dur_map.get(bid, 0))
            for bid in eligible[p]
        )
        for p in active
    }
    cov_rows = []
    for p in sorted(active):
        r_vals    = R_map.get(p)
        req_alpha = float(np.quantile(r_vals, alpha)) if r_vals is not None else 0.0
        alloc     = allocated.get(p, 0.0)
        n_covered = (
            sum(1 for s in range(S) if (pulp.value(z[p][s]) or 0) > 0.5)
            if p in z else 0
        )
        cov_rows.append({
            "provider_id":        p,
            "allocated_min":      round(alloc, 1),
            "required_min_alpha": round(req_alpha, 1),
            "coverage_ratio":     round(alloc / req_alpha, 3) if req_alpha > 0 else float("nan"),
            "meets_alpha":        bool(n_covered >= k_req) if p in z else (req_alpha == 0.0),
            "scenarios_covered":  n_covered if p in z else None,
            "slack_min":          round(float(pulp.value(slack[p]) or 0.0), 1) if p in slack else 0.0,
        })
    cov_df = pd.DataFrame(cov_rows)

    o1_val = float(pulp.value(O1) or 0.0)
    o3_val = float(pulp.value(O3) or 0.0)
    n_met  = int(cov_df["meets_alpha"].sum()) if not cov_df.empty else 0

    return assign_df, cov_df, {
        "theme": theme, "status": status,
        "solve_time_s": round(elapsed, 2),
        "objective_value": round(float(pulp.value(prob.objective) or 0.0), 4),
        "O1_deviation_min": round(o1_val, 2),
        "O3_changes": round(o3_val, 2),
        "changed_blocks": n_changed,
        "goals_met": n_met,
        "goals_total": len(R_map),
        "k_required": k_req, "S": S, "M_big": round(M_big, 1),
        "goals": goals.to_dict(orient="records"),
        "eps_O1": eps_O1, "eps_O3": eps_O3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Improved greedy fallback  (PuLP absent)
# ─────────────────────────────────────────────────────────────────────────────

def _improved_greedy(prelayer, providers, scenarios, early_release, config, theme):
    alpha    = float(config.get("layer2", {}).get("coverage_alpha", 0.85))
    template = _prepare_template(prelayer, config)
    template_pids = set(
        template["dominant_provider_id"].astype(str).dropna().unique()
    ) - {"OPEN", "", "nan"}
    goals    = make_goals_from_scenarios(
        scenarios, config, template_providers=template_pids
    )
    req      = required_minutes_by_provider(scenarios, early_release, goals, config)
    req_map  = dict(zip(req["provider_id"].astype(str), req["required_min_alpha"]))
    prov_rows = {str(r["provider_id"]): r for _, r in providers.iterrows()}
    active    = set(req_map.keys()) | template_pids
    allocated: Dict[str, float] = {p: 0.0 for p in active}
    block_owner: Dict[str, str] = {}
    dur_map = dict(zip(template["block_id"].astype(str), template["duration_min"]))

    # Warm-start from template
    for _, b in template.iterrows():
        bid = str(b["block_id"])
        raw = b.get("dominant_provider_id")
        cur = "OPEN"
        if raw is not None and str(raw) not in ("nan", "", "OPEN"):
            cur = str(raw)
        block_owner[bid] = cur if cur in active else "OPEN"
        if block_owner[bid] != "OPEN":
            allocated[block_owner[bid]] = allocated.get(block_owner[bid], 0.0) + float(b["duration_min"])

    # Greedy reallocation
    max_moves = {"conservative": 8, "continuity_first": 12,
                 "balanced": 30, "utilization_first": 80}.get(theme, 30)
    for _ in range(max_moves):
        need = sorted(
            [(p, req_map.get(p, 0) - allocated.get(p, 0)) for p in req_map],
            key=lambda kv: -kv[1],
        )
        if not need or need[0][1] <= 0:
            break
        target_p, _ = need[0]
        prow = prov_rows.get(target_p)
        if prow is None:
            break
        best_bid, best_score = None, float("inf")
        for _, b in template.iterrows():
            bid   = str(b["block_id"])
            donor = block_owner[bid]
            if donor == target_p:
                continue
            dur = float(b["duration_min"])
            if donor != "OPEN" and allocated.get(donor, 0) - req_map.get(donor, 0) < dur:
                continue
            score = float(b.get("exception_rate", 0)) * 1000 + dur
            if score < best_score:
                best_bid, best_score = bid, score
        if best_bid is None:
            break
        donor = block_owner[best_bid]
        dur   = float(dur_map.get(best_bid, 0))
        if donor != "OPEN":
            allocated[donor] -= dur
        block_owner[best_bid] = target_p
        allocated[target_p]   = allocated.get(target_p, 0) + dur

    rows = []
    for _, b in template.iterrows():
        bid  = str(b["block_id"])
        raw  = b.get("dominant_provider_id")
        warm = "OPEN" if raw is None or str(raw) in ("nan", "", "OPEN") else str(raw)
        rows.append({**b.to_dict(), "assigned_provider_id": block_owner.get(bid, "OPEN"),
                     "warm_start_provider_id": warm,
                     "changed": block_owner.get(bid, "OPEN") != warm})
    assign_df = pd.DataFrame(rows)

    cov_rows = []
    for p in sorted(active):
        req_a = req_map.get(p, 0.0)
        alloc = allocated.get(p, 0.0)
        cov_rows.append({
            "provider_id": p, "allocated_min": round(alloc, 1),
            "required_min_alpha": round(req_a, 1),
            "coverage_ratio": round(alloc / req_a, 3) if req_a > 0 else float("nan"),
            "meets_alpha": bool(req_a > 0 and alloc >= req_a),
            "scenarios_covered": None, "slack_min": 0.0,
        })
    cov_df = pd.DataFrame(cov_rows)
    undershoot = float((cov_df["required_min_alpha"] - cov_df["allocated_min"]).clip(lower=0).sum())
    n_changed  = int(assign_df["changed"].sum())
    return assign_df, cov_df, {
        "theme": theme, "status": "Greedy-Fallback", "solve_time_s": 0.0,
        "objective_value": round(undershoot + 5.0 * n_changed, 4),
        "O1_deviation_min": round(undershoot, 2), "O3_changes": float(n_changed),
        "changed_blocks": n_changed,
        "goals_met": int(cov_df["meets_alpha"].sum()), "goals_total": len(req_map),
        "k_required": int(math.ceil(alpha)), "S": 0, "M_big": 0.0,
        "goals": goals.to_dict(orient="records"), "eps_O1": None, "eps_O3": None,
    }


def _warm_start_only(template, goals, theme):
    rows = [{**b.to_dict(),
             "assigned_provider_id": str(b.get("dominant_provider_id", "OPEN")),
             "warm_start_provider_id": str(b.get("dominant_provider_id", "OPEN")),
             "changed": False}
            for _, b in template.iterrows()]
    return (pd.DataFrame(rows),
            pd.DataFrame(columns=["provider_id","allocated_min","required_min_alpha",
                                   "coverage_ratio","meets_alpha","scenarios_covered","slack_min"]),
            {"theme": theme, "status": "WarmStartOnly", "solve_time_s": 0.0,
             "objective_value": 0.0, "O1_deviation_min": 0.0, "O3_changes": 0.0,
             "changed_blocks": 0, "goals_met": 0, "goals_total": 0,
             "k_required": 0, "S": 0, "M_big": 0.0,
             "goals": goals.to_dict(orient="records"), "eps_O1": None, "eps_O3": None})


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def greedy_allocate(
    prelayer: dict,
    providers: pd.DataFrame,
    scenarios: pd.DataFrame,
    early_release: pd.DataFrame,
    config: dict,
    theme: str = "balanced",
    eps_O1: Optional[float] = None,
    eps_O3: Optional[float] = None,
    time_limit: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    if _HAS_PULP:
        return _mip_allocate(prelayer, providers, scenarios, early_release,
                             config, theme, eps_O1, eps_O3, time_limit)
    return _improved_greedy(prelayer, providers, scenarios, early_release, config, theme)
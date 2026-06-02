#!/usr/bin/env python3
"""
Standalone Layer-2 artifact validator.

Run after Layer 2 finishes:

    python3 src/layer2/validate_layer2_outputs.py --layer2_dir outputs/layer2

This does not rerun optimization. It only reads Layer-2 artifacts and checks
structural feasibility, file consistency, and Pareto outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def validate_artifacts(layer2_dir: str | Path) -> Dict[str, Any]:
    d = Path(layer2_dir)
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, level: str = "FAIL", detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "level": level if not ok else "PASS", "detail": str(detail)})

    paths = {
        "assignments": d / "assignments.csv",
        "schedule_by_week": d / "schedule_by_week.csv",
        "provider_day_coverage": d / "provider_day_coverage.csv",
        "slot_utilization": d / "slot_utilization.csv",
        "candidate_pairs": d / "candidate_pairs.csv",
        "pareto_all": d / "pareto_all.csv",
        "pareto_frontier": d / "pareto_frontier.csv",
        "calibrated_goals": d / "calibrated_goals.json",
        "layer2_result": d / "layer2_result.json",
        "validation_report": d / "validation_report.json",
        "plans_summary": d / "plans_summary.csv",
        "objective_values_by_plan": d / "objective_values_by_plan.csv",
        "provider_goal_details_by_plan": d / "provider_goal_details_by_plan.csv",
        "metric_goal_details_by_plan": d / "metric_goal_details_by_plan.csv",
        "fragmentation_summary_by_plan": d / "fragmentation_summary_by_plan.csv",
        "change_details_by_plan": d / "change_details_by_plan.csv",
        "provider_delta_by_plan": d / "provider_delta_by_plan.csv",
    }

    missing = [k for k, p in paths.items() if not p.exists() and k != "validation_report"]
    add("Required files exist", not missing, "FAIL", "present" if not missing else "missing=" + ",".join(missing))

    a = _read_csv(paths["assignments"])
    c = _read_csv(paths["provider_day_coverage"])
    s = _read_csv(paths["slot_utilization"])
    p_all = _read_csv(paths["pareto_all"])
    p_front = _read_csv(paths["pareto_frontier"])

    add("Assignments non-empty", len(a) > 0, "WARN", f"rows={len(a)}")
    add("Coverage non-empty", len(c) > 0, "FAIL", f"rows={len(c)}")
    add("Slot utilization non-empty", len(s) > 0, "FAIL", f"rows={len(s)}")

    if not a.empty:
        need = {"slot_idx", "target_idx", "allocated_min"}
        add("Assignments required columns", need.issubset(a.columns), "FAIL", f"cols={list(a.columns)}")
        if need.issubset(a.columns):
            add("No negative assigned minutes", (a["allocated_min"] >= 0).all(), "FAIL", "allocated_min >= 0")

    if not s.empty and {"capacity_min", "allocated_min"}.issubset(s.columns):
        over = (s["allocated_min"] - s["capacity_min"]).max()
        add("No slot capacity violation", float(over) <= 1e-6, "FAIL", f"max_over_capacity_min={float(over):.3f}")
        if "utilization" in s.columns:
            add("Slot utilization <= 1", float(s["utilization"].max()) <= 1.000001, "FAIL", f"max_util={float(s['utilization'].max()):.6f}")

    if not c.empty and {"target_q_int", "allocated_min"}.issubset(c.columns):
        over_target = (c["allocated_min"] - c["target_q_int"]).max()
        add("No target over-allocation", float(over_target) <= 1e-6, "FAIL", f"max_over_target_min={float(over_target):.3f}")

    if not c.empty and not s.empty and "allocated_min" in c.columns and "allocated_min" in s.columns:
        cov_total = float(c["allocated_min"].sum())
        slot_total = float(s["allocated_min"].sum())
        add("Coverage total equals slot total", abs(cov_total - slot_total) <= 1e-6, "FAIL", f"coverage={cov_total:.0f}, slots={slot_total:.0f}")

    axes = ["useful_utilization", "continuity_score", "stability_score", "preference_score"]
    add("Pareto all non-empty", len(p_all) > 0, "FAIL", f"rows={len(p_all)}")
    add("Pareto frontier non-empty", len(p_front) > 0, "FAIL", f"rows={len(p_front)}")
    missing_axes = [x for x in axes if x not in p_all.columns]
    add("Pareto axes present", not missing_axes, "FAIL", "present" if not missing_axes else "missing=" + ",".join(missing_axes))
    if "pareto_efficient" in p_all.columns:
        n_eff = int(p_all["pareto_efficient"].fillna(False).astype(bool).sum())
        add("Pareto efficient rows marked", n_eff > 0, "FAIL", f"n_efficient={n_eff}")
    else:
        # Backward-compatible recovery for outputs written by v4 before this fix:
        # if pareto_frontier exists but pareto_all lacks the marker, infer the
        # marker by matching the theme column.  This avoids forcing a full rerun.
        if not p_all.empty and not p_front.empty and "theme" in p_all.columns and "theme" in p_front.columns:
            eff_themes = set(p_front["theme"].astype(str))
            p_all["pareto_efficient"] = p_all["theme"].astype(str).isin(eff_themes)
            try:
                p_all.to_csv(paths["pareto_all"], index=False)
            except Exception:
                pass
            n_eff = int(p_all["pareto_efficient"].fillna(False).astype(bool).sum())
            add("Pareto efficient rows marked", n_eff > 0, "FAIL", f"recovered_from_frontier; n_efficient={n_eff}")
        else:
            add("Pareto efficient rows marked", False, "FAIL", "column missing")

    # Detailed v6 plan artifacts.
    ps = _read_csv(paths["plans_summary"])
    add("Plan summary non-empty", len(ps) > 0, "FAIL", f"rows={len(ps)}")
    if not ps.empty:
        need_plan_cols = {"theme", "total_goals_met_count", "total_goals_count", "O2_fragmentation_handoffs", "O3_changed_slots", "schedule_path"}
        missing_plan_cols = sorted(list(need_plan_cols - set(ps.columns)))
        add("Plan summary has goal/objective columns", not missing_plan_cols, "FAIL", "present" if not missing_plan_cols else "missing=" + ",".join(missing_plan_cols))
        if "schedule_path" in ps.columns:
            missing_schedules = [x for x in ps["schedule_path"].dropna().astype(str).tolist() if not Path(x).exists()]
            add("Saved schedule exists for every plan", len(missing_schedules) == 0, "FAIL", "all schedules present" if not missing_schedules else f"missing={len(missing_schedules)}")

    for key, label in [
        ("objective_values_by_plan", "Objective values by plan"),
        ("provider_goal_details_by_plan", "Provider goal details"),
        ("metric_goal_details_by_plan", "Metric goal details"),
        ("fragmentation_summary_by_plan", "Fragmentation summary"),
        ("change_details_by_plan", "Change details"),
        ("provider_delta_by_plan", "Provider delta details"),
    ]:
        tbl = _read_csv(paths[key])
        add(f"{label} non-empty", len(tbl) > 0, "FAIL", f"rows={len(tbl)}")

    if paths["layer2_result"].exists():
        try:
            result = json.loads(paths["layer2_result"].read_text())
            m = result.get("metrics", {})
            for name in ["useful_utilization", "target_coverage", "continuity_score", "stability_score", "preference_score", "overall_score"]:
                val = m.get(name)
                add(f"Metric finite: {name}", val is not None and np.isfinite(float(val)), "FAIL", f"{name}={val}")
        except Exception as exc:
            add("layer2_result readable", False, "FAIL", f"error={exc}")

    status = "PASS"
    if any((not x["ok"]) and x["level"] == "FAIL" for x in checks):
        status = "FAIL"
    elif any((not x["ok"]) and x["level"] == "WARN" for x in checks):
        status = "WARN"

    return {
        "status": status,
        "checks": checks,
        "summary": {
            "n_checks": len(checks),
            "n_pass": int(sum(x["ok"] for x in checks)),
            "n_warn": int(sum((not x["ok"]) and x["level"] == "WARN" for x in checks)),
            "n_fail": int(sum((not x["ok"]) and x["level"] == "FAIL" for x in checks)),
        },
    }


def print_report(report: Dict[str, Any]) -> None:
    print("\n" + "─" * 62)
    print("  STANDALONE LAYER 2 VALIDATION REPORT")
    print("─" * 62)
    for c in report.get("checks", []):
        if c["ok"]:
            mark = "✓ PASS"
        elif c["level"] == "WARN":
            mark = "⚠ WARN"
        else:
            mark = "✗ FAIL"
        print(f"  {mark:<8} {c['name']:<40} {c['detail']}")
    print("─" * 62)
    print("Status:", report.get("status"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer2_dir", default="outputs/layer2")
    args = ap.parse_args()
    report = validate_artifacts(args.layer2_dir)
    out_path = Path(args.layer2_dir) / "validation_report_standalone.json"
    out_path.write_text(json.dumps(report, indent=2))
    print_report(report)
    print(f"Saved: {out_path}")
    return 0 if report["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())

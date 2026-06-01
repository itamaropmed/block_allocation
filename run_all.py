"""Run the full Gen3 block-allocation pipeline.

This version is compatible with the updated standalone layer scripts.
The layer scripts use same-folder imports, so they must be executed by file path,
not with `python -m src.layer...`.

Typical usage:
    python3 run_all.py --force
    python3 run_all.py --force --clean
    python3 run_all.py --from-layer layer2
    python3 run_all.py --only layer2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "default_config.yaml"
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class LayerSpec:
    name: str
    script: Path
    marker: Path
    args: tuple[str, ...]


LAYERS: list[LayerSpec] = [
    LayerSpec(
        name="pre_layer",
        script=ROOT / "src" / "pre_layer" / "run_prelayer.py",
        marker=OUTPUTS / "pre_layer" / "prelayer_result.json",
        args=("--config", str(CONFIG)),
    ),
    LayerSpec(
        name="layer1",
        script=ROOT / "src" / "layer1" / "run_layer1.py",
        marker=OUTPUTS / "layer1" / "scenarios_long.csv",
        args=(
            "--config", str(CONFIG),
            "--pre_layer_result", str(OUTPUTS / "pre_layer" / "prelayer_result.json"),
        ),
    ),
    LayerSpec(
        name="layer2",
        script=ROOT / "src" / "layer2" / "run_layer2.py",
        marker=OUTPUTS / "layer2" / "recommended_candidate.json",
        args=(
            "--config", str(CONFIG),
            "--pre_layer_result", str(OUTPUTS / "pre_layer" / "prelayer_result.json"),
            "--layer1_dir", str(OUTPUTS / "layer1"),
        ),
    ),
    LayerSpec(
        name="layer3",
        script=ROOT / "src" / "layer3" / "run_layer3.py",
        marker=OUTPUTS / "layer3" / "explanation_report.md",
        args=(
            "--config", str(CONFIG),
            "--layer1_dir", str(OUTPUTS / "layer1"),
            "--layer2_dir", str(OUTPUTS / "layer2"),
        ),
    ),
]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(THREAD_ENV)
    # Keep project root importable for optional package-style imports.
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not old else str(ROOT) + os.pathsep + old
    return env


def _layer_names() -> list[str]:
    return [layer.name for layer in LAYERS]


def _select_layers(only: str | None, from_layer: str | None, to_layer: str | None) -> list[LayerSpec]:
    layers = LAYERS
    names = _layer_names()

    if only:
        requested = [x.strip() for x in only.split(",") if x.strip()]
        bad = [x for x in requested if x not in names]
        if bad:
            raise ValueError(f"Unknown --only layer(s): {bad}. Valid: {names}")
        return [layer for layer in layers if layer.name in requested]

    start = 0
    end = len(layers)

    if from_layer:
        if from_layer not in names:
            raise ValueError(f"Unknown --from-layer {from_layer!r}. Valid: {names}")
        start = names.index(from_layer)

    if to_layer:
        if to_layer not in names:
            raise ValueError(f"Unknown --to-layer {to_layer!r}. Valid: {names}")
        end = names.index(to_layer) + 1

    if start >= end:
        raise ValueError("Layer range is empty. Check --from-layer and --to-layer.")

    return layers[start:end]


def _check_required_files() -> None:
    required = [
        CONFIG,
        ROOT / "data" / "raw" / "geisinger-users_blocks.json",
        ROOT / "data" / "raw" / "geisinger-users_providers.json",
        ROOT / "data" / "raw" / "geisinger-users_cases.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\nMissing required files:", flush=True)
        for p in missing:
            print(f"  - {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}", flush=True)
        raise FileNotFoundError("Required input files are missing.")

    missing_scripts = [layer.script for layer in LAYERS if not layer.script.exists()]
    if missing_scripts:
        print("\nMissing layer scripts:", flush=True)
        for p in missing_scripts:
            print(f"  - {p}", flush=True)
        raise FileNotFoundError("Layer script files are missing.")


def _clean_selected_outputs(layers: Iterable[LayerSpec]) -> None:
    for layer in layers:
        out_dir = OUTPUTS / layer.name
        if layer.name == "pre_layer":
            out_dir = OUTPUTS / "pre_layer"
        if out_dir.exists():
            print(f"Removing {out_dir.relative_to(ROOT)}", flush=True)
            shutil.rmtree(out_dir)


def _run_script(layer: LayerSpec) -> None:
    print("\n" + "=" * 72, flush=True)
    print(f"RUNNING {layer.name}: {layer.script.relative_to(ROOT)}", flush=True)
    print("=" * 72, flush=True)

    cmd = [sys.executable, str(layer.script), *layer.args]
    subprocess.run(cmd, cwd=str(ROOT), env=_env(), check=True)


def _is_open_provider(pid: object) -> bool:
    if pid is None:
        return True
    s = str(pid).strip()
    return s == "" or s.upper() in {"OPEN", "NONE", "NAN", "NULL", "CLOSED"}


def _is_group_or_service(pid: object) -> bool:
    s = str(pid).strip().upper()
    return s.startswith("SVC-") or s.startswith("GRP-")


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _load_provider_sites() -> dict[str, set[str]]:
    path = ROOT / "data" / "raw" / "geisinger-users_providers.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: dict[str, set[str]] = {}
    for row in data:
        pid = str(row.get("provider_id", "")).strip()
        sites = row.get("exclusive_sites", []) or []
        if isinstance(sites, str):
            sites = [sites]
        out[pid] = {str(s).strip() for s in sites if str(s).strip()}
    return out


def _validate_assignment_file(path: Path, provider_sites: dict[str, set[str]]) -> tuple[dict, list[dict]]:
    import pandas as pd

    df = pd.read_csv(path)
    theme = path.stem.replace("assignments_", "")

    required_cols = {
        "assigned_provider_id",
        "physical_site",
        "room_type",
        "day_of_week",
        "rotation_phase",
        "canon_start_min",
        "canon_end_min",
        "block_id",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        return {
            "file": path.name,
            "theme": theme,
            "assigned_blocks": 0,
            "open_blocks": 0,
            "bad_time_intervals": None,
            "room_overlap_violations": None,
            "individual_provider_overlap_violations": None,
            "group_or_service_overlap_violations": None,
            "site_incompatible_violations": None,
            "missing_provider_violations": None,
            "feasible_individual_strict": False,
            "feasible_full_strict": False,
            "note": "missing required columns: " + ", ".join(missing),
        }, []

    d = df.copy()
    d["assigned_provider_id"] = d["assigned_provider_id"].fillna("OPEN").astype(str).str.strip()
    d["canon_start_min"] = pd.to_numeric(d["canon_start_min"], errors="coerce")
    d["canon_end_min"] = pd.to_numeric(d["canon_end_min"], errors="coerce")
    d["day_of_week"] = pd.to_numeric(d["day_of_week"], errors="coerce")
    d["rotation_phase"] = pd.to_numeric(d["rotation_phase"], errors="coerce")

    assigned = d[~d["assigned_provider_id"].map(_is_open_provider)].copy()
    violations: list[dict] = []

    # Bad intervals.
    bad_time = assigned[
        assigned["canon_start_min"].isna()
        | assigned["canon_end_min"].isna()
        | (assigned["canon_start_min"] >= assigned["canon_end_min"])
    ]
    for _, r in bad_time.iterrows():
        violations.append({
            "theme": theme,
            "violation_type": "bad_time_interval",
            "provider_id": r["assigned_provider_id"],
            "block_id": r.get("block_id", ""),
            "details": f"start={r.get('canon_start_min')} end={r.get('canon_end_min')}",
        })

    # Site compatibility.
    site_bad = 0
    missing_provider = 0
    for _, r in assigned.iterrows():
        pid = str(r["assigned_provider_id"])
        site = str(r.get("physical_site", "")).strip()
        sites = provider_sites.get(pid)
        if sites is None:
            missing_provider += 1
            violations.append({
                "theme": theme,
                "violation_type": "missing_provider_in_providers_json",
                "provider_id": pid,
                "block_id": r.get("block_id", ""),
                "details": f"assigned to site={site}",
            })
        elif sites and site not in sites:
            site_bad += 1
            violations.append({
                "theme": theme,
                "violation_type": "site_incompatible",
                "provider_id": pid,
                "block_id": r.get("block_id", ""),
                "details": f"site={site}; allowed={sorted(sites)}",
            })

    # Room overlaps: same room/site/day/phase cannot contain overlapping assigned blocks.
    room_overlap = 0
    room_keys = ["physical_site", "room_type", "rotation_phase", "day_of_week"]
    for key, g in assigned.groupby(room_keys, dropna=False):
        rows = g.sort_values("canon_start_min").to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if float(b["canon_start_min"]) >= float(a["canon_end_min"]):
                    break
                if _overlap(float(a["canon_start_min"]), float(a["canon_end_min"]),
                            float(b["canon_start_min"]), float(b["canon_end_min"])):
                    room_overlap += 1
                    violations.append({
                        "theme": theme,
                        "violation_type": "room_overlap",
                        "provider_id": f"{a['assigned_provider_id']} | {b['assigned_provider_id']}",
                        "block_id": f"{a.get('block_id','')} | {b.get('block_id','')}",
                        "details": f"room_key={key}; intervals=({a['canon_start_min']},{a['canon_end_min']}) and ({b['canon_start_min']},{b['canon_end_min']})",
                    })

    # Provider overlaps. Individual overlaps are strict. SVC/GRP overlaps are reported separately.
    individual_overlap = 0
    group_overlap = 0
    prov_keys = ["assigned_provider_id", "rotation_phase", "day_of_week"]
    for key, g in assigned.groupby(prov_keys, dropna=False):
        pid = str(key[0]) if isinstance(key, tuple) else str(key)
        rows = g.sort_values("canon_start_min").to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if float(b["canon_start_min"]) >= float(a["canon_end_min"]):
                    break
                if _overlap(float(a["canon_start_min"]), float(a["canon_end_min"]),
                            float(b["canon_start_min"]), float(b["canon_end_min"])):
                    if _is_group_or_service(pid):
                        group_overlap += 1
                        vtype = "group_or_service_provider_overlap"
                    else:
                        individual_overlap += 1
                        vtype = "individual_provider_overlap"
                    violations.append({
                        "theme": theme,
                        "violation_type": vtype,
                        "provider_id": pid,
                        "block_id": f"{a.get('block_id','')} | {b.get('block_id','')}",
                        "details": f"phase/day=({a['rotation_phase']},{a['day_of_week']}); rooms={a.get('room_type')} | {b.get('room_type')}; intervals=({a['canon_start_min']},{a['canon_end_min']}) and ({b['canon_start_min']},{b['canon_end_min']})",
                    })

    summary = {
        "file": path.name,
        "theme": theme,
        "assigned_blocks": int(len(assigned)),
        "open_blocks": int(len(d) - len(assigned)),
        "bad_time_intervals": int(len(bad_time)),
        "room_overlap_violations": int(room_overlap),
        "individual_provider_overlap_violations": int(individual_overlap),
        "group_or_service_overlap_violations": int(group_overlap),
        "site_incompatible_violations": int(site_bad),
        "missing_provider_violations": int(missing_provider),
        "feasible_individual_strict": bool(len(bad_time) == 0 and room_overlap == 0 and individual_overlap == 0 and site_bad == 0 and missing_provider == 0),
        "feasible_full_strict": bool(len(bad_time) == 0 and room_overlap == 0 and individual_overlap == 0 and group_overlap == 0 and site_bad == 0 and missing_provider == 0),
        "note": "SVC-* and GRP-* overlaps are reported separately because they may represent service/group buckets, not individual surgeons.",
    }
    return summary, violations


def validate_layer2_outputs(fail_on_infeasible: bool = False) -> None:
    import pandas as pd

    layer2 = OUTPUTS / "layer2"
    if not layer2.exists():
        print("Layer 2 output directory does not exist; skipping feasibility validation.", flush=True)
        return

    files = sorted(layer2.glob("assignments_*.csv"))
    files = [p for p in files if not p.name.startswith("assignments_recommended")]
    if not files:
        print("No Layer 2 assignment files found; skipping feasibility validation.", flush=True)
        return

    provider_sites = _load_provider_sites()
    summaries: list[dict] = []
    violations: list[dict] = []

    for path in files:
        summary, v = _validate_assignment_file(path, provider_sites)
        summaries.append(summary)
        violations.extend(v)

    summary_df = pd.DataFrame(summaries)
    violations_df = pd.DataFrame(violations)

    summary_path = layer2 / "feasibility_summary.csv"
    violations_path = layer2 / "feasibility_violations.csv"
    summary_df.to_csv(summary_path, index=False)
    violations_df.to_csv(violations_path, index=False)

    print("\n" + "=" * 72, flush=True)
    print("LAYER 2 FEASIBILITY VALIDATION", flush=True)
    print("=" * 72, flush=True)
    show_cols = [
        "theme",
        "assigned_blocks",
        "open_blocks",
        "room_overlap_violations",
        "individual_provider_overlap_violations",
        "group_or_service_overlap_violations",
        "site_incompatible_violations",
        "missing_provider_violations",
        "feasible_individual_strict",
        "feasible_full_strict",
    ]
    print(summary_df[show_cols].to_string(index=False), flush=True)
    print(f"\nSaved: {summary_path.relative_to(ROOT)}", flush=True)
    print(f"Saved: {violations_path.relative_to(ROOT)}", flush=True)

    infeasible = not bool(summary_df["feasible_individual_strict"].all())
    if infeasible:
        msg = (
            "Layer 2 produced at least one assignment file that is not feasible "
            "under individual-provider strict checks. See feasibility_violations.csv."
        )
        if fail_on_infeasible:
            raise RuntimeError(msg)
        print("\nWARNING: " + msg, flush=True)
    else:
        print("\nAll theme files passed individual-provider strict feasibility checks.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gen3 block-allocation pipeline")
    parser.add_argument("--force", action="store_true", help="recompute selected layers even if marker outputs exist")
    parser.add_argument("--clean", action="store_true", help="delete selected layer output folders before running")
    parser.add_argument("--only", default=None, help="comma-separated layer names to run, e.g. layer2 or layer1,layer2")
    parser.add_argument("--from-layer", default=None, choices=_layer_names(), help="start from this layer")
    parser.add_argument("--to-layer", default=None, choices=_layer_names(), help="stop after this layer")
    parser.add_argument("--skip-validation", action="store_true", help="skip Layer 2 assignment feasibility validation")
    parser.add_argument("--fail-on-infeasible", action="store_true", help="exit non-zero if Layer 2 feasibility validation fails")
    args = parser.parse_args()

    os.chdir(ROOT)
    OUTPUTS.mkdir(exist_ok=True)
    _check_required_files()

    selected = _select_layers(args.only, args.from_layer, args.to_layer)

    print("Project root:", ROOT, flush=True)
    print("Selected layers:", ", ".join(layer.name for layer in selected), flush=True)

    if args.clean:
        _clean_selected_outputs(selected)

    ran_layer2 = False
    for layer in selected:
        if layer.marker.exists() and not args.force:
            print(
                f"\nSkipping {layer.name}; found {layer.marker.relative_to(ROOT)}. "
                "Use --force to recompute.",
                flush=True,
            )
        else:
            _run_script(layer)
        if layer.name == "layer2":
            ran_layer2 = True

    if ran_layer2 and not args.skip_validation:
        validate_layer2_outputs(fail_on_infeasible=args.fail_on_infeasible)

    print("\nDONE. See outputs/ for all artifacts.", flush=True)


if __name__ == "__main__":
    main()

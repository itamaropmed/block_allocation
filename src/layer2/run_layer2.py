"""
Layer 2 — Run script (standalone, no src imports)
==================================================
Place this file in the same folder as optimizer.py and pareto_frontier.py.
All helpers (config loading, JSON IO, table loading, plotting) are inlined here.

Usage (from the folder containing these files):
    python run_layer2.py
    python run_layer2.py --config path/to/config.yaml
    python run_layer2.py --pre_layer_result path/to/prelayer_result.json \
                         --layer1_dir       path/to/layer1_output/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml  # pip install pyyaml

from pareto_frontier import build_pareto_candidates   # same-folder import

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Inlined IO helpers  (no src.common needed)
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_raw_tables(
    blocks_json: str,
    providers_json: str,
    cases_json: Optional[str] = None,
):
    """
    Load blocks, providers (and optionally cases) from JSON files.
    Returns (blocks_df, providers_df, cases_df).
    """
    with open(blocks_json, encoding="utf-8") as f:
        blocks_df = pd.DataFrame(json.load(f))

    with open(providers_json, encoding="utf-8") as f:
        providers_df = pd.DataFrame(json.load(f))

    cases_df = pd.DataFrame()
    if cases_json and Path(cases_json).exists():
        with open(cases_json, encoding="utf-8") as f:
            cases_df = pd.DataFrame(json.load(f))

    log.info("Loaded blocks=%d  providers=%d  cases=%d",
             len(blocks_df), len(providers_df), len(cases_df))
    return blocks_df, providers_df, cases_df


def save_pareto_plot(frontier: pd.DataFrame, path: Path) -> None:
    """Save a Pareto scatter plot (O1 vs O3, coloured by goals_met)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        cmap   = plt.cm.RdYlGn
        colors = frontier["goals_met"] / max(frontier["goals_total"].max(), 1)

        sc = ax.scatter(
            frontier["O1_deviation_min"],
            frontier["O3_changes"],
            c=colors, cmap=cmap, s=120, zorder=3,
        )
        for _, row in frontier.iterrows():
            ax.annotate(
                row["theme"],
                (row["O1_deviation_min"], row["O3_changes"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )
        plt.colorbar(sc, ax=ax, label="Goals met (normalised)")
        ax.set_xlabel("O1 — Deviation from BlockTimeRequired (min)")
        ax.set_ylabel("O3 — Block changes from warm-start")
        ax.set_title("Pareto Frontier — 4 themes")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log.info("Pareto plot saved → %s", path)
    except Exception as exc:
        log.warning("Pareto plot skipped (%s)", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main run function
# ─────────────────────────────────────────────────────────────────────────────

def run(
    config: Optional[dict] = None,
    pre_layer_result_path: Optional[str] = None,
    layer1_dir: Optional[str] = None,
) -> dict:

    # ── Config ────────────────────────────────────────────────────────────────
    if config is None:
        # Look for config relative to this file, then CWD.
        for candidate in [
            Path(__file__).parent / "config" / "default_config.yaml",
            Path("config") / "default_config.yaml",
            Path("default_config.yaml"),
        ]:
            if candidate.exists():
                config = load_config(str(candidate))
                log.info("Config loaded from %s", candidate)
                break
        if config is None:
            raise FileNotFoundError(
                "Could not find default_config.yaml. "
                "Pass --config explicitly or place it at config/default_config.yaml"
            )

    out = ensure_dir(Path(config["paths"]["output_dir"]) / "layer2")

    # ── Pre-Layer result ─────────────────────────────────────────────────────
    pre_path = Path(
        pre_layer_result_path
        or Path(config["paths"]["output_dir"]) / "pre_layer" / "prelayer_result.json"
    )
    if not pre_path.exists():
        raise FileNotFoundError(f"Pre-Layer result not found: {pre_path}")
    prelayer = load_json(pre_path)
    log.info("Pre-Layer result loaded: %s  (blocks=%d)",
             pre_path, len(prelayer.get("block_template", [])))

    # ── Providers ────────────────────────────────────────────────────────────
    _, providers_df, _ = load_raw_tables(
        config["paths"]["blocks_json"],
        config["paths"]["providers_json"],
        config["paths"].get("cases_json"),
    )

    # ── Layer 1 artefacts ────────────────────────────────────────────────────
    l1_dir = Path(layer1_dir or Path(config["paths"]["output_dir"]) / "layer1")

    scenarios_path = l1_dir / "scenarios_long.csv"
    if not scenarios_path.exists():
        raise FileNotFoundError(
            f"Layer 1 scenarios not found: {scenarios_path}\n"
            "Run Layer 1 first."
        )
    scenarios = pd.read_csv(scenarios_path)
    log.info("Scenarios: %d rows, %d unique",
             len(scenarios),
             scenarios["scenario_id"].nunique() if "scenario_id" in scenarios.columns else -1)

    er_path = l1_dir / "early_release_projected.csv"
    if er_path.exists():
        early_release = pd.read_csv(er_path)
        log.info("Early release: %d providers", len(early_release))
    else:
        log.warning(
            "early_release_projected.csv not found at %s — "
            "using 0 for all providers.", er_path
        )
        early_release = pd.DataFrame(
            columns=["provider_id", "early_release_projected_min"]
        )

    # ── Run Pareto frontier ──────────────────────────────────────────────────
    log.info("Running Layer 2: epsilon-constraint Pareto frontier …")
    result = build_pareto_candidates(
        prelayer, providers_df, scenarios, early_release, out, config
    )

    # ── Pareto plot ──────────────────────────────────────────────────────────
    save_pareto_plot(result["frontier"], out / "pareto_candidates.png")

    # ── Print summary ────────────────────────────────────────────────────────
    rec       = result["recommended"]
    dominated = result.get("dominated", [])
    frontier  = result["frontier"]

    print("\n" + "=" * 60)
    print("  Layer 2 complete")
    print(f"  Candidates  : {len(frontier)}")
    print(f"  Dominated   : {dominated or 'none'}")
    print(f"  Recommended : {rec['theme']}")
    print(f"  Goals met   : {rec.get('goals_met',0)} / {rec.get('goals_total',0)}")
    print(f"  O1 deviation: {rec.get('O1_deviation_min',0):.1f} min")
    print(f"  O3 changes  : {rec.get('O3_changes',0):.0f} blocks")
    print(f"  Status      : {rec.get('status','?')}")
    print(f"  Output dir  : {out}")
    print("=" * 60)

    cols = ["theme", "status", "O1_deviation_min", "O3_changes",
            "goals_met", "goals_total", "solve_time_s", "dominated"]
    avail = [c for c in cols if c in frontier.columns]
    print("\n" + frontier[avail].to_string(index=False) + "\n")

    return {"output_dir": str(out)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 2 — Pareto frontier MIP")
    parser.add_argument("--config",            default=None, help="Path to config YAML")
    parser.add_argument("--pre_layer_result",  default=None, help="Path to prelayer_result.json")
    parser.add_argument("--layer1_dir",        default=None, help="Path to Layer 1 output dir")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    run(
        config=cfg,
        pre_layer_result_path=args.pre_layer_result,
        layer1_dir=args.layer1_dir,
    )

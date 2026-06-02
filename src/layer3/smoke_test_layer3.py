"""Create tiny fake Layer 1/Layer 2 artifacts and run Layer 3 end-to-end."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from .explainability import Layer3Config, run_layer3
except ImportError:
    from explainability import Layer3Config, run_layer3


def main() -> int:
    rng = np.random.default_rng(42)
    root = Path(tempfile.mkdtemp(prefix="layer3_smoke_"))
    l1 = root / "outputs" / "layer1"
    l2 = root / "outputs" / "layer2"
    out = root / "outputs" / "layer3"
    l1.mkdir(parents=True)
    l2.mkdir(parents=True)

    providers = ["PRV-001", "PRV-002", "PRV-003"]
    rows = []
    for week in range(8):
        for p in providers:
            for d in range(5):
                util = rng.uniform(0.2, 0.9)
                rows.append({
                    "provider_id": p,
                    "service_line": "Cardiac" if p != "PRV-003" else "Ortho",
                    "week_index": week,
                    "day_of_week": d,
                    "rotation_phase": week % 4,
                    "attn_weighted_util": util,
                    "attn_entropy": rng.uniform(0.1, 1.0),
                    "attn_turn_weighted": rng.uniform(5, 30),
                    "trailing_4w_mean_case": util * 200 + rng.normal(0, 10),
                    "trailing_12w_mean_case": util * 180 + rng.normal(0, 10),
                    "trailing_4w_std_case": rng.uniform(5, 40),
                    "trailing_12w_std_case": rng.uniform(10, 30),
                    "service_line_wk_mean": rng.uniform(100, 200),
                    "exception_rate_series": rng.uniform(0, 0.3),
                    "sin_woy": np.sin(week / 52 * 2 * np.pi),
                    "cos_woy": np.cos(week / 52 * 2 * np.pi),
                })
    features = pd.DataFrame(rows)
    y_case = features["attn_weighted_util"] * 250 + features["service_line_wk_mean"] * 0.2 + rng.normal(0, 5, len(features))
    y_turn = features["attn_turn_weighted"] + rng.normal(0, 2, len(features))
    features["casetime_min"] = y_case
    features["turnover_min"] = y_turn
    features.to_csv(l1 / "features.csv", index=False)

    X = features[[c for c in features.columns if c not in ["provider_id", "service_line", "casetime_min", "turnover_min"] and pd.api.types.is_numeric_dtype(features[c])]]
    for target, y in [("casetime_min", y_case), ("turnover_min", y_turn)]:
        model = Pipeline([("imputer", SimpleImputer()), ("model", RandomForestRegressor(n_estimators=25, random_state=42))])
        model.fit(X, y)
        joblib.dump(model, l1 / f"{target}_model.joblib")

    scen = []
    for p in providers:
        for d in range(5):
            base = 120 if p == "PRV-001" else 80 if p == "PRV-002" else 150
            for s in range(30):
                scen.append({
                    "provider_id": p,
                    "service_line": "Cardiac" if p != "PRV-003" else "Ortho",
                    "day_of_week": d,
                    "scenario_id": s,
                    "demand_casetime_min": max(0, rng.normal(base, 25)),
                    "demand_turnover_min": max(0, rng.normal(15, 4)),
                })
    pd.DataFrame(scen).to_csv(l1 / "scenarios_long.csv", index=False)
    pd.DataFrame({"provider_id": providers, "sigma_case": [20, 10, 30], "sigma_turn": [3, 2, 4]}).to_csv(l1 / "hierarchical_sigma_estimates.csv", index=False)
    pd.DataFrame({"provider_id": providers, "early_release_projected_min": [30, 0, 60]}).to_csv(l1 / "early_release_projected.csv", index=False)
    (l1 / "model_metrics.json").write_text(json.dumps({"casetime": {"r_hat_max": 1.01}, "turnover": {"r_hat_max": 1.02}}))

    coverage = pd.DataFrame({
        "provider_id": providers,
        "allocated_min": [2400, 960, 1440],
        "required_min_alpha": [2200, 1200, 2600],
        "coverage_ratio": [1.09, 0.8, 0.55],
        "meets_alpha": [True, False, False],
        "target_utilization": [0.7, 0.7, 0.7],
    })
    coverage.to_csv(l2 / "coverage_recommended.csv", index=False)
    sched = []
    for i, p in enumerate(["PRV-001"] * 5 + ["PRV-002"] * 2 + ["PRV-003"] * 3):
        sched.append({"block_id": f"B{i}", "week": i % 4, "day_of_week": i % 5, "room": f"OR-{i%2}", "current_blockholder": providers[i % 3], "assigned_provider": p, "capacity_min": 480})
    pd.DataFrame(sched).to_csv(l2 / "schedule_recommended.csv", index=False)
    (l2 / "recommended_candidate.json").write_text(json.dumps({"theme": "utilization_first", "recommended": True}))
    candidates = [
        {"theme": "utilization_first", "utilization_score": 0.92, "stability_score": 0.50, "continuity_score": 0.40, "preference_score": 0.90, "fragmentation": 8, "changes": 10, "goals_met_count": 1},
        {"theme": "stability_first", "utilization_score": 0.91, "stability_score": 0.75, "continuity_score": 0.45, "preference_score": 0.88, "fragmentation": 5, "changes": 6, "goals_met_count": 1},
        {"theme": "continuity_first", "utilization_score": 0.89, "stability_score": 0.60, "continuity_score": 0.80, "preference_score": 0.85, "fragmentation": 4, "changes": 5, "goals_met_count": 1},
    ]
    (l2 / "candidates_all.json").write_text(json.dumps(candidates))

    cfg = Layer3Config(layer1_dir=l1, layer2_dir=l2, out_dir=out, optimization_weeks=4, alpha=0.85, max_shap_rows=1000)
    res = run_layer3(cfg)
    print("Smoke test output:", res["out_dir"])
    print("Summary:", res["summary_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

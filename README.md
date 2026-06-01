# Gen3 Block Allocation Engine - Code Bundle

This bundle contains a runnable implementation scaffold for the four algorithmic folders:

1. `src/pre_layer` - template reconstruction from historical block records.
2. `src/layer1` - stochastic demand forecasting: weekly observations, attention features, point forecasts, posterior-style scenarios.
3. `src/layer2` - stochastic allocation optimizer and Pareto-style candidate generation.
4. `src/layer3` - explanation package: feature attribution, confidence bands, goal attribution, and narrative reports.

The uploaded JSON files are included under `data/raw/`:

- `geisinger-users_blocks.json`
- `geisinger-users_providers.json`

A third `geisinger-users_cases.json` was expected by the Notion specification, because Layer 1 uses `block_case_minutes` and `turnover_time`. It was not present in the mounted files. The code is therefore written in two modes:

- **Full mode**: add `data/raw/geisinger-users_cases.json`, and Layer 1 uses real case minutes and turnover times.
- **Proxy mode**: without a cases file, Layer 1 builds a deterministic proxy demand from allocated block minutes so the whole pipeline still runs end-to-end. Proxy outputs are useful for testing code and data plumbing, not for clinical decisions.

## Quick start

```bash
cd gen3_block_allocation_code_bundle
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py          # uses included/generated outputs when present
python run_all.py --force  # recompute all layers from raw JSONs
```

Outputs are written to `outputs/`:

- `outputs/pre_layer/`
- `outputs/layer1/`
- `outputs/layer2/`
- `outputs/layer3/`

## Run layers separately

```bash
python -m src.pre_layer.run_prelayer
python -m src.layer1.run_layer1
python -m src.layer2.run_layer2
python -m src.layer3.run_layer3
```

## Important configuration

Edit `config/default_config.yaml`:

- `template_weeks: 0` means automatic BIC period detection.
- `canonical_time_margin_minutes: 120` implements the 2-hour canonical clustering margin.
- `n_scenarios: 200` matches the Gen3 stochastic design.
- `coverage_alpha: 0.85` is the SAA chance-constraint coverage target.
- `max_active_providers` and `max_template_blocks_for_mip` keep the MIP tractable during local tests.

## Notes on dependencies

The code prefers optional packages when installed:

- `xgboost` for Layer 1 point forecasts.
- `ortools` for Layer 2 CP-SAT optimization.
- `shap` for Layer 3 SHAP explanations.

If these are missing, the code falls back to scikit-learn models, a greedy allocation heuristic, and feature-importance explanations.

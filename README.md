# Gen3 Block Allocation Engine

A layered block-allocation research and optimization pipeline for reconstructing historical operating-room block templates, forecasting uncertain provider demand, solving stochastic allocation plans, and explaining the resulting recommendations.

The project is organized as an end-to-end pipeline:

1. **Pre-Layer — Template Reconstruction**  
   Reconstructs the repeating historical block template and detects the rotation period `T*`.

2. **Layer 1 — Stochastic Demand Forecasting**  
   Builds phase-aligned features, trains point-forecast models such as XGBoost, fits a hierarchical Bayesian residual model, and generates demand scenarios for SAA optimization.

3. **Layer 2 — Stochastic Optimization + Pareto Frontier**  
   Uses the Layer 1 scenarios to solve robust block-allocation plans with chance-constraint logic, utilization-first optimization, and ε-constraint Pareto alternatives.

4. **Layer 3 — Explainability + Recommendations**  
   Produces SHAP explanations, confidence bands, goal-attribution tables, provider risk summaries, Pareto diagnostics, and human-readable recommendations.

---

## 1. Data First: What the Raw Data Contains

The project expects three raw JSON files under `data/raw/`.

```text
data/raw/
├── geisinger-users_blocks.json
├── geisinger-users_cases.json
└── geisinger-users_providers.json
```

### 1.1 Providers file

Expected file:

```text
data/raw/geisinger-users_providers.json
```

The providers file is a list of providers, surgeons, or service-line-level block holders.

Each provider record has this structure:

```json
{
  "provider_id": "PRV-0001",
  "name": "General Surgery",
  "service_line": [],
  "exclusive_sites": [
    "ENDOSCOPY GCMC",
    "ENDOSCOPY GWV"
  ]
}
```

Main fields:

| Field | Meaning |
|---|---|
| `provider_id` | Unique provider identifier used across blocks and cases |
| `name` | Provider or service-line display name |
| `service_line` | Optional service-line metadata |
| `exclusive_sites` | Sites where the provider is allowed or historically associated |

Current dataset summary:

| Item | Value |
|---|---:|
| Provider rows | 336 |
| Unique provider IDs | 336 |
| Providers with exclusive sites | 336 |
| Providers with non-empty service-line metadata | 0 in the current raw file |

---

### 1.2 Cases file

Expected file:

```text
data/raw/geisinger-users_cases.json
```

The cases file is a list of historical surgical or procedural cases.

Each case record has this structure:

```json
{
  "case_id": "1000000",
  "provider_id": "",
  "turnover_time": 15,
  "block_case_minutes": 100,
  "day_of_week": "Friday"
}
```

Main fields:

| Field | Meaning |
|---|---|
| `case_id` | Unique case identifier |
| `provider_id` | Provider responsible for the case. Some rows may be blank. |
| `turnover_time` | Room turnover time in minutes |
| `block_case_minutes` | Case duration in block minutes |
| `day_of_week` | Weekday of the case |

Current dataset summary:

| Item | Value |
|---|---:|
| Case rows | 36,051 |
| Rows with blank `provider_id` | 14,505 |
| Unique non-blank case provider IDs | 330 |
| Turnover time | 15 minutes for all rows in current data |
| Minimum case minutes | 5 |
| Median case minutes | 70 |
| Mean case minutes | about 101.2 |
| Maximum case minutes | 720 |
| Days present | Monday–Saturday |

Important caveat:

The current cases file has provider and weekday information, but no direct `block_id` or timestamp. Layer 1 therefore reconstructs weekly observations by distributing provider/day case demand across matching historical block weeks. The code logs this clearly as a data warning, not as a crash.

---

### 1.3 Blocks file

Expected file:

```text
data/raw/geisinger-users_blocks.json
```

The blocks file is the main historical block-schedule file. It contains every historical block occurrence.

Each block record has this structure:

```json
{
  "block_historical_id": "SYN-00001_GMC_HFAM_ENDO_08_2026-01-09_7_12",
  "is_open": false,
  "site": "ENDOSCOPY GMC",
  "occurrence": {
    "start": "2026-01-09T07:30:00+00:00",
    "end": "2026-01-09T12:00:00+00:00"
  },
  "current_blockholder": {
    "provider_id": "PRV-0227",
    "name": "Provider PRV-0227"
  },
  "room": {
    "type": "GMC HFAM ENDO 08"
  },
  "manual_early_release": null
}
```

Main fields:

| Field | Meaning |
|---|---|
| `block_historical_id` | Historical block occurrence ID |
| `is_open` | Whether the block is open/unassigned |
| `site` | Site or facility |
| `occurrence.start` | Block start datetime |
| `occurrence.end` | Block end datetime |
| `current_blockholder.provider_id` | Provider assigned to the block |
| `room.type` | Room type/name |
| `manual_early_release` | Optional early-release timestamp or duration |

Current dataset summary:

| Item | Value |
|---|---:|
| Block rows | 8,190 |
| Date range | 2026-01-05 to 2026-04-11 |
| Unique calendar dates | 84 |
| Sites | 20 |
| Room types | 125 |
| Open blocks | 326 |
| Non-open blocks | 7,864 |
| Unique non-open block provider IDs | 330 |

---

## 2. Project Folder Structure

The repository should be structured like this:

```text
gen3_block_allocation_code_bundle/
├── README.md
├── run_all.py
├── config/
│   └── default_config.yaml
├── data/
│   └── raw/
│       ├── geisinger-users_blocks.json
│       ├── geisinger-users_cases.json
│       └── geisinger-users_providers.json
├── docs/
│   └── ...
├── outputs/
│   ├── pre_layer/
│   ├── layer1/
│   ├── layer2/
│   └── layer3/
├── src/
│   ├── pre_layer/
│   │   └── run_prelayer.py
│   ├── layer1/
│   │   ├── run_layer1.py
│   │   └── ...
│   ├── layer2/
│   │   ├── run_layer2.py
│   │   ├── validate_layer2_outputs.py
│   │   └── ...
│   └── layer3/
│       ├── run_layer3.py
│       └── ...
└── tests/
    └── ...
```

### Folder meanings

| Folder | Purpose |
|---|---|
| `config/` | Default paths, model settings, optimization parameters, validation thresholds |
| `data/raw/` | Raw input JSON files |
| `docs/` | Architecture notes, design documents, diagrams, and planning files |
| `outputs/` | Generated artifacts from each layer |
| `src/` | Source code for each pipeline layer |
| `tests/` | Unit tests, smoke tests, and validation tests |
| `run_all.py` | End-to-end orchestration script |

---

## 3. Clean Environment Setup

Recommended Python version:

```text
Python 3.11 or 3.12
```

Python 3.13 may work for this codebase, but Python 3.11/3.12 is safer for scientific packages such as PyMC, ArviZ, SHAP, OR-Tools, and XGBoost.

### 3.1 Create a virtual environment

Mac/Linux:

```bash
cd /path/to/gen3_block_allocation_code_bundle

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Windows PowerShell:

```powershell
cd C:\path\to\gen3_block_allocation_code_bundle

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

---

## 4. Install Dependencies

Install the core dependencies:

```bash
pip install \
  numpy \
  pandas \
  scipy \
  scikit-learn \
  xgboost \
  optuna \
  pymc \
  arviz \
  pytensor \
  ortools \
  shap \
  matplotlib \
  seaborn \
  joblib \
  pyyaml \
  tqdm
```

Optional developer/test dependencies:

```bash
pip install pytest black ruff ipykernel
```

Recommended `requirements.txt` content:

```text
numpy
pandas
scipy
scikit-learn
xgboost
optuna
pymc
arviz
pytensor
ortools
shap
matplotlib
seaborn
joblib
pyyaml
tqdm
pytest
```

After installing, verify the important packages:

```bash
python - <<'PY'
import xgboost, pymc, arviz, ortools, shap
print("xgboost:", xgboost.__version__)
print("pymc:", pymc.__version__)
print("arviz:", arviz.__version__)
print("ortools: OK")
print("shap:", shap.__version__)
PY
```

---

## 5. Put the Data in the Correct Location

Create the raw data folder:

```bash
mkdir -p data/raw
```

Copy or move the three JSON files into `data/raw/`:

```text
data/raw/geisinger-users_blocks.json
data/raw/geisinger-users_cases.json
data/raw/geisinger-users_providers.json
```

If the downloaded files have names like this:

```text
geisinger-users_blocks (1).json
geisinger-users_cases (1).json
```

rename them to:

```text
geisinger-users_blocks.json
geisinger-users_cases.json
```

The scripts expect the clean names above unless you override paths manually from the command line.

---

## 6. Running the Whole Pipeline

The easiest way to run everything is:

```bash
python3 run_all.py
```

Before running the real pipeline, check the commands without executing them:

```bash
python3 run_all.py --dry_run
```

Run only one layer:

```bash
python3 run_all.py --only prelayer
python3 run_all.py --only layer1
python3 run_all.py --only layer2
python3 run_all.py --only layer3
```

Show available options:

```bash
python3 run_all.py --help
```

Important:

`run_all.py` should be in the project root, next to `src/`, not inside `src/`.

Correct:

```text
gen3_block_allocation_code_bundle/
├── run_all.py
└── src/
```

Incorrect:

```text
gen3_block_allocation_code_bundle/
└── src/
    └── run_all.py
```

---

## 7. Running Each Layer Manually

Manual runs are useful for debugging.

---

### 7.1 Pre-Layer: Template Reconstruction

Purpose:

The Pre-Layer reconstructs the historical repeating block template. It detects the rotation period `T*`, assigns each recurring slot to its dominant provider, computes phase labels, and records exceptions.

Run:

```bash
python3 src/pre_layer/run_prelayer.py \
  --blocks data/raw/geisinger-users_blocks.json \
  --providers data/raw/geisinger-users_providers.json \
  --cases data/raw/geisinger-users_cases.json \
  --out outputs \
  --expect_T 4
```

Main outputs:

```text
outputs/pre_layer/
├── prelayer_result.json
├── bic_scores.csv
├── bic_scores.png
├── block_template.csv
├── exceptions.csv
└── validation_report.json
```

Expected successful behavior:

The validation report should pass. In the current dataset, the expected detected rotation period is:

```text
T* = 4
```

The Pre-Layer output is required by Layer 1 and Layer 2.

---

### 7.2 Layer 1: Stochastic Demand Forecasting

Purpose:

Layer 1 creates provider/day demand forecasts and uncertainty scenarios. It reads the Pre-Layer phase labels, builds weekly observations, creates phase-aligned features, trains point-forecast models, fits a hierarchical Bayesian residual model, and generates SAA scenarios.

Fast smoke test:

```bash
python3 src/layer1/run_layer1.py \
  --blocks data/raw/geisinger-users_blocks.json \
  --providers data/raw/geisinger-users_providers.json \
  --cases data/raw/geisinger-users_cases.json \
  --out outputs \
  --pre_layer_result outputs/pre_layer/prelayer_result.json \
  --n_scenarios 20 \
  --optuna_trials 0 \
  --point_model xgboost \
  --sigma_model bayesian \
  --mcmc_draws 200 \
  --mcmc_tune 200 \
  --mcmc_chains 2 \
  --mcmc_cores 2
```

Full recommended run:

```bash
python3 src/layer1/run_layer1.py \
  --blocks data/raw/geisinger-users_blocks.json \
  --providers data/raw/geisinger-users_providers.json \
  --cases data/raw/geisinger-users_cases.json \
  --out outputs \
  --pre_layer_result outputs/pre_layer/prelayer_result.json \
  --n_scenarios 200 \
  --optuna_trials 0 \
  --point_model xgboost \
  --sigma_model bayesian \
  --mcmc_draws 1000 \
  --mcmc_tune 1000 \
  --mcmc_chains 4 \
  --mcmc_cores 4
```

Main outputs:

```text
outputs/layer1/
├── weekly_observations.csv
├── features.csv
├── point_forecasts.csv
├── model_metrics.json
├── casetime_min_model.joblib
├── turnover_min_model.joblib
├── hierarchical_sigma_estimates.csv
├── scenarios_long.csv
├── casetime_residuals.png
├── turnover_residuals.png
├── posterior_casetime_min.nc / .pkl
└── posterior_turnover_min.nc / .pkl
```

Expected successful behavior:

For `n_scenarios = 200`, the scenario file should have:

```text
number of rows = forecast_rows × 200
```

In the current dataset, the typical Layer 1 output has approximately:

```text
Forecast rows: 1719 provider×day rows
Scenario rows with 200 scenarios: 343800
```

Bayesian diagnostics:

Layer 1 stores posterior diagnostics such as:

```text
R-hat
ESS bulk
ESS tail
divergences
posterior summaries
sigma estimates
```

For quick smoke tests with only 2 chains and 200 draws, R-hat warnings may appear. That is acceptable for debugging. For final experiments, use 4 chains and larger draw/tune values.

---

### 7.3 Layer 2: Stochastic Optimization + Pareto Frontier

Purpose:

Layer 2 reads the Layer 1 scenarios and solves the block-allocation optimization. It uses stochastic demand, capacity constraints, shortage variables, utilization-first logic, and ε-constraint Pareto alternatives for stability, continuity, and preference.

Run:

```bash
python3 src/layer2/run_layer2.py \
  --config config/default_config.yaml \
  --pre_layer_result outputs/pre_layer/prelayer_result.json \
  --layer1_dir outputs/layer1 \
  --blocks data/raw/geisinger-users_blocks.json \
  --providers data/raw/geisinger-users_providers.json \
  --out outputs/layer2 \
  --alpha 0.85 \
  --time_limit_s 180 \
  --workers 8 \
  --max_slots 0 \
  --candidate_top_per_day 120 \
  --pareto_grid 6
```

Main outputs:

```text
outputs/layer2/
├── provider_day_coverage.csv
├── objective_summary.csv
├── goals_met_summary.csv
├── pareto_frontier.csv
├── pareto_frontier.json
├── frontier_diagnostics.json
├── validation_report.json
└── plans/
    ├── utilization_first_selected/
    │   ├── schedule_by_week.csv
    │   ├── assignments.csv
    │   ├── coverage.csv
    │   └── objective_components.json
    ├── stability_selected/
    ├── continuity_selected/
    └── preference_selected/
```

Layer 2 themes:

| Theme | Meaning |
|---|---|
| `utilization_first` | Main plan. Prioritizes demand coverage and utilization first. |
| `stability` | Attempts to stay closer to the current reconstructed template. |
| `continuity` | Attempts to reduce fragmentation and preserve provider consistency across recurring blocks. |
| `preference` | Attempts to favor preferred/provider-compatible assignments. |

Capacity warning:

Layer 2 may print a warning like:

```text
Total capacity is smaller than alpha-quantile demand target.
Solver will minimize shortage, but 100% alpha coverage is physically impossible.
```

This is not necessarily an error. It means the available block capacity is physically smaller than the full demand target. In that situation, the optimizer should still produce the best feasible allocation and minimize shortage.

Validate Layer 2 outputs:

```bash
python3 src/layer2/validate_layer2_outputs.py \
  --layer2_dir outputs/layer2
```

---

### 7.4 Layer 3: Explainability + Recommendations

Purpose:

Layer 3 explains the Layer 1 forecasts and Layer 2 allocation decisions. It creates SHAP explanations, provider risk profiles, confidence bands, goal attribution, Pareto diagnostics, counterfactuals, and recommendation tables.

Run:

```bash
python3 src/layer3/run_layer3.py \
  --layer1_dir outputs/layer1 \
  --layer2_dir outputs/layer2 \
  --pre_layer_result outputs/pre_layer/prelayer_result.json \
  --out outputs/layer3 \
  --alpha 0.85 \
  --optimization_weeks 4 \
  --selected_theme utilization_first \
  --top_k_features 12 \
  --max_shap_rows 5000 \
  --make_interactions
```

For a faster run without interaction SHAP:

```bash
python3 src/layer3/run_layer3.py \
  --layer1_dir outputs/layer1 \
  --layer2_dir outputs/layer2 \
  --pre_layer_result outputs/pre_layer/prelayer_result.json \
  --out outputs/layer3 \
  --alpha 0.85 \
  --optimization_weeks 4 \
  --selected_theme utilization_first \
  --top_k_features 12 \
  --max_shap_rows 3000
```

Main outputs:

```text
outputs/layer3/run_YYYYMMDD_HHMMSS/
├── SUMMARY.md
├── explanations.json
├── provenance.json
├── recommendations.csv
├── goals_met_summary.csv
├── coverage_by_provider_day.csv
├── goal_attribution_by_provider.csv
├── confidence_bands.csv
├── provider_risk_profiles.csv
├── frontier_diagnostics.json
├── shap_summary_casetime_min.csv
├── shap_summary_turnover_min.csv
├── shap_global_importance.csv
├── plots/
│   ├── shap_bar_casetime_min.png
│   ├── shap_beeswarm_casetime_min.png
│   ├── shap_bar_turnover_min.png
│   ├── shap_beeswarm_turnover_min.png
│   ├── confidence_bands.png
│   ├── coverage_distribution.png
│   └── pareto_frontier.png
└── validation_report.json
```

Expected successful behavior:

Layer 3 should print:

```text
Validation : PASS
```

A warning about missing coverage usually means Layer 2 did not write `provider_day_coverage.csv`, or Layer 3 is pointing to the wrong `outputs/layer2` directory.

---

## 8. Configuration

The default configuration lives here:

```text
config/default_config.yaml
```

Typical configuration categories:

```yaml
paths:
  blocks_json: data/raw/geisinger-users_blocks.json
  providers_json: data/raw/geisinger-users_providers.json
  cases_json: data/raw/geisinger-users_cases.json
  output_dir: outputs

pre_layer:
  max_T: 13
  expect_T: 4

layer1:
  point_model: xgboost
  sigma_model: bayesian
  n_scenarios: 200
  optuna_trials: 0
  mcmc_draws: 1000
  mcmc_tune: 1000
  mcmc_chains: 4
  mcmc_cores: 4
  rhat_threshold: 1.05

layer2:
  alpha: 0.85
  time_limit_s: 180
  workers: 8
  candidate_top_per_day: 120
  pareto_grid: 6

layer3:
  selected_theme: utilization_first
  top_k_features: 12
  max_shap_rows: 5000
  make_interactions: true
```

Command-line arguments override config values.

---

## 9. Expected End-to-End Outputs

After a successful run, the outputs folder should look like this:

```text
outputs/
├── pre_layer/
│   ├── prelayer_result.json
│   ├── bic_scores.csv
│   ├── bic_scores.png
│   └── validation_report.json
├── layer1/
│   ├── features.csv
│   ├── point_forecasts.csv
│   ├── scenarios_long.csv
│   ├── hierarchical_sigma_estimates.csv
│   ├── model_metrics.json
│   └── validation_report.json
├── layer2/
│   ├── provider_day_coverage.csv
│   ├── pareto_frontier.csv
│   ├── goals_met_summary.csv
│   ├── validation_report.json
│   └── plans/
└── layer3/
    └── run_YYYYMMDD_HHMMSS/
        ├── SUMMARY.md
        ├── explanations.json
        ├── recommendations.csv
        ├── confidence_bands.csv
        ├── provider_risk_profiles.csv
        └── plots/
```

---

## 10. Testing

Run unit tests:

```bash
python3 -m pytest tests -q
```

Compile-check all Python files:

```bash
python3 -m compileall src run_all.py
```

Run a dry-run of the orchestrator:

```bash
python3 run_all.py --dry_run
```

Run a fast smoke test:

```bash
python3 src/pre_layer/run_prelayer.py \
  --blocks data/raw/geisinger-users_blocks.json \
  --providers data/raw/geisinger-users_providers.json \
  --cases data/raw/geisinger-users_cases.json \
  --out outputs \
  --expect_T 4

python3 src/layer1/run_layer1.py \
  --blocks data/raw/geisinger-users_blocks.json \
  --providers data/raw/geisinger-users_providers.json \
  --cases data/raw/geisinger-users_cases.json \
  --out outputs \
  --pre_layer_result outputs/pre_layer/prelayer_result.json \
  --n_scenarios 20 \
  --optuna_trials 0 \
  --point_model xgboost \
  --sigma_model bayesian \
  --mcmc_draws 200 \
  --mcmc_tune 200 \
  --mcmc_chains 2 \
  --mcmc_cores 2
```

---

## 11. Runtime Expectations

Approximate runtimes depend heavily on hardware and parameter choices.

| Stage | Smoke test | Full run |
|---|---:|---:|
| Pre-Layer | seconds | seconds |
| Layer 1 XGBoost | seconds to minutes | minutes |
| Layer 1 Bayesian MCMC | seconds to minutes | minutes to longer |
| Layer 2 CP-SAT optimization | 30–180 seconds per theme | several minutes |
| Layer 2 Pareto grid | depends on grid size | longer with larger grid |
| Layer 3 SHAP | seconds to minutes | longer with interaction SHAP |

For fast debugging, use:

```text
n_scenarios = 20
mcmc_draws = 200
mcmc_tune = 200
mcmc_chains = 2
time_limit_s = 30
pareto_grid = 3
max_shap_rows = 1000
make_interactions = false
```

For final runs, use:

```text
n_scenarios = 200
mcmc_draws >= 1000
mcmc_tune >= 1000
mcmc_chains = 4
time_limit_s >= 180
pareto_grid >= 6
```

---

## 12. Troubleshooting

### XGBoost is installed but Layer 1 still uses Ridge

Make sure the run command includes:

```bash
--point_model xgboost
```

Also verify XGBoost is importable:

```bash
python -c "import xgboost; print(xgboost.__version__)"
```

---

### PyMC or ArviZ fails to import

Reinstall the Bayesian stack:

```bash
pip install --upgrade pymc arviz pytensor
```

A clean Python 3.11 environment is recommended.

---

### R-hat warning in Layer 1

Example warning:

```text
Bayesian casetime R-hat acceptable: max_r_hat=1.06
```

For smoke tests, this is common and not fatal.

For final results, increase MCMC quality:

```bash
--mcmc_draws 1000 \
--mcmc_tune 1000 \
--mcmc_chains 4 \
--mcmc_cores 4
```

---

### Layer 2 says capacity is smaller than demand

This is not automatically a bug. It means available OR capacity is smaller than the target demand at the requested alpha level. The optimizer should still minimize shortage and produce the best feasible schedule.

Possible fixes:

1. Increase available slots.
2. Include more open blocks.
3. Lower the alpha target.
4. Increase optimization horizon.
5. Relax stability or continuity constraints.
6. Increase `time_limit_s`.

---

### Layer 2 runs too long

Try a smaller candidate set or smaller Pareto grid:

```bash
--candidate_top_per_day 60 \
--pareto_grid 3 \
--time_limit_s 60
```

For final runs, increase time:

```bash
--time_limit_s 300
```

---

### Layer 3 warns that coverage is missing

Make sure this file exists:

```text
outputs/layer2/provider_day_coverage.csv
```

If it does not exist, rerun Layer 2.

---

### SHAP interaction values are slow

Run Layer 3 without interactions:

```bash
python3 src/layer3/run_layer3.py \
  --layer1_dir outputs/layer1 \
  --layer2_dir outputs/layer2 \
  --pre_layer_result outputs/pre_layer/prelayer_result.json \
  --out outputs/layer3 \
  --alpha 0.85 \
  --optimization_weeks 4 \
  --selected_theme utilization_first \
  --top_k_features 12 \
  --max_shap_rows 3000
```

Only use:

```bash
--make_interactions
```

for deeper final analysis.

---

## 13. GitHub Hygiene

Do not commit large generated outputs or sensitive raw data unless they are synthetic and approved for release.

Recommended `.gitignore`:

```gitignore
# Python
__pycache__/
*.pyc
.venv/
.env

# Outputs
outputs/
*.joblib
*.pkl
*.nc

# Local data
data/raw/*.json

# OS/editor
.DS_Store
.idea/
.vscode/
```

For public GitHub, use one of these options:

1. Commit only a small synthetic sample under `data/sample/`.
2. Keep real data outside the repository.
3. Document how to place the raw files locally under `data/raw/`.
4. Do not commit `outputs/`, posterior files, or trained model files unless they are intentionally released.

---

## 14. Reproducibility Notes

To make experiments reproducible:

1. Save the exact command used.
2. Save `config/default_config.yaml`.
3. Save `model_metrics.json`.
4. Save Layer 1 scenario files.
5. Save Layer 2 objective summaries.
6. Save Layer 3 `provenance.json`.
7. Record package versions:

```bash
pip freeze > requirements.lock.txt
```

For research comparisons, keep each experiment in a separate output folder:

```text
outputs/
├── experiment_001/
├── experiment_002/
└── experiment_003/
```

---

## 15. Pipeline Summary

### Pre-Layer

Input:

```text
blocks + providers + cases
```

Output:

```text
T*
phase labels
block template
exceptions
BIC scores
```

### Layer 1

Input:

```text
Pre-Layer result
blocks
providers
cases
```

Output:

```text
features
XGBoost point forecasts
Bayesian sigma estimates
posterior diagnostics
200 SAA scenarios
```

### Layer 2

Input:

```text
Pre-Layer template
Layer 1 scenarios
providers
blocks
```

Output:

```text
utilization-first plan
stability plan
continuity plan
preference plan
provider-day coverage
Pareto frontier
goals-met summary
```

### Layer 3

Input:

```text
Layer 1 models and scenarios
Layer 2 schedules and coverage
Pre-Layer template
```

Output:

```text
SHAP explanations
confidence bands
goal attribution
risk profiles
recommendations
plain-language summaries
plots
```

---

## 16. Main Command Reference

Dry run:

```bash
python3 run_all.py --dry_run
```

Full run:

```bash
python3 run_all.py
```

Pre-Layer only:

```bash
python3 run_all.py --only prelayer
```

Layer 1 only:

```bash
python3 run_all.py --only layer1
```

Layer 2 only:

```bash
python3 run_all.py --only layer2
```

Layer 3 only:

```bash
python3 run_all.py --only layer3
```

Manual help:

```bash
python3 run_all.py --help
python3 src/pre_layer/run_prelayer.py --help
python3 src/layer1/run_layer1.py --help
python3 src/layer2/run_layer2.py --help
python3 src/layer3/run_layer3.py --help
```

---

## 17. Project Status

Current implemented pipeline:

| Component | Status |
|---|---|
| Pre-Layer template reconstruction | Implemented |
| T* detection and validation | Implemented |
| Phase labels and phase offset | Implemented |
| Layer 1 feature matrix | Implemented |
| XGBoost point forecasting | Implemented |
| Hierarchical Bayesian residual model | Implemented |
| SAA scenario generation | Implemented |
| Layer 2 stochastic optimization | Implemented |
| Utilization-first plan | Implemented |
| ε-constraint Pareto alternatives | Implemented |
| Layer 2 validation | Implemented |
| Layer 3 SHAP explanations | Implemented |
| Layer 3 confidence/risk/recommendations | Implemented |
| End-to-end `run_all.py` | Implemented |

---

## 18. Owner

Project owner:

```text
Zernitsky Itamar
```

Repository:

```text
Block Allocation / Gen3 Block Allocation Engine
```

Layer 1 Bayesian update v2

This version fixes the ArviZ/PyMC compatibility warnings from the run log:
- uses idata.to_netcdf(...) when az.to_netcdf is not available
- uses ci_prob for ArviZ 1.x, with hdi_prob fallback
- computes max_r_hat, ESS, and divergences directly even if az.summary changes column names

Replace:
  cp bayesian_residual_model.py src/layer1/bayesian_residual_model.py
  cp run_layer1.py src/layer1/run_layer1.py

Smoke test:
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

Full validation run:
  python3 src/layer1/run_layer1.py \
    --blocks data/raw/geisinger-users_blocks.json \
    --providers data/raw/geisinger-users_providers.json \
    --cases data/raw/geisinger-users_cases.json \
    --out outputs \
    --pre_layer_result outputs/pre_layer/prelayer_result.json \
    --n_scenarios 200 \
    --optuna_trials 20 \
    --point_model xgboost \
    --sigma_model bayesian \
    --mcmc_draws 1000 \
    --mcmc_tune 1000 \
    --mcmc_chains 4 \
    --mcmc_cores 4

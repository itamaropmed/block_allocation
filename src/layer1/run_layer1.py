from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.block_schema import load_raw_tables
from src.common.io import ensure_dir, load_config, load_json, save_df, save_json
from src.common.plotting import save_residual_plot
from src.layer1.weekly_observations import build_weekly_observations, early_release_projected
from src.layer1.feature_engineering import build_features
from src.layer1.forecasting import train_point_forecasts, make_next_week_forecasts
from src.layer1.scenarios import estimate_sigma, generate_scenarios


def run(config: dict | None = None, pre_layer_result_path: str | None = None) -> dict:
    config = config or load_config('config/default_config.yaml')
    out = ensure_dir(Path(config['paths']['output_dir']) / 'layer1')
    pre_path = pre_layer_result_path or Path(config['paths']['output_dir']) / 'pre_layer' / 'prelayer_result.json'
    print('Layer1: loading prelayer', flush=True)
    prelayer = load_json(pre_path)
    print('Layer1: loading raw tables', flush=True)
    blocks_df, providers_df, cases_df = load_raw_tables(
        config['paths']['blocks_json'],
        config['paths']['providers_json'],
        config['paths'].get('cases_json'),
    )
    print('Layer1: building weekly observations', flush=True)
    obs, obs_meta = build_weekly_observations(blocks_df, providers_df, cases_df, prelayer, config)
    print('Layer1: building features', flush=True)
    features, attn_meta = build_features(obs, prelayer, config)
    save_df(obs, out / 'weekly_observations.csv')
    save_df(features, out / 'features.csv')
    er = early_release_projected(blocks_df, optimization_weeks=int(config['layer2'].get('optimization_weeks', 13)))
    save_df(er, out / 'early_release_projected.csv')
    print('Layer1: training point forecasts', flush=True)
    model_result = train_point_forecasts(features, out, config)
    print('Layer1: making point forecasts', flush=True)
    point = make_next_week_forecasts(features, model_result['models'], model_result['feature_columns'], out)
    print('Layer1: estimating sigma', flush=True)
    sigma = estimate_sigma(features, model_result['predictions'])
    save_df(sigma, out / 'hierarchical_sigma_estimates.csv')
    print('Layer1: generating scenarios', flush=True)
    scenarios = generate_scenarios(point, sigma, out, config)
    save_json({'warnings': obs_meta['warnings'], 'attention': attn_meta, 'n_features': len(features), 'n_scenarios': int(config['layer1'].get('n_scenarios', 200))}, out / 'layer1_metadata.json')
    save_residual_plot(model_result['predictions'], out / 'casetime_residuals.png', 'casetime_min')
    save_residual_plot(model_result['predictions'], out / 'turnover_residuals.png', 'turnover_min')
    print(f"Layer 1 complete: features={len(features)}, scenarios={len(scenarios)}")
    return {'output_dir': str(out)}


if __name__ == '__main__':
    # Some numerical libraries can leave non-daemon worker threads alive in certain server shells.
    # os._exit guarantees the CLI returns after all files are written.
    import os, sys
    run()
    print('Layer1: exiting now', flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.block_schema import load_raw_tables
from src.common.io import ensure_dir, load_config, load_json
from src.common.plotting import save_pareto_plot
from src.layer2.pareto_frontier import build_pareto_candidates


def run(config: dict | None = None, pre_layer_result_path: str | None = None, layer1_dir: str | None = None) -> dict:
    config = config or load_config('config/default_config.yaml')
    out = ensure_dir(Path(config['paths']['output_dir']) / 'layer2')
    pre_path = pre_layer_result_path or Path(config['paths']['output_dir']) / 'pre_layer' / 'prelayer_result.json'
    l1 = Path(layer1_dir or Path(config['paths']['output_dir']) / 'layer1')
    prelayer = load_json(pre_path)
    _, providers_df, _ = load_raw_tables(config['paths']['blocks_json'], config['paths']['providers_json'], config['paths'].get('cases_json'))
    scenarios = pd.read_csv(l1 / 'scenarios_long.csv')
    early_release = pd.read_csv(l1 / 'early_release_projected.csv')
    result = build_pareto_candidates(prelayer, providers_df, scenarios, early_release, out, config)
    save_pareto_plot(result['frontier'], out / 'pareto_candidates.png')
    print(f"Layer 2 complete: candidates={len(result['frontier'])}, recommended={result['recommended']['theme']}")
    return {'output_dir': str(out)}


if __name__ == '__main__':
    run()

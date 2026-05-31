from __future__ import annotations

from pathlib import Path

from src.common.io import ensure_dir, load_config
from src.layer3.explainability import feature_attribution, confidence_bands, goal_attribution, counterfactual_targets, write_narrative


def run(config: dict | None = None, layer1_dir: str | None = None, layer2_dir: str | None = None) -> dict:
    config = config or load_config('config/default_config.yaml')
    out = ensure_dir(Path(config['paths']['output_dir']) / 'layer3')
    l1 = Path(layer1_dir or Path(config['paths']['output_dir']) / 'layer1')
    l2 = Path(layer2_dir or Path(config['paths']['output_dir']) / 'layer2')
    attr = feature_attribution(l1, out, top_k=int(config['layer3'].get('top_k_features', 10)))
    bands = confidence_bands(l1, out)
    goals = goal_attribution(l2, out)
    counterfactual_targets(l1, out, targets=config['layer3'].get('counterfactual_targets', [0.6, 0.7, 0.8]), optimization_weeks=int(config['layer2'].get('optimization_weeks', 13)))
    report = write_narrative(out, attr, bands, goals)
    print(f'Layer 3 complete: {report}')
    return {'output_dir': str(out), 'report': str(report)}


if __name__ == '__main__':
    run()

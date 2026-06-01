"""
Layer 3 — Run script  (standalone, no src imports)
===================================================
Place this file in the same folder as explainability.py.
All helpers from src.common.* are inlined here.

Usage:
    python run_layer3.py
    python run_layer3.py --config path/to/config.yaml
    python run_layer3.py --layer1_dir path/to/layer1/ \
                         --layer2_dir path/to/layer2/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import yaml

from explainability import (     # same-folder import — no src prefix
    feature_attribution,
    confidence_bands,
    goal_attribution,
    counterfactual_targets,
    provider_risk_profiles,
    drift_detection,
    write_narrative,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
)


# ─────────────────────────────────────────────────────────────────────────────
# Inlined IO helpers  (no src.common needed)
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main run function
# ─────────────────────────────────────────────────────────────────────────────

def run(
    config: Optional[dict] = None,
    layer1_dir: Optional[str] = None,
    layer2_dir: Optional[str] = None,
) -> dict:

    # ── Config ────────────────────────────────────────────────────────────────
    if config is None:
        for candidate in [
            Path(__file__).parent / 'config' / 'default_config.yaml',
            Path('config') / 'default_config.yaml',
            Path('default_config.yaml'),
        ]:
            if candidate.exists():
                config = load_config(str(candidate))
                log.info('Config loaded from %s', candidate)
                break
        if config is None:
            raise FileNotFoundError(
                'Could not find default_config.yaml. '
                'Pass --config explicitly or place it at config/default_config.yaml'
            )

    out = ensure_dir(Path(config['paths']['output_dir']) / 'layer3')
    l1  = Path(layer1_dir or Path(config['paths']['output_dir']) / 'layer1')
    l2  = Path(layer2_dir or Path(config['paths']['output_dir']) / 'layer2')

    cfg_l1 = config.get('layer1', {})
    cfg_l2 = config.get('layer2', {})
    cfg_l3 = config.get('layer3', {})

    # ── Stage 0: feature attribution ──────────────────────────────────────────
    log.info('Stage 0 — Feature attribution …')
    try:
        attr = feature_attribution(
            l1, out, top_k=int(cfg_l3.get('top_k_features', 10))
        )
        log.info('  Attribution rows: %d', len(attr))
    except Exception as exc:
        log.warning('Feature attribution failed (%s) — skipping.', exc)
        import pandas as pd
        attr = pd.DataFrame()

    # ── Stage 1: confidence bands ─────────────────────────────────────────────
    log.info('Stage 1 — Confidence bands …')
    try:
        bands = confidence_bands(l1, out)
        log.info('  Providers with bands: %d', bands['provider_id'].nunique())
    except Exception as exc:
        log.warning('Confidence bands failed (%s) — skipping.', exc)
        import pandas as pd
        bands = pd.DataFrame()

    # ── Stage 2: goal attribution ─────────────────────────────────────────────
    log.info('Stage 2 — Goal attribution …')
    try:
        goals = goal_attribution(l2, out)
        n_met = int(goals['meets_alpha'].sum()) if 'meets_alpha' in goals.columns else 0
        log.info('  Goal providers: %d  met: %d', len(goals), n_met)
    except Exception as exc:
        log.warning('Goal attribution failed (%s) — skipping.', exc)
        import pandas as pd
        goals = pd.DataFrame()

    # ── Stage 3: counterfactual target curve ──────────────────────────────────
    log.info('Stage 3 — Counterfactual target curve …')
    try:
        cf_targets = cfg_l3.get('counterfactual_targets', [0.60, 0.70, 0.80])
        er_path    = l1 / 'early_release_projected.csv'
        cf = counterfactual_targets(
            l1, out,
            targets=cf_targets,
            optimization_weeks=int(cfg_l2.get('optimization_weeks', 13)),
            early_release_path=er_path if er_path.exists() else None,
        )
        log.info('  Counterfactual rows: %d', len(cf))
    except Exception as exc:
        log.warning('Counterfactual targets failed (%s) — skipping.', exc)

    # ── Stage 4: provider risk profiles (NEW) ──────────────────────────────────
    log.info('Stage 4 — Provider risk profiles …')
    try:
        risk = provider_risk_profiles(l1, out)
        if not risk.empty:
            ha = int((risk['risk_quadrant'] == 'high_attention').sum())
            log.info('  Risk profiles: %d providers, %d high-attention', len(risk), ha)
    except Exception as exc:
        log.warning('Risk profiles failed (%s) — skipping.', exc)
        import pandas as pd
        risk = pd.DataFrame()

    # ── Stage 5: drift detection (NEW) ────────────────────────────────────────
    log.info('Stage 5 — Drift detection …')
    try:
        drift = drift_detection(
            l1, out,
            drift_threshold=float(cfg_l3.get('drift_threshold', 0.50)),
        )
        log.info('  Drift flags: %d providers flagged', len(drift))
    except Exception as exc:
        log.warning('Drift detection failed (%s) — skipping.', exc)
        import pandas as pd
        drift = pd.DataFrame()

    # ── Stage 6: narrative report ─────────────────────────────────────────────
    log.info('Stage 6 — Writing narrative report …')
    report = write_narrative(out, attr, bands, goals, risk, drift)

    # ── Print summary ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  Layer 3 complete')
    print(f'  Output dir : {out}')
    print(f'  Report     : {report}')
    if not goals.empty and 'meets_alpha' in goals.columns:
        n_met   = int(goals['meets_alpha'].sum())
        n_total = int(len(goals))
        print(f'  Goals met  : {n_met} / {n_total}')
    if not risk.empty and 'risk_quadrant' in risk.columns:
        ha = int((risk['risk_quadrant'] == 'high_attention').sum())
        print(f'  High-attention providers : {ha}')
    if not drift.empty:
        print(f'  Drift flags : {len(drift)}')
    print('=' * 60 + '\n')

    return {'output_dir': str(out), 'report': str(report)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Layer 3 — Explainability')
    parser.add_argument('--config',      default=None, help='Path to config YAML')
    parser.add_argument('--layer1_dir',  default=None, help='Path to Layer 1 output dir')
    parser.add_argument('--layer2_dir',  default=None, help='Path to Layer 2 output dir')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    run(config=cfg, layer1_dir=args.layer1_dir, layer2_dir=args.layer2_dir)

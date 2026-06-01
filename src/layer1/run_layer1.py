"""
Layer 1 — Run script  (standalone, no src imports)
===================================================
All helpers from src.common.* are inlined here.
All layer1 imports use same-folder names (no src prefix).

Usage (from the folder containing these files):
    python run_layer1.py
    python run_layer1.py --config path/to/config.yaml
    python run_layer1.py --pre_layer_result path/to/prelayer_result.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# ── same-folder imports (no src prefix) ──────────────────────────────────────
from weekly_observations import build_weekly_observations, early_release_projected
from feature_engineering  import build_features
from forecasting          import train_point_forecasts, make_next_week_forecasts
from scenarios            import estimate_sigma, generate_scenarios

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
)


# ─────────────────────────────────────────────────────────────────────────────
# Inlined IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_json(path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: dict, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)


def save_df(df: pd.DataFrame, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


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
    Load blocks, providers, and optionally cases from JSON files.
    Returns raw DataFrames — normalization of nested Geisinger JSON fields
    (occurrence.start, current_blockholder.provider_id, etc.) happens
    inside weekly_observations._normalize_blocks_from_raw_json().
    """
    with open(blocks_json, encoding='utf-8') as f:
        blocks_raw = json.load(f)

    # Geisinger blocks may have nested dicts — use json_normalize for safety
    try:
        blocks_df = pd.json_normalize(blocks_raw)
    except Exception:
        blocks_df = pd.DataFrame(blocks_raw)

    with open(providers_json, encoding='utf-8') as f:
        providers_raw = json.load(f)
    providers_df = pd.DataFrame(providers_raw)

    cases_df = pd.DataFrame()
    if cases_json and Path(cases_json).exists():
        with open(cases_json, encoding='utf-8') as f:
            cases_raw = json.load(f)
        cases_df = pd.DataFrame(cases_raw)

    log.info('Loaded: blocks=%d  providers=%d  cases=%d',
             len(blocks_df), len(providers_df), len(cases_df))
    return blocks_df, providers_df, cases_df


def save_residual_plot(predictions: pd.DataFrame, path, target_col: str = 'casetime_min') -> None:
    """Save a simple actual-vs-predicted residuals plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        pred_col = f'pred_{target_col}'
        if pred_col not in predictions.columns or target_col not in predictions.columns:
            return

        actual = predictions[target_col].values
        pred   = predictions[pred_col].values
        resid  = actual - pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(pred, actual, alpha=0.4, s=10)
        mn, mx = min(pred.min(), actual.min()), max(pred.max(), actual.max())
        axes[0].plot([mn, mx], [mn, mx], 'r--', lw=1.5)
        axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
        axes[0].set_title(f'{target_col} — Predicted vs Actual')

        axes[1].hist(resid, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
        axes[1].axvline(0, color='red', lw=1.5, ls='--')
        axes[1].set_xlabel('Residual (actual − predicted)')
        axes[1].set_title(f'{target_col} — Residuals')

        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log.info('Residual plot saved → %s', path)
    except Exception as exc:
        log.warning('Residual plot skipped (%s)', exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main run function
# ─────────────────────────────────────────────────────────────────────────────

def run(
    config: Optional[dict] = None,
    pre_layer_result_path: Optional[str] = None,
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

    out = ensure_dir(Path(config['paths']['output_dir']) / 'layer1')

    # ── Pre-Layer result ──────────────────────────────────────────────────────
    pre_path = Path(
        pre_layer_result_path
        or Path(config['paths']['output_dir']) / 'pre_layer' / 'prelayer_result.json'
    )
    if not pre_path.exists():
        raise FileNotFoundError(f'Pre-Layer result not found: {pre_path}')

    log.info('Loading Pre-Layer result: %s', pre_path)
    prelayer = load_json(pre_path)

    # ── Raw data ──────────────────────────────────────────────────────────────
    log.info('Loading raw tables …')
    blocks_df, providers_df, cases_df = load_raw_tables(
        config['paths']['blocks_json'],
        config['paths']['providers_json'],
        config['paths'].get('cases_json'),
    )

    # ── Stage 0: weekly observations ─────────────────────────────────────────
    log.info('Stage 0 — Building weekly observations …')
    obs, obs_meta = build_weekly_observations(
        blocks_df, providers_df, cases_df, prelayer, config
    )
    save_df(obs, out / 'weekly_observations.csv')

    if obs_meta['warnings']:
        log.warning('Observation warnings:')
        for w in obs_meta['warnings']:
            log.warning('  • %s', w)

    # ── Early release projection ──────────────────────────────────────────────
    er = early_release_projected(
        blocks_df,
        optimization_weeks=int(config.get('layer2', {}).get('optimization_weeks', 13)),
    )
    save_df(er, out / 'early_release_projected.csv')

    # ── Stage 1+2: feature engineering ───────────────────────────────────────
    log.info('Stage 1+2 — Building features …')
    features, attn_meta = build_features(obs, prelayer, config)
    save_df(features, out / 'features.csv')
    log.info('  Features: %d rows × %d columns', *features.shape)

    # ── Stage 3: point forecasts ──────────────────────────────────────────────
    log.info('Stage 3 — Training point forecast models …')
    model_result = train_point_forecasts(features, out, config)

    log.info('Stage 3 — Making point forecasts …')
    point = make_next_week_forecasts(
        features,
        model_result['models'],
        model_result['feature_columns'],
        out,
    )

    # ── Stage 4: sigma estimation ─────────────────────────────────────────────
    log.info('Stage 4 — Estimating σ_pd (hierarchical shrinkage) …')
    sigma = estimate_sigma(features, model_result['predictions'])
    save_df(sigma, out / 'hierarchical_sigma_estimates.csv')

    # ── Stage 5: scenario generation ──────────────────────────────────────────
    log.info('Stage 5 — Generating scenarios …')
    n_scen = int(config.get('layer1', {}).get('n_scenarios', 200))
    scenarios = generate_scenarios(point, sigma, out, config)

    # ── Metadata ──────────────────────────────────────────────────────────────
    save_json(
        {
            'warnings':    obs_meta['warnings'],
            'attention':   attn_meta,
            'n_obs_rows':  int(len(obs)),
            'n_features':  int(len(features)),
            'n_scenarios': n_scen,
            'model_metrics': model_result['metrics'],
        },
        out / 'layer1_metadata.json',
    )

    # ── Residual plots ────────────────────────────────────────────────────────
    save_residual_plot(model_result['predictions'], out / 'casetime_residuals.png',  'casetime_min')
    save_residual_plot(model_result['predictions'], out / 'turnover_residuals.png',  'turnover_min')

    log.info('=' * 60)
    log.info('Layer 1 complete')
    log.info('  Observation rows : %d', len(obs))
    log.info('  Feature rows     : %d', len(features))
    log.info('  Scenario rows    : %d', len(scenarios))
    log.info('  Output dir       : %s', out)
    log.info('=' * 60)

    return {'output_dir': str(out)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Layer 1 — Stochastic Demand Forecasting')
    parser.add_argument('--config',            default=None, help='Path to config YAML')
    parser.add_argument('--pre_layer_result',  default=None, help='Path to prelayer_result.json')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    run(config=cfg, pre_layer_result_path=args.pre_layer_result)

    sys.stdout.flush()
    sys.stderr.flush()

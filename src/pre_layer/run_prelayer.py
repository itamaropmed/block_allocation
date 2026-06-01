"""
Pre-Layer — Run script  (standalone, no src imports)
=====================================================
Place this file in the same folder as template_reconstruction.py.
All helpers from src.common.* are inlined here, including
a proper Geisinger JSON normalizer for the nested block schema.

Usage:
    python run_prelayer.py
    python run_prelayer.py --config path/to/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from template_reconstruction import reconstruct_template   # same-folder import

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


# ─────────────────────────────────────────────────────────────────────────────
# Geisinger JSON normalizer  (inlined from Layer 1 weekly_observations)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_blocks(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the Geisinger nested JSON block schema into the flat columns
    that template_reconstruction.py expects:
        provider_id, physical_site, room_type,
        start_min, end_min, duration_min,
        manual_early_release_min,
        week_start (YYYY-MM-DD, Monday of occurrence week),
        week_index (int 0-based),
        day_of_week (int 0=Mon),
        date (YYYY-MM-DD of occurrence),
        service_line_from_block
    """
    b = raw_df.copy()

    # ── provider_id ──────────────────────────────────────────────────────────
    if 'provider_id' not in b.columns:
        if 'current_blockholder' in b.columns:
            b['provider_id'] = b['current_blockholder'].map(
                lambda v: str(v.get('provider_id', 'OPEN') or 'OPEN').strip()
                if isinstance(v, dict) else 'OPEN'
            )
        elif 'current_blockholder.provider_id' in b.columns:
            b['provider_id'] = b['current_blockholder.provider_id'].fillna('OPEN').astype(str)
        else:
            b['provider_id'] = 'OPEN'
    b['provider_id'] = b['provider_id'].fillna('OPEN').astype(str).str.strip()
    b.loc[b['provider_id'].isin(['', 'None', 'nan', 'NaN', 'NULL']), 'provider_id'] = 'OPEN'

    # ── physical_site ────────────────────────────────────────────────────────
    if 'physical_site' not in b.columns:
        for col in ('site', 'location', 'facility'):
            if col in b.columns:
                b['physical_site'] = b[col].fillna('Unknown').astype(str)
                break
        else:
            b['physical_site'] = 'Unknown'

    # ── room_type ────────────────────────────────────────────────────────────
    if 'room_type' not in b.columns:
        if 'room' in b.columns:
            b['room_type'] = b['room'].map(
                lambda v: str(v.get('type', 'Unknown') or 'Unknown').strip()
                if isinstance(v, dict) else 'Unknown'
            )
        elif 'room.type' in b.columns:
            b['room_type'] = b['room.type'].fillna('Unknown').astype(str)
        else:
            b['room_type'] = 'Unknown'

    # ── service_line_from_block ───────────────────────────────────────────────
    if 'service_line_from_block' not in b.columns:
        for col in ('service_line', 'service', 'specialty', 'room_service_line'):
            if col in b.columns:
                b['service_line_from_block'] = b[col].fillna('Unknown').astype(str)
                break
        else:
            b['service_line_from_block'] = 'Unknown'

    # ── occurrence start / end datetimes ─────────────────────────────────────
    if 'occurrence_start' not in b.columns:
        if 'occurrence' in b.columns:
            b['occurrence_start'] = b['occurrence'].map(
                lambda v: v.get('start') if isinstance(v, dict) else None
            )
            b['occurrence_end'] = b['occurrence'].map(
                lambda v: v.get('end')   if isinstance(v, dict) else None
            )
        elif 'occurrence.start' in b.columns:
            b['occurrence_start'] = b['occurrence.start']
            b['occurrence_end']   = b.get('occurrence.end', pd.Series([None] * len(b)))
        else:
            b['occurrence_start'] = None
            b['occurrence_end']   = None

    b['occurrence_start'] = pd.to_datetime(b['occurrence_start'], utc=True, errors='coerce')
    b['occurrence_end']   = pd.to_datetime(b['occurrence_end'],   utc=True, errors='coerce')

    # ── start_min / end_min (minutes since midnight) ──────────────────────────
    has_ts = b['occurrence_start'].notna()
    b['start_min'] = 0.0
    b['end_min']   = 0.0
    b.loc[has_ts, 'start_min'] = (
        b.loc[has_ts, 'occurrence_start'].dt.hour * 60
        + b.loc[has_ts, 'occurrence_start'].dt.minute
    ).astype(float)
    b.loc[has_ts, 'end_min'] = (
        b.loc[has_ts, 'occurrence_end'].dt.hour * 60
        + b.loc[has_ts, 'occurrence_end'].dt.minute
    ).where(b.loc[has_ts, 'occurrence_end'].notna(), b.loc[has_ts, 'start_min'] + 480)

    # ── duration_min ─────────────────────────────────────────────────────────
    if 'duration_min' not in b.columns:
        valid = has_ts & b['occurrence_end'].notna()
        b['duration_min'] = 0.0
        b.loc[valid, 'duration_min'] = (
            (b.loc[valid, 'occurrence_end'] - b.loc[valid, 'occurrence_start'])
            .dt.total_seconds() / 60.0
        ).clip(lower=0.0)

    # ── manual_early_release_min ──────────────────────────────────────────────
    if 'manual_early_release_min' not in b.columns:
        if 'manual_early_release' in b.columns:
            er = pd.to_datetime(b['manual_early_release'], utc=True, errors='coerce')
            valid = er.notna() & b['occurrence_end'].notna()
            b['manual_early_release_min'] = 0.0
            b.loc[valid, 'manual_early_release_min'] = (
                (b.loc[valid, 'occurrence_end'] - er[valid])
                .dt.total_seconds() / 60.0
            ).clip(lower=0.0)
        else:
            b['manual_early_release_min'] = 0.0

    # ── date, week_start, week_index, day_of_week ─────────────────────────────
    b['date'] = b['occurrence_start'].dt.date.astype(str).where(has_ts, '')

    b['_monday'] = b['occurrence_start'] - pd.to_timedelta(
        b['occurrence_start'].dt.weekday, unit='D'
    )
    b['week_start'] = b['_monday'].dt.date.astype(str).where(has_ts, None)

    unique_mondays  = sorted(b.loc[has_ts, 'week_start'].dropna().unique())
    monday_to_idx   = {w: i for i, w in enumerate(unique_mondays)}
    b['week_index'] = b['week_start'].map(monday_to_idx).fillna(0).astype(int)
    b['day_of_week'] = b['occurrence_start'].dt.weekday.where(has_ts, -1).fillna(-1).astype(int)

    b.drop(columns=['_monday'], inplace=True, errors='ignore')
    return b.reset_index(drop=True)


def load_raw_tables(
    blocks_json: str,
    providers_json: str,
    cases_json: Optional[str] = None,
):
    """Load + normalize blocks/providers/cases from JSON files."""
    with open(blocks_json, encoding='utf-8') as f:
        raw_blocks = json.load(f)

    # json_normalize flattens nested dicts (occurrence.start etc.)
    try:
        blocks_df = pd.json_normalize(raw_blocks)
    except Exception:
        blocks_df = pd.DataFrame(raw_blocks)

    # Normalize Geisinger nested schema → flat columns
    blocks_df = _normalize_blocks(blocks_df)

    with open(providers_json, encoding='utf-8') as f:
        providers_df = pd.DataFrame(json.load(f))

    cases_df = pd.DataFrame()
    if cases_json and Path(cases_json).exists():
        with open(cases_json, encoding='utf-8') as f:
            cases_df = pd.DataFrame(json.load(f))

    log.info(
        'Loaded: blocks=%d  providers=%d  cases=%d',
        len(blocks_df), len(providers_df), len(cases_df),
    )
    return blocks_df, providers_df, cases_df


# ─────────────────────────────────────────────────────────────────────────────
# BIC plot  (inlined from src.common.plotting)
# ─────────────────────────────────────────────────────────────────────────────

def save_bic_plot(bic_df: pd.DataFrame, path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: BIC by T
        axes[0].bar(bic_df['T'], bic_df['bic'], color='steelblue',
                    edgecolor='white', linewidth=0.6)
        best_T = int(bic_df.loc[bic_df['bic'].idxmin(), 'T'])
        axes[0].axvline(best_T, color='crimson', lw=2, ls='--',
                        label=f'T* = {best_T}')
        axes[0].set_xlabel('Candidate period T (weeks)')
        axes[0].set_ylabel('BIC score  (lower = better)')
        axes[0].set_title('BIC Period Detection')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: deviations by T
        axes[1].bar(bic_df['T'], bic_df['deviations'],
                    color='darkorange', edgecolor='white', linewidth=0.6)
        axes[1].axvline(best_T, color='crimson', lw=2, ls='--',
                        label=f'T* = {best_T}')
        axes[1].set_xlabel('Candidate period T (weeks)')
        axes[1].set_ylabel('Total deviations D(T)')
        axes[1].set_title('Deviation Count by Period')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log.info('BIC plot saved → %s', path)
    except Exception as exc:
        log.warning('BIC plot skipped (%s)', exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main run function
# ─────────────────────────────────────────────────────────────────────────────

def run(config: Optional[dict] = None) -> dict:

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

    out = ensure_dir(Path(config['paths']['output_dir']) / 'pre_layer')

    # ── Load + normalize raw data ─────────────────────────────────────────────
    log.info('Loading and normalizing raw tables …')
    blocks_df, providers_df, _ = load_raw_tables(
        config['paths']['blocks_json'],
        config['paths']['providers_json'],
        config['paths'].get('cases_json'),
    )
    log.info(
        'Blocks ready: %d rows  |  week_index range [%d, %d]  |  '
        'unique cells preview (site/room/dow): %d',
        len(blocks_df),
        int(blocks_df['week_index'].min()),
        int(blocks_df['week_index'].max()),
        int(blocks_df.groupby(['physical_site', 'room_type', 'day_of_week']).ngroups),
    )

    # ── Template reconstruction ───────────────────────────────────────────────
    log.info('Running template reconstruction …')
    result = reconstruct_template(blocks_df, providers_df, config)

    # ── Extract private DataFrames ────────────────────────────────────────────
    canonical_blocks = result.pop('_canonical_blocks_df')
    template_df      = result.pop('_template_df')
    bic_df           = result.pop('_bic_df')
    offset_df        = result.pop('_offset_df')

    # ── Save artefacts ────────────────────────────────────────────────────────
    save_df(canonical_blocks, out / 'canonical_blocks.csv')
    save_df(template_df,      out / 'block_template.csv')
    save_df(bic_df,           out / 'bic_scores.csv')
    save_df(offset_df,        out / 'phase_offset_scores.csv')
    save_json(result,          out / 'prelayer_result.json')
    save_bic_plot(bic_df,     out / 'bic_scores.png')

    # ── Print summary ─────────────────────────────────────────────────────────
    T_star   = result['T_star']
    k_offset = result['phase_offset']
    n_tmpl   = len(template_df)
    n_exc    = len(result['exceptions_list'])
    warnings = result.get('warnings', [])

    print('\n' + '=' * 60)
    print('  Pre-Layer complete')
    print(f'  T*              : {T_star}  (rotation period in weeks)')
    print(f'  Phase offset    : {k_offset}')
    print(f'  Template blocks : {n_tmpl}')
    print(f'  Exceptions      : {n_exc}')
    print(f'  Output dir      : {out}')
    if warnings:
        print(f'  Warnings ({len(warnings)}):')
        for w in warnings:
            print(f'    • {w}')
    print('=' * 60 + '\n')

    print('\nBIC scores:')
    print(bic_df[['T', 'deviations', 'bic']].to_string(index=False))

    return {
        'output_dir':  str(out),
        'result_json': str(out / 'prelayer_result.json'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-Layer — Template Reconstruction')
    parser.add_argument('--config', default=None, help='Path to config YAML')
    args   = parser.parse_args()
    cfg    = load_config(args.config) if args.config else None
    run(config=cfg)

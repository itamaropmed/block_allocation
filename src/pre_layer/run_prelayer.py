"""
Pre-Layer — Run script  (standalone, no src imports)
=====================================================
Place this file in the same folder as template_reconstruction.py.

TWO ways to point it at your data:
  1. Direct args (no config needed):
       python run_prelayer.py \
           --blocks   ../../data/raw/geisinger-users_blocks.json \
           --providers ../../data/raw/geisinger-users_providers.json \
           --cases    ../../data/raw/geisinger-users_cases.json \
           --out      ../../outputs/pre_layer

  2. Config file:
       python run_prelayer.py --config ../../config/default_config.yaml

If neither is supplied the script tries to auto-locate the JSONs by
walking up from its own directory looking for data/raw/*.json.

Critical fix in this version:
  pd.json_normalize can create both current_blockholder.provider_id and
  current_blockholder. In the Geisinger raw data, current_blockholder is often
  an all-NaN helper column, while current_blockholder.provider_id contains the
  real PRV-* ids. The normalizer therefore prefers the flattened provider_id
  column first and only falls back to the nested dict column if needed.
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
# IO helpers
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
# Auto-locate data files (walks up from script dir looking for data/raw/)
# ─────────────────────────────────────────────────────────────────────────────

def _find_data_dir() -> Optional[Path]:
    """
    Locate the project's data/raw folder.

    This supports both common layouts:
      project_root/run_prelayer.py
      project_root/src/pre_layer/run_prelayer.py

    It also checks from the current working directory, because many runs are
    launched from the project root while the script itself lives in src/.
    """
    anchors = [Path(__file__).parent.resolve(), Path.cwd().resolve()]
    seen = set()

    for anchor in anchors:
        for parent in [anchor] + list(anchor.parents)[:8]:
            for candidate in (
                parent / 'data' / 'raw',
                parent / '..' / 'data' / 'raw',
                parent / '..' / '..' / 'data' / 'raw',
            ):
                candidate = candidate.resolve()
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate.is_dir():
                    return candidate
    return None


def _auto_paths() -> dict[str, Optional[str]]:
    data_dir = _find_data_dir()
    if data_dir is None:
        return {'blocks': None, 'providers': None, 'cases': None}
    return {
        'blocks':    str(data_dir / 'geisinger-users_blocks.json'),
        'providers': str(data_dir / 'geisinger-users_providers.json'),
        'cases':     str(data_dir / 'geisinger-users_cases.json'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Block JSON normalizer
# ─────────────────────────────────────────────────────────────────────────────

_MISSING_ID_STRINGS = {'', 'None', 'none', 'nan', 'NaN', 'NULL', 'null', '<NA>'}


def _clean_id_series(s: pd.Series, default: str = 'OPEN') -> pd.Series:
    """Clean id-like string columns while preserving real ids such as PRV-0001."""
    out = s.fillna(default).astype(str).str.strip()
    out = out.mask(out.isin(_MISSING_ID_STRINGS), default)
    return out


def _prefer_non_missing(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    """Use fallback where primary is missing/empty/OPEN."""
    p = primary.copy()
    p_as_str = p.fillna('').astype(str).str.strip()
    bad = p.isna() | p_as_str.isin(_MISSING_ID_STRINGS | {'OPEN'})
    return p.where(~bad, fallback)


def _normalize_blocks(raw_df: pd.DataFrame) -> pd.DataFrame:
    b = raw_df.copy()

    # Provider extraction — this is the important bug fix.
    # pd.json_normalize may create a current_blockholder column that is all NaN
    # and a separate current_blockholder.provider_id column with the real PRV-* ids.
    # The old code checked current_blockholder first and therefore converted all
    # blocks to OPEN. We now prefer the flattened provider_id column first.
    raw_provider = pd.Series(['OPEN'] * len(b), index=b.index, dtype='object')

    if 'provider_id' in b.columns:
        raw_provider = b['provider_id'].copy()

    if 'current_blockholder.provider_id' in b.columns:
        nested_provider = b['current_blockholder.provider_id']
        raw_provider = _prefer_non_missing(raw_provider, nested_provider)
    elif 'current_blockholder' in b.columns:
        nested_provider = b['current_blockholder'].map(
            lambda v: v.get('provider_id') if isinstance(v, dict) else None
        )
        raw_provider = _prefer_non_missing(raw_provider, nested_provider)

    b['provider_id'] = _clean_id_series(raw_provider, default='OPEN')

    if 'provider_name' not in b.columns:
        if 'current_blockholder.name' in b.columns:
            b['provider_name'] = b['current_blockholder.name'].fillna('').astype(str).str.strip()
        elif 'current_blockholder' in b.columns:
            b['provider_name'] = b['current_blockholder'].map(
                lambda v: str(v.get('name', '') or '').strip()
                if isinstance(v, dict) else ''
            )
        else:
            b['provider_name'] = ''

    if 'physical_site' not in b.columns:
        for col in ('site', 'physical_site', 'location', 'facility'):
            if col in b.columns:
                b['physical_site'] = b[col].fillna('Unknown').astype(str).str.strip()
                break
        else:
            b['physical_site'] = 'Unknown'
    b.loc[b['physical_site'].isin(['', 'None', 'nan', 'NaN', 'NULL']), 'physical_site'] = 'Unknown'

    if 'room_type' not in b.columns:
        if 'room.type' in b.columns:
            b['room_type'] = b['room.type'].fillna('Unknown').astype(str).str.strip()
        elif 'room' in b.columns:
            b['room_type'] = b['room'].map(
                lambda v: str(v.get('type', 'Unknown') or 'Unknown').strip()
                if isinstance(v, dict) else 'Unknown'
            )
        else:
            b['room_type'] = 'Unknown'
    b.loc[b['room_type'].isin(['', 'None', 'nan', 'NaN', 'NULL']), 'room_type'] = 'Unknown'

    if 'service_line_from_block' not in b.columns:
        for col in ('service_line', 'service', 'specialty', 'room_service_line'):
            if col in b.columns:
                b['service_line_from_block'] = b[col].fillna('Unknown').astype(str).str.strip()
                break
        else:
            b['service_line_from_block'] = 'Unknown'

    if 'occurrence_start' not in b.columns:
        if 'occurrence.start' in b.columns:
            b['occurrence_start'] = b['occurrence.start']
            b['occurrence_end']   = b.get('occurrence.end', pd.Series([None] * len(b), index=b.index))
        elif 'occurrence' in b.columns:
            b['occurrence_start'] = b['occurrence'].map(
                lambda v: v.get('start') if isinstance(v, dict) else None)
            b['occurrence_end']   = b['occurrence'].map(
                lambda v: v.get('end')   if isinstance(v, dict) else None)
        else:
            b['occurrence_start'] = None
            b['occurrence_end']   = None

    b['occurrence_start'] = pd.to_datetime(b['occurrence_start'], utc=True, errors='coerce')
    b['occurrence_end']   = pd.to_datetime(b['occurrence_end'],   utc=True, errors='coerce')

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

    if 'duration_min' not in b.columns:
        valid = has_ts & b['occurrence_end'].notna()
        b['duration_min'] = 0.0
        b.loc[valid, 'duration_min'] = (
            (b.loc[valid, 'occurrence_end'] - b.loc[valid, 'occurrence_start'])
            .dt.total_seconds() / 60.0
        ).clip(lower=0.0)

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

    b['date'] = b['occurrence_start'].dt.date.astype(str).where(has_ts, '')
    b['_monday'] = b['occurrence_start'] - pd.to_timedelta(
        b['occurrence_start'].dt.weekday, unit='D')
    b['week_start'] = b['_monday'].dt.date.astype(str).where(has_ts, None)
    unique_mondays  = sorted(b.loc[has_ts, 'week_start'].dropna().unique())
    monday_to_idx   = {w: i for i, w in enumerate(unique_mondays)}
    b['week_index']  = b['week_start'].map(monday_to_idx).fillna(0).astype(int)
    b['day_of_week'] = b['occurrence_start'].dt.weekday.where(has_ts, -1).fillna(-1).astype(int)
    b.drop(columns=['_monday'], inplace=True, errors='ignore')
    return b.reset_index(drop=True)


def _log_block_diagnostics(blocks_df: pd.DataFrame) -> None:
    """Print diagnostics that catch the all-OPEN provider bug immediately."""
    if blocks_df.empty or 'provider_id' not in blocks_df.columns:
        log.warning('Block diagnostics skipped: no provider_id column')
        return

    pid = blocks_df['provider_id'].fillna('OPEN').astype(str).str.strip()
    n_rows = int(len(pid))
    n_unique = int(pid.nunique(dropna=True))
    n_open = int((pid == 'OPEN').sum())
    n_non_open = n_rows - n_open
    open_rate = n_open / max(1, n_rows)

    log.info(
        'Provider extraction: rows=%d  unique_provider_ids=%d  non_OPEN=%d  OPEN=%d (%.1f%%)',
        n_rows, n_unique, n_non_open, n_open, 100.0 * open_rate,
    )

    top = pid.value_counts(dropna=False).head(10)
    log.info('Top provider_id values after normalization:\n%s', top.to_string())

    if n_non_open == 0:
        log.error(
            'All blocks are OPEN after normalization. T* will be meaningless. '
            'Check current_blockholder.provider_id parsing.'
        )
    elif n_unique <= 2:
        log.warning(
            'Very low provider diversity after normalization. T* may be unreliable.'
        )


def load_raw_tables(blocks_json: str, providers_json: str,
                    cases_json: Optional[str] = None):
    with open(blocks_json, encoding='utf-8') as f:
        raw_blocks = json.load(f)
    try:
        blocks_df = pd.json_normalize(raw_blocks)
    except Exception:
        blocks_df = pd.DataFrame(raw_blocks)
    blocks_df = _normalize_blocks(blocks_df)
    _log_block_diagnostics(blocks_df)

    with open(providers_json, encoding='utf-8') as f:
        providers_df = pd.DataFrame(json.load(f))

    cases_df = pd.DataFrame()
    if cases_json and Path(cases_json).exists():
        with open(cases_json, encoding='utf-8') as f:
            cases_df = pd.DataFrame(json.load(f))

    log.info('Loaded: blocks=%d  providers=%d  cases=%d',
             len(blocks_df), len(providers_df), len(cases_df))
    return blocks_df, providers_df, cases_df


# ─────────────────────────────────────────────────────────────────────────────
# BIC plot
# ─────────────────────────────────────────────────────────────────────────────

def save_bic_plot(bic_df: pd.DataFrame, path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        best_T = int(bic_df.loc[bic_df['bic'].idxmin(), 'T'])
        axes[0].bar(bic_df['T'], bic_df['bic'], color='steelblue',
                    edgecolor='white', linewidth=0.6)
        axes[0].axvline(best_T, color='crimson', lw=2, ls='--', label=f'T* = {best_T}')
        axes[0].set_xlabel('Candidate period T (weeks)')
        axes[0].set_ylabel('BIC score')
        axes[0].set_title('BIC Period Detection')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].bar(bic_df['T'], bic_df['deviations'],
                    color='darkorange', edgecolor='white', linewidth=0.6)
        axes[1].axvline(best_T, color='crimson', lw=2, ls='--', label=f'T* = {best_T}')
        axes[1].set_xlabel('Candidate period T (weeks)')
        axes[1].set_ylabel('Total deviations D(T)')
        axes[1].set_title('Deviation Count by Period')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        log.info('BIC plot saved → %s', path)
    except Exception as exc:
        log.warning('BIC plot skipped (%s)', exc)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(result: dict, providers_df: pd.DataFrame, expected_T: Optional[int] = None) -> bool:
    """
    Run sanity checks and print a pass/fail table.
    Returns True if all critical checks pass.
    """
    T_star         = result['T_star']
    template       = result['block_template']
    warnings_list  = result.get('warnings', [])
    n_tmpl         = len(template)
    exceptions     = result['exceptions_list']
    exc_rate_vals  = list(result['exception_rate_per_series'].values())
    avg_exc_rate   = sum(exc_rate_vals) / max(len(exc_rate_vals), 1)

    prov_pids      = set(providers_df['provider_id'].astype(str).unique())
    tmpl_pids      = {
        row['dominant_provider_id']
        for row in template
        if row.get('dominant_provider_id') not in ('OPEN', '', 'nan', None)
    }
    missing_in_providers = tmpl_pids - prov_pids

    # Extra guard: T*=1 with zero deviations is often caused by all providers
    # accidentally becoming OPEN during normalization. Check the canonical block
    # DataFrame when it is available.
    canonical_df = result.get('_canonical_blocks_df')
    if canonical_df is not None and not canonical_df.empty and 'provider_id' in canonical_df.columns:
        provider_series = canonical_df['provider_id'].fillna('OPEN').astype(str).str.strip()
        non_open_count = int((provider_series != 'OPEN').sum())
        unique_provider_count = int(provider_series.nunique())
    else:
        non_open_count = None
        unique_provider_count = None

    checks = [
        ('T* detected',
         f'T* = {T_star}',
         T_star >= 1,
         'PASS' if T_star >= 1 else 'FAIL'),

        ('T* matches expected' if expected_T else None,
         f'T* = {T_star}, expected = {expected_T}' if expected_T else None,
         T_star == expected_T if expected_T else True,
         ('PASS' if T_star == expected_T else 'FAIL') if expected_T else 'SKIP'),

        ('Template non-empty',
         f'{n_tmpl} template blocks',
         n_tmpl > 0,
         'PASS' if n_tmpl > 0 else 'FAIL'),

        ('Provider extraction non-empty',
         f'non-OPEN blocks = {non_open_count}' if non_open_count is not None else 'canonical blocks unavailable',
         (non_open_count is None) or (non_open_count > 0),
         'PASS' if (non_open_count is None or non_open_count > 0) else 'FAIL'),

        ('Provider diversity reasonable',
         f'unique provider values = {unique_provider_count}' if unique_provider_count is not None else 'canonical blocks unavailable',
         (unique_provider_count is None) or (unique_provider_count > 2),
         'PASS' if (unique_provider_count is None or unique_provider_count > 2) else 'WARN'),

        ('No missing provider IDs',
         f'{len(missing_in_providers)} dominant providers not in providers.json',
         len(missing_in_providers) == 0,
         'PASS' if len(missing_in_providers) == 0 else 'WARN'),

        ('Exception rate reasonable',
         f'avg exception rate = {avg_exc_rate:.1%}',
         avg_exc_rate < 0.30,
         'PASS' if avg_exc_rate < 0.30 else 'WARN'),

        ('No pre-layer warnings',
         f'{len(warnings_list)} warning(s)',
         len(warnings_list) == 0,
         'PASS' if len(warnings_list) == 0 else 'WARN'),

        ('phase_labels consistent',
         'phase_labels keyed by week_index int strings',
         all(isinstance(k, str) and k.isdigit()
             for k in result.get('phase_labels', {}).keys()),
         'PASS'),
    ]

    print('\n' + '─' * 62)
    print('  VALIDATION REPORT')
    print('─' * 62)
    all_pass = True
    for check in checks:
        if check[0] is None:
            continue
        name, detail, ok, label = check
        icon = '✓' if label == 'PASS' else ('⚠' if label == 'WARN' else '✗')
        print(f'  {icon} {label:4s}  {name:35s}  {detail}')
        if label == 'FAIL':
            all_pass = False
    print('─' * 62)
    if missing_in_providers:
        print(f'  Missing PIDs (first 10): {sorted(missing_in_providers)[:10]}')
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Main run function
# ─────────────────────────────────────────────────────────────────────────────

def run(
    config: Optional[dict] = None,
    blocks_path:    Optional[str] = None,
    providers_path: Optional[str] = None,
    cases_path:     Optional[str] = None,
    out_dir:        Optional[str] = None,
    expected_T:     Optional[int] = None,
) -> dict:
    """
    Parameters
    ----------
    config         : Config dict (loaded from YAML if not given).
    blocks_path    : Direct path to blocks JSON  — overrides config.
    providers_path : Direct path to providers JSON — overrides config.
    cases_path     : Direct path to cases JSON — overrides config.
    out_dir        : Output directory — overrides config.
    expected_T     : If set, validation checks that T* equals this value.
    """

    # ── Resolve data paths ────────────────────────────────────────────────────
    # Priority: direct args > config > auto-detect
    if blocks_path and providers_path:
        # Direct args supplied — build a minimal config
        _out = out_dir or str(Path(__file__).parent.parent.parent / 'outputs')
        cfg = {
            'paths': {
                'blocks_json':    blocks_path,
                'providers_json': providers_path,
                'cases_json':     cases_path,
                'output_dir':     _out,
            },
            'pre_layer': {},
            'layer2':    {'optimization_weeks': 13},
        }
    else:
        if config is None:
            for candidate in [
                Path(__file__).parent / 'config' / 'default_config.yaml',
                Path(__file__).parent.parent.parent / 'config' / 'default_config.yaml',
                Path('config') / 'default_config.yaml',
                Path('default_config.yaml'),
            ]:
                if candidate.exists():
                    config = load_config(str(candidate))
                    log.info('Config loaded from %s', candidate)
                    break

        if config is None:
            # Last resort: auto-detect data files
            auto = _auto_paths()
            if auto['blocks'] and Path(auto['blocks']).exists():
                log.info('Auto-detected data dir: %s',
                         Path(auto['blocks']).parent)
                _out = out_dir or str(Path(__file__).parent.parent.parent / 'outputs')
                cfg = {
                    'paths': {
                        'blocks_json':    auto['blocks'],
                        'providers_json': auto['providers'],
                        'cases_json':     auto['cases'],
                        'output_dir':     _out,
                    },
                    'pre_layer': {},
                    'layer2':    {'optimization_weeks': 13},
                }
            else:
                raise FileNotFoundError(
                    'Cannot find data files.\n'
                    'Run with:  python run_prelayer.py \\\n'
                    '    --blocks   path/to/geisinger-users_blocks.json \\\n'
                    '    --providers path/to/geisinger-users_providers.json \\\n'
                    '    --cases    path/to/geisinger-users_cases.json'
                )
        else:
            cfg = config

    if out_dir:
        cfg['paths']['output_dir'] = out_dir

    out = ensure_dir(Path(cfg['paths']['output_dir']) / 'pre_layer')

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info('Loading and normalizing raw tables …')
    blocks_df, providers_df, _ = load_raw_tables(
        cfg['paths']['blocks_json'],
        cfg['paths']['providers_json'],
        cfg['paths'].get('cases_json'),
    )
    log.info(
        'Blocks ready: %d rows  |  week_index range [%d, %d]  |  '
        'unique (site/room/dow) cells: %d',
        len(blocks_df),
        int(blocks_df['week_index'].min()),
        int(blocks_df['week_index'].max()),
        int(blocks_df.groupby(['physical_site', 'room_type', 'day_of_week']).ngroups),
    )

    # ── Template reconstruction ───────────────────────────────────────────────
    log.info('Running template reconstruction …')
    result = reconstruct_template(blocks_df, providers_df, cfg)

    # ── Extract DataFrames ────────────────────────────────────────────────────
    canonical_blocks = result.pop('_canonical_blocks_df')
    template_df      = result.pop('_template_df')
    bic_df           = result.pop('_bic_df')
    offset_df        = result.pop('_offset_df')

    # ── Save ──────────────────────────────────────────────────────────────────
    save_df(canonical_blocks, out / 'canonical_blocks.csv')
    save_df(template_df,      out / 'block_template.csv')
    save_df(bic_df,           out / 'bic_scores.csv')
    save_df(offset_df,        out / 'phase_offset_scores.csv')
    save_json(result,          out / 'prelayer_result.json')
    save_bic_plot(bic_df,     out / 'bic_scores.png')

    # ── Print summary ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  Pre-Layer complete')
    print(f"  T*              : {result['T_star']}  (rotation period in weeks)")
    print(f"  Phase offset    : {result['phase_offset']}")
    print(f'  Template blocks : {len(template_df)}')
    print(f"  Exceptions      : {len(result['exceptions_list'])}")
    print(f'  Output dir      : {out}')
    for w in result.get('warnings', []):
        print(f'  ⚠  {w}')
    print('=' * 60)

    print('\nBIC scores:')
    print(bic_df[['T', 'deviations', 'bic']].to_string(index=False))

    # ── Validation ────────────────────────────────────────────────────────────
    # The JSON-safe result has no private DataFrames, but validation benefits
    # from checking the canonical provider_id distribution.
    validation_result = dict(result)
    validation_result['_canonical_blocks_df'] = canonical_blocks
    validate(validation_result, providers_df, expected_T=expected_T)

    return {
        'output_dir':  str(out),
        'result_json': str(out / 'prelayer_result.json'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-Layer — Template Reconstruction',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--config',    default=None,
                        help='Path to config YAML (optional if --blocks etc. supplied)')
    parser.add_argument('--blocks',    default=None,
                        help='Path to blocks JSON  (overrides config)')
    parser.add_argument('--providers', default=None,
                        help='Path to providers JSON (overrides config)')
    parser.add_argument('--cases',     default=None,
                        help='Path to cases JSON (overrides config, optional)')
    parser.add_argument('--out',       default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--expect_T',  type=int, default=None,
                        help='Expected T* — validation fails if BIC selects a different value')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None

    run(
        config         = cfg,
        blocks_path    = args.blocks,
        providers_path = args.providers,
        cases_path     = args.cases,
        out_dir        = args.out,
        expected_T     = args.expect_T,
    )
from __future__ import annotations

from pathlib import Path

from src.common.block_schema import load_raw_tables
from src.common.io import ensure_dir, load_config, save_df, save_json
from src.common.plotting import save_bic_plot
from src.pre_layer.template_reconstruction import reconstruct_template


def run(config: dict | None = None) -> dict:
    config = config or load_config('config/default_config.yaml')
    out = ensure_dir(Path(config['paths']['output_dir']) / 'pre_layer')
    blocks_df, providers_df, _ = load_raw_tables(
        config['paths']['blocks_json'],
        config['paths']['providers_json'],
        config['paths'].get('cases_json'),
    )
    result = reconstruct_template(blocks_df, providers_df, config)
    canonical_blocks = result.pop('_canonical_blocks_df')
    template_df = result.pop('_template_df')
    bic_df = result.pop('_bic_df')
    offset_df = result.pop('_offset_df')

    save_df(canonical_blocks, out / 'canonical_blocks.csv')
    save_df(template_df, out / 'block_template.csv')
    save_df(bic_df, out / 'bic_scores.csv')
    save_df(offset_df, out / 'phase_offset_scores.csv')
    save_json(result, out / 'prelayer_result.json')
    save_bic_plot(bic_df, out / 'bic_scores.png')
    print(f"Pre-Layer complete: T*={result['T_star']}, phase_offset={result['phase_offset']}, template_blocks={len(template_df)}")
    return {'output_dir': str(out), 'result_json': str(out / 'prelayer_result.json')}


if __name__ == '__main__':
    run()

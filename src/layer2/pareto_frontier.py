from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.layer2.optimizer import greedy_allocate
from src.common.io import save_df, save_json


def build_pareto_candidates(prelayer: dict, providers: pd.DataFrame, scenarios: pd.DataFrame, early_release: pd.DataFrame, out_dir: str | Path, config: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    themes = ['conservative', 'continuity_first', 'balanced', 'utilization_first']
    rows = []
    best = None
    for theme in themes:
        assignments, coverage, meta = greedy_allocate(prelayer, providers, scenarios, early_release, config, theme=theme)
        assignments.to_csv(out / f'assignments_{theme}.csv', index=False)
        coverage.to_csv(out / f'coverage_{theme}.csv', index=False)
        rows.append(meta)
        if best is None or meta['objective_value'] < best['objective_value']:
            best = meta
    frontier = pd.DataFrame(rows).sort_values('objective_value')
    frontier.to_csv(out / 'pareto_candidates.csv', index=False)
    save_json(best, out / 'recommended_candidate.json')
    return {'frontier': frontier, 'recommended': best}

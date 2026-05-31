"""Run the full Gen3 pipeline.

By default this uses any already-created upstream artifacts and only recomputes
missing layers. Use `python run_all.py --force` to recompute everything.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

THREAD_ENV = {
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'PYTHONUNBUFFERED': '1',
}

LAYERS = [
    ('src.pre_layer.run_prelayer', Path('outputs/pre_layer/prelayer_result.json')),
    ('src.layer1.run_layer1', Path('outputs/layer1/scenarios_long.csv')),
    ('src.layer2.run_layer2', Path('outputs/layer2/recommended_candidate.json')),
    ('src.layer3.run_layer3', Path('outputs/layer3/explanation_report.md')),
]


def run_module(module: str) -> None:
    print(f"\n=== {module} ===", flush=True)
    env = os.environ.copy()
    env.update(THREAD_ENV)
    subprocess.run([sys.executable, '-m', module], check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='recompute all layers even when outputs already exist')
    args = parser.parse_args()
    Path('outputs').mkdir(exist_ok=True)
    for module, marker in LAYERS:
        if marker.exists() and not args.force:
            print(f"Skipping {module}; found {marker}. Use --force to recompute.", flush=True)
            continue
        run_module(module)
    print('\nDONE. See outputs/ for all artifacts.', flush=True)


if __name__ == '__main__':
    main()

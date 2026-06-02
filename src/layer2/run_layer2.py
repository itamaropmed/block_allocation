#!/usr/bin/env python3
"""Thin entrypoint for Layer 2.

Place this file next to optimizer.py inside src/layer2/ and run:

python3 src/layer2/run_layer2.py \
  --config config/default_config.yaml \
  --pre_layer_result outputs/pre_layer/prelayer_result.json \
  --layer1_dir outputs/layer1 \
  --blocks data/raw/geisinger-users_blocks.json \
  --providers data/raw/geisinger-users_providers.json \
  --out outputs/layer2 \
  --alpha 0.85 \
  --candidate_top_per_day 120 \
  --pareto_grid 6
"""

from __future__ import annotations

try:
    from optimizer import main
except ImportError:  # allows package-style import too
    from .optimizer import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())

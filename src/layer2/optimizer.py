#!/usr/bin/env python3
"""
Layer 2 — Stochastic OR Block Allocation Optimizer
===================================================

This module reads:
  1. Pre-Layer reconstructed template blocks / T* rotation information.
  2. Layer-1 SAA demand scenarios from outputs/layer1/scenarios_long.csv.
  3. Raw blocks/providers JSON for metadata fallbacks.

It solves the next T* rotation horizon as a capacity allocation problem:
  - Slots are the reconstructed template blocks across the T* rotation horizon.
  - Targets are provider × day-of-week alpha-quantile SAA demand minutes.
  - The optimizer assigns OR minutes from slots to provider-day targets.
  - Unmet demand is shortage, not a hard failure.
  - Utilization/coverage dominates the score.
  - Continuity, stability, and preference are explicit secondary Pareto axes.

The primary solver is Google OR-Tools min-cost flow when available. A deterministic
fallback greedy allocator is included so the pipeline still produces artifacts if
OR-Tools is unavailable or the graph API differs across versions.

Outputs are written to the requested Layer-2 output directory:
  - assignments.csv
  - schedule_by_week.csv
  - provider_day_coverage.csv
  - slot_utilization.csv
  - pareto_frontier.csv
  - pareto_all.csv
  - calibrated_goals.json
  - validation_report.json
  - layer2_result.json
  - layer2_summary.md

Run through run_layer2.py, not this file directly, although this file also has a
main() entrypoint for convenience.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

LOG = logging.getLogger("layer2")

OPEN_VALUES = {"", "OPEN", "NONE", "NULL", "NAN", "UNASSIGNED", "NO PROVIDER", "NO_PROVIDER"}
DAY_ALIASES = ["day_of_week", "dow", "weekday", "day"]
PROVIDER_ALIASES = ["provider_id", "provider", "surgeon_id", "blockholder_id", "current_provider_id"]
SERVICE_LINE_ALIASES = ["service_line", "service", "specialty", "department", "division"]
ROOM_ALIASES = ["room", "room_id", "or_room", "operating_room"]
SITE_ALIASES = ["site", "site_id", "facility", "location"]
PHASE_ALIASES = ["horizon_week", "template_week", "rotation_phase", "phase", "phase_label", "week_phase"]
START_ALIASES = ["start_min", "start_time_min", "start_minutes", "block_start_min", "start_time"]
END_ALIASES = ["end_min", "end_time_min", "end_minutes", "block_end_min", "end_time"]
DURATION_ALIASES = ["duration_min", "allocated_min", "capacity_min", "block_minutes", "minutes", "duration"]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Layer2Config:
    pre_layer_result: str = "outputs/pre_layer/prelayer_result.json"
    layer1_dir: str = "outputs/layer1"
    blocks_json: str = "data/raw/geisinger-users_blocks.json"
    providers_json: str = "data/raw/geisinger-users_providers.json"
    output_dir: str = "outputs/layer2"

    alpha: float = 0.85
    time_limit_s: float = 180.0
    workers: int = 8
    max_slots: int = 0
    candidate_top_per_day: int = 120
    pareto_grid: int = 6
    random_seed: int = 42

    # Costs are per allocated minute. Lower is better.
    # Objective knobs. They are used both in the utilization-first solve and in
    # the Pareto grid.
    # - noncurrent_penalty_per_min controls continuity: keep incumbent/current holder.
    # - service_line_mismatch_penalty_per_min controls preference: keep provider in
    #   matching service line/specialty when available.
    # - stability_fit_penalty_per_min controls stability: prefer slot→target pairs
    #   whose block capacity fits the provider-day demand, which reduces avoidable
    #   fragmentation in a min-cost-flow model. The final stability metric still
    #   directly measures split slots.
    noncurrent_penalty_per_min: int = 25
    service_line_mismatch_penalty_per_min: int = 40
    stability_fit_penalty_per_min: int = 15
    open_slot_bonus_per_min: int = 5
    slack_penalty_per_min: int = 1_000_000

    # Pareto grid uses the candidate depth that achieved the utilization-first
    # optimum unless this is set to a positive value. 0 means inherit/adaptive.
    pareto_candidate_top_per_day: int = 0

    # Epsilon-constraint Pareto settings. The first solve maximizes utilization.
    # Every secondary Pareto solve then enforces:
    #     allocated_minutes >= epsilon_floor_fraction * allocated_minutes_optimum
    # and optimizes continuity/preference/stability inside that utilization floor.
    # This makes utilization lexicographically dominant but still allows controlled
    # secondary trade-offs when the floor is below 1.0.
    pareto_min_utilization_fraction: float = 0.95
    pareto_include_full_utilization_floor: bool = True
    pareto_slack_cost_per_min: int = 0

    # Validation/scoring
    utilization_warn_floor: float = 0.98
    coverage_warn_floor: float = 0.80

    # If the Pre-Layer template is missing phase labels, replicate one base week T* times.
    replicate_if_no_phase: bool = True

    run_pareto: bool = True


# =============================================================================
# Generic helpers
# =============================================================================


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    return str(x)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        LOG.warning("Config file not found: %s", p)
        return {}
    if yaml is None:
        LOG.warning("PyYAML not installed; ignoring YAML config %s", p)
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def as_list_payload(obj: Any) -> List[Dict[str, Any]]:
    """Accepts JSON list or common wrapper dicts and returns list[dict]."""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ["data", "rows", "items", "blocks", "providers", "cases", "records"]:
            val = obj.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def load_json_table(path: str | Path) -> pd.DataFrame:
    obj = read_json(path)
    return pd.DataFrame(as_list_payload(obj))


def get_nested(row: Any, path: Sequence[str], default: Any = None) -> Any:
    cur = row
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            return default
    return cur


def clean_provider_id(value: Any) -> str:
    if value is None:
        return "OPEN"
    if isinstance(value, float) and math.isnan(value):
        return "OPEN"
    s = str(value).strip()
    if s.upper() in OPEN_VALUES:
        return "OPEN"
    return s


def clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return str(value).strip()


def normalize_day_value(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    s = str(value).strip().lower()
    if s == "":
        return None
    names = {
        "mon": 0, "monday": 0,
        "tue": 1, "tues": 1, "tuesday": 1,
        "wed": 2, "wednesday": 2,
        "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
        "fri": 4, "friday": 4,
        "sat": 5, "saturday": 5,
        "sun": 6, "sunday": 6,
    }
    if s in names:
        return names[s]
    m = re.search(r"\d+", s)
    if m:
        return int(m.group())
    return None


def parse_datetime(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts)
    except Exception:
        return None


def parse_time_min(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(round(value))
    s = str(value).strip()
    if s == "":
        return None
    # ISO datetime
    ts = parse_datetime(s)
    if ts is not None and ("T" in s or re.search(r"\d{4}-\d{2}-\d{2}", s)):
        return int(ts.hour * 60 + ts.minute)
    # HH:MM or HH:MM:SS
    m = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    # Numeric string
    try:
        return int(round(float(s)))
    except Exception:
        return None


def duration_from_start_end(start_min: Optional[int], end_min: Optional[int]) -> Optional[int]:
    if start_min is None or end_min is None:
        return None
    dur = end_min - start_min
    if dur < 0:
        dur += 24 * 60
    return int(max(0, dur))


def find_col(df: pd.DataFrame, aliases: Sequence[str], required: bool = False, name: str = "column") -> Optional[str]:
    if df.empty:
        if required:
            raise ValueError(f"Could not find {name}; DataFrame is empty")
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    # loose normalized match
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for a in aliases:
        key = re.sub(r"[^a-z0-9]", "", a.lower())
        if key in norm:
            return norm[key]
    if required:
        raise ValueError(f"Could not find {name}. Tried aliases={list(aliases)}. Columns={list(df.columns)}")
    return None


def first_existing(path_candidates: Sequence[str | Path]) -> Optional[Path]:
    for p in path_candidates:
        pp = Path(p)
        if pp.exists():
            return pp
    return None


# =============================================================================
# Provider metadata
# =============================================================================


def normalize_providers(providers_raw: pd.DataFrame) -> pd.DataFrame:
    if providers_raw.empty:
        return pd.DataFrame(columns=["provider_id", "provider_name", "service_line", "site"])

    rows: List[Dict[str, Any]] = []
    for _, r in providers_raw.iterrows():
        d = r.to_dict()
        pid = (
            d.get("provider_id")
            or d.get("id")
            or d.get("providerId")
            or d.get("npi")
            or get_nested(d, ["provider", "id"])
        )
        service_line = None
        for c in SERVICE_LINE_ALIASES:
            if c in d:
                service_line = d.get(c)
                break
        if service_line is None:
            service_line = (
                get_nested(d, ["service_line", "name"])
                or get_nested(d, ["specialty", "name"])
                or get_nested(d, ["department", "name"])
            )
        site = None
        for c in SITE_ALIASES:
            if c in d:
                site = d.get(c)
                break
        rows.append(
            {
                "provider_id": clean_provider_id(pid),
                "provider_name": clean_str(d.get("name") or d.get("display_name") or d.get("provider_name"), ""),
                "service_line": clean_str(service_line, "UNKNOWN"),
                "site": clean_str(site, ""),
            }
        )

    out = pd.DataFrame(rows)
    out = out[out["provider_id"] != "OPEN"].drop_duplicates("provider_id")
    return out.reset_index(drop=True)


# =============================================================================
# Pre-Layer template extraction and slot normalization
# =============================================================================


def extract_template_from_prelayer(pre_layer: Dict[str, Any], raw_blocks: pd.DataFrame) -> pd.DataFrame:
    """Extracts the reconstructed template table from multiple possible key names."""
    candidate_keys = [
        "block_template",
        "template_blocks",
        "template",
        "blocks_template",
        "reconstructed_template",
        "T_star_template",
    ]
    for key in candidate_keys:
        val = pre_layer.get(key)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            LOG.info("Template blocks extracted from pre-layer key=%s rows=%s", key, len(val))
            return pd.DataFrame(val)
        if isinstance(val, dict):
            nested = as_list_payload(val)
            if nested:
                LOG.info("Template blocks extracted from nested pre-layer key=%s rows=%s", key, len(nested))
                return pd.DataFrame(nested)

    path_keys = ["block_template_path", "template_blocks_path", "template_path"]
    for key in path_keys:
        p = pre_layer.get(key)
        if p and Path(str(p)).exists():
            pp = Path(str(p))
            LOG.info("Template blocks loaded from pre-layer path key=%s path=%s", key, pp)
            if pp.suffix.lower() == ".csv":
                return pd.read_csv(pp)
            return pd.DataFrame(as_list_payload(read_json(pp)))

    LOG.warning("No template_blocks object found in pre_layer_result; falling back to raw blocks rows=%s", len(raw_blocks))
    return raw_blocks.copy()


def infer_t_star(pre_layer: Dict[str, Any]) -> int:
    for key in ["T_star", "T*", "T", "rotation_period_weeks", "rotation_period", "t_star"]:
        if key in pre_layer:
            try:
                val = int(pre_layer[key])
                if val > 0:
                    return val
            except Exception:
                pass
    return 1


def normalize_raw_block_row(d: Dict[str, Any], min_week_start: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    occurrence = d.get("occurrence") if isinstance(d.get("occurrence"), dict) else {}
    holder = d.get("current_blockholder") if isinstance(d.get("current_blockholder"), dict) else {}
    room_obj = d.get("room") if isinstance(d.get("room"), dict) else {}

    start_raw = occurrence.get("start") or occurrence.get("start_time") or d.get("start") or d.get("start_time")
    end_raw = occurrence.get("end") or occurrence.get("end_time") or d.get("end") or d.get("end_time")
    start_ts = parse_datetime(start_raw)
    end_ts = parse_datetime(end_raw)

    start_min = parse_time_min(start_raw) or parse_time_min(d.get("start_time_min"))
    end_min = parse_time_min(end_raw) or parse_time_min(d.get("end_time_min"))

    if start_ts is not None:
        dow = int(start_ts.weekday())
        week_start = pd.Timestamp(start_ts.date()) - pd.Timedelta(days=dow)
    else:
        dow = None
        for c in DAY_ALIASES:
            if c in d:
                dow = normalize_day_value(d.get(c))
                break
        week_start = None

    if end_ts is not None and start_ts is not None:
        duration_min = int(max(0, round((end_ts - start_ts).total_seconds() / 60)))
    else:
        duration_min = None
        for c in DURATION_ALIASES:
            if c in d:
                try:
                    duration_min = int(round(float(d.get(c))))
                    break
                except Exception:
                    pass
        if duration_min is None:
            duration_min = duration_from_start_end(start_min, end_min)

    provider_id = (
        holder.get("provider_id")
        or holder.get("id")
        or holder.get("providerId")
        or d.get("provider_id")
        or d.get("current_provider_id")
        or d.get("blockholder_id")
    )
    is_open = bool(d.get("is_open", False)) or clean_provider_id(provider_id) == "OPEN"
    provider_id = "OPEN" if is_open else clean_provider_id(provider_id)

    service_line = (
        holder.get("service_line")
        or holder.get("specialty")
        or d.get("service_line")
        or d.get("specialty")
        or d.get("service")
    )

    site = d.get("site") or d.get("site_id") or get_nested(d, ["site", "name"], "")
    room = (
        room_obj.get("id")
        or room_obj.get("name")
        or room_obj.get("type")
        or d.get("room_id")
        or d.get("room")
        or d.get("or_room")
    )

    manual_er = d.get("manual_early_release")
    early_release_min = 0
    try:
        if manual_er not in [None, ""] and not (isinstance(manual_er, float) and math.isnan(manual_er)):
            # Could be boolean or minute value. Boolean True means no reliable minutes; keep 0.
            if not isinstance(manual_er, bool):
                early_release_min = int(round(float(manual_er)))
    except Exception:
        early_release_min = 0

    return {
        "source_block_id": clean_str(d.get("block_historical_id") or d.get("block_id") or d.get("id"), ""),
        "site": clean_str(site, "UNKNOWN"),
        "room": clean_str(room, "UNKNOWN_ROOM"),
        "day_of_week": dow,
        "week_start": str(week_start.date()) if week_start is not None else "",
        "start_min": start_min if start_min is not None else 8 * 60,
        "end_min": end_min if end_min is not None else (8 * 60 + int(duration_min or 0)),
        "duration_min": int(duration_min or 0),
        "current_provider_id": provider_id,
        "slot_service_line": clean_str(service_line, "UNKNOWN"),
        "is_open": bool(is_open),
        "early_release_min": early_release_min,
    }


def normalize_any_template_row(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes either a Pre-Layer template row or a raw block row into slot-like fields."""
    # If it has occurrence/current_blockholder, parse as raw nested row.
    if isinstance(d.get("occurrence"), dict) or isinstance(d.get("current_blockholder"), dict):
        base = normalize_raw_block_row(d)
    else:
        base: Dict[str, Any] = {}
        day = None
        for c in DAY_ALIASES:
            if c in d:
                day = normalize_day_value(d.get(c))
                break
        if day is None:
            # Try timestamp columns.
            for c in ["start", "start_time", "occurrence_start"]:
                if c in d:
                    ts = parse_datetime(d.get(c))
                    if ts is not None:
                        day = int(ts.weekday())
                        break

        start_min = None
        for c in START_ALIASES:
            if c in d:
                start_min = parse_time_min(d.get(c))
                if start_min is not None:
                    break
        end_min = None
        for c in END_ALIASES:
            if c in d:
                end_min = parse_time_min(d.get(c))
                if end_min is not None:
                    break

        duration_min = None
        for c in DURATION_ALIASES:
            if c in d:
                try:
                    duration_min = int(round(float(d.get(c))))
                    break
                except Exception:
                    pass
        if duration_min is None:
            duration_min = duration_from_start_end(start_min, end_min)
        if duration_min is None:
            duration_min = 0
        if start_min is None:
            start_min = 8 * 60
        if end_min is None:
            end_min = start_min + int(duration_min)

        provider = None
        for c in PROVIDER_ALIASES:
            if c in d:
                provider = d.get(c)
                break
        is_open = bool(d.get("is_open", False)) or clean_provider_id(provider) == "OPEN"

        site = None
        for c in SITE_ALIASES:
            if c in d:
                site = d.get(c)
                break
        room = None
        for c in ROOM_ALIASES:
            if c in d:
                room = d.get(c)
                break
        sl = None
        for c in SERVICE_LINE_ALIASES:
            if c in d:
                sl = d.get(c)
                break
        if sl is None:
            sl = d.get("slot_service_line") or d.get("current_service_line")

        base = {
            "source_block_id": clean_str(d.get("block_historical_id") or d.get("block_id") or d.get("template_block_id") or d.get("id"), ""),
            "site": clean_str(site, "UNKNOWN"),
            "room": clean_str(room, "UNKNOWN_ROOM"),
            "day_of_week": day,
            "week_start": clean_str(d.get("week_start"), ""),
            "start_min": int(start_min),
            "end_min": int(end_min),
            "duration_min": int(duration_min),
            "current_provider_id": "OPEN" if is_open else clean_provider_id(provider),
            "slot_service_line": clean_str(sl, "UNKNOWN"),
            "is_open": bool(is_open),
            "early_release_min": 0,
        }

    phase = None
    for c in PHASE_ALIASES:
        if c in d:
            try:
                phase = int(float(d.get(c)))
                break
            except Exception:
                phase = normalize_day_value(d.get(c))
    base["rotation_phase"] = phase
    return base


def normalize_template_to_slots(
    template: pd.DataFrame,
    pre_layer: Dict[str, Any],
    providers: pd.DataFrame,
    cfg: Layer2Config,
    early_release: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    t_star = infer_t_star(pre_layer)
    if template.empty:
        raise ValueError("Template table is empty; cannot build Layer-2 slots.")

    rows = [normalize_any_template_row(r.to_dict()) for _, r in template.iterrows()]
    slots = pd.DataFrame(rows)

    slots = slots[slots["day_of_week"].notna()].copy()
    slots["day_of_week"] = slots["day_of_week"].astype(int)
    slots["duration_min"] = pd.to_numeric(slots["duration_min"], errors="coerce").fillna(0).astype(int)
    slots = slots[slots["duration_min"] > 0].copy()

    # If the template has phases, keep every phase row. Do NOT collapse to one week.
    has_phase = slots["rotation_phase"].notna().any()
    if has_phase:
        slots["horizon_week"] = slots["rotation_phase"].fillna(0).astype(int) % max(1, t_star)
    elif cfg.replicate_if_no_phase and t_star > 1:
        base = slots.copy()
        reps = []
        for w in range(t_star):
            tmp = base.copy()
            tmp["horizon_week"] = w
            tmp["rotation_phase"] = w
            reps.append(tmp)
        slots = pd.concat(reps, ignore_index=True)
    else:
        slots["horizon_week"] = 0
        slots["rotation_phase"] = 0

    # Apply early release when usable columns exist.
    slots["early_release_min"] = pd.to_numeric(slots.get("early_release_min", 0), errors="coerce").fillna(0).astype(int)
    if early_release is not None and not early_release.empty:
        slots = apply_early_release(slots, early_release)

    slots["capacity_min"] = (slots["duration_min"] - slots["early_release_min"]).clip(lower=0).astype(int)
    slots = slots[slots["capacity_min"] > 0].copy()

    # Provider service-line fallback.
    if not providers.empty:
        p_sl = providers.set_index("provider_id")["service_line"].to_dict()
        mask_unknown = slots["slot_service_line"].isna() | (slots["slot_service_line"].astype(str).str.upper().isin(["", "UNKNOWN", "NAN"]))
        slots.loc[mask_unknown, "slot_service_line"] = slots.loc[mask_unknown, "current_provider_id"].map(p_sl).fillna("UNKNOWN")

    slots = slots.sort_values(["horizon_week", "day_of_week", "site", "room", "start_min", "current_provider_id"]).reset_index(drop=True)

    if cfg.max_slots and cfg.max_slots > 0 and len(slots) > cfg.max_slots:
        LOG.warning("max_slots=%s is active; truncating normalized slots from %s to %s", cfg.max_slots, len(slots), cfg.max_slots)
        slots = slots.head(int(cfg.max_slots)).copy()

    slots["slot_idx"] = np.arange(len(slots), dtype=int)
    slots["slot_id"] = slots.apply(
        lambda r: f"W{int(r.horizon_week)}_D{int(r.day_of_week)}_{r.site}_{r.room}_{int(r.start_min)}_{int(r.slot_idx)}",
        axis=1,
    )

    LOG.info(
        "Normalized slots: rows=%s total_capacity_min=%s open_slots=%s days=%s horizon_weeks=%s",
        len(slots),
        int(slots["capacity_min"].sum()),
        int((slots["current_provider_id"] == "OPEN").sum()),
        sorted(slots["day_of_week"].unique().tolist()),
        sorted(slots["horizon_week"].unique().tolist()),
    )
    return slots


def apply_early_release(slots: pd.DataFrame, er: pd.DataFrame) -> pd.DataFrame:
    """Best-effort early-release integration. If columns are insufficient, leave slots unchanged."""
    er = er.copy()
    minute_col = find_col(er, ["early_release_min", "manual_early_release_min", "release_min", "projected_early_release_min"], required=False)
    if minute_col is None:
        LOG.warning("early_release_projected.csv present but columns are insufficient to create slots")
        return slots
    er["_early_release_min"] = pd.to_numeric(er[minute_col], errors="coerce").fillna(0).astype(int).clip(lower=0)

    provider_col = find_col(er, PROVIDER_ALIASES, required=False)
    day_col = find_col(er, DAY_ALIASES, required=False)
    horizon_col = find_col(er, ["horizon_week", "rotation_phase", "phase"], required=False)
    room_col = find_col(er, ROOM_ALIASES, required=False)

    out = slots.copy()

    if provider_col and day_col:
        er["_provider_id"] = er[provider_col].map(clean_provider_id)
        er["_day_of_week"] = er[day_col].map(normalize_day_value).astype("Int64")
        group_cols = ["_provider_id", "_day_of_week"]
        if horizon_col:
            er["_horizon_week"] = pd.to_numeric(er[horizon_col], errors="coerce").astype("Int64")
            group_cols.append("_horizon_week")
        agg = er.groupby(group_cols, dropna=True)["_early_release_min"].max().reset_index()
        out["_provider_id"] = out["current_provider_id"]
        out["_day_of_week"] = out["day_of_week"].astype("Int64")
        if horizon_col:
            out["_horizon_week"] = out["horizon_week"].astype("Int64")
        out = out.merge(agg, on=group_cols, how="left")
        out["early_release_min"] = np.maximum(out["early_release_min"], out["_early_release_min"].fillna(0).astype(int))
        drop_cols = [c for c in ["_provider_id", "_day_of_week", "_horizon_week", "_early_release_min"] if c in out.columns]
        out = out.drop(columns=drop_cols)
        return out

    if room_col and day_col:
        er["_room"] = er[room_col].map(lambda x: clean_str(x, "UNKNOWN_ROOM"))
        er["_day_of_week"] = er[day_col].map(normalize_day_value).astype("Int64")
        agg = er.groupby(["_room", "_day_of_week"], dropna=True)["_early_release_min"].max().reset_index()
        out["_room"] = out["room"]
        out["_day_of_week"] = out["day_of_week"].astype("Int64")
        out = out.merge(agg, on=["_room", "_day_of_week"], how="left")
        out["early_release_min"] = np.maximum(out["early_release_min"], out["_early_release_min"].fillna(0).astype(int))
        out = out.drop(columns=["_room", "_day_of_week", "_early_release_min"])
        return out

    LOG.warning("early_release_projected.csv present but columns are insufficient to match provider/day or room/day")
    return slots


# =============================================================================
# Layer-1 SAA target loading
# =============================================================================


def find_scenarios_file(layer1_dir: str | Path) -> Path:
    p = Path(layer1_dir)
    candidates = [
        p / "scenarios_long.csv",
        p / "saa_scenarios_long.csv",
        p / "demand_scenarios_long.csv",
        p / "scenarios.csv",
        p / "demand_scenarios.csv",
    ]
    found = first_existing(candidates)
    if found is None:
        raise FileNotFoundError(f"Could not find Layer-1 scenarios file in {p}. Tried {[str(x) for x in candidates]}")
    return found


def load_layer1_targets(layer1_dir: str | Path, alpha: float, providers: pd.DataFrame) -> pd.DataFrame:
    scenarios_path = find_scenarios_file(layer1_dir)
    df = pd.read_csv(scenarios_path)
    LOG.info("Layer-1 scenarios loaded: %s rows=%s", scenarios_path, len(df))
    if df.empty:
        raise ValueError("Layer-1 scenarios file is empty.")

    provider_col = find_col(df, PROVIDER_ALIASES, required=True, name="provider_id")
    day_col = find_col(df, DAY_ALIASES, required=True, name="day_of_week")
    scenario_col = find_col(df, ["scenario_id", "scenario", "saa_scenario", "draw", "sample_id"], required=False)

    # Demand column detection.
    total_col = find_col(
        df,
        ["total_demand_min", "demand_total_min", "demand_min", "scenario_total_min", "total_min"],
        required=False,
    )
    if total_col is not None:
        demand = pd.to_numeric(df[total_col], errors="coerce").fillna(0)
    else:
        case_candidates = [c for c in df.columns if "case" in str(c).lower() and "min" in str(c).lower()]
        turn_candidates = [c for c in df.columns if ("turn" in str(c).lower() or "turnover" in str(c).lower()) and "min" in str(c).lower()]
        if not case_candidates:
            # Last resort: any numeric demand-ish column.
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            numeric = [c for c in numeric if c not in [day_col, scenario_col]]
            if not numeric:
                raise ValueError(f"Could not infer demand columns in scenarios file. Columns={list(df.columns)}")
            LOG.warning("Could not find casetime/turnover columns. Using numeric column %s as demand.", numeric[0])
            demand = pd.to_numeric(df[numeric[0]], errors="coerce").fillna(0)
        else:
            case_col = prefer_demand_col(case_candidates)
            turn_col = prefer_demand_col(turn_candidates) if turn_candidates else None
            demand = pd.to_numeric(df[case_col], errors="coerce").fillna(0)
            if turn_col:
                demand = demand + pd.to_numeric(df[turn_col], errors="coerce").fillna(0)
            LOG.info("Using demand columns: case=%s turnover=%s", case_col, turn_col)

    work = pd.DataFrame(
        {
            "provider_id": df[provider_col].map(clean_provider_id),
            "day_of_week": df[day_col].map(normalize_day_value),
            "scenario_id": df[scenario_col] if scenario_col else np.arange(len(df)),
            "scenario_demand_min": demand.clip(lower=0),
        }
    )
    work = work[(work["provider_id"] != "OPEN") & work["day_of_week"].notna()].copy()
    work["day_of_week"] = work["day_of_week"].astype(int)

    grouped = work.groupby(["provider_id", "day_of_week"], as_index=False).agg(
        target_q_min=("scenario_demand_min", lambda x: float(np.quantile(x.to_numpy(dtype=float), alpha))),
        mean_demand_min=("scenario_demand_min", "mean"),
        p50_demand_min=("scenario_demand_min", "median"),
        p95_demand_min=("scenario_demand_min", lambda x: float(np.quantile(x.to_numpy(dtype=float), 0.95))),
        n_scenarios=("scenario_demand_min", "count"),
    )
    grouped["target_q_int"] = grouped["target_q_min"].round().clip(lower=0).astype(int)
    grouped = grouped[grouped["target_q_int"] > 0].copy()

    if not providers.empty:
        meta = providers[["provider_id", "service_line", "site"]].drop_duplicates("provider_id")
        grouped = grouped.merge(meta, on="provider_id", how="left")
    else:
        grouped["service_line"] = "UNKNOWN"
        grouped["site"] = ""
    grouped["service_line"] = grouped["service_line"].fillna("UNKNOWN").astype(str)
    grouped["site"] = grouped["site"].fillna("").astype(str)
    grouped = grouped.sort_values(["day_of_week", "target_q_int", "provider_id"], ascending=[True, False, True]).reset_index(drop=True)
    grouped["target_idx"] = np.arange(len(grouped), dtype=int)

    LOG.info(
        "SAA targets: provider_day_rows=%s providers=%s scenarios=%s alpha=%.3f total_target_q=%s",
        len(grouped),
        grouped["provider_id"].nunique(),
        int(work["scenario_id"].nunique()),
        alpha,
        int(grouped["target_q_int"].sum()),
    )
    return grouped


def prefer_demand_col(cols: Sequence[str]) -> str:
    if not cols:
        raise ValueError("No candidate columns supplied")
    priority = ["demand", "scenario", "sample", "draw", "min"]
    scored = []
    for c in cols:
        name = str(c).lower()
        score = sum(1 for p in priority if p in name)
        if "mu" in name or "pred" in name or "forecast" in name:
            score -= 1
        scored.append((score, c))
    scored.sort(key=lambda x: (-x[0], str(x[1])))
    return str(scored[0][1])


# =============================================================================
# Candidate pair construction
# =============================================================================


def build_candidate_pairs(
    slots: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: Layer2Config,
    noncurrent_penalty_per_min: Optional[int] = None,
    service_line_mismatch_penalty_per_min: Optional[int] = None,
    stability_fit_penalty_per_min: Optional[int] = None,
    open_slot_bonus_per_min: Optional[int] = None,
) -> pd.DataFrame:
    noncurrent_penalty = int(cfg.noncurrent_penalty_per_min if noncurrent_penalty_per_min is None else noncurrent_penalty_per_min)
    sl_penalty = int(cfg.service_line_mismatch_penalty_per_min if service_line_mismatch_penalty_per_min is None else service_line_mismatch_penalty_per_min)
    stability_penalty = int(cfg.stability_fit_penalty_per_min if stability_fit_penalty_per_min is None else stability_fit_penalty_per_min)
    open_bonus = int(cfg.open_slot_bonus_per_min if open_slot_bonus_per_min is None else open_slot_bonus_per_min)

    target_by_day: Dict[int, pd.DataFrame] = {
        int(d): g.copy() for d, g in targets.groupby("day_of_week", sort=False)
    }
    rows: List[Dict[str, Any]] = []

    for _, s in slots.iterrows():
        day = int(s["day_of_week"])
        candidates = target_by_day.get(day)
        if candidates is None or candidates.empty:
            continue

        current_provider = clean_provider_id(s["current_provider_id"])
        slot_sl = clean_str(s.get("slot_service_line", "UNKNOWN"), "UNKNOWN")
        slot_open = current_provider == "OPEN" or bool(s.get("is_open", False))

        cand = candidates.copy()
        cand["same_holder"] = (cand["provider_id"] == current_provider).astype(int)
        cand["service_line_match"] = (
            (cand["service_line"].fillna("UNKNOWN").astype(str) == slot_sl)
            | (slot_sl.upper() == "UNKNOWN")
            | (cand["service_line"].fillna("UNKNOWN").astype(str).str.upper() == "UNKNOWN")
        ).astype(int)
        cand["is_open_slot"] = int(slot_open)

        cand["unit_cost"] = 0
        # Continuity penalty only when the slot has a real current holder.
        cand.loc[(cand["same_holder"] == 0) & (~slot_open), "unit_cost"] += noncurrent_penalty
        # Preference/specialty penalty. In this project this is the available
        # preference proxy because the Layer-1 demand is provider×day and the
        # template has blockholder/service-line metadata, not a separate ranked
        # preference file.
        cand.loc[cand["service_line_match"] == 0, "unit_cost"] += sl_penalty

        # Stability proxy: min-cost-flow has no fixed-charge variable for
        # “do not split this block”. The closest linear objective is to prefer
        # slot→target pairs where one block naturally fits one provider-day
        # target. This makes the stability knob visible in Pareto without
        # destroying the fast Google optimizer path.
        slot_cap = max(1, int(s["capacity_min"]))
        tq = cand["target_q_int"].clip(lower=1).astype(float)
        cand["fit_score"] = (np.minimum(slot_cap, tq) / np.maximum(slot_cap, tq)).clip(0, 1)
        if stability_penalty > 0:
            cand["unit_cost"] += np.round(stability_penalty * (1.0 - cand["fit_score"])).astype(int)

        # Open slots should be attractive because they avoid disrupting an incumbent.
        if slot_open and open_bonus > 0:
            cand["unit_cost"] -= open_bonus
        cand["unit_cost"] = cand["unit_cost"].astype(int)

        cand = cand.sort_values(
            ["unit_cost", "same_holder", "service_line_match", "fit_score", "target_q_int"],
            ascending=[True, False, False, False, False],
        )
        if cfg.candidate_top_per_day and cfg.candidate_top_per_day > 0:
            cand = cand.head(int(cfg.candidate_top_per_day))

        for _, t in cand.iterrows():
            rows.append(
                {
                    "slot_idx": int(s["slot_idx"]),
                    "target_idx": int(t["target_idx"]),
                    "pair_capacity": int(min(int(s["capacity_min"]), int(t["target_q_int"]))),
                    "unit_cost": int(t["unit_cost"]),
                    "same_holder": int(t["same_holder"]),
                    "service_line_match": int(t["service_line_match"]),
                    "is_open_slot": int(slot_open),
                    "fit_score": float(t.get("fit_score", 0.0)),
                }
            )

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError("No candidate slot-target pairs generated. Check day/provider normalization.")

    LOG.info(
        "Candidate pairs: rows=%s slots_with_candidates=%s provider_days_with_candidates=%s avg_pairs_per_slot=%.1f",
        len(pairs),
        pairs["slot_idx"].nunique(),
        pairs["target_idx"].nunique(),
        len(pairs) / max(1, slots["slot_idx"].nunique()),
    )
    return pairs


# =============================================================================
# Optimization
# =============================================================================


def _import_min_cost_flow():
    try:
        from ortools.graph.python import min_cost_flow  # type: ignore
        return min_cost_flow.SimpleMinCostFlow(), "ortools.graph.python.min_cost_flow"
    except Exception:
        try:
            from ortools.graph import pywrapgraph  # type: ignore
            return pywrapgraph.SimpleMinCostFlow(), "ortools.graph.pywrapgraph"
        except Exception:
            return None, "unavailable"


def _call_method(obj: Any, lower: str, upper: str, *args: Any) -> Any:
    if hasattr(obj, lower):
        return getattr(obj, lower)(*args)
    return getattr(obj, upper)(*args)


def solve_min_cost_flow(
    slots: pd.DataFrame,
    targets: pd.DataFrame,
    pairs: pd.DataFrame,
    cfg: Layer2Config,
    theme: str = "utilization_first",
    *,
    required_allocated_min: Optional[int] = None,
    slack_penalty_per_min: Optional[int] = None,
    epsilon_floor_fraction: Optional[float] = None,
    utilization_reference_allocated_min: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Min-cost max-flow with an optional ε-constraint utilization floor.

    The graph sends ``useful_flow = min(total capacity, total target)`` units.
    Units routed through slot→target arcs become allocated OR minutes. Units routed
    through the direct source→sink slack arc become shortage/unallocated minutes.

    For utilization-first optimization we use a very high slack penalty so the
    solver maximizes allocated minutes first. For ε-constraint Pareto points, pass
    ``required_allocated_min``. The direct slack arc is then capped at:

        useful_flow - required_allocated_min

    so the secondary objective can trade continuity/preference/stability only
    inside a guaranteed utilization floor.
    """
    t0 = time.time()
    total_capacity = int(slots["capacity_min"].sum())
    total_target = int(targets["target_q_int"].sum())
    useful_flow = int(min(total_capacity, total_target))

    if useful_flow <= 0:
        raise ValueError("No useful flow to optimize: capacity or target is zero.")

    if required_allocated_min is None:
        required_allocated_min = 0
    required_allocated_min = int(max(0, min(int(required_allocated_min), useful_flow)))
    slack_cap_min = int(max(0, useful_flow - required_allocated_min))
    effective_slack_penalty = int(cfg.slack_penalty_per_min if slack_penalty_per_min is None else slack_penalty_per_min)

    solver, backend = _import_min_cost_flow()
    if solver is None:
        LOG.warning("OR-Tools min-cost flow unavailable; using greedy fallback.")
        return solve_greedy(
            slots, targets, pairs, cfg, theme=theme, backend="greedy_no_ortools", t0=t0,
            required_allocated_min=required_allocated_min,
            slack_penalty_per_min=effective_slack_penalty,
            epsilon_floor_fraction=epsilon_floor_fraction,
            utilization_reference_allocated_min=utilization_reference_allocated_min,
        )

    try:
        n_slots = len(slots)
        n_targets = len(targets)
        source = 0
        slot_offset = 1
        target_offset = slot_offset + n_slots
        sink = target_offset + n_targets

        slot_node = {int(r.slot_idx): slot_offset + i for i, r in slots.reset_index(drop=True).iterrows()}
        target_node = {int(r.target_idx): target_offset + i for i, r in targets.reset_index(drop=True).iterrows()}

        # source → slots
        for _, s in slots.iterrows():
            _call_method(solver, "add_arc_with_capacity_and_unit_cost", "AddArcWithCapacityAndUnitCost", source, slot_node[int(s.slot_idx)], int(s.capacity_min), 0)

        # slots → targets
        arc_meta: Dict[int, Tuple[int, int]] = {}
        for _, p in pairs.iterrows():
            if int(p.pair_capacity) <= 0:
                continue
            tail = slot_node[int(p.slot_idx)]
            head = target_node[int(p.target_idx)]
            arc_idx = _call_method(
                solver,
                "add_arc_with_capacity_and_unit_cost",
                "AddArcWithCapacityAndUnitCost",
                tail,
                head,
                int(p.pair_capacity),
                int(p.unit_cost),
            )
            arc_meta[int(arc_idx)] = (int(p.slot_idx), int(p.target_idx))

        # targets → sink
        for _, t in targets.iterrows():
            _call_method(solver, "add_arc_with_capacity_and_unit_cost", "AddArcWithCapacityAndUnitCost", target_node[int(t.target_idx)], sink, int(t.target_q_int), 0)

        # Slack guarantees feasibility. In utilization-first mode its cost is
        # very high, so the solver maximizes assigned minutes. In ε-constraint
        # mode its capacity is capped, which enforces the utilization floor, and
        # its cost can be zero so secondary objectives are optimized inside that
        # floor instead of always forcing the absolute maximum allocation.
        slack_arc_idx = _call_method(
            solver,
            "add_arc_with_capacity_and_unit_cost",
            "AddArcWithCapacityAndUnitCost",
            source,
            sink,
            slack_cap_min,
            effective_slack_penalty,
        )

        _call_method(solver, "set_node_supply", "SetNodeSupply", source, useful_flow)
        _call_method(solver, "set_node_supply", "SetNodeSupply", sink, -useful_flow)

        status = _call_method(solver, "solve", "Solve")
        optimal_const = getattr(solver, "OPTIMAL", None)
        status_name = str(status)
        if optimal_const is not None and status == optimal_const:
            status_name = "OPTIMAL"
        elif str(status).upper().endswith("OPTIMAL"):
            status_name = "OPTIMAL"

        if status_name != "OPTIMAL":
            LOG.warning("Min-cost flow status=%s using backend=%s; falling back to greedy.", status, backend)
            return solve_greedy(
                slots, targets, pairs, cfg, theme=theme, backend=f"greedy_after_mcf_status_{status}", t0=t0,
                required_allocated_min=required_allocated_min,
                slack_penalty_per_min=effective_slack_penalty,
                epsilon_floor_fraction=epsilon_floor_fraction,
                utilization_reference_allocated_min=utilization_reference_allocated_min,
            )

        assign_rows: List[Dict[str, Any]] = []
        slack_flow = 0
        num_arcs = int(_call_method(solver, "num_arcs", "NumArcs"))
        for a in range(num_arcs):
            flow = int(_call_method(solver, "flow", "Flow", a))
            if flow <= 0:
                continue
            if int(a) == int(slack_arc_idx):
                slack_flow = flow
                continue
            meta = arc_meta.get(int(a))
            if meta is None:
                continue
            slot_idx, target_idx = meta
            assign_rows.append(
                {
                    "slot_idx": slot_idx,
                    "target_idx": target_idx,
                    "allocated_min": flow,
                }
            )

        assignments = pd.DataFrame(assign_rows)
        if assignments.empty:
            assignments = pd.DataFrame(columns=["slot_idx", "target_idx", "allocated_min"])

        metrics = make_metrics(
            slots,
            targets,
            pairs,
            assignments,
            theme=theme,
            solver_status=status_name,
            solver_backend=backend,
            runtime_s=time.time() - t0,
            slack_flow_min=slack_flow,
        )
        metrics["epsilon_required_allocated_min"] = int(required_allocated_min)
        metrics["epsilon_slack_cap_min"] = int(slack_cap_min)
        metrics["epsilon_slack_penalty_per_min"] = int(effective_slack_penalty)
        metrics["epsilon_floor_fraction_of_optimum"] = None if epsilon_floor_fraction is None else float(epsilon_floor_fraction)
        metrics["utilization_reference_allocated_min"] = None if utilization_reference_allocated_min is None else int(utilization_reference_allocated_min)
        metrics["epsilon_constraint_satisfied"] = bool(int(metrics.get("total_allocated_min", 0)) + 1e-9 >= int(required_allocated_min))
        if utilization_reference_allocated_min is not None:
            ref_alloc = max(1, int(utilization_reference_allocated_min))
            metrics["utilization_loss_from_optimum_min"] = int(max(0, ref_alloc - int(metrics.get("total_allocated_min", 0))))
            metrics["utilization_loss_from_optimum_pct"] = float(metrics["utilization_loss_from_optimum_min"] / ref_alloc)
        coverage, slot_util = make_coverage_tables(slots, targets, assignments)
        return assignments, coverage, slot_util, metrics
    except Exception as exc:
        LOG.exception("Min-cost flow failed with backend=%s; using greedy fallback. Error: %s", backend, exc)
        return solve_greedy(
            slots, targets, pairs, cfg, theme=theme, backend="greedy_after_exception", t0=t0,
            required_allocated_min=required_allocated_min,
            slack_penalty_per_min=effective_slack_penalty,
            epsilon_floor_fraction=epsilon_floor_fraction,
            utilization_reference_allocated_min=utilization_reference_allocated_min,
        )


def solve_greedy(
    slots: pd.DataFrame,
    targets: pd.DataFrame,
    pairs: pd.DataFrame,
    cfg: Layer2Config,
    theme: str,
    backend: str,
    t0: Optional[float] = None,
    required_allocated_min: Optional[int] = None,
    slack_penalty_per_min: Optional[int] = None,
    epsilon_floor_fraction: Optional[float] = None,
    utilization_reference_allocated_min: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Deterministic fallback: useful when OR-Tools is unavailable.

    In ε-constraint mode with zero slack cost, the fallback stops once the
    required utilization floor is reached. With a positive/high slack cost it
    behaves like the old greedy allocator and fills as much as possible.
    """
    if t0 is None:
        t0 = time.time()
    slot_remaining = slots.set_index("slot_idx")["capacity_min"].astype(int).to_dict()
    target_remaining = targets.set_index("target_idx")["target_q_int"].astype(int).to_dict()

    # Continuity and low cost first, then large targets.
    target_size = targets.set_index("target_idx")["target_q_int"].to_dict()
    pp = pairs.copy()
    pp["target_size"] = pp["target_idx"].map(target_size).fillna(0).astype(int)
    pp = pp.sort_values(["unit_cost", "same_holder", "service_line_match", "target_size"], ascending=[True, False, False, False])

    useful_flow = int(min(int(slots["capacity_min"].sum()), int(targets["target_q_int"].sum())))
    req = int(max(0, min(int(required_allocated_min or 0), useful_flow)))
    effective_slack_penalty = int(cfg.slack_penalty_per_min if slack_penalty_per_min is None else slack_penalty_per_min)
    stop_at_floor = bool(req > 0 and effective_slack_penalty <= 0)

    rows: List[Dict[str, Any]] = []
    allocated_so_far = 0
    for _, p in pp.iterrows():
        if stop_at_floor and allocated_so_far >= req:
            break
        si = int(p.slot_idx)
        ti = int(p.target_idx)
        amount = min(slot_remaining.get(si, 0), target_remaining.get(ti, 0), int(p.pair_capacity))
        if stop_at_floor:
            amount = min(amount, req - allocated_so_far)
        if amount <= 0:
            continue
        slot_remaining[si] -= amount
        target_remaining[ti] -= amount
        allocated_so_far += int(amount)
        rows.append({"slot_idx": si, "target_idx": ti, "allocated_min": int(amount)})

    assignments = pd.DataFrame(rows)
    if assignments.empty:
        assignments = pd.DataFrame(columns=["slot_idx", "target_idx", "allocated_min"])
    metrics = make_metrics(
        slots,
        targets,
        pairs,
        assignments,
        theme=theme,
        solver_status="FEASIBLE_GREEDY",
        solver_backend=backend,
        runtime_s=time.time() - t0,
        slack_flow_min=int(min(int(slots["capacity_min"].sum()), int(targets["target_q_int"].sum())) - assignments["allocated_min"].sum()),
    )
    metrics["epsilon_required_allocated_min"] = int(req) if req > 0 else None
    metrics["epsilon_slack_cap_min"] = int(max(0, useful_flow - req)) if req > 0 else None
    metrics["epsilon_slack_penalty_per_min"] = int(effective_slack_penalty)
    metrics["epsilon_floor_fraction_of_optimum"] = None if epsilon_floor_fraction is None else float(epsilon_floor_fraction)
    metrics["utilization_reference_allocated_min"] = None if utilization_reference_allocated_min is None else int(utilization_reference_allocated_min)
    metrics["epsilon_constraint_satisfied"] = bool(int(metrics.get("total_allocated_min", 0)) + 1e-9 >= int(req))
    if utilization_reference_allocated_min is not None:
        ref_alloc = max(1, int(utilization_reference_allocated_min))
        metrics["utilization_loss_from_optimum_min"] = int(max(0, ref_alloc - int(metrics.get("total_allocated_min", 0))))
        metrics["utilization_loss_from_optimum_pct"] = float(metrics["utilization_loss_from_optimum_min"] / ref_alloc)
    coverage, slot_util = make_coverage_tables(slots, targets, assignments)
    return assignments, coverage, slot_util, metrics


def make_coverage_tables(slots: pd.DataFrame, targets: pd.DataFrame, assignments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    targets_base = targets.copy()
    if assignments.empty:
        alloc_by_target = pd.Series(dtype=float)
        alloc_by_slot = pd.Series(dtype=float)
    else:
        alloc_by_target = assignments.groupby("target_idx")["allocated_min"].sum()
        alloc_by_slot = assignments.groupby("slot_idx")["allocated_min"].sum()

    coverage = targets_base.copy()
    coverage["allocated_min"] = coverage["target_idx"].map(alloc_by_target).fillna(0).astype(int)
    coverage["shortage_min"] = (coverage["target_q_int"] - coverage["allocated_min"]).clip(lower=0).astype(int)
    coverage["coverage_ratio"] = coverage["allocated_min"] / coverage["target_q_int"].clip(lower=1)

    slot_util = slots.copy()
    slot_util["allocated_min"] = slot_util["slot_idx"].map(alloc_by_slot).fillna(0).astype(int)
    slot_util["unused_min"] = (slot_util["capacity_min"] - slot_util["allocated_min"]).clip(lower=0).astype(int)
    slot_util["utilization"] = slot_util["allocated_min"] / slot_util["capacity_min"].clip(lower=1)
    return coverage, slot_util


def make_metrics(
    slots: pd.DataFrame,
    targets: pd.DataFrame,
    pairs: pd.DataFrame,
    assignments: pd.DataFrame,
    theme: str,
    solver_status: str,
    solver_backend: str,
    runtime_s: float,
    slack_flow_min: int = 0,
) -> Dict[str, Any]:
    total_capacity = int(slots["capacity_min"].sum())
    total_target = int(targets["target_q_int"].sum())
    allocated = int(assignments["allocated_min"].sum()) if not assignments.empty else 0

    pair_cols = ["slot_idx", "target_idx", "unit_cost", "same_holder", "service_line_match", "is_open_slot"]
    if "fit_score" in pairs.columns:
        pair_cols.append("fit_score")
    pair_meta = pairs[pair_cols].drop_duplicates(["slot_idx", "target_idx"])
    if assignments.empty:
        joined = pd.DataFrame(columns=["allocated_min", "same_holder", "service_line_match", "is_open_slot", "unit_cost"])
    else:
        joined = assignments.merge(pair_meta, on=["slot_idx", "target_idx"], how="left")

    same_holder_min = int((joined["allocated_min"] * joined["same_holder"].fillna(0)).sum()) if not joined.empty else 0
    sl_match_min = int((joined["allocated_min"] * joined["service_line_match"].fillna(0)).sum()) if not joined.empty else 0
    open_slot_min = int((joined["allocated_min"] * joined["is_open_slot"].fillna(0)).sum()) if not joined.empty else 0
    weighted_cost = int((joined["allocated_min"] * joined["unit_cost"].fillna(0)).sum()) if not joined.empty else 0
    weighted_fit_score = float((joined["allocated_min"] * joined.get("fit_score", pd.Series(0, index=joined.index)).fillna(0)).sum() / max(1, allocated)) if not joined.empty else 0.0

    if assignments.empty:
        used_slots = 0
        split_slots = 0
    else:
        per_slot_count = assignments[assignments["allocated_min"] > 0].groupby("slot_idx").size()
        used_slots = int((per_slot_count > 0).sum())
        split_slots = int((per_slot_count > 1).sum())

    metrics = {
        "theme": theme,
        "solver_status": solver_status,
        "solver_backend": solver_backend,
        "runtime_s": float(runtime_s),
        "total_capacity_min": total_capacity,
        "total_target_q_min": total_target,
        "theoretical_best_allocated_min": int(min(total_capacity, total_target)),
        "total_allocated_min": allocated,
        "total_shortage_min": int(max(0, total_target - allocated)),
        "slack_flow_min": int(max(0, slack_flow_min)),
        "raw_capacity_utilization": float(allocated / max(1, total_capacity)),
        "target_coverage": float(allocated / max(1, total_target)),
        "same_holder_min": same_holder_min,
        "same_holder_share": float(same_holder_min / max(1, allocated)),
        "service_line_match_min": sl_match_min,
        "service_line_match_share": float(sl_match_min / max(1, allocated)),
        "open_slot_min": open_slot_min,
        "open_slot_share": float(open_slot_min / max(1, allocated)),
        "used_slots": used_slots,
        "split_slots": split_slots,
        "split_slot_share_among_used": float(split_slots / max(1, used_slots)),
        "weighted_fit_score": float(weighted_fit_score),
        "assigned_pairs": int(len(assignments)),
        "weighted_assignment_cost": weighted_cost,
    }
    return score_solution_metrics(metrics)


# =============================================================================
# Scoring, adaptive solve, Pareto
# =============================================================================


def score_solution_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds robust scores that still make sense when not all goals are achievable.

    Important distinction:
    - raw_capacity_utilization can be low when demand < capacity.
    - useful_utilization measures allocated / min(capacity, demand target).
    """
    capacity = float(metrics.get("total_capacity_min", 0))
    target = float(metrics.get("total_target_q_min", 0))
    allocated = float(metrics.get("total_allocated_min", 0))
    shortage = float(metrics.get("total_shortage_min", max(0.0, target - allocated)))

    useful_denominator = max(1.0, min(capacity, target))
    useful_utilization = max(0.0, min(1.0, allocated / useful_denominator))
    raw_capacity_utilization = max(0.0, min(1.0, allocated / max(1.0, capacity)))
    target_coverage = max(0.0, min(1.0, allocated / max(1.0, target)))
    shortage_score = max(0.0, min(1.0, 1.0 - shortage / max(1.0, target)))

    continuity_score = max(0.0, min(1.0, float(metrics.get("same_holder_share", 0.0))))
    split_slot_share = max(0.0, min(1.0, float(metrics.get("split_slot_share_among_used", 0.0))))
    no_split_score = 1.0 - split_slot_share
    fit_score = max(0.0, min(1.0, float(metrics.get("weighted_fit_score", 0.0))))
    # Stability combines the direct observed split metric and the capacity-fit
    # proxy used by the min-cost-flow objective.
    stability_score = 0.70 * no_split_score + 0.30 * fit_score
    sl_match_score = max(0.0, min(1.0, float(metrics.get("service_line_match_share", 0.0))))
    preference_score = sl_match_score

    # Utilization/coverage dominates. Secondary goals improve score only after useful work is high.
    overall_score = (
        0.66 * useful_utilization
        + 0.14 * shortage_score
        + 0.08 * continuity_score
        + 0.06 * stability_score
        + 0.06 * preference_score
    )

    out = dict(metrics)
    out["useful_utilization"] = float(useful_utilization)
    out["raw_capacity_utilization"] = float(raw_capacity_utilization)
    out["target_coverage"] = float(target_coverage)
    out["shortage_score"] = float(shortage_score)
    out["continuity_score"] = float(continuity_score)
    out["stability_score"] = float(stability_score)
    out["no_split_score"] = float(no_split_score)
    out["fit_score"] = float(fit_score)
    out["preference_score"] = float(preference_score)
    out["service_line_score"] = float(sl_match_score)
    out["overall_score"] = float(overall_score)
    return out


def calibrated_goal_report(metrics: Dict[str, Any]) -> Dict[str, Any]:
    metrics = score_solution_metrics(metrics)
    return {
        "attainable_allocated_min": int(metrics["total_allocated_min"]),
        "theoretical_best_allocated_min": int(metrics.get("theoretical_best_allocated_min", 0)),
        "attainable_target_coverage": float(metrics["target_coverage"]),
        "attainable_useful_utilization": float(metrics["useful_utilization"]),
        "attainable_raw_capacity_utilization": float(metrics["raw_capacity_utilization"]),
        "attainable_same_holder_share": float(metrics.get("same_holder_share", 0.0)),
        "attainable_stability_score": float(metrics.get("stability_score", 0.0)),
        "recommended_coverage_floor": float(0.995 * metrics["target_coverage"]),
        "recommended_useful_utilization_floor": float(0.995 * metrics["useful_utilization"]),
        "recommended_same_holder_goal": float(min(0.95, metrics.get("same_holder_share", 0.0) + 0.05)),
        "recommended_stability_goal": float(max(0.0, metrics.get("stability_score", 0.0) - 0.03)),
        "note": (
            "Goals are calibrated from the utilization-first optimum. Use these as soft goals or Pareto floors. "
            "Do not force impossible epsilon constraints when capacity or candidate graph prevents full coverage."
        ),
    }


def solve_utilization_first_adaptive(
    slots: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: Layer2Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    capacity = int(slots["capacity_min"].sum())
    target = int(targets["target_q_int"].sum())
    theoretical_best = min(capacity, target)

    if cfg.candidate_top_per_day and cfg.candidate_top_per_day > 0:
        candidate_grid = [
            int(cfg.candidate_top_per_day),
            max(250, int(cfg.candidate_top_per_day)),
            max(500, int(cfg.candidate_top_per_day)),
            max(1000, int(cfg.candidate_top_per_day)),
            0,
        ]
    else:
        candidate_grid = [0]

    seen = set()
    candidate_grid = [x for x in candidate_grid if not (x in seen or seen.add(x))]
    best: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]] = None

    for top_k in candidate_grid:
        trial_cfg = replace(cfg, candidate_top_per_day=int(top_k))
        LOG.info("Adaptive utilization-first attempt: candidate_top_per_day=%s", "ALL" if top_k == 0 else top_k)
        pairs = build_candidate_pairs(slots, targets, trial_cfg)
        assignments, coverage, slot_util, metrics = solve_min_cost_flow(slots, targets, pairs, trial_cfg, theme=f"utilization_first_top_{top_k if top_k else 'ALL'}")
        metrics = score_solution_metrics(metrics)
        metrics["candidate_top_per_day_used"] = int(top_k)

        LOG.info(
            "Adaptive attempt result: top_k=%s allocated=%s theoretical_best=%s useful_utilization=%.4f target_coverage=%.4f overall_score=%.4f",
            "ALL" if top_k == 0 else top_k,
            metrics["total_allocated_min"],
            theoretical_best,
            metrics["useful_utilization"],
            metrics["target_coverage"],
            metrics["overall_score"],
        )

        if best is None or metrics["overall_score"] > best[-1]["overall_score"]:
            best = (assignments, coverage, slot_util, pairs, metrics)

        if int(metrics["total_allocated_min"]) >= int(0.995 * theoretical_best):
            break

    if best is None:
        raise RuntimeError("Adaptive utilization-first solve failed to produce any solution.")
    return best


def _epsilon_floor_grid(grid_n: int, min_fraction: float, include_full: bool = True) -> List[float]:
    """Return utilization floors for the ε-constraint frontier.

    Fractions are relative to the utilization-first optimum, not to total demand.
    The default grid intentionally puts several points very close to 1.0 because
    this is where thesis users usually want to see the price of continuity or
    stability while utilization still dominates.
    """
    grid_n = max(1, int(grid_n))
    min_fraction = float(max(0.0, min(1.0, min_fraction)))
    canonical = [1.0, 0.995, 0.990, 0.980, 0.970, 0.950, 0.925, 0.900, 0.850, 0.800]
    vals = [x for x in canonical if x >= min_fraction - 1e-12]
    if min_fraction not in vals:
        vals.append(min_fraction)
    vals = sorted(set(round(float(x), 6) for x in vals), reverse=True)
    if not include_full:
        vals = [v for v in vals if v < 1.0 - 1e-12]
    if len(vals) < grid_n:
        extra = np.linspace(1.0 if include_full else 0.995, min_fraction, grid_n).tolist()
        vals = sorted(set(vals + [round(float(x), 6) for x in extra]), reverse=True)
    return vals[:grid_n]


def run_pareto_grid(
    slots: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: Layer2Config,
    out_dir: Optional[str | Path] = None,
    goal_reference: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """Run the ε-constraint Pareto frontier with utilization dominant.

    Method used here:

      1. Layer 2 first solves the real utilization-first problem separately.
         That gives A* = best attainable allocated minutes.
      2. For each ε floor q in the grid, every secondary solve receives the hard
         constraint allocated_minutes >= q × A*.
      3. Inside that utilization floor, we optimize one secondary axis at a time:
         continuity, preference, stability, or a balanced combination.

    Implementation detail: because this Layer-2 formulation is a min-cost-flow
    minute-allocation problem, the ε constraint is enforced by capping the direct
    source→sink slack arc. The total graph flow is still min(capacity, target),
    but at most ``useful_flow - ceil(q*A*)`` minutes may go through slack. This
    is mathematically the same as requiring at least ceil(q*A*) assigned minutes.
    """
    grid_n = max(1, int(cfg.pareto_grid))
    base_alloc = int((goal_reference or {}).get("attainable_allocated_min", 0) or 0)
    if base_alloc <= 0:
        # Compatibility path for direct calls into pareto_frontier.py.
        # The normal run_layer2 pipeline always passes goal_reference.
        LOG.warning("No utilization-first goal_reference supplied to run_pareto_grid; solving reference optimum now.")
        ref_assign, ref_cov, ref_su, ref_pairs, ref_metrics = solve_utilization_first_adaptive(slots, targets, cfg)
        base_alloc = int(ref_metrics.get("total_allocated_min", 0))
    if base_alloc <= 0:
        raise ValueError("Cannot build ε-constraint Pareto frontier because utilization-first allocation is zero.")

    floors = _epsilon_floor_grid(
        grid_n=grid_n,
        min_fraction=float(cfg.pareto_min_utilization_fraction),
        include_full=bool(cfg.pareto_include_full_utilization_floor),
    )

    # Penalty profiles are not the Pareto constraints; they define which secondary
    # objective is optimized after the utilization floor is enforced.
    secondary_scale = 100
    profiles: List[Dict[str, Any]] = [
        {
            "pareto_family": "utilization_anchor",
            "theme_prefix": "eps_utilization_anchor",
            "continuity_penalty_per_min": 0,
            "preference_penalty_per_min": 0,
            "stability_penalty_per_min": 0,
            "slack_penalty_per_min": int(cfg.slack_penalty_per_min),
            "description": "Anchor: maximize utilization with the full slack penalty.",
        },
        {
            "pareto_family": "continuity",
            "theme_prefix": "eps_continuity",
            "continuity_penalty_per_min": max(1, int(cfg.noncurrent_penalty_per_min * secondary_scale)),
            "preference_penalty_per_min": 0,
            "stability_penalty_per_min": 0,
            "slack_penalty_per_min": int(cfg.pareto_slack_cost_per_min),
            "description": "Maximize continuity subject to the utilization floor.",
        },
        {
            "pareto_family": "preference",
            "theme_prefix": "eps_preference",
            "continuity_penalty_per_min": 0,
            "preference_penalty_per_min": max(1, int(cfg.service_line_mismatch_penalty_per_min * secondary_scale)),
            "stability_penalty_per_min": 0,
            "slack_penalty_per_min": int(cfg.pareto_slack_cost_per_min),
            "description": "Maximize service-line/preference proxy subject to the utilization floor.",
        },
        {
            "pareto_family": "stability",
            "theme_prefix": "eps_stability",
            "continuity_penalty_per_min": 0,
            "preference_penalty_per_min": 0,
            "stability_penalty_per_min": max(1, int(cfg.stability_fit_penalty_per_min * secondary_scale)),
            "slack_penalty_per_min": int(cfg.pareto_slack_cost_per_min),
            "description": "Maximize stability/capacity-fit proxy subject to the utilization floor.",
        },
        {
            "pareto_family": "balanced",
            "theme_prefix": "eps_balanced",
            "continuity_penalty_per_min": max(1, int(cfg.noncurrent_penalty_per_min * secondary_scale)),
            "preference_penalty_per_min": max(1, int(cfg.service_line_mismatch_penalty_per_min * secondary_scale)),
            "stability_penalty_per_min": max(1, int(cfg.stability_fit_penalty_per_min * secondary_scale)),
            "slack_penalty_per_min": int(cfg.pareto_slack_cost_per_min),
            "description": "Balanced secondary objective subject to the utilization floor.",
        },
    ]

    rows: List[Dict[str, Any]] = []
    plan_records: List[Dict[str, Any]] = []
    useful_flow = int(min(int(slots["capacity_min"].sum()), int(targets["target_q_int"].sum())))

    for floor_frac in floors:
        required_alloc = int(math.ceil(base_alloc * float(floor_frac)))
        required_alloc = int(max(0, min(required_alloc, useful_flow)))
        allowed_slack = int(max(0, useful_flow - required_alloc))
        LOG.info(
            "ε-constraint floor %.3f: require allocated >= %s of A*=%s; allowed_slack<=%s",
            floor_frac,
            required_alloc,
            base_alloc,
            allowed_slack,
        )
        for prof in profiles:
            # The utilization anchor only needs the full floor. Otherwise it repeats
            # the already-solved utilization-first behavior for every ε row.
            if prof["pareto_family"] == "utilization_anchor" and abs(float(floor_frac) - 1.0) > 1e-12:
                continue

            theme = f"{prof['theme_prefix']}_floor_{floor_frac:.3f}".replace(".", "p")
            LOG.info(
                "Solving ε-Pareto theme=%s family=%s floor=%.3f required_alloc=%s penalties=(continuity=%s preference=%s stability=%s) slack_cost=%s",
                theme,
                prof["pareto_family"],
                floor_frac,
                required_alloc,
                prof["continuity_penalty_per_min"],
                prof["preference_penalty_per_min"],
                prof["stability_penalty_per_min"],
                prof["slack_penalty_per_min"],
            )
            pairs = build_candidate_pairs(
                slots,
                targets,
                cfg,
                noncurrent_penalty_per_min=int(prof["continuity_penalty_per_min"]),
                service_line_mismatch_penalty_per_min=int(prof["preference_penalty_per_min"]),
                stability_fit_penalty_per_min=int(prof["stability_penalty_per_min"]),
                open_slot_bonus_per_min=0,
            )
            assignments, coverage, slot_util, metrics = solve_min_cost_flow(
                slots,
                targets,
                pairs,
                cfg,
                theme=theme,
                required_allocated_min=required_alloc,
                slack_penalty_per_min=int(prof["slack_penalty_per_min"]),
                epsilon_floor_fraction=float(floor_frac),
                utilization_reference_allocated_min=base_alloc,
            )
            metrics = score_solution_metrics(metrics)
            metrics.update({
                "pareto_method": "epsilon_constraint_utilization_dominant",
                "pareto_family": prof["pareto_family"],
                "pareto_description": prof["description"],
                "epsilon_floor_fraction_of_optimum": float(floor_frac),
                "epsilon_required_allocated_min": int(required_alloc),
                "epsilon_allowed_slack_min": int(allowed_slack),
                "utilization_reference_allocated_min": int(base_alloc),
                "utilization_loss_from_optimum_min": int(max(0, base_alloc - int(metrics.get("total_allocated_min", 0)))),
                "utilization_loss_from_optimum_pct": float(max(0, base_alloc - int(metrics.get("total_allocated_min", 0))) / max(1, base_alloc)),
                "epsilon_constraint_satisfied": bool(int(metrics.get("total_allocated_min", 0)) + 1e-9 >= required_alloc),
                "continuity_penalty_per_min": int(prof["continuity_penalty_per_min"]),
                "preference_penalty_per_min": int(prof["preference_penalty_per_min"]),
                "stability_penalty_per_min": int(prof["stability_penalty_per_min"]),
                "noncurrent_penalty_per_min": int(prof["continuity_penalty_per_min"]),
                "service_line_mismatch_penalty_per_min": int(prof["preference_penalty_per_min"]),
                "stability_fit_penalty_per_min": int(prof["stability_penalty_per_min"]),
            })

            if out_dir is not None:
                schedule = build_schedule(assignments, slots, targets, pairs)
                rec = save_plan_artifacts(out_dir, theme, assignments, coverage, slot_util, schedule, slots, targets, pairs, metrics, cfg, goal_reference)
                plan_records.append(rec)
                for k, v in rec["summary_row"].items():
                    if k not in metrics:
                        metrics[k] = v
            rows.append(metrics)

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        return all_df, all_df, []

    all_df = pareto_filter(all_df)
    # Extra helpful ordering for reading CSVs: utilization floor first, then efficient/frontier points.
    frontier = all_df[all_df["pareto_efficient"].fillna(False).astype(bool)].copy()
    sort_cols = [
        "pareto_efficient",
        "epsilon_floor_fraction_of_optimum",
        "useful_utilization",
        "continuity_score",
        "stability_score",
        "preference_score",
        "overall_score",
    ]
    for c in sort_cols:
        if c not in all_df.columns:
            all_df[c] = 0.0
            frontier[c] = 0.0
    all_df = all_df.sort_values(sort_cols, ascending=[False, False, False, False, False, False, False]).reset_index(drop=True)
    frontier = frontier.sort_values(sort_cols, ascending=[False, False, False, False, False, False, False]).reset_index(drop=True)
    return all_df, frontier, plan_records

def pareto_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    objectives = ["total_allocated_min", "continuity_score", "stability_score", "preference_score"]
    work = df.copy().reset_index(drop=True)
    vals = work[objectives].fillna(0).to_numpy(dtype=float)
    efficient = np.ones(len(work), dtype=bool)
    for i in range(len(work)):
        for j in range(len(work)):
            if i == j:
                continue
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                efficient[i] = False
                break
    work["pareto_efficient"] = efficient
    return work



# =============================================================================
# Detailed plan analysis artifacts
# =============================================================================


def safe_name(value: Any) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return s or "plan"


def build_fragmentation_tables(theme: str, schedule: pd.DataFrame, slots: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds series-level fragmentation details.

    Fragmentation is measured on the dominant assigned provider per recurring
    series instance.  A series is approximated by site × room × day × start × end
    across horizon weeks.  This mirrors the planning document's O2 idea while
    still working with the current minute-allocation flow model, where a slot can
    be split across several provider-day targets.
    """
    base_cols = [
        "slot_idx", "slot_id", "horizon_week", "rotation_phase", "site", "room",
        "day_of_week", "start_min", "end_min", "current_provider_id", "capacity_min",
    ]
    slot_base = slots[[c for c in base_cols if c in slots.columns]].copy()
    if schedule.empty:
        details = slot_base.copy()
        details["theme"] = theme
        details["assigned_provider_id"] = ""
        details["dominant_allocated_min"] = 0
        details["n_assigned_providers"] = 0
        details["is_split_slot"] = False
    else:
        tmp = schedule.copy()
        tmp["allocated_min"] = pd.to_numeric(tmp["allocated_min"], errors="coerce").fillna(0)
        # Dominant provider for each slot = largest allocated minutes in that slot.
        tmp = tmp.sort_values(["slot_idx", "allocated_min"], ascending=[True, False])
        dom = tmp.groupby("slot_idx", as_index=False).first()[["slot_idx", "assigned_provider_id", "allocated_min"]]
        dom = dom.rename(columns={"allocated_min": "dominant_allocated_min"})
        nprov = tmp[tmp["allocated_min"] > 0].groupby("slot_idx")["assigned_provider_id"].nunique().rename("n_assigned_providers")
        details = slot_base.merge(dom, on="slot_idx", how="left")
        details = details.merge(nprov.reset_index(), on="slot_idx", how="left")
        details["theme"] = theme
        details["assigned_provider_id"] = details["assigned_provider_id"].fillna("")
        details["dominant_allocated_min"] = pd.to_numeric(details["dominant_allocated_min"], errors="coerce").fillna(0).astype(int)
        details["n_assigned_providers"] = pd.to_numeric(details["n_assigned_providers"], errors="coerce").fillna(0).astype(int)
        details["is_split_slot"] = details["n_assigned_providers"] > 1

    for c in ["site", "room", "day_of_week", "start_min", "end_min"]:
        if c not in details.columns:
            details[c] = ""
    details["series_key"] = (
        details["site"].astype(str) + "|" + details["room"].astype(str) + "|D" +
        details["day_of_week"].astype(str) + "|" + details["start_min"].astype(str) + "-" + details["end_min"].astype(str)
    )
    details = details.sort_values(["series_key", "horizon_week", "slot_idx"]).reset_index(drop=True)
    details["previous_assigned_provider_id"] = details.groupby("series_key")["assigned_provider_id"].shift(1).fillna("")
    details["fragmentation_event"] = (
        (details["previous_assigned_provider_id"] != "")
        & (details["assigned_provider_id"] != "")
        & (details["assigned_provider_id"] != details["previous_assigned_provider_id"])
    )
    details["split_extra_provider_count"] = (details["n_assigned_providers"] - 1).clip(lower=0).astype(int)

    if details.empty:
        summary = pd.DataFrame(columns=["theme", "series_key", "n_instances", "n_handoffs", "n_split_slots", "split_extra_provider_count", "total_capacity_min", "dominant_sequence"])
    else:
        summary = details.groupby("series_key", as_index=False).agg(
            n_instances=("slot_idx", "count"),
            n_handoffs=("fragmentation_event", "sum"),
            n_split_slots=("is_split_slot", "sum"),
            split_extra_provider_count=("split_extra_provider_count", "sum"),
            total_capacity_min=("capacity_min", "sum"),
        )
        seq = details.groupby("series_key")["assigned_provider_id"].apply(lambda x: " -> ".join([str(v) for v in x.tolist() if str(v) != ""])).rename("dominant_sequence")
        summary = summary.merge(seq.reset_index(), on="series_key", how="left")
        summary.insert(0, "theme", theme)
    details.insert(0, "theme", details.pop("theme"))
    return details, summary


def build_change_and_delta_tables(theme: str, schedule: pd.DataFrame, slots: pd.DataFrame, targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detailed changed-block and provider-delta tables.

    Because the optimizer allocates minutes rather than binary ownership, a
    "change" is counted in minutes and also as a dominant-slot change.  This is
    more informative than only a binary changed/not-changed flag.
    """
    if schedule.empty:
        change_details = pd.DataFrame(columns=[
            "theme", "slot_idx", "horizon_week", "day_of_week", "room", "start_hhmm", "end_hhmm",
            "current_provider_id", "assigned_provider_id", "allocated_min", "changed_min", "same_holder",
        ])
    else:
        change_details = schedule.copy()
        change_details["theme"] = theme
        change_details["current_provider_id"] = change_details.get("current_provider_id", "OPEN").map(clean_provider_id)
        change_details["assigned_provider_id"] = change_details.get("assigned_provider_id", "").map(clean_provider_id)
        change_details["allocated_min"] = pd.to_numeric(change_details["allocated_min"], errors="coerce").fillna(0).astype(int)
        change_details["changed"] = (
            (change_details["current_provider_id"] != "OPEN")
            & (change_details["assigned_provider_id"] != change_details["current_provider_id"])
        )
        change_details["changed_min"] = np.where(change_details["changed"], change_details["allocated_min"], 0).astype(int)
        keep = [
            "theme", "slot_idx", "slot_id", "horizon_week", "rotation_phase", "site", "room", "day_of_week",
            "start_hhmm", "end_hhmm", "capacity_min", "current_provider_id", "assigned_provider_id",
            "allocated_min", "changed", "changed_min", "same_holder", "service_line_match", "unit_cost",
        ]
        change_details = change_details[[c for c in keep if c in change_details.columns]].copy()

    current = slots.copy()
    current["current_provider_id"] = current["current_provider_id"].map(clean_provider_id)
    current = current[current["current_provider_id"] != "OPEN"].copy()
    current_by_provider = current.groupby("current_provider_id")["capacity_min"].sum().rename("current_template_min")

    if schedule.empty:
        proposed_by_provider = pd.Series(dtype=float, name="proposed_allocated_min")
        same_holder_by_provider = pd.Series(dtype=float, name="same_holder_min")
        moved_in_by_provider = pd.Series(dtype=float, name="moved_in_min")
        moved_out_by_provider = pd.Series(dtype=float, name="moved_out_min")
    else:
        proposed_by_provider = schedule.groupby("assigned_provider_id")["allocated_min"].sum().rename("proposed_allocated_min")
        same_rows = schedule[schedule.get("same_holder", 0).fillna(0).astype(int) == 1] if "same_holder" in schedule.columns else schedule.iloc[0:0]
        same_holder_by_provider = same_rows.groupby("assigned_provider_id")["allocated_min"].sum().rename("same_holder_min")
        moved_in = schedule[(schedule["current_provider_id"].map(clean_provider_id) != "OPEN") & (schedule["assigned_provider_id"].map(clean_provider_id) != schedule["current_provider_id"].map(clean_provider_id))]
        moved_in_by_provider = moved_in.groupby("assigned_provider_id")["allocated_min"].sum().rename("moved_in_min")
        moved_out_by_provider = moved_in.groupby("current_provider_id")["allocated_min"].sum().rename("moved_out_min")

    target_by_provider = targets.groupby("provider_id")["target_q_int"].sum().rename("target_q_min") if not targets.empty else pd.Series(dtype=float, name="target_q_min")
    providers = sorted(set(current_by_provider.index) | set(proposed_by_provider.index) | set(target_by_provider.index))
    delta = pd.DataFrame({"provider_id": providers})
    for series in [current_by_provider, proposed_by_provider, target_by_provider, same_holder_by_provider, moved_in_by_provider, moved_out_by_provider]:
        delta = delta.merge(series.reset_index().rename(columns={series.index.name or "index": "provider_id"}), on="provider_id", how="left")
    for c in ["current_template_min", "proposed_allocated_min", "target_q_min", "same_holder_min", "moved_in_min", "moved_out_min"]:
        if c not in delta.columns:
            delta[c] = 0
        delta[c] = pd.to_numeric(delta[c], errors="coerce").fillna(0).astype(int)
    delta["theme"] = theme
    delta["delta_added_min"] = delta["proposed_allocated_min"] - delta["current_template_min"]
    delta["delta_added_hours"] = delta["delta_added_min"] / 60.0
    delta["delta_vs_target_min"] = delta["proposed_allocated_min"] - delta["target_q_min"]
    delta["coverage_ratio"] = delta["proposed_allocated_min"] / delta["target_q_min"].clip(lower=1)
    delta["status_vs_target"] = np.select(
        [delta["coverage_ratio"] >= 0.99, delta["proposed_allocated_min"] > 0],
        ["FULL", "PARTIAL"],
        default="NOT_MET",
    )
    delta = delta[["theme"] + [c for c in delta.columns if c != "theme"]]
    return change_details, delta.sort_values(["delta_added_min", "provider_id"], ascending=[False, True]).reset_index(drop=True)


def build_provider_goal_details(theme: str, coverage: pd.DataFrame) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame(columns=["theme", "provider_id", "target_q_min", "allocated_min", "shortage_min", "coverage_ratio", "goal_status", "goal_met"])
    g = coverage.groupby("provider_id", as_index=False).agg(
        target_q_min=("target_q_int", "sum"),
        allocated_min=("allocated_min", "sum"),
        shortage_min=("shortage_min", "sum"),
    )
    g["coverage_ratio"] = g["allocated_min"] / g["target_q_min"].clip(lower=1)
    g["goal_status"] = np.select(
        [g["coverage_ratio"] >= 0.99, g["allocated_min"] > 0],
        ["FULL", "PARTIAL"],
        default="NOT_MET",
    )
    g["goal_met"] = g["goal_status"] == "FULL"
    g.insert(0, "theme", theme)
    return g.sort_values(["goal_status", "shortage_min", "target_q_min"], ascending=[True, False, False]).reset_index(drop=True)


def build_metric_goal_details(theme: str, metrics: Dict[str, Any], cfg: Layer2Config, goal_reference: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    ref = goal_reference or {}
    # Calibrated floors come from the utilization-first optimum.  Secondary
    # floors are deliberately soft because not every Pareto point should satisfy
    # every secondary goal.
    goals = [
        ("G1_useful_utilization_floor", float(metrics.get("useful_utilization", 0.0)), float(ref.get("recommended_useful_utilization_floor", cfg.utilization_warn_floor)), ">="),
        ("G2_target_coverage_floor", float(metrics.get("target_coverage", 0.0)), float(ref.get("recommended_coverage_floor", cfg.coverage_warn_floor)), ">="),
        ("G3_no_physical_overallocation", float(metrics.get("total_allocated_min", 0.0)), float(metrics.get("theoretical_best_allocated_min", 0.0)), "<="),
        ("G4_low_shortage", float(metrics.get("shortage_score", 0.0)), 0.95, ">="),
        ("G5_continuity_soft_goal", float(metrics.get("continuity_score", 0.0)), float(ref.get("recommended_same_holder_goal", 0.0)), ">="),
        ("G6_stability_soft_goal", float(metrics.get("stability_score", 0.0)), float(ref.get("recommended_stability_goal", 0.0)), ">="),
        ("G7_preference_service_line_goal", float(metrics.get("preference_score", 0.0)), 0.95, ">="),
    ]
    rows = []
    for name, value, threshold, sense in goals:
        met = value >= threshold if sense == ">=" else value <= threshold + 1e-9
        rows.append({
            "theme": theme,
            "goal_name": name,
            "value": float(value),
            "threshold": float(threshold),
            "sense": sense,
            "met": bool(met),
            "gap_to_threshold": float(value - threshold if sense == ">=" else threshold - value),
        })
    return pd.DataFrame(rows)


def build_objective_values(theme: str, metrics: Dict[str, Any], frag_summary: pd.DataFrame, change_details: pd.DataFrame) -> Dict[str, Any]:
    o2_frag = int(frag_summary["n_handoffs"].sum()) if not frag_summary.empty and "n_handoffs" in frag_summary.columns else 0
    o2_split = int(frag_summary["split_extra_provider_count"].sum()) if not frag_summary.empty and "split_extra_provider_count" in frag_summary.columns else 0
    if change_details.empty:
        changed_min = 0
        changed_slots = 0
    else:
        changed_series = pd.to_numeric(
            change_details["changed_min"] if "changed_min" in change_details.columns else pd.Series(0, index=change_details.index),
            errors="coerce",
        ).fillna(0)
        changed_min = int(changed_series.sum())
        changed_slots = int(change_details.loc[changed_series > 0, "slot_idx"].nunique()) if "slot_idx" in change_details.columns else 0
    return {
        "theme": theme,
        "O1_shortage_deviation_min": int(metrics.get("total_shortage_min", 0)),
        "O1_target_coverage": float(metrics.get("target_coverage", 0.0)),
        "O2_fragmentation_handoffs": o2_frag,
        "O2_split_extra_provider_count": o2_split,
        "O3_changed_slots": changed_slots,
        "O3_changed_minutes": changed_min,
        "O4_preference_proxy_service_line_match_min": int(metrics.get("service_line_match_min", 0)),
        "O4_preference_proxy_score": float(metrics.get("preference_score", 0.0)),
        "O5_consistency_proxy_same_holder_min": int(metrics.get("same_holder_min", 0)),
        "O5_consistency_proxy_score": float(metrics.get("continuity_score", 0.0)),
        "O6_open_slot_minutes_proxy": int(metrics.get("open_slot_min", 0)),
        "O7_time_preference_proxy_available": False,
        "objective_proxy_note": "O4-O7 are proxy metrics because the current inputs do not include declared provider day/time preference or OOB-day tables.",
    }


def analyze_plan(
    theme: str,
    assignments: pd.DataFrame,
    coverage: pd.DataFrame,
    slot_util: pd.DataFrame,
    schedule: pd.DataFrame,
    slots: pd.DataFrame,
    targets: pd.DataFrame,
    pairs: pd.DataFrame,
    metrics: Dict[str, Any],
    cfg: Layer2Config,
    goal_reference: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    frag_details, frag_summary = build_fragmentation_tables(theme, schedule, slots)
    change_details, provider_delta = build_change_and_delta_tables(theme, schedule, slots, targets)
    provider_goals = build_provider_goal_details(theme, coverage)
    metric_goals = build_metric_goal_details(theme, metrics, cfg, goal_reference)
    objective_values = build_objective_values(theme, metrics, frag_summary, change_details)

    n_provider_full = int((provider_goals["goal_status"] == "FULL").sum()) if not provider_goals.empty else 0
    n_provider_partial = int((provider_goals["goal_status"] == "PARTIAL").sum()) if not provider_goals.empty else 0
    n_provider_not = int((provider_goals["goal_status"] == "NOT_MET").sum()) if not provider_goals.empty else 0
    n_metric_met = int(metric_goals["met"].sum()) if not metric_goals.empty else 0
    n_metric_total = int(len(metric_goals))

    plan_summary = {
        "theme": theme,
        "metric_goals_met": n_metric_met,
        "metric_goals_total": n_metric_total,
        "provider_goals_fully_met": n_provider_full,
        "provider_goals_partial": n_provider_partial,
        "provider_goals_not_met": n_provider_not,
        "provider_goals_total": int(len(provider_goals)),
        "total_goals_met_count": int(n_metric_met + n_provider_full),
        "total_goals_count": int(n_metric_total + len(provider_goals)),
        "O1_shortage_deviation_min": objective_values["O1_shortage_deviation_min"],
        "O2_fragmentation_handoffs": objective_values["O2_fragmentation_handoffs"],
        "O3_changed_slots": objective_values["O3_changed_slots"],
        "O3_changed_minutes": objective_values["O3_changed_minutes"],
        "allocated_min": int(metrics.get("total_allocated_min", 0)),
        "useful_utilization": float(metrics.get("useful_utilization", 0.0)),
        "target_coverage": float(metrics.get("target_coverage", 0.0)),
        "continuity_score": float(metrics.get("continuity_score", 0.0)),
        "stability_score": float(metrics.get("stability_score", 0.0)),
        "preference_score": float(metrics.get("preference_score", 0.0)),
        "overall_score": float(metrics.get("overall_score", 0.0)),
    }
    return {
        "plan_summary": plan_summary,
        "objective_values": objective_values,
        "provider_goals": provider_goals,
        "metric_goals": metric_goals,
        "fragmentation_details": frag_details,
        "fragmentation_summary": frag_summary,
        "change_details": change_details,
        "provider_delta": provider_delta,
    }


def save_plan_artifacts(
    base_out_dir: str | Path,
    theme: str,
    assignments: pd.DataFrame,
    coverage: pd.DataFrame,
    slot_util: pd.DataFrame,
    schedule: pd.DataFrame,
    slots: pd.DataFrame,
    targets: pd.DataFrame,
    pairs: pd.DataFrame,
    metrics: Dict[str, Any],
    cfg: Layer2Config,
    goal_reference: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    plan_name = safe_name(theme)
    plan_dir = ensure_dir(Path(base_out_dir) / "plans" / plan_name)
    analysis = analyze_plan(theme, assignments, coverage, slot_util, schedule, slots, targets, pairs, metrics, cfg, goal_reference)

    paths = {
        "plan_dir": str(plan_dir),
        "schedule": str(plan_dir / "schedule_by_week.csv"),
        "assignments": str(plan_dir / "assignments.csv"),
        "coverage": str(plan_dir / "provider_day_coverage.csv"),
        "slot_utilization": str(plan_dir / "slot_utilization.csv"),
        "metrics": str(plan_dir / "metrics.json"),
        "objectives": str(plan_dir / "objective_values.json"),
        "objective_values_csv": str(plan_dir / "objective_values.csv"),
        "provider_goals": str(plan_dir / "provider_goal_details.csv"),
        "metric_goals": str(plan_dir / "metric_goal_details.csv"),
        "fragmentation_details": str(plan_dir / "fragmentation_details.csv"),
        "fragmentation_summary": str(plan_dir / "fragmentation_summary.csv"),
        "change_details": str(plan_dir / "change_details.csv"),
        "provider_delta": str(plan_dir / "provider_delta.csv"),
        "summary": str(plan_dir / "plan_summary.json"),
    }

    assignments.to_csv(paths["assignments"], index=False)
    schedule.to_csv(paths["schedule"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    slot_util.to_csv(paths["slot_utilization"], index=False)
    analysis["provider_goals"].to_csv(paths["provider_goals"], index=False)
    analysis["metric_goals"].to_csv(paths["metric_goals"], index=False)
    analysis["fragmentation_details"].to_csv(paths["fragmentation_details"], index=False)
    analysis["fragmentation_summary"].to_csv(paths["fragmentation_summary"], index=False)
    analysis["change_details"].to_csv(paths["change_details"], index=False)
    analysis["provider_delta"].to_csv(paths["provider_delta"], index=False)
    pd.DataFrame([analysis["objective_values"]]).to_csv(paths["objective_values_csv"], index=False)
    write_json(metrics, paths["metrics"])
    write_json(analysis["objective_values"], paths["objectives"])
    write_json(analysis["plan_summary"], paths["summary"])

    row = dict(analysis["plan_summary"])
    row.update({
        "plan_dir": paths["plan_dir"],
        "schedule_path": paths["schedule"],
        "provider_delta_path": paths["provider_delta"],
        "fragmentation_summary_path": paths["fragmentation_summary"],
        "change_details_path": paths["change_details"],
        "provider_goals_path": paths["provider_goals"],
        "metric_goals_path": paths["metric_goals"],
    })
    return {
        "summary_row": row,
        "analysis": analysis,
        "paths": paths,
    }


def write_aggregate_plan_tables(base_out_dir: str | Path, records: List[Dict[str, Any]]) -> Dict[str, str]:
    out = Path(base_out_dir)
    files = {
        "plans_summary": str(out / "plans_summary.csv"),
        "objective_values_by_plan": str(out / "objective_values_by_plan.csv"),
        "provider_goal_details_by_plan": str(out / "provider_goal_details_by_plan.csv"),
        "metric_goal_details_by_plan": str(out / "metric_goal_details_by_plan.csv"),
        "fragmentation_summary_by_plan": str(out / "fragmentation_summary_by_plan.csv"),
        "fragmentation_details_by_plan": str(out / "fragmentation_details_by_plan.csv"),
        "change_details_by_plan": str(out / "change_details_by_plan.csv"),
        "provider_delta_by_plan": str(out / "provider_delta_by_plan.csv"),
    }
    if not records:
        for path in files.values():
            pd.DataFrame().to_csv(path, index=False)
        return files

    pd.DataFrame([r["summary_row"] for r in records]).to_csv(files["plans_summary"], index=False)
    pd.DataFrame([r["analysis"]["objective_values"] for r in records]).to_csv(files["objective_values_by_plan"], index=False)

    def concat_table(key: str) -> pd.DataFrame:
        parts = [r["analysis"][key] for r in records if isinstance(r.get("analysis", {}).get(key), pd.DataFrame) and not r["analysis"][key].empty]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    concat_table("provider_goals").to_csv(files["provider_goal_details_by_plan"], index=False)
    concat_table("metric_goals").to_csv(files["metric_goal_details_by_plan"], index=False)
    concat_table("fragmentation_summary").to_csv(files["fragmentation_summary_by_plan"], index=False)
    concat_table("fragmentation_details").to_csv(files["fragmentation_details_by_plan"], index=False)
    concat_table("change_details").to_csv(files["change_details_by_plan"], index=False)
    concat_table("provider_delta").to_csv(files["provider_delta_by_plan"], index=False)
    return files


# =============================================================================
# Output assembly and validation
# =============================================================================


def build_schedule(assignments: pd.DataFrame, slots: pd.DataFrame, targets: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        return pd.DataFrame()
    s_cols = [
        "slot_idx", "slot_id", "horizon_week", "rotation_phase", "site", "room", "day_of_week",
        "start_min", "end_min", "duration_min", "capacity_min", "current_provider_id", "slot_service_line", "is_open",
    ]
    t_cols = ["target_idx", "provider_id", "service_line", "target_q_int", "mean_demand_min", "p50_demand_min", "p95_demand_min"]
    p_cols = ["slot_idx", "target_idx", "unit_cost", "same_holder", "service_line_match", "is_open_slot"]
    if "fit_score" in pairs.columns:
        p_cols.append("fit_score")
    out = assignments.merge(slots[s_cols], on="slot_idx", how="left")
    out = out.merge(targets[t_cols], on="target_idx", how="left")
    out = out.merge(pairs[p_cols].drop_duplicates(["slot_idx", "target_idx"]), on=["slot_idx", "target_idx"], how="left")
    out = out.rename(columns={"provider_id": "assigned_provider_id", "service_line": "assigned_service_line"})
    out["start_hhmm"] = out["start_min"].map(min_to_hhmm)
    out["end_hhmm"] = out["end_min"].map(min_to_hhmm)
    out = out.sort_values(["horizon_week", "day_of_week", "site", "room", "start_min", "allocated_min"], ascending=[True, True, True, True, True, False])
    return out.reset_index(drop=True)


def min_to_hhmm(x: Any) -> str:
    try:
        m = int(x) % (24 * 60)
        return f"{m // 60:02d}:{m % 60:02d}"
    except Exception:
        return ""


def validate_layer2_outputs(
    metrics: Dict[str, Any],
    slots: pd.DataFrame,
    targets: pd.DataFrame,
    assignments: pd.DataFrame,
    files: Dict[str, str],
    cfg: Layer2Config,
) -> Dict[str, Any]:
    """
    Layer-2 validation is intentionally stricter than just "files exist".

    It validates four things:
      1. Inputs and required artifacts exist.
      2. The allocation is structurally feasible: no negative minutes, no slot
         capacity violation, no target over-allocation, and IDs are valid.
      3. The objective output is meaningful even when full coverage is physically
         impossible: useful utilization, coverage, shortage, continuity,
         stability, and preference are all finite.
      4. The Pareto artifacts are real: all Pareto axes exist and at least one
         efficient solution is present when the Pareto sweep is enabled.

    WARN means the run is usable but the situation deserves attention, for example
    demand target > total capacity or useful utilization below the configured floor.
    FAIL means the artifacts are inconsistent or infeasible and should not be used
    downstream.
    """
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, level: str, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "level": level if not ok else "PASS", "detail": str(detail)})

    def finite_metric(name: str) -> bool:
        try:
            return bool(np.isfinite(float(metrics.get(name, np.nan))))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Required input/output existence
    # ------------------------------------------------------------------
    add("Slots non-empty", len(slots) > 0, "FAIL", f"slots={len(slots)}")
    add("Targets non-empty", len(targets) > 0, "FAIL", f"targets={len(targets)}")

    required_files = [
        "assignments",
        "schedule_by_week",
        "provider_day_coverage",
        "slot_utilization",
        "candidate_pairs",
        "pareto_all",
        "pareto_frontier",
        "calibrated_goals",
        "layer2_result",
        "plans_summary",
        "objective_values_by_plan",
        "provider_goal_details_by_plan",
        "metric_goal_details_by_plan",
        "fragmentation_summary_by_plan",
        "change_details_by_plan",
        "provider_delta_by_plan",
    ]
    missing = [k for k in required_files if k not in files or not Path(files[k]).exists()]
    add("Required artifact files written", len(missing) == 0, "FAIL", "present" if not missing else "missing=" + ",".join(missing))

    # ------------------------------------------------------------------
    # Core feasibility checks
    # ------------------------------------------------------------------
    add("Capacity positive", metrics.get("total_capacity_min", 0) > 0, "FAIL", f"capacity={metrics.get('total_capacity_min')}")
    add("Target positive", metrics.get("total_target_q_min", 0) > 0, "FAIL", f"target={metrics.get('total_target_q_min')}")

    has_required_assignment_cols = assignments.empty or {"slot_idx", "target_idx", "allocated_min"}.issubset(assignments.columns)
    add("Assignments have required columns", has_required_assignment_cols, "FAIL", f"cols={list(assignments.columns)}")

    if has_required_assignment_cols:
        add("Non-negative assignments", assignments.empty or (assignments["allocated_min"] >= 0).all(), "FAIL", "all allocated_min >= 0")

        if not assignments.empty and len(slots) > 0:
            valid_slots = set(slots["slot_idx"].astype(int).tolist())
            used_slots = set(assignments["slot_idx"].astype(int).tolist())
            bad_slots = sorted(list(used_slots - valid_slots))[:10]
            add("All assigned slots exist", len(bad_slots) == 0, "FAIL", f"bad_slot_idx_sample={bad_slots}" if bad_slots else "ok")

            slot_capacity = slots.set_index("slot_idx")["capacity_min"].astype(float)
            slot_alloc = assignments.groupby("slot_idx")["allocated_min"].sum().astype(float)
            slot_delta = slot_alloc - slot_capacity.reindex(slot_alloc.index).fillna(0.0)
            max_over = float(slot_delta.max()) if len(slot_delta) else 0.0
            add("No slot capacity violation", max_over <= 1e-6, "FAIL", f"max_over_capacity_min={max_over:.3f}")

        if not assignments.empty and len(targets) > 0:
            valid_targets = set(targets["target_idx"].astype(int).tolist())
            used_targets = set(assignments["target_idx"].astype(int).tolist())
            bad_targets = sorted(list(used_targets - valid_targets))[:10]
            add("All assigned targets exist", len(bad_targets) == 0, "FAIL", f"bad_target_idx_sample={bad_targets}" if bad_targets else "ok")

            target_q = targets.set_index("target_idx")["target_q_int"].astype(float)
            target_alloc = assignments.groupby("target_idx")["allocated_min"].sum().astype(float)
            target_delta = target_alloc - target_q.reindex(target_alloc.index).fillna(0.0)
            max_over_target = float(target_delta.max()) if len(target_delta) else 0.0
            add("No target over-allocation", max_over_target <= 1e-6, "FAIL", f"max_over_target_min={max_over_target:.3f}")

    # ------------------------------------------------------------------
    # Metrics and physical sanity
    # ------------------------------------------------------------------
    for m in [
        "useful_utilization",
        "raw_capacity_utilization",
        "target_coverage",
        "continuity_score",
        "stability_score",
        "preference_score",
        "overall_score",
    ]:
        add(f"Metric finite: {m}", finite_metric(m), "FAIL", f"{m}={metrics.get(m)}")

    total_capacity = float(metrics.get("total_capacity_min", 0) or 0)
    total_target = float(metrics.get("total_target_q_min", 0) or 0)
    total_allocated = float(metrics.get("total_allocated_min", 0) or 0)
    theoretical_best = min(total_capacity, total_target)
    allocation_gap = theoretical_best - total_allocated

    add(
        "Allocated minutes not above physical bound",
        total_allocated <= theoretical_best + 1e-6,
        "FAIL",
        f"allocated={total_allocated:.0f}, min(capacity,target)={theoretical_best:.0f}",
    )

    if total_target > total_capacity:
        add(
            "Physical capacity vs target",
            False,
            "WARN",
            "Target exceeds total capacity; full alpha coverage is impossible without more OR capacity/open slots.",
        )
    else:
        add("Physical capacity vs target", True, "PASS", "capacity >= target")

    if allocation_gap > max(60.0, 0.02 * max(1.0, theoretical_best)):
        add(
            "Allocation close to theoretical best",
            False,
            "WARN",
            f"gap_to_min(capacity,target)={allocation_gap:.0f} minutes; inspect candidate pruning/compatibility.",
        )
    else:
        add("Allocation close to theoretical best", True, "PASS", f"gap={allocation_gap:.0f} minutes")

    if float(metrics.get("useful_utilization", 0.0) or 0.0) < cfg.utilization_warn_floor:
        add(
            "Useful utilization above warning floor",
            False,
            "WARN",
            f"useful_utilization={metrics.get('useful_utilization'):.4f}, floor={cfg.utilization_warn_floor:.4f}",
        )
    else:
        add("Useful utilization above warning floor", True, "PASS", f"useful_utilization={metrics.get('useful_utilization'):.4f}")

    # ------------------------------------------------------------------
    # Cross-file consistency checks
    # ------------------------------------------------------------------
    try:
        if "provider_day_coverage" in files and Path(files["provider_day_coverage"]).exists():
            cov = pd.read_csv(files["provider_day_coverage"])
            cov_total = float(cov.get("allocated_min", pd.Series(dtype=float)).sum())
            add("Coverage file matches total allocation", abs(cov_total - total_allocated) <= 1e-6, "FAIL", f"coverage_alloc={cov_total:.0f}, metrics_alloc={total_allocated:.0f}")
    except Exception as exc:
        add("Coverage file readable", False, "FAIL", f"error={exc}")

    try:
        if "slot_utilization" in files and Path(files["slot_utilization"]).exists():
            su = pd.read_csv(files["slot_utilization"])
            su_total = float(su.get("allocated_min", pd.Series(dtype=float)).sum())
            max_util = float(su.get("utilization", pd.Series([0.0])).max())
            add("Slot utilization file matches total allocation", abs(su_total - total_allocated) <= 1e-6, "FAIL", f"slot_alloc={su_total:.0f}, metrics_alloc={total_allocated:.0f}")
            add("Slot utilization never exceeds 1", max_util <= 1.000001, "FAIL", f"max_slot_utilization={max_util:.6f}")
    except Exception as exc:
        add("Slot utilization file readable", False, "FAIL", f"error={exc}")

    # ------------------------------------------------------------------
    # Pareto validation
    # ------------------------------------------------------------------
    pareto_axes = ["useful_utilization", "continuity_score", "stability_score", "preference_score"]
    if cfg.run_pareto:
        try:
            pareto_all_path = Path(files.get("pareto_all", ""))
            pareto_front_path = Path(files.get("pareto_frontier", ""))
            pareto_all = pd.read_csv(pareto_all_path) if pareto_all_path.exists() else pd.DataFrame()
            pareto_front = pd.read_csv(pareto_front_path) if pareto_front_path.exists() else pd.DataFrame()

            add("Pareto all non-empty", len(pareto_all) > 0, "FAIL", f"rows={len(pareto_all)}")
            add("Pareto frontier non-empty", len(pareto_front) > 0, "FAIL", f"rows={len(pareto_front)}")
            missing_axes = [a for a in pareto_axes if a not in pareto_all.columns]
            add("Pareto axes present", len(missing_axes) == 0, "FAIL", "present" if not missing_axes else "missing=" + ",".join(missing_axes))
            if "pareto_efficient" in pareto_all.columns:
                n_eff = int(pareto_all["pareto_efficient"].fillna(False).astype(bool).sum())
                add("Pareto efficient rows marked", n_eff > 0, "FAIL", f"n_efficient={n_eff}")
            else:
                add("Pareto efficient rows marked", False, "FAIL", "pareto_efficient column missing")

            if not pareto_all.empty and not missing_axes:
                finite = np.isfinite(pareto_all[pareto_axes].to_numpy(dtype=float)).all()
                add("Pareto metrics finite", finite, "FAIL", "all Pareto axes finite" if finite else "non-finite Pareto score found")

            if "epsilon_required_allocated_min" in pareto_all.columns and "total_allocated_min" in pareto_all.columns:
                req = pd.to_numeric(pareto_all["epsilon_required_allocated_min"], errors="coerce").fillna(0)
                alloc = pd.to_numeric(pareto_all["total_allocated_min"], errors="coerce").fillna(0)
                min_gap = float((alloc - req).min()) if len(pareto_all) else 0.0
                add("Epsilon utilization floors respected", min_gap >= -1e-6, "FAIL", f"min_alloc_minus_required={min_gap:.3f}")
            else:
                add("Epsilon utilization floor columns present", False, "WARN", "old Pareto format without epsilon columns")
        except Exception as exc:
            add("Pareto files readable", False, "FAIL", f"error={exc}")
    else:
        add("Pareto validation skipped", True, "PASS", "cfg.run_pareto=False")

    # ------------------------------------------------------------------
    # Detailed plan-artifact validation
    # ------------------------------------------------------------------
    try:
        ps = pd.read_csv(files.get("plans_summary", "")) if Path(files.get("plans_summary", "")).exists() else pd.DataFrame()
        add("Plan summary non-empty", len(ps) > 0, "FAIL", f"rows={len(ps)}")
        if not ps.empty:
            need_cols = ["theme", "total_goals_met_count", "total_goals_count", "O2_fragmentation_handoffs", "O3_changed_slots", "schedule_path"]
            missing_cols = [c for c in need_cols if c not in ps.columns]
            add("Plan summary has goal/objective columns", len(missing_cols) == 0, "FAIL", "present" if not missing_cols else "missing=" + ",".join(missing_cols))
            if "schedule_path" in ps.columns:
                missing_sched = [x for x in ps["schedule_path"].dropna().astype(str).tolist() if not Path(x).exists()]
                add("Saved schedule exists for every plan", len(missing_sched) == 0, "FAIL", "all plan schedules present" if not missing_sched else f"missing={len(missing_sched)}")
        for key, label in [
            ("objective_values_by_plan", "Objective values by plan"),
            ("provider_goal_details_by_plan", "Provider goal details"),
            ("metric_goal_details_by_plan", "Metric goal details"),
            ("fragmentation_summary_by_plan", "Fragmentation summary"),
            ("change_details_by_plan", "Change details"),
            ("provider_delta_by_plan", "Provider delta details"),
        ]:
            pp = Path(files.get(key, ""))
            add(f"{label} written", pp.exists(), "FAIL", str(pp) if pp.exists() else "missing")
    except Exception as exc:
        add("Detailed plan artifacts readable", False, "FAIL", f"error={exc}")

    status = "PASS"
    if any((not c["ok"]) and c["level"] == "FAIL" for c in checks):
        status = "FAIL"
    elif any((not c["ok"]) and c["level"] == "WARN" for c in checks):
        status = "WARN"

    return {
        "status": status,
        "checks": checks,
        "summary": {
            "n_checks": len(checks),
            "n_fail": int(sum((not c["ok"]) and c["level"] == "FAIL" for c in checks)),
            "n_warn": int(sum((not c["ok"]) and c["level"] == "WARN" for c in checks)),
            "n_pass": int(sum(c["ok"] for c in checks)),
            "total_capacity_min": metrics.get("total_capacity_min"),
            "total_target_q_min": metrics.get("total_target_q_min"),
            "total_allocated_min": metrics.get("total_allocated_min"),
            "useful_utilization": metrics.get("useful_utilization"),
            "target_coverage": metrics.get("target_coverage"),
        },
    }

def print_validation_report(report: Dict[str, Any]) -> None:
    print("\n" + "─" * 62)
    print("  LAYER 2 VALIDATION REPORT")
    print("─" * 62)
    for c in report.get("checks", []):
        if c["ok"]:
            mark = "✓ PASS"
        elif c["level"] == "WARN":
            mark = "⚠ WARN"
        else:
            mark = "✗ FAIL"
        print(f"  {mark:<8} {c['name']:<42} {c['detail']}")
    print("─" * 62)


def write_summary_md(path: str | Path, metrics: Dict[str, Any], goals: Dict[str, Any], validation: Dict[str, Any]) -> None:
    lines = [
        "# Layer 2 Summary",
        "",
        f"Validation status: **{validation.get('status', 'UNKNOWN')}**",
        "",
        "## Utilization-first solution",
        "",
        f"- Solver backend: `{metrics.get('solver_backend')}`",
        f"- Solver status: `{metrics.get('solver_status')}`",
        f"- Runtime seconds: `{metrics.get('runtime_s'):.2f}`",
        f"- Total capacity minutes: `{metrics.get('total_capacity_min')}`",
        f"- Total alpha target minutes: `{metrics.get('total_target_q_min')}`",
        f"- Allocated minutes: `{metrics.get('total_allocated_min')}`",
        f"- Shortage minutes: `{metrics.get('total_shortage_min')}`",
        f"- Useful utilization: `{metrics.get('useful_utilization'):.4f}`",
        f"- Raw capacity utilization: `{metrics.get('raw_capacity_utilization'):.4f}`",
        f"- Target coverage: `{metrics.get('target_coverage'):.4f}`",
        f"- Same-holder share: `{metrics.get('same_holder_share'):.4f}`",
        f"- Stability score: `{metrics.get('stability_score'):.4f}`",
        f"- Preference score: `{metrics.get('preference_score'):.4f}`",
        f"- Overall score: `{metrics.get('overall_score'):.4f}`",
        "",
        "## Calibrated goals",
        "",
        f"- Recommended coverage floor: `{goals.get('recommended_coverage_floor'):.4f}`",
        f"- Recommended useful-utilization floor: `{goals.get('recommended_useful_utilization_floor'):.4f}`",
        f"- Recommended same-holder goal: `{goals.get('recommended_same_holder_goal'):.4f}`",
        "",
        "> These goals are calibrated from the attainable utilization-first optimum. They should be used as soft goals or Pareto floors, not impossible hard constraints.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_pareto_plot(pareto_all: pd.DataFrame, pareto_frontier: pd.DataFrame, out_path: str | Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if pareto_all.empty:
            return
        plt.figure(figsize=(8, 5))
        x_col = "continuity_score" if "continuity_score" in pareto_all.columns else "same_holder_share"
        plt.scatter(pareto_all[x_col], pareto_all["useful_utilization"], label="all profiles")
        if not pareto_frontier.empty:
            plt.scatter(pareto_frontier[x_col], pareto_frontier["useful_utilization"], marker="x", s=90, label="Pareto frontier")
        plt.xlabel("Continuity score")
        plt.ylabel("Useful utilization")
        plt.title("Layer 2 Pareto frontier: utilization vs continuity")
        plt.legend()
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=180)
        plt.close()
    except Exception as exc:
        LOG.warning("Could not save Pareto plot: %s", exc)


# =============================================================================
# Main pipeline
# =============================================================================


def run_layer2(cfg: Layer2Config) -> Dict[str, Any]:
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    out_dir = ensure_dir(cfg.output_dir)

    pre_layer = read_json(cfg.pre_layer_result)
    t_star = infer_t_star(pre_layer)
    LOG.info("Pre-layer result loaded: %s T*=%s", cfg.pre_layer_result, t_star)

    raw_blocks = load_json_table(cfg.blocks_json)
    raw_providers = load_json_table(cfg.providers_json)
    providers = normalize_providers(raw_providers)
    LOG.info("Raw blocks loaded: %s rows=%s", cfg.blocks_json, len(raw_blocks))
    LOG.info("Providers loaded: %s rows=%s normalized=%s", cfg.providers_json, len(raw_providers), len(providers))

    layer1_dir = Path(cfg.layer1_dir)
    early_release_path = first_existing([
        layer1_dir / "early_release_projected.csv",
        layer1_dir / "early_release.csv",
        layer1_dir / "projected_early_release.csv",
    ])
    early_release = pd.read_csv(early_release_path) if early_release_path else None
    if early_release_path:
        LOG.info("Early release loaded: %s rows=%s", early_release_path, len(early_release))

    template = extract_template_from_prelayer(pre_layer, raw_blocks)
    slots = normalize_template_to_slots(template, pre_layer, providers, cfg, early_release=early_release)
    targets = load_layer1_targets(cfg.layer1_dir, cfg.alpha, providers)

    capacity = int(slots["capacity_min"].sum())
    target = int(targets["target_q_int"].sum())
    LOG.info(
        "Capacity sanity: total_slot_capacity=%s total_alpha_target=%s capacity/target=%.3f",
        capacity,
        target,
        capacity / max(1, target),
    )
    if capacity < target:
        LOG.warning(
            "Total capacity is smaller than alpha-quantile demand target. Solver will minimize shortage, "
            "but 100%% alpha coverage is physically impossible unless more OR capacity/open slots are added."
        )

    assignments, coverage, slot_util, pairs, metrics = solve_utilization_first_adaptive(slots, targets, cfg)
    goal_report = calibrated_goal_report(metrics)

    schedule = build_schedule(assignments, slots, targets, pairs)

    # Save the main utilization-first plan in the same structure as all Pareto
    # plans.  This gives downstream notebooks a uniform way to inspect schedules,
    # goals, deltas, fragmentation, and changes for every candidate.
    plan_records: List[Dict[str, Any]] = []
    util_record = save_plan_artifacts(out_dir, "utilization_first_selected", assignments, coverage, slot_util, schedule, slots, targets, pairs, metrics, cfg, goal_report)
    plan_records.append(util_record)
    for k, v in util_record["summary_row"].items():
        if k not in metrics:
            metrics[k] = v

    # Pareto grid uses the same slots/targets but varies continuity, stability,
    # and preference penalties. Use the candidate depth that reached the
    # utilization-first optimum, otherwise the Pareto sweep can look artificially
    # bad simply because candidates were pruned too aggressively.
    if cfg.run_pareto:
        pareto_top = int(cfg.pareto_candidate_top_per_day or metrics.get("candidate_top_per_day_used", cfg.candidate_top_per_day) or 0)
        pareto_cfg = replace(cfg, candidate_top_per_day=pareto_top)
        pareto_all, pareto_frontier, pareto_records = run_pareto_grid(slots, targets, pareto_cfg, out_dir=out_dir, goal_reference=goal_report)
        plan_records.extend(pareto_records)
    else:
        pareto_all = pd.DataFrame([metrics])
        pareto_frontier = pd.DataFrame([metrics])
        pareto_all["pareto_efficient"] = True
        pareto_frontier["pareto_efficient"] = True

    aggregate_plan_files = write_aggregate_plan_tables(out_dir, plan_records)

    files = {
        "assignments": str(out_dir / "assignments.csv"),
        "schedule_by_week": str(out_dir / "schedule_by_week.csv"),
        "provider_day_coverage": str(out_dir / "provider_day_coverage.csv"),
        "slot_utilization": str(out_dir / "slot_utilization.csv"),
        "candidate_pairs": str(out_dir / "candidate_pairs.csv"),
        "pareto_all": str(out_dir / "pareto_all.csv"),
        "pareto_frontier": str(out_dir / "pareto_frontier.csv"),
        "pareto_plot": str(out_dir / "pareto_frontier.png"),
        "calibrated_goals": str(out_dir / "calibrated_goals.json"),
        "validation_report": str(out_dir / "validation_report.json"),
        "layer2_result": str(out_dir / "layer2_result.json"),
        "summary_md": str(out_dir / "layer2_summary.md"),
        **aggregate_plan_files,
    }

    assignments.to_csv(files["assignments"], index=False)
    schedule.to_csv(files["schedule_by_week"], index=False)
    coverage.to_csv(files["provider_day_coverage"], index=False)
    slot_util.to_csv(files["slot_utilization"], index=False)
    # Candidate pairs can be large but useful for debugging solver behavior.
    pairs.to_csv(files["candidate_pairs"], index=False)
    pareto_all.to_csv(files["pareto_all"], index=False)
    pareto_frontier.to_csv(files["pareto_frontier"], index=False)
    save_pareto_plot(pareto_all, pareto_frontier, files["pareto_plot"])
    write_json(goal_report, files["calibrated_goals"])

    # Write a preliminary layer2_result before validation so the required-file
    # validation can check it. Then overwrite with the final validation status.
    result = {
        "layer": "layer2",
        "status": "PENDING_VALIDATION",
        "t_star": t_star,
        "config": asdict(cfg),
        "metrics": metrics,
        "calibrated_goals": goal_report,
        "files": files,
        "pareto_axes": ["useful_utilization", "continuity_score", "stability_score", "preference_score"],
        "n_slots": int(len(slots)),
        "n_targets": int(len(targets)),
        "n_candidate_pairs": int(len(pairs)),
        "n_assignments": int(len(assignments)),
        "n_saved_plans": int(len(plan_records)),
        "plan_artifact_files": aggregate_plan_files,
    }
    write_json(result, files["layer2_result"])

    validation = validate_layer2_outputs(metrics, slots, targets, assignments, files, cfg)
    write_json(validation, files["validation_report"])

    result["status"] = validation["status"]
    result["validation"] = validation
    write_json(result, files["layer2_result"])
    write_summary_md(files["summary_md"], metrics, goal_report, validation)

    print_validation_report(validation)
    LOG.info("=" * 60)
    LOG.info("Layer 2 complete")
    LOG.info("  Slots          : %s", len(slots))
    LOG.info("  Targets        : %s", len(targets))
    LOG.info("  Candidate pairs: %s", len(pairs))
    LOG.info("  Assignments    : %s", len(assignments))
    LOG.info("  Allocated      : %s", metrics["total_allocated_min"])
    LOG.info("  Useful util    : %.4f", metrics["useful_utilization"])
    LOG.info("  Target coverage: %.4f", metrics["target_coverage"])
    LOG.info("  Validation     : %s", validation["status"])
    LOG.info("  Output dir     : %s", out_dir)
    LOG.info("=" * 60)
    return result


# =============================================================================
# CLI
# =============================================================================


def apply_config_defaults(cfg: Layer2Config, yaml_cfg: Dict[str, Any]) -> Layer2Config:
    data = asdict(cfg)

    # Support both flat config and config/default_config.yaml style nested paths/layer2.
    paths = yaml_cfg.get("paths", {}) if isinstance(yaml_cfg.get("paths"), dict) else {}
    layer2 = yaml_cfg.get("layer2", {}) if isinstance(yaml_cfg.get("layer2"), dict) else {}

    mapping = {
        "blocks_json": paths.get("blocks_json") or paths.get("blocks") or paths.get("blocks_path"),
        "providers_json": paths.get("providers_json") or paths.get("providers") or paths.get("providers_path"),
        "output_dir": paths.get("output_dir") or layer2.get("output_dir"),
        "pre_layer_result": layer2.get("pre_layer_result"),
        "layer1_dir": layer2.get("layer1_dir"),
    }
    for k, v in mapping.items():
        if v:
            data[k] = v

    for k in data.keys():
        if k in layer2 and layer2[k] is not None:
            data[k] = layer2[k]
        elif k in yaml_cfg and yaml_cfg[k] is not None:
            data[k] = yaml_cfg[k]
    return Layer2Config(**data)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Layer 2 stochastic OR block allocation optimizer.")
    p.add_argument("--config", default=None, help="Optional YAML config path.")
    p.add_argument("--pre_layer_result", default=None)
    p.add_argument("--layer1_dir", default=None)
    p.add_argument("--blocks", "--blocks_json", dest="blocks_json", default=None)
    p.add_argument("--providers", "--providers_json", dest="providers_json", default=None)
    p.add_argument("--out", "--output_dir", dest="output_dir", default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--time_limit_s", type=float, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--max_slots", type=int, default=None)
    p.add_argument("--candidate_top_per_day", type=int, default=None, help="0 means all candidates.")
    p.add_argument("--pareto_grid", type=int, default=None)
    p.add_argument("--noncurrent_penalty_per_min", type=int, default=None)
    p.add_argument("--service_line_mismatch_penalty_per_min", type=int, default=None)
    p.add_argument("--stability_fit_penalty_per_min", type=int, default=None)
    p.add_argument("--open_slot_bonus_per_min", type=int, default=None)
    p.add_argument("--pareto_candidate_top_per_day", type=int, default=None, help="0 means inherit the utilization-first adaptive candidate depth.")
    p.add_argument("--pareto_min_utilization_fraction", type=float, default=None, help="Lowest ε floor as a fraction of the utilization-first optimum, e.g. 0.95.")
    p.add_argument("--pareto_slack_cost_per_min", type=int, default=None, help="Slack cost inside ε-Pareto secondary solves. 0 means optimize secondary objective at the floor.")
    p.add_argument("--random_seed", type=int, default=None)
    p.add_argument("--no_pareto", action="store_true", help="Disable Pareto grid and only solve utilization-first.")
    return p


def config_from_args(args: argparse.Namespace) -> Layer2Config:
    cfg = Layer2Config()
    yaml_cfg = load_config(args.config)
    if args.config:
        LOG.info("Config loaded from %s", args.config)
    cfg = apply_config_defaults(cfg, yaml_cfg)

    overrides = {
        "pre_layer_result": args.pre_layer_result,
        "layer1_dir": args.layer1_dir,
        "blocks_json": args.blocks_json,
        "providers_json": args.providers_json,
        "output_dir": args.output_dir,
        "alpha": args.alpha,
        "time_limit_s": args.time_limit_s,
        "workers": args.workers,
        "max_slots": args.max_slots,
        "candidate_top_per_day": args.candidate_top_per_day,
        "pareto_grid": args.pareto_grid,
        "noncurrent_penalty_per_min": args.noncurrent_penalty_per_min,
        "service_line_mismatch_penalty_per_min": args.service_line_mismatch_penalty_per_min,
        "stability_fit_penalty_per_min": args.stability_fit_penalty_per_min,
        "open_slot_bonus_per_min": args.open_slot_bonus_per_min,
        "pareto_candidate_top_per_day": args.pareto_candidate_top_per_day,
        "pareto_min_utilization_fraction": args.pareto_min_utilization_fraction,
        "pareto_slack_cost_per_min": args.pareto_slack_cost_per_min,
        "random_seed": args.random_seed,
    }
    data = asdict(cfg)
    for k, v in overrides.items():
        if v is not None:
            data[k] = v
    if args.no_pareto:
        data["run_pareto"] = False
    cfg = Layer2Config(**data)

    LOG.info("Using Google OR-Tools min-cost-flow optimizer when available")
    LOG.info(
        "Effective alpha=%.3f max_slots=%s candidate_top_per_day=%s pareto_grid=%s output_dir=%s secondary_penalties=(continuity=%s, preference=%s, stability=%s) epsilon_min_fraction=%.3f",
        cfg.alpha, cfg.max_slots, cfg.candidate_top_per_day, cfg.pareto_grid, cfg.output_dir,
        cfg.noncurrent_penalty_per_min, cfg.service_line_mismatch_penalty_per_min, cfg.stability_fit_penalty_per_min,
        cfg.pareto_min_utilization_fraction,
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    result = run_layer2(cfg)
    return 0 if result.get("status") in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())

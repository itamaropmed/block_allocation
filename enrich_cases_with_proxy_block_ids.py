#!/usr/bin/env python3
"""
Enrich cases by filling BOTH:
  1. provider_id
  2. block_historical_id

Output structure:
{
  "case_id": "...",
  "provider_id": "...",
  "turnover_time": 15,
  "block_case_minutes": 330,
  "day_of_week": "Thursday",
  "block_historical_id": "..."
}

Important:
- This uses REAL provider_id and REAL block_historical_id values from geisinger-users_blocks.json.
- It does not leave provider_id/block_historical_id empty unless the blocks file has no usable blocks at all.
- For cases with existing provider_id:
    first tries same provider + same weekday block.
- For cases with blank provider_id:
    assigns to a real block on the same weekday, then fills provider_id from that blockholder.
- Because cases have no real date/start/block id, this is still PROXY linkage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DAY_TO_INT = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "items", "records", "cases", "blocks", "providers", "results"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Could not read JSON list from: {path}")


def save_json_list(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def clean_provider_id(x: Any) -> str:
    if x is None:
        return ""

    s = str(x).strip()

    if s.lower() in {"none", "nan", "null"}:
        return ""

    return s


def normalize_day(x: Any) -> int | None:
    if x is None:
        return None

    s = str(x).strip()

    if not s:
        return None

    try:
        v = int(float(s))
        if 0 <= v <= 6:
            return v
        if 1 <= v <= 7:
            return v - 1
    except ValueError:
        pass

    return DAY_TO_INT.get(s.lower())


def prepare_cases(cases_raw: list[dict[str, Any]]) -> pd.DataFrame:
    cases = pd.DataFrame(cases_raw).copy()

    required_cols = [
        "case_id",
        "provider_id",
        "turnover_time",
        "block_case_minutes",
        "day_of_week",
    ]

    missing = [c for c in required_cols if c not in cases.columns]
    if missing:
        raise ValueError(f"Cases JSON missing required columns: {missing}")

    cases["_row_id"] = range(len(cases))

    cases["case_id"] = cases["case_id"].fillna("").astype(str)
    cases["provider_id"] = cases["provider_id"].fillna("").astype(str)
    cases["provider_id_clean"] = cases["provider_id"].map(clean_provider_id)

    cases["day_int"] = cases["day_of_week"].map(normalize_day)

    cases["turnover_time_num"] = pd.to_numeric(
        cases["turnover_time"],
        errors="coerce",
    ).fillna(0.0)

    cases["block_case_minutes_num"] = pd.to_numeric(
        cases["block_case_minutes"],
        errors="coerce",
    ).fillna(0.0)

    cases["total_minutes"] = cases["turnover_time_num"] + cases["block_case_minutes_num"]

    return cases


def prepare_blocks(blocks_raw: list[dict[str, Any]]) -> pd.DataFrame:
    blocks = pd.json_normalize(blocks_raw).copy()

    required_cols = [
        "block_historical_id",
        "current_blockholder.provider_id",
        "occurrence.start",
        "occurrence.end",
    ]

    missing = [c for c in required_cols if c not in blocks.columns]
    if missing:
        raise ValueError(f"Blocks JSON missing required columns: {missing}")

    blocks["block_historical_id"] = blocks["block_historical_id"].fillna("").astype(str)
    blocks["provider_id"] = blocks["current_blockholder.provider_id"].map(clean_provider_id)

    if "current_blockholder.name" in blocks.columns:
        blocks["provider_name"] = blocks["current_blockholder.name"].fillna("").astype(str)
    else:
        blocks["provider_name"] = ""

    if "site" in blocks.columns:
        blocks["site"] = blocks["site"].fillna("").astype(str)
    else:
        blocks["site"] = ""

    if "room.type" in blocks.columns:
        blocks["room_type"] = blocks["room.type"].fillna("").astype(str)
    else:
        blocks["room_type"] = ""

    blocks["start_dt"] = pd.to_datetime(
        blocks["occurrence.start"],
        errors="coerce",
        utc=True,
    )

    blocks["end_dt"] = pd.to_datetime(
        blocks["occurrence.end"],
        errors="coerce",
        utc=True,
    )

    blocks["duration_min"] = (
        blocks["end_dt"] - blocks["start_dt"]
    ).dt.total_seconds() / 60.0

    blocks["day_int"] = blocks["start_dt"].dt.weekday

    usable = blocks[
        blocks["block_historical_id"].ne("")
        & blocks["provider_id"].ne("")
        & blocks["start_dt"].notna()
        & blocks["end_dt"].notna()
        & blocks["duration_min"].gt(0)
        & blocks["day_int"].notna()
    ].copy()

    usable["day_int"] = usable["day_int"].astype(int)

    # Prefer real surgeon/provider blocks first, but still allow SVC/GRP blocks.
    usable["is_service_or_group"] = usable["provider_id"].str.startswith(("SVC-", "GRP-"))

    usable = usable.sort_values(
        [
            "day_int",
            "is_service_or_group",
            "start_dt",
            "provider_id",
            "block_historical_id",
        ],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    return usable


def initialize_block_state(blocks: pd.DataFrame) -> pd.DataFrame:
    state = blocks.copy().reset_index(drop=True)

    state["remaining_min"] = state["duration_min"].astype(float)
    state["assigned_total_minutes"] = 0.0
    state["assigned_case_minutes"] = 0.0
    state["assigned_turnover_minutes"] = 0.0
    state["assigned_cases"] = 0

    return state


def choose_block(
    block_state: pd.DataFrame,
    needed_minutes: float,
    day_int: int | None,
    provider_id: str,
) -> tuple[int, str, float]:
    """
    Return:
      chosen block index,
      match method,
      confidence

    Matching priority:
    1. same provider + same weekday
    2. same provider any weekday
    3. same weekday any provider
    4. any block
    """

    provider_id = clean_provider_id(provider_id)

    candidate_sets: list[tuple[pd.DataFrame, str, float]] = []

    if provider_id and day_int is not None:
        candidate_sets.append((
            block_state[
                block_state["provider_id"].eq(provider_id)
                & block_state["day_int"].eq(day_int)
            ].copy(),
            "same_provider_same_day",
            0.90,
        ))

    if provider_id:
        candidate_sets.append((
            block_state[
                block_state["provider_id"].eq(provider_id)
            ].copy(),
            "same_provider_any_day",
            0.65,
        ))

    if day_int is not None:
        candidate_sets.append((
            block_state[
                block_state["day_int"].eq(day_int)
            ].copy(),
            "same_day_any_provider",
            0.35,
        ))

    candidate_sets.append((
        block_state.copy(),
        "any_block_fallback",
        0.10,
    ))

    for candidates, method, confidence in candidate_sets:
        if candidates.empty:
            continue

        feasible = candidates[candidates["remaining_min"] >= needed_minutes].copy()

        if not feasible.empty:
            feasible["leftover_after"] = feasible["remaining_min"] - needed_minutes
            chosen_idx = feasible.sort_values(
                [
                    "leftover_after",
                    "is_service_or_group",
                    "start_dt",
                    "provider_id",
                    "block_historical_id",
                ],
                ascending=[True, True, True, True, True],
            ).index[0]

            return int(chosen_idx), method + "_capacity_fit", confidence

        # If nothing fits exactly, still choose the best available real block.
        chosen_idx = candidates.sort_values(
            [
                "remaining_min",
                "is_service_or_group",
                "start_dt",
                "provider_id",
                "block_historical_id",
            ],
            ascending=[False, True, True, True, True],
        ).index[0]

        return int(chosen_idx), method + "_over_capacity", max(confidence - 0.20, 0.05)

    raise RuntimeError("No usable blocks found.")


def enrich_cases_with_real_block_values(
    cases: pd.DataFrame,
    blocks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill provider_id and block_historical_id using real block rows.

    For blank provider cases:
      provider_id is filled from chosen block's provider_id.

    For known provider cases:
      provider_id is preserved, unless there is no matching block for that provider
      and fallback is needed. In fallback, we still keep original provider_id by default,
      but block_historical_id comes from the chosen fallback block.

    To force provider_id to always match the chosen block, set:
      FORCE_PROVIDER_TO_CHOSEN_BLOCK = True
    """

    FORCE_PROVIDER_TO_CHOSEN_BLOCK = True

    enriched = cases.copy()
    enriched["block_historical_id"] = ""

    block_state = initialize_block_state(blocks)

    audit_rows: list[dict[str, Any]] = []

    # Sort biggest cases first so long cases get real capacity first.
    sorted_cases = enriched.sort_values(
        ["total_minutes", "case_id"],
        ascending=[False, True],
    )

    for _, case_row in sorted_cases.iterrows():
        row_id = int(case_row["_row_id"])

        original_provider = clean_provider_id(case_row["provider_id"])
        day_int = None if pd.isna(case_row["day_int"]) else int(case_row["day_int"])
        needed = float(case_row["total_minutes"])

        chosen_idx, method, confidence = choose_block(
            block_state=block_state,
            needed_minutes=needed,
            day_int=day_int,
            provider_id=original_provider,
        )

        chosen = block_state.loc[chosen_idx]

        chosen_provider_id = str(chosen["provider_id"])
        chosen_block_id = str(chosen["block_historical_id"])

        if FORCE_PROVIDER_TO_CHOSEN_BLOCK:
            final_provider_id = chosen_provider_id
        else:
            final_provider_id = original_provider if original_provider else chosen_provider_id

        enriched.loc[enriched["_row_id"].eq(row_id), "provider_id"] = final_provider_id
        enriched.loc[enriched["_row_id"].eq(row_id), "provider_id_clean"] = final_provider_id
        enriched.loc[enriched["_row_id"].eq(row_id), "block_historical_id"] = chosen_block_id

        block_state.loc[chosen_idx, "remaining_min"] -= needed
        block_state.loc[chosen_idx, "assigned_total_minutes"] += needed
        block_state.loc[chosen_idx, "assigned_case_minutes"] += float(case_row["block_case_minutes_num"])
        block_state.loc[chosen_idx, "assigned_turnover_minutes"] += float(case_row["turnover_time_num"])
        block_state.loc[chosen_idx, "assigned_cases"] += 1

        audit_rows.append({
            "case_id": case_row["case_id"],
            "original_provider_id": original_provider,
            "filled_provider_id": final_provider_id,
            "chosen_block_provider_id": chosen_provider_id,
            "block_historical_id": chosen_block_id,
            "day_of_week": case_row["day_of_week"],
            "case_day_int": day_int,
            "block_day_int": int(chosen["day_int"]),
            "turnover_time": case_row["turnover_time"],
            "block_case_minutes": case_row["block_case_minutes"],
            "total_minutes": needed,
            "match_method": method,
            "proxy_confidence": confidence,
            "block_remaining_after_assignment": float(block_state.loc[chosen_idx, "remaining_min"]),
            "provider_name": chosen.get("provider_name", ""),
            "site": chosen.get("site", ""),
            "room_type": chosen.get("room_type", ""),
        })

    audit = pd.DataFrame(audit_rows)

    return enriched, audit


def build_output_records(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    out = enriched[
        [
            "case_id",
            "provider_id",
            "turnover_time",
            "block_case_minutes",
            "day_of_week",
            "block_historical_id",
        ]
    ].copy()

    out["case_id"] = out["case_id"].fillna("").astype(str)
    out["provider_id"] = out["provider_id"].fillna("").astype(str)
    out["day_of_week"] = out["day_of_week"].fillna("").astype(str)
    out["block_historical_id"] = out["block_historical_id"].fillna("").astype(str)

    out["turnover_time"] = pd.to_numeric(out["turnover_time"], errors="coerce").fillna(0)
    out["block_case_minutes"] = pd.to_numeric(out["block_case_minutes"], errors="coerce").fillna(0)

    return out.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cases",
        default="data/raw/geisinger-users_cases.json",
        help="Original cases JSON.",
    )

    parser.add_argument(
        "--blocks",
        default="data/raw/geisinger-users_blocks.json",
        help="Blocks JSON containing real provider_id and block_historical_id.",
    )

    parser.add_argument(
        "--out-json",
        default="data/raw/geisinger-users_cases_enriched_real_ids.json",
        help="Output cases JSON with provider_id and block_historical_id filled.",
    )

    parser.add_argument(
        "--out-audit",
        default="outputs/data_linkage/cases_real_id_fill_audit.csv",
        help="Audit CSV explaining how each case was filled.",
    )

    args = parser.parse_args()

    print("Loading JSON files...")
    cases_raw = load_json_list(args.cases)
    blocks_raw = load_json_list(args.blocks)

    print(f"Loaded cases : {len(cases_raw):,}")
    print(f"Loaded blocks: {len(blocks_raw):,}")

    print("Preparing data...")
    cases = prepare_cases(cases_raw)
    blocks = prepare_blocks(blocks_raw)

    print(f"Usable real blocks: {len(blocks):,}")

    original_blank_provider = int(cases["provider_id_clean"].eq("").sum())
    print(f"Original blank provider_id rows: {original_blank_provider:,}")

    print("Filling provider_id and block_historical_id with real values from blocks...")
    enriched, audit = enrich_cases_with_real_block_values(cases, blocks)

    records = build_output_records(enriched)

    save_json_list(records, args.out_json)

    audit_path = Path(args.out_audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    final_blank_provider = sum(1 for r in records if not str(r["provider_id"]).strip())
    final_blank_block = sum(1 for r in records if not str(r["block_historical_id"]).strip())

    print("\nDONE")
    print(f"Saved enriched JSON: {args.out_json}")
    print(f"Saved audit CSV    : {args.out_audit}")

    print("\nSummary:")
    print(f"  Total cases                  : {len(records):,}")
    print(f"  Original blank provider_id   : {original_blank_provider:,}")
    print(f"  Final blank provider_id      : {final_blank_provider:,}")
    print(f"  Final blank block_id         : {final_blank_block:,}")

    print("\nMatch methods:")
    print(audit["match_method"].value_counts(dropna=False).to_string())

    print("\nFirst output row:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
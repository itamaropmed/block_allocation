"""
Synthetic Geisinger data generator
====================================
Generates three JSON files in the exact Geisinger format but with:

  1. No provider ID mismatches  — all block holders exist in providers.json
  2. site field = physical location code  — matches providers' exclusive_sites
     so the optimizer's site-eligibility filter works correctly
  3. Designed 4-week rotation  — BIC will select T* = 4

The design for T* = 4:
  - 14 weeks, week_index 0–13
  - phase = week_index % 4  →  phases {0,1,2,3}
  - Each cell (room × day × start_time) has a fixed dominant provider per phase
  - ~3% random exceptions added for realism
  - D(T=4) ≈ 0.03 × N_BLOCKS  (very small)
  - D(T=1) ≈ 0.70 × N_BLOCKS  (large — each provider appears only ~25% of weeks)
  - BIC gap is >> 3 × C × log(V) / log(H×C×V) ≈ 700  →  T*=4 wins clearly

Output (written to --out_dir):
  geisinger-users_blocks.json    (N_BLOCKS ≈ 8 187 blocks)
  geisinger-users_providers.json (N_PROVIDERS = 336 providers)
  geisinger-users_cases.json     (N_CASES ≈ 36 051 cases)

Usage:
    python generate_synthetic_data.py \\
        --blocks   geisinger-users_blocks.json \\
        --providers geisinger-users_providers.json \\
        --cases    geisinger-users_cases.json \\
        --out_dir  synthetic_data/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Design constants
# ─────────────────────────────────────────────────────────────────────────────
N_PROVIDERS   = 336
N_BLOCKS      = 8_187
N_CASES       = 36_051
T_DESIGN      = 4          # target rotation period (BIC should select this)
H_WEEKS       = 14         # Q1 2026: Jan 5 → Apr 5 (14 full Mon-weeks)
EXCEPTION_RATE = 0.03      # fraction of blocks that deviate from dominant
OPEN_RATE      = 0.04      # fraction of blocks that are open (no holder)
SEED           = 42

# Q1 2026: Monday 5 January 2026 is week 0
WEEK0_MONDAY = datetime(2026, 1, 5, tzinfo=timezone.utc)

DAYS_OF_WEEK  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday"]
WORKDAYS      = DAYS_OF_WEEK[:5]   # Mon-Fri (Saturday/Sunday very rare — added below)

# ─────────────────────────────────────────────────────────────────────────────
# Room → physical-site mapping  (matches providers' exclusive_sites exactly)
# ─────────────────────────────────────────────────────────────────────────────
def room_to_phys_site(room: str) -> str:
    r = room.upper()
    if "ENDO" in r:
        for prefix, site in [
            ("OSSC",  "ENDOSCOPY OSSC"),
            ("OSW",   "ENDOSCOPY OSW"),
            ("GWV",   "ENDOSCOPY GWV"),
            ("GCMC",  "ENDOSCOPY GCMC"),
            ("GECL",  "ENDOSCOPY GECL"),
            ("GSACH", "ENDOSCOPY GSACH"),
            ("GMC",   "ENDOSCOPY GMC"),
        ]:
            if r.startswith(prefix):
                return site
        return "ENDOSCOPY GMC"
    for prefix, site in [
        ("GCMC",  "OR GCMC"),
        ("GWV",   "OR GWV"),
        ("GMCM",  "OR GMCM"),
        ("GMC",   "OR GMC"),
        ("OSSC",  "OR OSSC"),
        ("OSCP",  "OR OSCP"),
        ("OSW",   "OR OSW"),
        ("OSHP",  "OR OSHP"),
        ("GLH",   "OR GLH"),
        ("GSWB",  "OR GSWB"),
        ("GSACH", "OR GSACH"),
        ("GBH",   "OR GBH"),
        ("GJSH",  "OR GJSH"),
    ]:
        if r.startswith(prefix):
            return site
    return "OR GMC"


# ─────────────────────────────────────────────────────────────────────────────
# Load original data to extract structural distributions
# ─────────────────────────────────────────────────────────────────────────────

def load_originals(
    blocks_path: str,
    providers_path: str,
    cases_path: str,
) -> Tuple[list, list, list]:
    with open(blocks_path, encoding="utf-8") as f:
        blocks = json.load(f)
    with open(providers_path, encoding="utf-8") as f:
        providers = json.load(f)
    with open(cases_path, encoding="utf-8") as f:
        cases = json.load(f)
    return blocks, providers, cases


def extract_room_pool(orig_blocks: list) -> List[Tuple[str, str, int, int]]:
    """
    Return (room_type, phys_site, start_min, end_min) tuples with original
    frequency weights, for realistic sampling.
    """
    from collections import Counter
    counter = Counter()
    for b in orig_blocks:
        room  = b["room"]["type"]
        site  = room_to_phys_site(room)
        start = datetime.fromisoformat(b["occurrence"]["start"].replace("Z", "+00:00"))
        end   = datetime.fromisoformat(b["occurrence"]["end"].replace("Z", "+00:00"))
        s_min = start.hour * 60 + start.minute
        e_min = end.hour   * 60 + end.minute
        if e_min <= s_min:
            e_min = s_min + 480
        counter[(room, site, s_min, e_min)] += 1
    return list(counter.items())   # [((room, site, s, e), count), ...]


def extract_duration_weights(orig_blocks: list) -> List[Tuple[int, int, float]]:
    """
    Return [(start_min, dur_min, weight)] for realistic block durations.
    """
    from collections import Counter
    c = Counter()
    for b in orig_blocks:
        start = datetime.fromisoformat(b["occurrence"]["start"].replace("Z", "+00:00"))
        end   = datetime.fromisoformat(b["occurrence"]["end"].replace("Z", "+00:00"))
        s_min = start.hour * 60 + start.minute
        dur   = int((end - start).total_seconds() // 60)
        c[(s_min, dur)] += 1
    return [(s, d, w) for (s, d), w in c.items()]


def extract_case_minutes(orig_cases: list) -> List[int]:
    return [c["block_case_minutes"] for c in orig_cases if c.get("block_case_minutes")]


def extract_provider_names(orig_providers: list) -> List[str]:
    return [p["name"] for p in orig_providers]


def extract_exclusive_site_combos(orig_providers: list) -> List[Tuple[str, ...]]:
    """Return list of exclusive_site tuples from original providers."""
    combos = []
    for p in orig_providers:
        s = tuple(sorted(p.get("exclusive_sites", [])))
        if s:
            combos.append(s)
    return combos


# ─────────────────────────────────────────────────────────────────────────────
# Provider generation
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_NAMES = [
    "General Surgery", "Orthopaedics", "Cardiac Surgery", "Neurosurgery",
    "Gynecology", "Otolaryngology", "Urology", "Ophthalmology",
    "Vascular Surgery", "Thoracic Surgery", "Transplant Surgery",
    "Pediatric Surgery", "Plastic Surgery", "Gastroenterology",
    "Colorectal Surgery", "Bariatric Surgery", "Hand Surgery",
    "Spine Surgery", "Joint Replacement", "Sports Medicine",
    "Trauma Surgery", "Oral Surgery", "Podiatry",
    "Interventional Radiology", "Pain Management",
]

LAST_NAMES = [
    "SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER",
    "DAVIS", "MARTINEZ", "HERNANDEZ", "WILSON", "ANDERSON", "TAYLOR",
    "THOMAS", "MOORE", "JACKSON", "MARTIN", "LEE", "PEREZ", "THOMPSON",
    "WHITE", "HARRIS", "SANCHEZ", "CLARK", "LEWIS", "ROBINSON", "WALKER",
    "YOUNG", "ALLEN", "KING", "WRIGHT", "SCOTT", "TORRES", "NGUYEN",
    "HILL", "FLORES", "GREEN", "ADAMS", "NELSON", "BAKER", "HALL",
    "RIVERA", "CAMPBELL", "MITCHELL", "CARTER", "ROBERTS", "CHEN",
    "PATEL", "KUMAR", "SHARMA", "GUPTA", "COHEN", "LEVY", "KIM",
]

FIRST_NAMES = [
    "JAMES", "JOHN", "ROBERT", "MICHAEL", "WILLIAM", "DAVID", "RICHARD",
    "JOSEPH", "THOMAS", "CHARLES", "SARAH", "KAREN", "LISA", "NANCY",
    "BETTY", "MARGARET", "SANDRA", "ASHLEY", "DOROTHY", "KIMBERLY",
    "EMILY", "DONNA", "CAROL", "MICHELLE", "AMANDA", "MELISSA", "DEBORAH",
    "STEPHANIE", "REBECCA", "SHARON", "LAURA", "CYNTHIA", "KATHLEEN",
    "AMY", "ANGELA", "SHIRLEY", "ANNA", "BRENDA", "PAMELA", "EMMA",
    "ELIZABETH", "ALICE", "HELEN", "JESSICA", "MARY", "LINDA",
]

MIDDLE_INITIALS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def generate_providers(
    n: int,
    orig_site_combos: List[Tuple[str, ...]],
    rng: random.Random,
) -> List[dict]:
    """
    Generate N providers with a realistic mix of service names, individual
    names, and exclusive_sites drawn from the original distribution.
    """
    # Shuffle and cycle original site combos for realistic distribution
    site_pool = orig_site_combos.copy()
    rng.shuffle(site_pool)
    # Extend if needed
    while len(site_pool) < n:
        site_pool.extend(orig_site_combos)

    providers = []
    svc_name_idx = 0
    used_names   = set()

    for i in range(n):
        pid = f"PRV-{i+1:04d}"

        # Alternate between service names and individual names
        if i % 3 == 0 and svc_name_idx < len(SERVICE_NAMES):
            name = SERVICE_NAMES[svc_name_idx]
            svc_name_idx += 1
        else:
            # Individual name — keep unique
            for _ in range(100):
                last  = rng.choice(LAST_NAMES)
                first = rng.choice(FIRST_NAMES)
                mid   = rng.choice(MIDDLE_INITIALS)
                name  = f"{last}, {first} {mid}"
                if name not in used_names:
                    used_names.add(name)
                    break

        excl_sites = list(site_pool[i % len(site_pool)])

        providers.append({
            "provider_id":    pid,
            "name":           name,
            "service_line":   [],
            "exclusive_sites": excl_sites,
        })

    return providers


# ─────────────────────────────────────────────────────────────────────────────
# Cell and rotation design
# ─────────────────────────────────────────────────────────────────────────────

def build_site_to_providers(providers: List[dict]) -> Dict[str, List[str]]:
    """Map each physical site to the list of provider IDs eligible for it."""
    mapping: Dict[str, List[str]] = defaultdict(list)
    for p in providers:
        for site in p["exclusive_sites"]:
            mapping[site].append(p["provider_id"])
    return dict(mapping)


def design_cells_and_rotation(
    orig_rooms: List[Tuple],          # [((room, site, s_min, e_min), count), ...]
    site_to_providers: Dict[str, List[str]],
    n_blocks_target: int,
    h_weeks: int,
    t_design: int,
    rng: random.Random,
    exception_rate: float = 0.03,
    open_rate: float = 0.04,
    workday_weights: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """
    Design the full block schedule so BIC selects T* = t_design.

    Returns a list of dicts, one per block occurrence, with fields:
      block_historical_id, week_index, date, room_type, phys_site,
      start_min, end_min, day_of_week, phase, provider_id, is_open,
      manual_early_release
    """
    if workday_weights is None:
        workday_weights = {
            "Monday": 1.0, "Tuesday": 1.0, "Wednesday": 0.95,
            "Thursday": 1.0, "Friday": 1.0,
            "Saturday": 0.018, "Sunday": 0.008,
        }

    # ── Step 1: determine how many cells we need ──────────────────────────────
    # Total occurrences ≈ n_blocks_target
    # Each cell appears in all h_weeks (so total = n_cells × h_weeks)
    n_cells = math.ceil(n_blocks_target / h_weeks)

    # ── Step 2: sample cells from the original room pool ──────────────────────
    # Weighted sample of (room, phys_site, start_min, end_min)
    room_keys     = [item[0] for item in orig_rooms]
    room_weights  = [item[1] for item in orig_rooms]
    total_rw      = sum(room_weights)
    room_probs    = [w / total_rw for w in room_weights]

    # Sample day-of-week with original-like weights
    days_ordered = list(workday_weights.keys())
    day_weights  = [workday_weights[d] for d in days_ordered]
    dw_total     = sum(day_weights)
    day_probs    = [w / dw_total for w in day_weights]

    cells = []          # (room, phys_site, start_min, end_min, day_of_week)
    used_cell_keys = set()

    for _ in range(n_cells * 5):    # over-sample, deduplicate
        if len(cells) >= n_cells:
            break
        # Pick a room/time
        idx     = rng.choices(range(len(room_keys)), weights=room_probs)[0]
        room, phys_site, s_min, e_min = room_keys[idx]
        # Pick a day
        day = rng.choices(days_ordered, weights=day_probs)[0]
        key = (room, phys_site, s_min, e_min, day)
        if key not in used_cell_keys:
            used_cell_keys.add(key)
            cells.append(key)

    # Top-up if needed with duplicated cells (different days on same room/time)
    extra_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    while len(cells) < n_cells:
        idx  = rng.randrange(len(cells))
        room, phys_site, s_min, e_min, _ = cells[idx]
        day  = rng.choice(extra_days)
        key  = (room, phys_site, s_min, e_min, day)
        if key not in used_cell_keys:
            used_cell_keys.add(key)
            cells.append(key)

    # ── Step 3: assign t_design providers per cell (one per phase) ────────────
    # For each cell, pick t_design providers from the matching site pool.
    cell_phase_providers: List[List[Optional[str]]] = []

    for cell in cells:
        room, phys_site, s_min, e_min, day = cell
        pool = site_to_providers.get(phys_site, [])
        if not pool:
            # Fallback: any provider
            pool = [p for ps in site_to_providers.values() for p in ps]
        if len(pool) < t_design:
            pool = (pool * math.ceil(t_design / len(pool)))[:t_design]
        # Sample t_design distinct providers
        phase_provs = rng.sample(pool, min(t_design, len(pool)))
        while len(phase_provs) < t_design:
            phase_provs.append(rng.choice(pool))
        cell_phase_providers.append(phase_provs)

    # ── Step 4: generate one block per (cell, week) ───────────────────────────
    blocks = []
    block_counter = 0

    for week_idx in range(h_weeks):
        phase       = week_idx % t_design
        week_monday = WEEK0_MONDAY + timedelta(weeks=week_idx)

        for cell_idx, (room, phys_site, s_min, e_min, day) in enumerate(cells):

            # Date for this block
            dow_to_delta = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2,
                "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
            }
            block_date = week_monday + timedelta(days=dow_to_delta[day])

            # Provider assignment
            dominant_prov = cell_phase_providers[cell_idx][phase]

            # Exception: with exception_rate probability, swap for another provider
            is_exception = (rng.random() < exception_rate)
            if is_exception:
                pool = site_to_providers.get(phys_site, [dominant_prov])
                other = [p for p in pool if p != dominant_prov]
                if other:
                    actual_prov = rng.choice(other)
                else:
                    actual_prov = dominant_prov
                    is_exception = False
            else:
                actual_prov = dominant_prov

            # Open block: no holder
            is_open = (rng.random() < open_rate)
            if is_open:
                actual_prov = None

            # Build occurrence datetimes
            occ_start = block_date.replace(
                hour=s_min // 60, minute=s_min % 60, second=0, microsecond=0
            )
            occ_end   = block_date.replace(
                hour=e_min // 60, minute=e_min % 60, second=0, microsecond=0
            )
            if occ_end <= occ_start:
                occ_end = occ_start + timedelta(hours=8)

            block_counter += 1
            bid = f"SYN-{block_counter:05d}_{room.replace(' ','_')}_{block_date.date()}_{s_min//60}_{e_min//60}"

            blocks.append({
                "_cell_idx":    cell_idx,
                "_week_idx":    week_idx,
                "_phase":       phase,
                "_is_exception": is_exception,
                "block_historical_id": bid,
                "is_open":      is_open,
                "site":         phys_site,          # ← physical site (matches exclusive_sites)
                "occurrence": {
                    "start": occ_start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "end":   occ_end.strftime(  "%Y-%m-%dT%H:%M:%S+00:00"),
                },
                "current_blockholder": (
                    {"provider_id": actual_prov,
                     "name":        f"Provider {actual_prov}"}
                    if actual_prov else None
                ),
                "room":    {"type": room},
                "manual_early_release": None,
            })

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Case generation
# ─────────────────────────────────────────────────────────────────────────────

def lognormal_sample(rng: random.Random, target_median: float, sigma: float) -> float:
    """Sample from a log-normal with the given median and shape."""
    mu  = math.log(target_median)
    val = rng.lognormvariate(mu, sigma)
    return max(5.0, min(val, 720.0))


def generate_cases(
    blocks: List[dict],
    n_cases_target: int,
    orig_case_minutes: List[int],
    provider_ids: List[str],
    rng: random.Random,
) -> List[dict]:
    """
    Generate cases with realistic distributions.
    - block_case_minutes: log-normal matching original (median≈68, mean≈95)
    - turnover_time: always 15 (matching original)
    - provider_id: drawn from block holders (not empty, unlike original where 40% blank)
      We keep ~40% without provider_id to match original behaviour,
      but only use valid provider IDs when present.
    """
    # Build pool of (provider_id, day_of_week) from non-open blocks
    prov_day_pool = []
    for b in blocks:
        if b["current_blockholder"] and b["current_blockholder"].get("provider_id"):
            pid = b["current_blockholder"]["provider_id"]
            occ_start = datetime.fromisoformat(b["occurrence"]["start"])
            dow = occ_start.strftime("%A")
            prov_day_pool.append((pid, dow))

    if not prov_day_pool:
        prov_day_pool = [(pid, "Monday") for pid in provider_ids]

    # Estimate log-normal parameters from original data
    if orig_case_minutes:
        sorted_m = sorted(orig_case_minutes)
        target_median = float(sorted_m[len(sorted_m) // 2])
    else:
        target_median = 68.0
    sigma = 0.9  # matches observed spread in Geisinger

    # Fraction of cases with blank provider_id (matching original ≈ 40%)
    blank_rate = 0.40

    cases = []
    for i in range(n_cases_target):
        case_id = str(1_000_000 + i)

        # Case minutes
        raw_minutes = lognormal_sample(rng, target_median, sigma)
        block_case_minutes = int(round(raw_minutes / 5) * 5)

        # Provider
        if rng.random() < blank_rate:
            provider_id = ""
        else:
            pid, dow = rng.choice(prov_day_pool)
            provider_id = pid

        # Day of week
        _, dow = rng.choice(prov_day_pool)

        cases.append({
            "case_id":             case_id,
            "provider_id":         provider_id,
            "turnover_time":       15,
            "block_case_minutes":  block_case_minutes,
            "day_of_week":         dow,
        })

    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Strip internal fields before saving
# ─────────────────────────────────────────────────────────────────────────────

def clean_blocks_for_output(blocks: List[dict]) -> List[dict]:
    internal = {"_cell_idx", "_week_idx", "_phase", "_is_exception"}
    return [{k: v for k, v in b.items() if k not in internal} for b in blocks]


# ─────────────────────────────────────────────────────────────────────────────
# BIC sanity check
# ─────────────────────────────────────────────────────────────────────────────

def quick_bic_check(blocks: List[dict], t_design: int, h_weeks: int) -> None:
    """Print BIC scores for T=1..6 to verify T* = t_design is selected."""
    from collections import Counter as C

    # Build cell_id and week_index
    week0 = WEEK0_MONDAY
    rows = []
    for b in blocks:
        room = b["room"]["type"]
        s    = datetime.fromisoformat(b["occurrence"]["start"])
        s_min = s.hour * 60 + s.minute
        dow   = s.weekday()
        cell  = f"{room}|dow={dow}|s={s_min}"
        wi    = (s.date() - week0.date()).days // 7
        pid   = (b["current_blockholder"]["provider_id"]
                 if b["current_blockholder"] else "OPEN")
        rows.append((cell, wi, pid))

    H = h_weeks
    cells = set(r[0] for r in rows)
    pids  = set(r[2] for r in rows)
    C_n   = len(cells)
    V     = len(pids) + 2

    print("\n--- BIC Sanity Check ---")
    print(f"H={H}  C={C_n}  V={V}  total blocks={len(rows)}")
    best_t, best_bic = 1, float("inf")
    for T in range(1, min(8, H + 1)):
        phase_map = defaultdict(list)
        for cell, wi, pid in rows:
            phase = wi % T
            phase_map[(cell, phase)].append(pid)
        deviations = 0
        for vals in phase_map.values():
            cnt = C(vals)
            mode_count = max(cnt.values())
            deviations += len(vals) - mode_count
        bic = T * C_n * math.log(V) + deviations * math.log(max(2, H * C_n * V))
        marker = " ← selected T*" if bic < best_bic else ""
        if bic < best_bic:
            best_bic, best_t = bic, T
            marker = " ← selected T*"
        else:
            marker = ""
        print(f"  T={T:2d}  deviations={deviations:6d}  BIC={bic:12.1f}{marker}")
    print(f"  → BIC selects T* = {best_t}  (target = {t_design})")
    if best_t != t_design:
        print(f"  WARNING: T*={best_t} ≠ target {t_design} — "
              "consider lowering exception_rate or increasing H.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Default paths: same folder as this script
    _here = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(
        description="Generate synthetic Geisinger-format data with T*=4 rotation"
    )
    parser.add_argument("--blocks",
                        default=str(_here / "geisinger-users_blocks.json"),
                        help="Path to original geisinger-users_blocks.json "
                             "(default: same folder as this script)")
    parser.add_argument("--providers",
                        default=str(_here / "geisinger-users_providers.json"),
                        help="Path to original geisinger-users_providers.json "
                             "(default: same folder as this script)")
    parser.add_argument("--cases",
                        default=str(_here / "geisinger-users_cases.json"),
                        help="Path to original geisinger-users_cases.json "
                             "(default: same folder as this script)")
    parser.add_argument("--out_dir",
                        default=str(_here / "synthetic_data"),
                        help="Output directory (default: synthetic_data/ beside this script)")
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--n_blocks",  type=int, default=N_BLOCKS)
    parser.add_argument("--n_providers", type=int, default=N_PROVIDERS)
    parser.add_argument("--n_cases",   type=int, default=N_CASES)
    parser.add_argument("--t_design",  type=int, default=T_DESIGN,
                        help="Rotation period to design into the schedule")
    parser.add_argument("--exception_rate", type=float, default=EXCEPTION_RATE)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load originals ────────────────────────────────────────────────────────
    print("Loading original datasets …")
    orig_blocks, orig_providers, orig_cases = load_originals(
        args.blocks, args.providers, args.cases
    )
    print(f"  blocks={len(orig_blocks)}  "
          f"providers={len(orig_providers)}  cases={len(orig_cases)}")

    # Extract structural distributions from originals
    orig_rooms        = extract_room_pool(orig_blocks)
    orig_site_combos  = extract_exclusive_site_combos(orig_providers)
    orig_case_minutes = extract_case_minutes(orig_cases)

    # Day-of-week weights from original blocks
    dow_counts: Counter = Counter()
    for b in orig_blocks:
        s = datetime.fromisoformat(b["occurrence"]["start"].replace("Z", "+00:00"))
        dow_counts[s.strftime("%A")] += 1
    total_dow = sum(dow_counts.values())
    dow_weights = {d: dow_counts[d] / total_dow for d in DAYS_OF_WEEK}

    # ── Generate providers ────────────────────────────────────────────────────
    print(f"Generating {args.n_providers} providers …")
    providers = generate_providers(args.n_providers, orig_site_combos, rng)
    site_to_prov = build_site_to_providers(providers)
    provider_ids = [p["provider_id"] for p in providers]
    print(f"  Sites with providers: {sorted(site_to_prov.keys())}")

    # ── Generate blocks (4-week rotation) ────────────────────────────────────
    print(f"Generating {args.n_blocks} blocks with T={args.t_design} rotation …")
    raw_blocks = design_cells_and_rotation(
        orig_rooms        = orig_rooms,
        site_to_providers = site_to_prov,
        n_blocks_target   = args.n_blocks,
        h_weeks           = H_WEEKS,
        t_design          = args.t_design,
        rng               = rng,
        exception_rate    = args.exception_rate,
        open_rate         = OPEN_RATE,
        workday_weights   = dow_weights,
    )
    print(f"  Generated {len(raw_blocks)} block occurrences")

    # Sanity check: BIC should select T* = args.t_design
    quick_bic_check(raw_blocks, args.t_design, H_WEEKS)

    # Validate: all block holders exist in providers
    prov_id_set   = set(provider_ids)
    block_pid_set = {
        b["current_blockholder"]["provider_id"]
        for b in raw_blocks
        if b["current_blockholder"]
    }
    missing = block_pid_set - prov_id_set
    if missing:
        print(f"  WARNING: {len(missing)} block PIDs not in providers: {missing}")
    else:
        print(f"  ✓  All {len(block_pid_set)} block holder IDs exist in providers")

    # Validate: site consistency
    site_mismatches = 0
    for b in raw_blocks:
        if not b["current_blockholder"]:
            continue
        pid  = b["current_blockholder"]["provider_id"]
        site = b["site"]
        prow = next((p for p in providers if p["provider_id"] == pid), None)
        if prow and site not in prow["exclusive_sites"]:
            site_mismatches += 1
    if site_mismatches == 0:
        print(f"  ✓  Site consistency: 0 mismatches")
    else:
        print(f"  WARNING: {site_mismatches} site mismatches")

    # ── Generate cases ────────────────────────────────────────────────────────
    print(f"Generating {args.n_cases} cases …")
    cases = generate_cases(raw_blocks, args.n_cases, orig_case_minutes, provider_ids, rng)
    has_pid = sum(1 for c in cases if c["provider_id"])
    print(f"  Cases with provider_id: {has_pid}/{len(cases)} = {100*has_pid/len(cases):.1f}%")

    # ── Write outputs ─────────────────────────────────────────────────────────
    blocks_out    = out / "geisinger-users_blocks.json"
    providers_out = out / "geisinger-users_providers.json"
    cases_out     = out / "geisinger-users_cases.json"

    clean_b = clean_blocks_for_output(raw_blocks)
    with open(blocks_out,    "w", encoding="utf-8") as f:
        json.dump(clean_b, f, indent=2)
    with open(providers_out, "w", encoding="utf-8") as f:
        json.dump(providers, f, indent=2)
    with open(cases_out,     "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print(f"\n✓  Output written to {out}/")
    print(f"   blocks.json    : {len(clean_b)} records")
    print(f"   providers.json : {len(providers)} records")
    print(f"   cases.json     : {len(cases)} records")

    # Final summary
    open_b = sum(1 for b in clean_b if b["is_open"])
    exc_b  = sum(1 for b in raw_blocks if b["_is_exception"])
    print(f"\n   Open blocks      : {open_b} ({100*open_b/len(clean_b):.1f}%)")
    print(f"   Exception blocks : {exc_b}  ({100*exc_b/len(clean_b):.1f}%)")
    print(f"   Designed T       : {args.t_design}  (BIC should select this)\n")


if __name__ == "__main__":
    main()
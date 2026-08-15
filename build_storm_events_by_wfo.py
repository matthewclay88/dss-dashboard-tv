#!/usr/bin/env python3
"""
build_storm_events_by_wfo.py

Downloads NCEI's official Storm Events "details" CSVs (one file per
year, nationwide) and fans them out into one compact JSON file per NWS
office (WFO) — e.g. storm_events/BTV.json, storm_events/OUN.json, etc.
— covering all 120+ offices in a single pass per year, since the
source file is already nationwide.

WHY THIS EXISTS
NCEI's Storm Events Database (https://www.ncei.noaa.gov/stormevents/)
is the *official*, human-certified record — richer than raw LSRs (EF
ratings, damage dollar estimates, injuries/fatalities) but only
published as one big CSV per year, not a queryable API. This script
does the "download the whole thing and filter" step once, offline, so
the warning-verification dashboard can just fetch a small pre-built
per-office JSON file instead of parsing 10+ years of nationwide data
client-side.

USAGE
    python3 build_storm_events_by_wfo.py --start-year 2008 --end-year 2025
    python3 build_storm_events_by_wfo.py --start-year 2008 --end-year 2025 --wfo BTV,OUN

Re-running is safe: already-downloaded yearly CSVs are cached in
--cache-dir and not re-fetched. Output JSON is fully regenerated each
run (cheap once the CSVs are cached locally).

WHY 2008 AS THE DEFAULT START YEAR, NOT 2000
Storm-based (polygon) warnings — the kind this dashboard verifies —
didn't exist before 1 October 2007; before that, NWS warnings were
issued for whole counties. Storm Events data further back than that
still exists, but there's nothing in the polygon-verification side of
this dashboard to cross-reference it against, so pulling it would just
add processing time for no payoff. Override with --start-year if you
want it anyway (e.g. for a different kind of historical research).

OUTPUT SCHEMA (per WFO JSON file)
{
  "wfo": "BTV",
  "generated_at": "2026-08-15T12:00:00Z",
  "source": "NCEI Storm Events Database (official/certified)",
  "years_covered": [2008, ..., 2025],
  "events": [
    {
      "event_id": "1162071",
      "episode_id": "188834",
      "event_type": "Winter Storm",
      "state": "VERMONT",
      "cz_type": "Z",
      "cz_fips": "7",
      "cz_name": "CALEDONIA",
      "begin_date_time": "03-APR-24 19:00:00",
      "end_date_time": "05-APR-24 00:00:00",
      "cz_timezone": "EST-5",
      "injuries_direct": 0, "injuries_indirect": 0,
      "deaths_direct": 0, "deaths_indirect": 0,
      "damage_property_usd": 0.0, "damage_crops_usd": 0.0,
      "source": "CoCoRaHS",
      "magnitude": null, "magnitude_type": null,
      "tor_f_scale": null, "tor_length": null, "tor_width": null,
      "begin_lat": null, "begin_lon": null, "end_lat": null, "end_lon": null,
      "episode_narrative": "...", "event_narrative": "..."
    },
    ...
  ]
}

A NOTE ON TIME ZONES (read before doing precise time-based matching)
begin_date_time/end_date_time are in LOCAL time per cz_timezone (e.g.
"EST-5" = 5 hours behind UTC) — NCEI does not clearly document whether
this shifts for daylight saving or stays fixed year-round. This script
does not attempt to resolve that ambiguity; it passes the raw strings
through as-is. If you build exact-time matching against this dashboard's
UTC-based warning times, treat the converted UTC time as approximate
(±1hr) unless/until this gets nailed down against known events.
"""
import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
INDEX_URL = BASE_URL

# NCEI renames files with a new "created" (c*) date stamp whenever the
# year's data gets revised, so the filename isn't 100% predictable from
# the year alone. We fetch the directory listing once and regex out the
# real filename for each year, rather than guessing the c-date.
FNAME_RE = re.compile(
    r'StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz'
)


def find_year_filenames(start_year, end_year):
    """Fetch the directory listing once and map year -> real filename."""
    with urllib.request.urlopen(INDEX_URL, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    found = {}
    for m in FNAME_RE.finditer(html):
        year, cdate = int(m.group(1)), m.group(2)
        if start_year <= year <= end_year:
            fname = f"StormEvents_details-ftp_v1.0_d{year}_c{cdate}.csv.gz"
            # Keep the most-recently-created version if duplicates appear
            existing = found.get(year)
            if not existing or cdate > existing[1]:
                found[year] = (fname, cdate)
    return {y: v[0] for y, v in found.items()}


def download_year(year, fname, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, fname)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"  [{year}] using cached {fname}")
        return local_path
    url = BASE_URL + fname
    print(f"  [{year}] downloading {fname} ...")
    urllib.request.urlretrieve(url, local_path)
    return local_path


def parse_damage(val):
    """'0.00K' -> 0.0 ; '1.50M' -> 1500000.0 ; '' -> None"""
    if not val or not val.strip():
        return None
    val = val.strip().upper()
    mult = 1.0
    if val.endswith("K"):
        mult, val = 1_000.0, val[:-1]
    elif val.endswith("M"):
        mult, val = 1_000_000.0, val[:-1]
    elif val.endswith("B"):
        mult, val = 1_000_000_000.0, val[:-1]
    try:
        return float(val) * mult
    except ValueError:
        return None


def to_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def to_float_or_none(val):
    if val is None or not str(val).strip():
        return None
    try:
        return float(val)
    except ValueError:
        return None


def process_year_file(csv_path, wfo_filter, by_wfo):
    with gzip.open(csv_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            wfo = (row.get("WFO") or "").strip()
            if not wfo:
                continue
            if wfo_filter and wfo not in wfo_filter:
                continue
            record = {
                "event_id": row.get("EVENT_ID"),
                "episode_id": row.get("EPISODE_ID"),
                "event_type": row.get("EVENT_TYPE"),
                "state": row.get("STATE"),
                "cz_type": row.get("CZ_TYPE"),
                "cz_fips": row.get("CZ_FIPS"),
                "cz_name": row.get("CZ_NAME"),
                "begin_date_time": row.get("BEGIN_DATE_TIME"),
                "end_date_time": row.get("END_DATE_TIME"),
                "cz_timezone": row.get("CZ_TIMEZONE"),
                "injuries_direct": to_int(row.get("INJURIES_DIRECT")),
                "injuries_indirect": to_int(row.get("INJURIES_INDIRECT")),
                "deaths_direct": to_int(row.get("DEATHS_DIRECT")),
                "deaths_indirect": to_int(row.get("DEATHS_INDIRECT")),
                "damage_property_usd": parse_damage(row.get("DAMAGE_PROPERTY")),
                "damage_crops_usd": parse_damage(row.get("DAMAGE_CROPS")),
                "source": row.get("SOURCE"),
                "magnitude": to_float_or_none(row.get("MAGNITUDE")),
                "magnitude_type": row.get("MAGNITUDE_TYPE") or None,
                "tor_f_scale": row.get("TOR_F_SCALE") or None,
                "tor_length": to_float_or_none(row.get("TOR_LENGTH")),
                "tor_width": to_float_or_none(row.get("TOR_WIDTH")),
                "begin_lat": to_float_or_none(row.get("BEGIN_LAT")),
                "begin_lon": to_float_or_none(row.get("BEGIN_LON")),
                "end_lat": to_float_or_none(row.get("END_LAT")),
                "end_lon": to_float_or_none(row.get("END_LON")),
                "episode_narrative": (row.get("EPISODE_NARRATIVE") or "").strip() or None,
                "event_narrative": (row.get("EVENT_NARRATIVE") or "").strip() or None,
            }
            by_wfo[wfo].append(record)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year", type=int, default=2008,
                     help="First year to include (default 2008 — see docstring for why not 2000)")
    ap.add_argument("--end-year", type=int, default=datetime.now().year,
                     help="Last year to include (default: current year)")
    ap.add_argument("--wfo", type=str, default="",
                     help="Comma-separated WFO codes to limit output to (default: all 120+ offices)")
    ap.add_argument("--cache-dir", type=str, default="./ncei_cache",
                     help="Where to cache downloaded yearly CSVs")
    ap.add_argument("--out-dir", type=str, default="./storm_events",
                     help="Where to write per-WFO JSON output")
    ap.add_argument("--lean", action="store_true",
                     help="Drop episode/event narrative text (~50%% of file size) — keeps all structured "
                          "fields (dates, magnitude, EF-scale, damage, injuries/deaths, lat/lon) for fast "
                          "cross-referencing. Narratives are nice for human reading but not needed for "
                          "automated matching against warnings.")
    args = ap.parse_args()

    wfo_filter = set(w.strip().upper() for w in args.wfo.split(",") if w.strip()) or None

    print(f"Finding available years {args.start_year}-{args.end_year} on NCEI...")
    year_files = find_year_filenames(args.start_year, args.end_year)
    if not year_files:
        print("No matching files found — check the year range.", file=sys.stderr)
        sys.exit(1)

    by_wfo = defaultdict(list)
    total = 0
    for year in sorted(year_files):
        local_path = download_year(year, year_files[year], args.cache_dir)
        n = process_year_file(local_path, wfo_filter, by_wfo)
        total += n
        print(f"  [{year}] {n} matching events")

    print(f"\nTotal events classified: {total} across {len(by_wfo)} offices")

    os.makedirs(args.out_dir, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    years_covered = sorted(year_files.keys())
    for wfo, events in sorted(by_wfo.items()):
        events.sort(key=lambda e: (e.get("event_id") or ""))
        if args.lean:
            for e in events:
                e.pop("episode_narrative", None)
                e.pop("event_narrative", None)
        out = {
            "wfo": wfo,
            "generated_at": generated_at,
            "source": "NCEI Storm Events Database (official/certified)",
            "years_covered": years_covered,
            "events": events,
        }
        out_path = os.path.join(args.out_dir, f"{wfo}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"  wrote {out_path} ({len(events)} events, {os.path.getsize(out_path)/1024:.0f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()

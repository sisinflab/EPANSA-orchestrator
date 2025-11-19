"""
fsq_fetch.py

Batch enrichment of POIs using the Foursquare Places API.

Key features:
- We call /places/{id} using the legacy venue_id from our dataset.
- If that returns 200, we keep it.
- If that returns a "hard fail" (400/404 etc.), we try a fallback geosearch
  using the POI's lat/lon and category to locate the modern fsq_id, then fetch
  /places/{new_id}.
- We retry on rate limits (429) and transient 5xx.
- We run calls concurrently with ThreadPoolExecutor.
- We save all results in JSONL for downstream feature extraction.

Environment (must be loaded via python-dotenv in app.py):
    FSQ_TOKEN=...
    FSQ_API_VERSION=2025-06-17   (if missing, we default to 2025-06-17)

Input expected by download_all_enriched():
    venues_df with columns:
        - venue_id      (legacy Foursquare venue id from dataset)
        - latitude
        - longitude
        - category

Output:
    JSONL file where each line is either:
        {full fsq place data ...}
    or, if unrecoverable,
        {"_error": <code>, "_venue_id": <legacy_id>}
"""

from __future__ import annotations
import os
import time
import random
import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
from .io_utils import log, ensure_dir, read_jsonl, append_jsonl


API_BASE = "https://places-api.foursquare.com/places"


# ---------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------
def _headers() -> dict:
    """
    Build headers exactly like the working manual request:
        accept: application/json
        X-Places-Api-Version: <FSQ_API_VERSION or default>
        authorization: Bearer <FSQ_TOKEN>

    NOTE:
    - We intentionally DO NOT add params like ?fields=...
      because we already saw that your manual call without fields works.
    """
    token = os.getenv("FSQ_TOKEN")
    api_version = os.getenv("FSQ_API_VERSION", "2025-06-17")

    if not token:
        raise RuntimeError("Missing FSQ_TOKEN env var for Foursquare API access")

    return {
        "accept": "application/json",
        "X-Places-Api-Version": api_version,
        "authorization": f"Bearer {token}",
    }


# ---------------------------------------------------------------------
# Core HTTP helpers
# ---------------------------------------------------------------------
def _backoff_sleep(attempt_i: int) -> None:
    """
    Exponential backoff with jitter for rate limits / transient errors.
    """
    time.sleep((2 ** attempt_i) + random.random())


def _get(session: requests.Session, url: str, params: dict | None = None, retries: int = 5) -> tuple[int, dict | None]:
    """
    Perform a GET with retry logic for rate-limiting / transient errors.

    Returns:
        (status_code, json_response_or_None)

    If we fully exhaust retries, we return ("retry_exhausted", None)
    and let the caller decide how to log it.
    """
    for i in range(retries):
        try:
            r = session.get(url, headers=_headers(), params=params, timeout=20)
        except Exception as e:
            # Network error, just backoff + retry
            log.warning(f"Network error calling {url}: {e}")
            _backoff_sleep(i)
            continue

        # If OK -> parse
        if r.status_code == 200:
            try:
                return (200, r.json())
            except Exception as e:
                log.warning(f"JSON decode error from {url}: {e}")
                return (200, None)

        # Retryable server-side or rate limit
        if r.status_code in (429, 500, 502, 503, 504):
            _backoff_sleep(i)
            continue

        # Hard failure (400, 401, 404, etc.) -> stop immediately
        return (r.status_code, None)

    # Out of retries
    return ("retry_exhausted", None)


# ---------------------------------------------------------------------
# Foursquare helpers
# ---------------------------------------------------------------------
def _fetch_direct(session: requests.Session, fsq_id: str) -> tuple[int, dict | None]:
    """
    Try direct /places/{id}
    """
    url = f"{API_BASE}/{fsq_id}"
    return _get(session, url, params=None)


def _search_nearby(session: requests.Session, lat: float, lon: float, query: str) -> tuple[int, dict | None]:
    """
    Try /places/search with lat/lon and a query string (category hint).

    We'll ask for radius ~100m and limit=1.
    Different versions of the Places API may require
    slightly different params. We're using the common
    pattern:
        /places/search?ll=LAT,LON&radius=100&limit=1&query=query

    Returns:
        (status_code, top_place_dict_or_None)
    """
    url = f"{API_BASE}/search"
    params = {
        "ll": f"{lat},{lon}",
        "radius": 100,
        "limit": 1,
        "query": query or "",
    }

    status, payload = _get(session, url, params=params)
    if status != 200 or payload is None:
        return (status, None)

    # the API tends to return results under "results"
    results = payload.get("results") or payload.get("venues") or []
    if not results:
        return (200, None)

    return (200, results[0])


def _normalize_place(raw: dict, legacy_id: str | None = None, resolution: str | None = None) -> dict:
    """
    Take a raw Foursquare place JSON and produce a normalized record
    we will store in venue_details.jsonl.

    We:
    - expose fsq_id and also keep legacy_id for traceability
    - try to surface lat/lon from geocodes.main if present
    - attach a 'source_resolution' tag ("direct" or "geosearch")
    """
    if raw is None:
        return {}

    out = dict(raw)  # shallow copy

    # make sure we always have venue_id (for downstream code expecting that)
    if "fsq_id" in out and out["fsq_id"]:
        out["venue_id"] = out["fsq_id"]
    elif legacy_id:
        out["venue_id"] = legacy_id

    # propagate legacy id explicitly if we have it
    if legacy_id:
        out["legacy_venue_id"] = legacy_id

    if resolution:
        out["source_resolution"] = resolution

    # normalize lat/lon
    lat = None
    lon = None
    geocodes = out.get("geocodes", {})
    if isinstance(geocodes, dict):
        main_geo = geocodes.get("main") or {}
        lat = lat or main_geo.get("latitude")
        lon = lon or main_geo.get("longitude")

    # Some search responses might not use geocodes but have 'geocodes.main' anyway,
    # so check fallback
    if lat is None and "latitude" in out:
        lat = out.get("latitude")
    if lon is None and "longitude" in out:
        lon = out.get("longitude")

    out["latitude"] = lat
    out["longitude"] = lon

    return out


def fetch_one_with_fallback(session: requests.Session, row: pd.Series) -> dict:
    """
    High-level logic for a single POI row from venues_df.

    Steps:
    1. Try direct /places/{venue_id}
       - if 200 -> normalize and return
       - if hard fail (400/404/etc.) -> fallback

    2. Fallback:
       - /places/search near (lat, lon) using category as query
       - if we get a hit with fsq_id -> fetch /places/{fsq_id}
       - normalize result

    3. If still nothing useful, return an _error record.
    """
    legacy_id = str(row["venue_id"])
    lat = row["latitude"]
    lon = row["longitude"]
    query_hint = row.get("category", "")

    # Step 1: direct
    status_direct, data_direct = _fetch_direct(session, legacy_id)
    if status_direct == 200 and data_direct:
        return _normalize_place(
            data_direct,
            legacy_id=legacy_id,
            resolution="direct"
        )

    # If direct gave us a retryable or rate limit pattern,
    # status_direct could be "retry_exhausted" or 429/5xx.
    # We only Fallback if it's a "hard" fail, e.g. 400/404.
    if isinstance(status_direct, int) and status_direct in (400, 401, 403, 404):
        # Step 2: fallback via nearby search
        status_search, top_candidate = _search_nearby(session, lat, lon, query_hint)
        if status_search == 200 and top_candidate and "fsq_id" in top_candidate:
            new_id = top_candidate["fsq_id"]

            status_resolved, data_resolved = _fetch_direct(session, new_id)
            if status_resolved == 200 and data_resolved:
                return _normalize_place(
                    data_resolved,
                    legacy_id=legacy_id,
                    resolution="geosearch"
                )

    # Step 3: give up -> return error record (we still persist it so we don't retry forever)
    return {
        "_error": status_direct,
        "_venue_id": legacy_id,
        "latitude": lat,
        "longitude": lon,
        "category_hint": query_hint,
    }


# ---------------------------------------------------------------------
# Public batch API
# ---------------------------------------------------------------------
def download_all_enriched(venues_df: pd.DataFrame, out_jsonl: str, max_workers: int = 8) -> None:
    """
    Fetch details for all POIs in venues_df and write them to out_jsonl.

    This function:
    - creates the output directory if needed
    - skips POIs we already downloaded in previous runs (cache)
    - uses a thread pool for concurrency
    - appends each result line-by-line in JSONL

    Expected columns in venues_df:
        ['venue_id', 'latitude', 'longitude', 'category', ...]
    """
    ensure_dir(os.path.dirname(out_jsonl) or ".")

    required_cols = ["venue_id", "latitude", "longitude", "category"]
    for c in required_cols:
        if c not in venues_df.columns:
            raise ValueError(f"venues_df is missing required column '{c}'")

    # Load cache of what's already saved, so we don't re-fetch
    cached_records = read_jsonl(out_jsonl) if os.path.exists(out_jsonl) else []
    seen_ids = set()
    for rec in cached_records:
        if "venue_id" in rec and rec["venue_id"]:
            seen_ids.add(str(rec["venue_id"]))
        elif "_venue_id" in rec and rec["_venue_id"]:
            seen_ids.add(str(rec["_venue_id"]))

    # Subset to only not-yet-fetched venues
    todo_rows = [
        row for _, row in venues_df.iterrows()
        if str(row["venue_id"]) not in seen_ids
    ]

    log.info(f"Already cached: {len(seen_ids)} venues")
    log.info(f"To fetch now: {len(todo_rows)} venues")

    # We'll open the file in append mode so we don't destroy old data
    with requests.Session() as session, \
         open(out_jsonl, "a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=max_workers) as pool:

        futures = {pool.submit(fetch_one_with_fallback, session, row): row for row in todo_rows}

        for fut in as_completed(futures):
            data = fut.result()
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

            vid_for_log = data.get("venue_id") or data.get("_venue_id")
            log.info(f"Saved {vid_for_log} ({'OK' if '_error' not in data else 'ERR'})")

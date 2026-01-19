"""
Feature extraction utilities for venue metadata.

This module:
 - Parses raw JSONL venue records (from Foursquare or other sources).
 - Produces a compact pandas DataFrame with canonical fields.
 - Exposes `canonical_text()` which returns a single textual description
   for each venue (name + category + address + short location hints).
   The canonical text is what we encode to produce venue embeddings.
"""

from typing import Any, Dict
import pandas as pd

from .io_utils import log, read_jsonl, save_parquet


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _extract_primary_category(categories: list[dict] | None) -> str:
    """
    Given the 'categories' array from a places API record,
    return a single primary category string.
    Fallback to "unknown".
    """
    if not categories:
        return "unknown"
    cat = categories[0]
    name = cat.get("name") or cat.get("short_name")
    return name or "unknown"


def _normalize_location_block(location_block: dict | None) -> dict:
    """
    Normalize the typical 'location' dict into a small consistent dict.
    Keys: address, locality, region, postcode, country
    """
    if not location_block:
        return {
            "address": None,
            "locality": None,
            "region": None,
            "postcode": None,
            "country": None,
        }
    keys = ["address", "locality", "region", "postcode", "country"]
    return {k: location_block.get(k) for k in keys}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def build_venue_feature_frame(jsonl_path: str) -> pd.DataFrame:
    """
    Read a JSONL file containing enriched venue records and return
    a flattened DataFrame with core fields.

    Expected minimum fields present in each JSON record:
      - venue_id (or fsq_id)
      - name
      - categories (list)
      - latitude, longitude
      - location (dict)

    Returns
    -------
    pd.DataFrame with columns: venue_id, name, primary_category,
    latitude, longitude, location (dict)
    """
    raw_records = read_jsonl(jsonl_path)
    rows = []

    for rec in raw_records:
        venue_id = rec.get("venue_id") or rec.get("fsq_id")
        if not venue_id:
            continue

        row = {
            "venue_id": venue_id,
            "name": rec.get("name"),
            "primary_category": _extract_primary_category(rec.get("categories")),
            "latitude": rec.get("latitude") or rec.get("lat"),
            "longitude": rec.get("longitude") or rec.get("lon") or rec.get("lng"),
            "location": _normalize_location_block(rec.get("location")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["primary_category"] = df["primary_category"].fillna("unknown")
    log.info(f"Built venue feature frame with {len(df)} rows from {jsonl_path}")
    return df


def save_venue_features(jsonl_path: str, out_path: str) -> pd.DataFrame:
    """
    Convenience wrapper: parse JSONL -> DataFrame -> save parquet.
    Returns the DataFrame for in-memory chaining.
    """
    df = build_venue_feature_frame(jsonl_path)
    save_parquet(df, out_path)
    log.info(f"Saved {len(df)} venue feature rows -> {out_path}")
    return df


def canonical_text(row: Dict[str, Any]) -> str:
    """
    Build a simple, compact textual description for a single venue row.
    This is the text we feed to the embedding encoder.

    Example output:
      "Louvre Museum || Museum || Rue de Rivoli, Paris || FR"

    The function is intentionally conservative (short) to avoid huge tokenization cost.
    """
    parts = []
    name = row.get("name")
    if name:
        parts.append(str(name).strip())

    cat = row.get("primary_category")
    if cat:
        parts.append(str(cat).strip())

    loc = row.get("location") or {}
    addr = loc.get("address")
    locality = loc.get("locality")
    region = loc.get("region")
    country = loc.get("country")

    addr_parts = [p for p in [addr, locality, region] if p]
    if addr_parts:
        parts.append(", ".join(addr_parts))

    if country:
        parts.append(country)

    # join with separator chosen to be robust but not too verbose
    return " || ".join(parts)

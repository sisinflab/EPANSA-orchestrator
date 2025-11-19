"""
Tourism-focused POI selection pipeline.

We build the candidate POI set for our recommender in two stages:

1. CATEGORY FILTER (tourism / leisure relevance)
   Keep only POIs that correspond to things a traveler would actually
   want to visit or experience: museums, parks, landmarks, restaurants,
   nightlife, viewpoints, etc.
   We explicitly exclude logistical / functional places like airports,
   train stations, offices, hospitals, schools, etc.

2. POPULARITY FILTER (top fraction by check-ins)
   Within the "touristic" POIs, keep only the most popular ones
   based on the number of check-ins. We keep the top N% (e.g. 50%)
   so we focus on high-signal places.

The resulting POI list is what we will fetch metadata for,
embed, and recommend.

Inputs
------
- dataset_TIST2015_Checkins.txt
    Tab-separated, no header:
        0: user_id (anonymized)
        1: venue_id (Foursquare venue ID)
        2: utc_time (string timestamp)
        3: timezone_offset (minutes)

- POI.txt
    Tab-separated, no header:
        0: venue_id
        1: latitude
        2: longitude
        3: category
        4: country_code

Output
------
A DataFrame with columns like:
    venue_id
    latitude
    longitude
    category
    country_code
    checkin_count
Only containing:
    - touristic categories
    - top X% most popular within those categories

This DataFrame will be saved to cfg.venues_list by app.step_downsample
and then consumed by fsq_fetch.py.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_checkins(path: str) -> pd.DataFrame:
    """
    Load raw check-ins from the TIST2015 dataset.

    Columns:
        user_id, venue_id, utc_time, timezone_offset
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "venue_id", "utc_time", "timezone_offset"],
        dtype={"user_id": str, "venue_id": str},
        usecols=[0, 1, 2, 3],
    )
    return df


def load_poi_table(path: str) -> pd.DataFrame:
    """
    Load POI metadata from POI.txt.

    Columns:
        venue_id, latitude, longitude, category, country_code
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["venue_id", "latitude", "longitude", "category", "country_code"],
        dtype={
            "venue_id": str,
            "latitude": float,
            "longitude": float,
            "category": str,
            "country_code": str,
        },
    )
    return df


# ---------------------------------------------------------------------
# Category filtering (tourist relevance)
# ---------------------------------------------------------------------

# We APPROVE POIs if their category matches ANY of these keywords.
TOURISTIC_KEYWORDS = [
    # Culture / heritage / sightseeing
    "museum", "art", "gallery", "historic", "history",
    "monument", "landmark", "castle", "palace", "ruins",
    "cathedral", "church", "basilica", "temple", "mosque", "synagogue",

    # Nature / outdoors / views
    "park", "national park", "garden", "botanical",
    "beach", "lake", "mountain", "forest", "scenic", "lookout", "viewpoint",
    "plaza", "square", "waterfront", "pier",

    # Food & drink (core part of travel recs)
    "restaurant", "cafe", "coffee", "bar", "pub",
    "bakery", "patisserie", "ice cream", "gelato", "winery", "brewery",
    "food market", "market", "street food",

    # Entertainment / nightlife / leisure
    "theater", "theatre", "cinema", "movie theater",
    "concert hall", "music venue", "club", "nightclub",
    "event space", "stadium", "arena",

    # Attractions / activities
    "tourist attraction", "amusement park", "theme park",
    "zoo", "aquarium",

    # Shopping as experience (souvenirs, local crafts etc.)
    "souvenir", "gift shop", "flea market", "antique", "bazaar",
    "shopping mall", "shopping centre", "shopping center",

    # Stay / hospitality (still part of tourist planning)
    "hotel", "hostel", "resort", "bed & breakfast", "bed and breakfast", "bnb",
]


def is_touristic_category(cat: str) -> bool:
    """
    Returns True if the POI category looks like something
    a traveler would actively want to visit / experience.

    We do a lowercase substring match against TOURISTIC_KEYWORDS.
    """
    if not isinstance(cat, str):
        return False
    c = cat.lower()
    return any(kw in c for kw in TOURISTIC_KEYWORDS)


def filter_touristic_pois(poi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only POIs whose category matches the touristic whitelist.
    """
    df = poi_df.copy()
    df["is_touristic"] = df["category"].apply(is_touristic_category)
    df = df[df["is_touristic"]].drop(columns=["is_touristic"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Popularity computation
# ---------------------------------------------------------------------

def compute_poi_popularity(checkins_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute number of check-ins per venue.

    Output columns:
        venue_id | checkin_count
    """
    popularity = (
        checkins_df.groupby("venue_id")
        .size()
        .reset_index(name="checkin_count")
    )
    return popularity


def keep_top_fraction_by_popularity(
    poi_df: pd.DataFrame,
    popularity_df: pd.DataFrame,
    keep_fraction: float = 0.03,
) -> pd.DataFrame:
    """
    Merge touristic POIs with popularity, rank them, and keep top fraction.

    Steps:
    - left join so we attach checkin_count
    - fill missing popularity with 0 (in case a POI never appears in checkins)
    - sort by checkin_count desc
    - cut at keep_fraction

    Returns
    -------
    DataFrame subset of poi_df with an extra column checkin_count.
    """
    merged = poi_df.merge(popularity_df, on="venue_id", how="left")
    merged["checkin_count"] = merged["checkin_count"].fillna(0).astype(int)

    merged = merged.sort_values("checkin_count", ascending=False)

    n_total = len(merged)
    n_keep = max(1, int(n_total * keep_fraction))
    trimmed = merged.head(n_keep).reset_index(drop=True)

    return trimmed


# ---------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------

def build_filtered_poi_subset(
    checkins_path: str,
    poi_path: str,
    keep_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    High-level pipeline to produce the final POI subset we care about.

    Steps:
    1. Load POI.txt
    2. Filter POIs by touristic / leisure category (whitelist)
    3. Load check-ins
    4. Compute popularity per POI
    5. Keep only the top `keep_fraction` most popular among touristic POIs
    6. Clean up rows with missing coords / duplicates

    Returns
    -------
    DataFrame with columns:
        venue_id
        latitude
        longitude
        category
        country_code
        checkin_count
    """
    # Step 1-2: load POIs and keep only touristic ones
    poi_all = load_poi_table(poi_path)
    poi_touristic = filter_touristic_pois(poi_all)

    # Step 3-4: load check-ins and compute popularity
    checkins = load_checkins(checkins_path)
    popularity = compute_poi_popularity(checkins)

    # Step 5: retain only the most popular touristic POIs
    top_pois = keep_top_fraction_by_popularity(
        poi_df=poi_touristic,
        popularity_df=popularity,
        keep_fraction=keep_fraction,
    )

    # Step 6: final cleanup
    top_pois = top_pois.dropna(subset=["latitude", "longitude"])
    top_pois = top_pois.drop_duplicates(subset=["venue_id"]).reset_index(drop=True)

    return top_pois

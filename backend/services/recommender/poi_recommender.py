import asyncio
import os
import logging
import requests
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from thefuzz import process as fuzzy_process
from thefuzz import fuzz
from typing import Any, List
import pycountry_convert as pc
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from libs.llm_graph_builder.src.llm import get_llm
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
import jellyfish

from .retrieval import build_nn_index
from .ranker import rank_pois_by_relevance
from .io_utils import log
from .embeddings import cosine_sim
from .intent_extractor import _extract_and_parse_json


# A set of categories that are generally not considered tourist attractions.
NON_ATTRACTION_CATEGORIES = {
    'restaurant', 'food', 'dining', 'eatery', 'cafe', 'café', 'coffee shop', 'bar', 'pub', 'club', 'cocktail',
    'nightlife', 'fast food', 'pizzeria', 'bakery', 'dessert shop', 'food court', 'food stand', 'ice cream', 'diner',
    'shopping', 'mall', 'store', 'market', 'boutique', 'shop', 'department store', 'supermarket', 'grocery',
    'bank', 'atm', 'post office', 'laundry', 'gay bar', 'gym', 'fitness', 'office', 'business', 'medical', 'hospital',
    'hotel', 'motel', 'hostel', 'airport', 'train station', 'bus station', 'parking', 'other', 'unknown', 'marine terminal'
    'tech startup', 'apartment', 'event space', 'swimming pool', 'basketball stadium', 'soccer stadium',
    'resort', 'radio station', 'tech startup', 'conference', 'scenic lookout', 'breakfast spot', 'baseball stadium', 'music school'
}

# Prompt template for generating a personalized reason for recommending a CITY.
REASON_GENERATION_PROMPT_TEMPLATE = """
You are a travel expert writing a short, compelling recommendation.
Based on the user's profile and the recommended city, write a single, engaging sentence (max 25 words) explaining WHY this city is a great match for THEM.
Connect a specific user interest to a specific city feature.

**User Profile:**
{user_profile_summary}

**City:** {city_name}, {country}
**Key Attractions:** {attractions_list}

**Example:** "Given your love for ancient history, Rome's iconic Colosseum and Roman Forum will feel like stepping back in time."

**Your personalized sentence:**
"""

# Prompt template for generating a personalized reason for recommending a specific POINT OF INTEREST.
POI_REASON_GENERATION_PROMPT_TEMPLATE = """
You are a tour guide explaining why a specific Point of Interest is a great fit for a user.
In a single, short, engaging sentence (max 20 words), connect ONE of the user's interests to this specific POI.

**User Profile:**
{user_profile_summary}

**Point of Interest:** {poi_name} ({poi_category} in {city_name})

**Example:** "Since you appreciate modernist architecture, you'll be fascinated by the unique design of Casa Batlló."

**Your personalized sentence:**
"""

RERANKING_PROMPT_TEMPLATE = """
You are a pragmatic and expert travel planner. Your task is to re-rank a list of potential travel destinations to create a balanced and inspiring set of recommendations.

**User Profile:**
{user_profile_summary}

**Your Core Task: Analysis and Balanced Recommendation**
1.  **First, Analyze the Profile:** Carefully read the user profile above to identify all their distinct interests (e.g., history, food, nature, sports, art).
2.  **Then, Create a DIVERSE List:** Your main goal is to create a **balanced** list of recommendations that reflects the user's varied interests.
3.  **AVOID MONOTHEMATIC SUGGESTIONS:** Do not let a single interest, even if it seems dominant in the profile, monopolize all your suggestions. For example, if the user clearly loves history, do not suggest five ancient cities. Mix it up with suggestions that cater to their other potential interests.

**User's Trip Request:**
{intent_str}

**User's Home Base:** {home_base_str}

**Candidate Destinations (pre-ranked by thematic match):**
{candidates_str}

**Your Task and Rules:**
1.  **Analyze the Candidate Destinations provided above. You MUST ONLY use cities from this list.**
2.  **DO NOT invent or suggest any city that is NOT in the provided list.** This is a strict rule.
3.  Return a SINGLE, VALID JSON object with a single key "ranked_destinations".
4.  This key must hold a list of the TOP 5 most suitable and DIVERSE destinations selected **exclusively from the candidate list**.
5.  For each destination, provide:
    - "city": The name of the city, copied exactly from the candidate list.
    - "country": The name of the country, copied exactly from the candidate list.
    - "justification": A short, compelling sentence explaining WHY this city is a perfect fit for THIS specific trip, combining thematic and practical reasons. Ensure your justifications highlight different aspects of the user's profile.

Your entire response must be ONLY the JSON object. Do not add any commentary before or after.
"""

def _to_list(value: Any) -> List[str]:
    """
    Ensures the given value is a list of strings.
    - If None, returns an empty list.
    - If a list, returns it as is.
    - If a single value (e.g., a string), returns it wrapped in a list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [str(value)]


def _is_category_touristic(category_name: str) -> bool:
    """
    Checks if a single category name is considered touristic.
    """
    if not isinstance(category_name, str):
        return False
    cat_lower = category_name.lower()
    return not any(keyword in cat_lower for keyword in NON_ATTRACTION_CATEGORIES)

def extract_city_and_country_from_address(address_string: str) -> str | None:
    """Extracts 'City, Country' from a full address string."""
    if not isinstance(address_string, str): return None
    parts = [p.strip() for p in address_string.split(',')]
    if len(parts) >= 2:
        return f"{parts[-2]}, {parts[-1]}"
    return parts[0] if parts else None


def _check_if_city_has_attractions(series_of_categories: pd.Series) -> bool:
    """
    Checks if a city has at least one tourist attraction
    """
    return any(_is_category_touristic(cat) for cat in series_of_categories.dropna())


# continent mapping functions
def get_country_code(country_input: str) -> str | None:
    """
    Normalizes a country name (e.g., "Italy") or a 2-letter code to its
    standard ISO 3166-1 alpha-2 code (e.g., "IT").

    Args:
        country_input: The country name or code provided by the user/intent.

    Returns:
        The uppercase 2-letter country code, or None if conversion fails.
    """
    if not country_input or not isinstance(country_input, str) or len(country_input) < 2:
        return None

    # If it's already a 2-letter code, just standardize it to uppercase.
    if len(country_input) == 2:
        return country_input.upper()

    try:
        # Otherwise, attempt to convert the full name to a code.
        return pc.country_name_to_country_alpha2(country_input, cn_name_format="default")
    except (KeyError, Exception):
        logging.warning(f"Could not convert '{country_input}' to a valid country code.")
        return None


def get_continent_from_country(country_code: str) -> str | None:
    """
    Converts a 2-letter country alpha-2 code (e.g., 'IT', 'US') to its
    full continent name.

    Args:
        country_code: The 2-letter ISO code from the dataset.

    Returns:
        The full continent name (e.g., "Europe"), or None if conversion fails.
    """
    if not country_code or not isinstance(country_code, str) or len(country_code) != 2:
        return None

    try:
        continent_code = pc.country_alpha2_to_continent_code(country_code.upper())
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except (KeyError, Exception):
        # This function is called for every row in the dataframe,
        # so avoid logging warnings here to prevent spamming the console.
        return None


def _normalize_city_id(city_name: str, country_name: str) -> str | None:
    """Creates a consistent, lowercased ID for a city from its name and country."""
    c = str(city_name or '').strip()
    k = str(country_name or '').strip()
    return f"{c}|{k}".lower() if c or k else None


def maximal_marginal_relevance(
        user_embedding: np.ndarray,
        candidates_df: pd.DataFrame,
        lambda_param: float = 0.7,
        top_k: int = 10
) -> pd.DataFrame:
    """
    Re-ranks a list of candidates using the Maximal Marginal Relevance (MMR) algorithm.
    """
    if candidates_df.empty or 'city_embedding' not in candidates_df.columns:
        return pd.DataFrame()

    candidate_embeddings = np.stack(candidates_df['city_embedding'].values)
    user_embedding = user_embedding.reshape(1, -1)

    relevance_scores = cosine_similarity(user_embedding, candidate_embeddings)[0]
    ranked_indices, remaining_indices = [], list(range(len(candidates_df)))

    if not remaining_indices:
        return pd.DataFrame()

    first_choice_idx = np.argmax(relevance_scores)
    ranked_indices.append(first_choice_idx)
    remaining_indices.remove(first_choice_idx)

    while len(ranked_indices) < top_k and remaining_indices:
        mmr_scores = {}
        selected_embeddings = candidate_embeddings[ranked_indices]
        for idx in remaining_indices:
            relevance = relevance_scores[idx]
            similarity_to_selected = cosine_similarity(candidate_embeddings[idx].reshape(1, -1), selected_embeddings)
            max_similarity = np.max(similarity_to_selected)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores[idx] = mmr_score

        if not mmr_scores: break
        best_next_idx = max(mmr_scores, key=mmr_scores.get)
        ranked_indices.append(best_next_idx)
        remaining_indices.remove(best_next_idx)

    return candidates_df.iloc[ranked_indices]


# Geocoding function with retries
def geocode_location(location_name: str, retries: int = 3, backoff_factor: float = 0.5) -> tuple[float, float] | None:
    """
    Geocodes a location name with retry logic.
    """
    if not isinstance(location_name, str) or not location_name.strip(): return None

    url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=1&format=json&language=en"

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            data = r.json()
            results = data.get("results")
            if not results: return None
            best = results[0]
            lat, lon = best.get("latitude"), best.get("longitude")
            if lat is None or lon is None: return None
            return (float(lat), float(lon))
        except requests.exceptions.RequestException as e:
            log.warning(f"Geocoding attempt {attempt + 1} for '{location_name}' failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                log.error(f"Geocoding failed for '{location_name}' after {retries} attempts.")
                return None
    return None


class POIRecommender:
    def __init__(self, poi_data_path: str):
        logging.info(f"Initializing POIRecommender with data from {poi_data_path}...")
        try:
            features_df = pd.read_parquet(os.path.join(poi_data_path, "venue_features.parquet"))
            embeddings_df = pd.read_parquet(os.path.join(poi_data_path, "venue_embeddings.parquet"))

            if "location" in features_df.columns:
                loc_data = features_df['location'].apply(
                    lambda d: pd.Series([d.get('locality'), d.get('country'), d.get('address')]) if isinstance(d,
                                                                                                               dict) else pd.Series(
                        [None, None, None]))
                features_df[['locality', 'country', 'address']] = loc_data
                features_df['neighborhood'] = features_df['address']

            self.features_df = pd.merge(features_df, embeddings_df, on="venue_id", how="inner")
            logging.info(f"Data merged. Total POIs: {len(self.features_df)}")

            if 'locality' in self.features_df.columns:
                self.available_cities = self.features_df['locality'].dropna().str.lower().unique().tolist()
                logging.info(f"Found {len(self.available_cities)} unique cities for matching.")
            else:
                self.available_cities = []

            self.nn_index, self.item_vecs, self.venue_ids = build_nn_index(self.features_df)

            self._build_city_index()

        except Exception as e:
            logging.error(f"Failed to initialize POIRecommender: {e}", exc_info=True)

    def select_city_pois(self, city_id, user_profile_vector, n=5):
        """
        Selects a DIVERSE and HIGH-QUALITY list of POIs for a given city to showcase to the user.
        This function includes:
        1. A primary attempt to select from strictly 'touristic' POIs.
        2. A fallback to use all available POIs if no touristic ones are found.
        3. MMR-based selection to ensure the final list is not dominated by a single theme (like sports).
        """
        # Get all POIs for the city from the main dataframe
        full_city_subset = self.features_df[self.features_df["city_id"] == city_id].copy()
        if full_city_subset.empty:
            logging.warning(f"No POIs found for city_id {city_id} in the dataset.")
            return []

        # Filter for touristic POIs
        touristic_subset = full_city_subset[full_city_subset['primary_category'].apply(_is_category_touristic)].copy()

        subset_to_rank = None
        if not touristic_subset.empty:
            logging.info(
                f"Found {len(touristic_subset)} touristic POIs for city_id {city_id}. Proceeding with selection from this subset.")
            subset_to_rank = touristic_subset
        else:
            # Fallback Strategy: Use all POIs if no touristic ones are available
            logging.warning(f"No strictly 'touristic' POIs found for city_id {city_id}. "
                            f"Falling back to all {len(full_city_subset)} available POIs for that city.")
            subset_to_rank = full_city_subset

        # Ranking and Diversification (MMR)
        if "embedding" not in subset_to_rank.columns or subset_to_rank.empty:
            logging.error(f"Cannot perform ranking for city_id {city_id} due to missing embeddings or empty subset.")
            return []

        # If there are fewer POIs than requested, no need for complex ranking
        if len(subset_to_rank) <= n:
            return subset_to_rank.head(n)[['name', 'primary_category', 'neighborhood']].to_dict(orient="records")

        # Calculate relevance for all candidates
        subset_to_rank['relevance'] = subset_to_rank['embedding'].apply(
            lambda emb: cosine_sim(user_profile_vector, emb))
        subset_to_rank = subset_to_rank.sort_values('relevance', ascending=False).reset_index(drop=True)

        # Initialize MMR
        selected_indices = []
        candidate_indices = list(subset_to_rank.index)
        candidate_embeddings = np.stack(subset_to_rank['embedding'].values)
        lambda_param = 0.7  # Balance between relevance and diversity

        # Safety check for empty candidate list
        if not candidate_indices:
            return []

        # Add the most relevant item first
        first_choice_idx = candidate_indices.pop(0)
        selected_indices.append(first_choice_idx)

        # Iteratively select the rest using MMR
        while len(selected_indices) < n and candidate_indices:
            best_next_idx = -1
            max_mmr_score = -np.inf
            selected_embeddings = candidate_embeddings[selected_indices]

            for idx in candidate_indices:
                relevance_score = subset_to_rank.loc[idx, 'relevance']

                # Calculate diversity penalty (max similarity to already selected items)
                candidate_emb = candidate_embeddings[idx].reshape(1, -1)
                diversity_penalty = cosine_similarity(candidate_emb, selected_embeddings).max()

                # MMR formula
                mmr_score = (lambda_param * relevance_score) - ((1 - lambda_param) * diversity_penalty)

                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    best_next_idx = idx

            if best_next_idx != -1:
                selected_indices.append(best_next_idx)
                candidate_indices.remove(best_next_idx)
            else:
                # Stop if no more valid candidates can be found
                break

        # Return the selected POIs
        top_pois = subset_to_rank.iloc[selected_indices]
        return top_pois[['name', 'primary_category', 'neighborhood']].to_dict(orient="records")

    def _build_city_index(self):
        """
        Aggregates POI data to create a city-level index DataFrame.
        This method processes `self.features_df` to create `self.city_index`,
        which contains aggregated information for each city. Key steps:
        - Calculates total POI count and a specific count for 'touristic' POIs.
        - Calculates a city-level embedding by averaging the embeddings of ONLY its touristic POIs.
        """
        df = self.features_df
        if 'locality' not in df.columns:
            self.city_index = pd.DataFrame()
            logging.warning("'locality' column not found. City index will be empty.")
            return

        # Create a unique ID for each city to group by.
        df['city_id'] = df.apply(lambda row: _normalize_city_id(row.get('locality'), row.get('country')), axis=1)

        # Aggregate basic city information.
        agg_dict = {
            "city_name": ("locality", "first"),
            "country": ("country", "first"),
            "city_lat": ("latitude", "mean"),
            "city_lon": ("longitude", "mean"),
            "poi_count": ("venue_id", "count")  # Total POI count
        }
        city_index_df = df.dropna(subset=["city_id"]).groupby("city_id").agg(**agg_dict).reset_index()

        city_index_df.dropna(subset=["city_name"], inplace=True)
        if city_index_df.empty:
            self.city_index = pd.DataFrame()
            logging.warning("City index is empty after dropping entries with no city name.")
            return

        # Add continent information
        city_index_df['continent'] = city_index_df['country'].apply(get_continent_from_country)

        # Determine the top 3 themes (primary POI categories) for each city from all POIs
        themes = df.dropna(subset=['city_id', 'primary_category']).groupby('city_id')['primary_category'].apply(
            lambda s: s.value_counts().nlargest(3).index.tolist()
        ).rename("themes")
        city_index_df = city_index_df.merge(themes, on="city_id", how="left")

        # Identify which POIs are touristic at the POI level
        df['is_touristic'] = df['primary_category'].apply(_is_category_touristic)

        # Calculate the count of ONLY touristic POIs for each city
        touristic_poi_counts = df[df['is_touristic'] == True].groupby('city_id').size().rename('touristic_poi_count')
        city_index_df = city_index_df.merge(touristic_poi_counts, on='city_id', how='left').fillna(
            {'touristic_poi_count': 0})
        city_index_df['touristic_poi_count'] = city_index_df['touristic_poi_count'].astype(int)

        # Create 'has_attractions' based on the new touristic count for consistency
        city_index_df['has_attractions'] = city_index_df['touristic_poi_count'] > 0

        # Calculate city-level embedding by averaging embeddings of its touristic POIs
        touristic_pois_df = df[df['is_touristic'] == True].copy()

        if "embedding" not in touristic_pois_df.columns or touristic_pois_df.empty:
            logging.warning("No touristic POIs with embeddings found. City embeddings cannot be calculated.")
            emb_series = pd.Series(name="city_embedding", dtype=object)
        else:
            touristic_pois_df["_emb_arr"] = touristic_pois_df["embedding"].apply(
                lambda v: np.array(v, dtype=float) if v is not None else None)
            emb_series = touristic_pois_df.dropna(subset=['city_id', '_emb_arr']).groupby("city_id")["_emb_arr"].apply(
                lambda s: np.mean(np.stack(s.values), axis=0)
            ).rename("city_embedding")

        city_index_df = city_index_df.merge(emb_series, on="city_id", how="left")

        # Finalize the city index, dropping any cities that failed to get a (touristic) embedding.
        self.city_index = city_index_df.dropna(subset=["city_embedding"])

        logging.info(
            f"Built city_index with {len(self.city_index)} cities. "
            f"These cities have at least one touristic POI with an embedding."
        )

    async def get_recommendations(self, user_profile_summary: str, model_env_value: str, intent,
                                  user_profile_vector, user_location=None, city_name=None):
        if self.features_df is None or self.features_df.empty:
            return []

        candidate_pois_df = self.features_df.copy()
        logging.info(f"--- Starting POI Filtering for city: {city_name or 'Not specified'} ---")

        # Filter by City
        if city_name:
            canonical_city_name = _find_best_city_match(city_name.lower(), self.available_cities)
            if canonical_city_name:
                logging.info(f"Successfully matched '{city_name}' to '{canonical_city_name}'. Filtering POIs...")
                mask = candidate_pois_df["locality"].str.lower() == canonical_city_name
                candidate_pois_df = candidate_pois_df[mask.fillna(False)]
                logging.info(f"Found {len(candidate_pois_df)} POIs in city '{canonical_city_name}'.")
            else:
                logging.warning(f"Could not find a confident match for city '{city_name}'. Aborting.")
                return []

            if candidate_pois_df.empty:
                logging.warning(f"City '{canonical_city_name}' matched, but no POIs found in the dataset.")
                return []

        # Filter by User Intent (Category/Subcategory)
        search_pattern = None
        search_regex = False

        cat_from_intent = (intent.get("poi_category") or "").lower()
        subcat_from_intent = (intent.get("subcategory") or "").lower()

        if subcat_from_intent:
            search_pattern = subcat_from_intent
        elif cat_from_intent:
            if cat_from_intent == "restaurant":
                search_pattern = r"restaurant|food|dining|eatery|cafe|café|pizzeria|bakery"
                search_regex = True
            elif cat_from_intent == "attractions":
                search_pattern = r"architecture|attraction|plaza|square|landmark|monument|historic|church|museum|gallery|park|stadium|theater|building|art|site|viewpoint|cathedral|basilica|beach"
                search_regex = True
            else:
                search_pattern = cat_from_intent

        if search_pattern and 'primary_category' in candidate_pois_df.columns:
            logging.info(f"Applying category filter with pattern: '{search_pattern}' (regex={search_regex})")
            pois_before_filter = candidate_pois_df.copy()

            categories_series = candidate_pois_df['primary_category'].str.lower().fillna('')
            mask_cat = categories_series.str.contains(search_pattern, na=False, regex=search_regex)

            candidate_pois_df = candidate_pois_df[mask_cat]
            logging.info(f"After category filter, {len(candidate_pois_df)} POIs remain.")

            if candidate_pois_df.empty:
                logging.warning(
                    f"Category filter for '{search_pattern}' yielded 0 results. Falling back to pre-filter list.")
                candidate_pois_df = pois_before_filter

        # Filter to remove undesirable categories conditionally
        USER_SEARCHABLE_NON_TOURISTIC = {
            "restaurant", "nightlife", "shopping", "food", "dining", "bar", "cafe", "pub", "club"
        }

        if cat_from_intent not in USER_SEARCHABLE_NON_TOURISTIC:
            if not candidate_pois_df.empty:
                initial_count = len(candidate_pois_df)
                mask_quality = candidate_pois_df['primary_category'].apply(_is_category_touristic)
                candidate_pois_df = candidate_pois_df[mask_quality]
                logging.info(f"Applying quality filter. {len(candidate_pois_df)} of {initial_count} POIs remain.")
        else:
            logging.info(f"Skipping quality filter because user is searching for '{cat_from_intent}'.")

        if candidate_pois_df.empty:
            logging.warning("No POIs remained after all filters.")
            return []

        # Ranking and LLM Justification
        logging.info(f"Ranking {len(candidate_pois_df)} final candidates...")
        top_pois_df = rank_pois_by_relevance(
            candidate_pois_df=candidate_pois_df,
            user_profile_vector=user_profile_vector,
            user_location_coords=user_location,
            top_k=5
        )

        if top_pois_df.empty:
            return []

        tasks = [asyncio.create_task(generate_poi_llm_reason(user_profile_summary, poi_row, model_env_value)) for
                 _, poi_row in top_pois_df.iterrows()]
        reasons = await asyncio.gather(*tasks)

        results = top_pois_df[['name', 'primary_category', 'locality']].to_dict('records')
        for i, poi in enumerate(results):
            poi['why'] = reasons[i]

        return results

    async def get_destination_recommendations(self, user_profile_vector,
                                              topk=20, exclude_cities=None, intent=None, home_base_str=None):
        """
        Gets destination (city) recommendations based on user profile, location, and trip type.
        This process includes a sequential filtering and scoring pipeline:
        1. Quality Filtering: Ensures cities have a minimum number of touristic attractions.
        2. Geographic Inclusion: Narrows down by continent or country if specified.
        3. Geographic Exclusion: Removes cities from excluded continents or countries.
        4. Profile Exclusion: Removes cities the user has already visited or explicitly excluded.
        5. Scoring & Ranking: Ranks remaining candidates by thematic match, popularity, and practicality.
        6. Diversification: Uses MMR to ensure the final list is varied.
        """
        if not hasattr(self, "city_index") or self.city_index.empty:
            logging.warning("City index is not available or empty. Cannot provide destination recommendations.")
            return pd.DataFrame()

        cities = self.city_index.copy()

        # Quality Filter
        MIN_TOURISTIC_POI_COUNT = 3
        initial_city_count = len(cities)
        if 'touristic_poi_count' in cities.columns:
            cities = cities[cities['touristic_poi_count'] >= MIN_TOURISTIC_POI_COUNT]
            logging.info(f"After quality filter (>= {MIN_TOURISTIC_POI_COUNT} attractions), "
                         f"{len(cities)} of {initial_city_count} cities remain.")
        else:
            logging.warning("'touristic_poi_count' column not found. Skipping quality filter.")

        if cities.empty:
            logging.warning("No candidate cities remaining after quality filter.")
            return pd.DataFrame()

        # Geographic Inclusion & Exclusion Filters
        if intent:
            # Inclusion filters are applied first to narrow the search space
            continent_filter = intent.get('continent')
            if continent_filter:
                logging.info(f"Applying continent inclusion filter: '{continent_filter}'")
                if 'continent' in cities.columns:
                    mask = cities['continent'].str.lower() == continent_filter.lower()
                    cities = cities[mask.fillna(False)]

            country_filter_raw = intent.get('country')
            if country_filter_raw:
                country_code = get_country_code(country_filter_raw)
                logging.info(f"Applying country inclusion filter for '{country_filter_raw}' (code: {country_code})")
                if country_code and 'country' in cities.columns:
                    mask = cities['country'].str.upper() == country_code
                    cities = cities[mask.fillna(False)]

            # Exclusion filters are applied on the remaining candidates
            continents_to_exclude = _to_list(intent.get('exclude_continent'))
            if continents_to_exclude:
                logging.info(f"Applying continent exclusion for: {continents_to_exclude}")
                lower_continents = [c.lower() for c in continents_to_exclude]
                if 'continent' in cities.columns:
                    mask = cities['continent'].str.lower().isin(lower_continents)
                    cities = cities[~mask.fillna(False)]

            countries_to_exclude_raw = _to_list(intent.get('exclude_country'))
            if countries_to_exclude_raw:
                codes_to_exclude = [c for c in (get_country_code(name) for name in countries_to_exclude_raw) if c]
                logging.info(f"Applying country exclusion for {countries_to_exclude_raw} (codes: {codes_to_exclude})")
                if codes_to_exclude and 'country' in cities.columns:
                    mask = cities['country'].str.upper().isin(codes_to_exclude)
                    cities = cities[~mask.fillna(False)]

            logging.info(f"After all geographic filters, {len(cities)} cities remain.")

        # Profile-based City Exclusion (already visited or explicitly excluded in query)
        if exclude_cities:
            initial_count = len(cities)
            cities = cities[~cities['city_name'].str.strip().str.lower().isin(exclude_cities)]
            logging.info(
                f"Excluded {initial_count - len(cities)} visited/explicitly excluded cities. {len(cities)} candidates remain.")

        if cities.empty:
            logging.warning("No candidate cities remaining after all filters.")
            return pd.DataFrame()

        # Scoring Phase
        home_base = extract_city_and_country_from_address(home_base_str) or home_base_str
        home_base_coords = geocode_location(home_base)
        duration = intent.get("duration") if intent else None

        cities["theme_match"] = cities.apply(lambda row: cosine_sim(user_profile_vector, row.get("city_embedding")),
                                             axis=1)
        cities["popularity_score"] = np.log1p(cities["poi_count"]) / np.log1p(cities["poi_count"].max())
        cities["practicality"] = cities.apply(lambda row: practicality_score(row, home_base_coords, duration), axis=1)

        cities["initial_score"] = (
                0.65 * cities["theme_match"] +
                0.05 * cities["popularity_score"] +
                0.30 * cities["practicality"]
        )

        # Candidate Selection & Diversification
        initial_candidates = cities.sort_values("initial_score", ascending=False).head(topk)

        if initial_candidates.empty:
            logging.warning("No candidates left after initial scoring.")
            return pd.DataFrame()

        logging.info(f"Applying MMR to diversify the top {len(initial_candidates)} candidates...")

        return maximal_marginal_relevance(
            user_embedding=np.array(user_profile_vector),
            candidates_df=initial_candidates,
            lambda_param=0.75,
            top_k=topk
        )

    async def rerank_destinations_with_llm(self, candidates_df, user_profile_summary, intent, home_base_str,
                                           model_env_value, user_profile_vector):
        if candidates_df.empty: return []

        # Create a structured list of candidates for the prompt
        candidate_list_for_prompt = []
        for _, row in candidates_df.iterrows():
            candidate_list_for_prompt.append({
                "city": row['city_name'],
                "country": row['country'],
                "highlights": row.get('themes', []),
                "initial_score": round(row.get('initial_score', 0), 2)
            })
        candidates_str = json.dumps(candidate_list_for_prompt, indent=2)

        intent_str = json.dumps(intent)
        home_base_str = home_base_str or "Not specified"

        llm, _ = get_llm(model_env_value)
        prompt = ChatPromptTemplate.from_template(RERANKING_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()

        try:
            response_str = await chain.ainvoke({
                "user_profile_summary": user_profile_summary,
                "intent_str": intent_str,
                "home_base_str": home_base_str,
                "candidates_str": candidates_str
            })

            result_json = _extract_and_parse_json(response_str)
            llm_ranked_destinations = result_json.get("ranked_destinations", [])
            if not llm_ranked_destinations:
                raise ValueError("LLM did not return ranked_destinations.")

            final_recommendations = []
            for dest in llm_ranked_destinations:
                city_name_from_llm = dest.get('city')
                if not city_name_from_llm: continue

                city_info_rows = candidates_df[candidates_df['city_name'].str.lower() == city_name_from_llm.lower()]
                if city_info_rows.empty:
                    logging.warning(f"LLM returned city '{city_name_from_llm}' not in candidates. Skipping.")
                    continue
                city_info = city_info_rows.iloc[0]

                pois = self.select_city_pois(city_info['city_id'], np.array(user_profile_vector))
                final_recommendations.append({
                    "city": city_info['city_name'], "country": city_info['country'],
                    "justification": dest.get('justification', 'A great match for your interests.'),
                    "pois": pois
                })
            return final_recommendations

        except Exception as e:
            logging.error(f"Error during LLM re-ranking ({e}). Falling back to top 5 initial candidates.",
                          exc_info=True)
            top_5_fallback = candidates_df.head(5).to_dict('records')
            for item in top_5_fallback:
                item['justification'] = "This destination is a strong match for your general interests."
                item['pois'] = self.select_city_pois(item['city_id'], np.array(user_profile_vector))
            return top_5_fallback


def _find_best_city_match(city_name: str, available_cities: list[str]) -> str | None:
    """
    Finds the best city match using a hybrid approach:
    1. Fuzzy matching (thefuzz) to find top candidates.
    2. Phonetic matching (jellyfish) to re-rank candidates and select the best one.
    This is more robust for cross-lingual names like Rome/Roma.
    """
    if not city_name or not available_cities:
        return None

    city_name_lower = city_name.lower().strip()

    # exact match
    for city in available_cities:
        if city.lower().strip() == city_name_lower:
            logging.info(f"Found exact match for city '{city_name}'.")
            return city

    # extract top 3 candidates thefuzz
    candidates = fuzzy_process.extract(city_name, available_cities, limit=3, scorer=fuzz.WRatio)

    if not candidates or candidates[0][1] < 70:
        best_guess = candidates[0][0] if candidates else "N/A"
        score = candidates[0][1] if candidates else 0
        logging.warning(
            f"No confident fuzzy match for '{city_name}'. Best guess '{best_guess}' had a very low score of {score}.")
        return None

    logging.info(f"Fuzzy candidates for '{city_name}': {candidates}")

    # reorder using phonetic matching
    best_candidate = None
    highest_combined_score = 0

    for candidate_name, fuzzy_score in candidates:
        max_dist = len(city_name) // 2
        lev_dist = jellyfish.levenshtein_distance(city_name_lower, candidate_name.lower())

        if lev_dist > max_dist and fuzzy_score < 90:
            logging.info(
                f"  - Candidate '{candidate_name}' rejected by Levenshtein veto (distance: {lev_dist} > max: {max_dist}).")
            continue
        # Match Rating Approach
        phonetic_match_score = 100 if jellyfish.match_rating_comparison(city_name, candidate_name) else 0

        # highweight phonetic match
        combined_score = (0.4 * fuzzy_score) + (0.6 * phonetic_match_score)

        logging.info(
            f"  - Candidate '{candidate_name}': Fuzzy={fuzzy_score}, Phonetic={phonetic_match_score}, Combined={combined_score:.2f}")

        if combined_score > highest_combined_score:
            highest_combined_score = combined_score
            best_candidate = candidate_name

    # final threshold
    if highest_combined_score > 65:
        logging.info(
            f"Selected '{best_candidate}' for '{city_name}' with a combined score of {highest_combined_score:.2f}.")
        return best_candidate
    else:
        logging.warning(
            f"No candidate for '{city_name}' passed the final combined score threshold of 65. "
            f"Best candidate was '{best_candidate}' with score {highest_combined_score:.2f}."
        )
        return None


def practicality_score(city_row, home_base_coords, duration=None):
    """
    Calculates a practicality score based on distance.
    This version is very punitive for long distances on short trips.
    """
    if home_base_coords is None:
        logging.warning("Home base coordinates not available, returning neutral practicality score (0.5).")
        return 0.5

    city_coords = (city_row.get("city_lat"), city_row.get("city_lon"))
    if not (pd.notna(city_coords[0]) and pd.notna(city_coords[1])):
        return 0.5  # Neutral score if city coords are invalid

    try:
        km = geodesic(home_base_coords, city_coords).km
    except Exception:
        logging.warning("Could not calculate geodesic distance.")
        return 0.5

    if duration == 'short':
        if km <= 400:
            return 1.0
        if km <= 1500:
            return 1.0 - 0.9 * ((km - 400) / (1500 - 400))
        else:
            return 0.01
    else:  # long trip
        return 1.0


async def generate_poi_llm_reason(user_profile_summary: str, poi_row, model_env_value: str):
    try:
        llm, _ = get_llm(model_env_value)
        prompt = ChatPromptTemplate.from_template(POI_REASON_GENERATION_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        reason = await chain.ainvoke({
            "user_profile_summary": user_profile_summary, "poi_name": poi_row['name'],
            "poi_category": poi_row['primary_category'], "city_name": poi_row['locality']
        })
        return reason.strip().replace('"', '')
    except Exception as e:
        logging.warning(f"Failed to generate LLM reason for POI {poi_row.get('name')}: {e}")
        return f"A highly-rated attraction in {poi_row.get('locality')}."
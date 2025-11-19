import numpy as np
import pandas as pd
from geopy.distance import geodesic
from .embeddings import cosine_sim
import math
from typing import Optional, Tuple, Union, Sequence
from sklearn.metrics.pairwise import cosine_similarity


def safe_distance_km(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[float]:
    """Return distance in km or None if not computable, checking for valid coordinates."""
    # safeguard for missing/invalid coordinates
    if a is None or b is None:
        return None
    try:
        # Ensure coordinates are valid floats
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        if not (pd.notna(ax) and pd.notna(ay) and pd.notna(bx) and pd.notna(by)):
            return None
    except (TypeError, ValueError, IndexError):
        return None

    try:
        return geodesic((ax, ay), (bx, by)).km
    except Exception:
        return None


def weekend_distance_weight(d: Optional[float]) -> float:
    """Piecewise distance weight tuned for EU/weekend scenarios.

    Returns a score in [0, 1]. None -> 0.5 (neutral).
    - <=600 km: high score (plateau)
    - 600–1500 km: moderate decay
    - >1500 km: stronger decay
    """
    if d is None or not pd.notna(d):
        return 0.5  # Neutral score for unknown distance

    d = float(d)
    if d <= 600:
        # High score for short distances (e.g., drivable or short flight)
        return 0.95
    elif d <= 1500:
        # Moderate exponential decay for medium distances
        return math.exp(-(d - 600) / 900.0) * 0.95
    else:
        # Stronger decay for long distances
        # Starts from the value at 1500km and decays faster
        val_at_1500 = math.exp(-(1500 - 600) / 900.0) * 0.95
        return val_at_1500 * math.exp(-(d - 1500) / 1000.0)


def rank_pois_by_relevance(
        candidate_pois_df: pd.DataFrame,
        user_profile_vector: Union[np.ndarray, list],
        user_location_coords: Optional[Tuple[float, float]] = None,
        top_k: int = 5,
        alpha_similarity: float = 0.7,
        alpha_distance: float = 0.3,
        use_mmr: bool = True,
        lambda_param: float = 0.6
) -> pd.DataFrame:
    """
    Rank POIs. Can use a simple weighted score or MMR for diversity.
    """
    if candidate_pois_df is None or candidate_pois_df.empty:
        return pd.DataFrame()

    df = candidate_pois_df.copy()
    user_vec = np.asarray(user_profile_vector, dtype=float)

    if 'embedding' not in df.columns:
        df['similarity_score'] = 0.0
    else:
        df['similarity_score'] = df['embedding'].apply(lambda e: cosine_sim(user_vec, e))

    if user_location_coords and 'latitude' in df.columns and 'longitude' in df.columns:
        df['distance_km'] = df.apply(
            lambda row: safe_distance_km(user_location_coords, (row.get('latitude'), row.get('longitude'))), axis=1)
        df['distance_score'] = df['distance_km'].apply(weekend_distance_weight)
    else:
        df['distance_km'] = None
        df['distance_score'] = 0.5

    df['final_score'] = alpha_similarity * df['similarity_score'].fillna(0.0) + alpha_distance * df[
        'distance_score'].fillna(0.5)

    # sort by final score descending
    df = df.sort_values('final_score', ascending=False).reset_index(drop=True)

    if not use_mmr or 'embedding' not in df.columns or len(df) <= top_k:
        return df.head(top_k)

    candidate_embeddings = np.stack(df['embedding'].values)

    # Normalize embeddings
    norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = candidate_embeddings / norms

    relevance_scores = df['final_score'].values

    selected_indices = []
    candidate_indices = list(range(len(df)))

    # add the most relevant item first
    best_initial_idx = candidate_indices[0]
    selected_indices.append(best_initial_idx)
    candidate_indices.pop(0)

    while len(selected_indices) < top_k and candidate_indices:
        best_next_idx = -1
        max_mmr_score = -np.inf

        # calculate embeddings of selected items
        selected_embs = normalized_embeddings[selected_indices]

        for idx in candidate_indices:
            relevance = relevance_scores[idx]
            candidate_emb = normalized_embeddings[idx].reshape(1, -1)

            # Similarity to selected set
            similarity_to_selected = cosine_similarity(candidate_emb, selected_embs).max()

            # MMR
            mmr_score = lambda_param * relevance - (1 - lambda_param) * similarity_to_selected

            if mmr_score > max_mmr_score:
                max_mmr_score = mmr_score
                best_next_idx = idx

        if best_next_idx != -1:
            selected_indices.append(best_next_idx)
            candidate_indices.remove(best_next_idx)
        else:
            break

    return df.iloc[selected_indices].reset_index(drop=True)
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------
# Candidate Retrieval
#
# Given:
#   - all venue embeddings
#   - (optionally) a user profile embedding / query embedding
#
# We find top-K nearest venues by cosine similarity.
#
# This is "stage 1" in a typical recommender:
#   1. retrieve a small set of candidates quickly (ANN / nearest neighbors)
#   2. pass them to a ranker that applies richer scoring (distance, recency, etc.)
# ---------------------------------------------------------------------


def _cosine_normalize(mat: np.ndarray) -> np.ndarray:
    """
    Normalize rows to unit length to enable cosine similarity
    via dot product.

    If a row is all zeros or NaN, we leave it zero.
    """
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_nn_index(emb_df: pd.DataFrame) -> tuple[NearestNeighbors, np.ndarray, np.ndarray]:
    """
    Build a brute-force NearestNeighbors index over venue embeddings.

    Parameters
    ----------
    emb_df : DataFrame
        Must contain:
        - 'venue_id'
        - 'embedding' (list[float])

    Returns
    -------
    (nn_model, item_vecs, venue_ids)
        nn_model  : fitted NearestNeighbors
        item_vecs : np.ndarray of shape [num_items, dim]
        venue_ids : np.ndarray of shape [num_items]
    """
    venue_ids = emb_df["venue_id"].to_numpy()
    item_vecs = np.stack(emb_df["embedding"].to_numpy()).astype("float32")

    item_vecs = _cosine_normalize(item_vecs)

    nn = NearestNeighbors(
        n_neighbors=50,        # default K to build index; you can override at query time
        algorithm="auto",
        metric="cosine"
    )
    nn.fit(item_vecs)

    return nn, item_vecs, venue_ids


def retrieve_similar(
    query_vec: np.ndarray,
    nn_index: NearestNeighbors,
    item_vecs: np.ndarray,
    venue_ids: np.ndarray,
    top_k: int = 50,
) -> pd.DataFrame:
    """
    Retrieve the top-K most similar venues to a query vector.

    Parameters
    ----------
    query_vec : np.ndarray
        Shape [dim]. This should already be normalized.
    nn_index : NearestNeighbors
        Output of build_nn_index.
    item_vecs : np.ndarray
        Same array used to build the index.
    venue_ids : np.ndarray
        Parallel array of venue IDs.
    top_k : int
        How many candidates to return.

    Returns
    -------
    DataFrame with:
      - venue_id
      - distance  (cosine distance from sklearn)
      - score     (cosine similarity = 1 - distance)
    """
    q = query_vec.reshape(1, -1)
    # sklearn returns distances (0 = perfect match), not similarities
    distances, idxs = nn_index.kneighbors(q, n_neighbors=top_k, return_distance=True)

    distances = distances[0]
    idxs = idxs[0]

    sims = 1.0 - distances  # cosine similarity

    return pd.DataFrame({
        "venue_id": venue_ids[idxs],
        "distance": distances,
        "score": sims
    }).reset_index(drop=True)

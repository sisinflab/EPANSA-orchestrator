import os
import json
import logging
import pandas as pd


# ---------------------------------------------------------------------
# Basic logging setup
# ---------------------------------------------------------------------
# We centralize logging here so every module can import `log`
# and get consistent logging behavior.
log = logging.getLogger("epansa")
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """
    Create the directory `path` if it does not already exist.
    Works recursively (like `mkdir -p`).

    Parameters
    ----------
    path : str
        Directory path.
    """
    os.makedirs(path, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save a pandas DataFrame to a Parquet file, creating
    parent dirs if needed.
    """
    ensure_dir(os.path.dirname(path) or ".")
    df.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """
    Load a pandas DataFrame from Parquet.
    """
    return pd.read_parquet(path)


def append_jsonl(path: str, rec: dict) -> None:
    """
    Append a single JSON object (one line) to a .jsonl file.
    Creates parent directories if needed.

    This is used for:
    - caching fetched POI metadata
    - caching generated user profiles (LLM output, etc.)

    Parameters
    ----------
    path : str
        Output file path.
    rec : dict
        The record to append.
    """
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """
    Read an entire .jsonl file into memory as a list of dicts.

    Returns [] if the file does not exist.
    """
    if not os.path.exists(path):
        return []

    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning(f"Skipping malformed JSONL line in {path}")
                continue
    return out

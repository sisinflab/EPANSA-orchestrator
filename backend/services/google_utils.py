from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def load_json(p: Path, default):
    """
    Load a JSON file from disk and return its contents.

    Behavior
    --------
    - Returns `default` if the file does not exist or contains invalid JSON.
    - Logs unexpected errors and still returns `default`.
    - This helper is intentionally forgiving to avoid surfacing non-critical
      file issues to callers.
    """
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.debug(f"Could not load JSON from {p}: {e}. Returning default.")
        return default
    except Exception as e:
        logger.warning(f"Unexpected error loading JSON from {p}: {e}")
        return default

def save_json(p: Path, data):
    """
    Persist a Python object as pretty-printed JSON to disk.

    Notes
    -----
    - Ensures the parent directory exists before writing.
    - Uses UTF-8 encoding and stable indentation for human-readable diffs.
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def iso_to_date_time(iso_str: Optional[str]) -> Tuple[str, str]:
    """
    Convert an ISO 8601 string into separate date and time strings.

    Returns
    -------
    Tuple[str, str]
        A `(date_str, time_str)` pair where:
        - `date_str` is formatted as `DD-Mon-YYYY` (strftime `%d-%b-%Y`)
        - `time_str` is formatted as `HH:MM` (24-hour)
        If parsing fails or input is missing, returns `("", "")`.

    Caveat
    ------
    `%b` (month abbreviation) is locale-dependent. This function preserves the
    existing behavior and does not enforce a specific locale.
    """
    if not iso_str:
        return "", ""
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00')).astimezone(timezone.utc)
        return dt.strftime("%d-%b-%Y"), dt.strftime("%H:%M")
    except (ValueError, TypeError):
        logger.warning(f"Could not parse ISO date string: {iso_str}")
        return "", ""

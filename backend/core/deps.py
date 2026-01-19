from __future__ import annotations
from contextlib import contextmanager
from typing import Iterator, Optional, Dict, Any
import re
import logging
from neo4j import GraphDatabase, Session
from fastapi import Request
from backend.core.config import settings

# initialize neo4j driver
_driver = None

# In-memory cache of databases
_db_cache = set()

logger = logging.getLogger(__name__)

LABELS_WITH_UNIQUES = [
    "General",
    "User",
    "Event",
    "Contact",
    "Alarm",
    "Note",
    "PhoneCall",
    "Document",
    "Photo",
    "Counter",
]


def _get_driver():
    """
    Return a process-wide Neo4j driver instance.
    The driver manages connection pools under the hood and should be reused.
    """
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
    return _driver


def slug_db_name(user_id: str) -> str:
    """
    Build a per-user database name from a user identifier, enforcing:
      - Lowercase
      - Alphanumeric and underscore only (other chars mapped to '-')
      - Prefixed with 'u-' for consistency and to avoid reserved names.

    This creates a stable tenant DB name like 'user-1', 'user-2', etc.
    """
    s = re.sub(r"[^a-zA-Z0-9_]+", "-", user_id).lower()
    return f"user-{s}"


def get_user_slug(user_id: str) -> str:
    """
    Returns the slugged user identifier suitable for disk paths (e.g., 'user-1').
    Uses the same logic as slug_db_name for consistency.
    """
    return slug_db_name(user_id)


def ensure_user_database(
    db_name: str, user_details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Create the per-user database if it does not exist, apply model constraints,
    and optionally ensure the User node exists if details are provided.
    """
    if db_name in _db_cache:
        if user_details:
            with get_neo4j(database=db_name) as session:
                _create_user_node(session, user_details)
        return

    drv = _get_driver()
    with drv.session(database="system") as s:
        s.run(f"CREATE DATABASE `{db_name}` IF NOT EXISTS").consume()

    ensure_constraints(database=db_name)

    # create user node when login
    if user_details:
        with get_neo4j(database=db_name) as session:
            _create_user_node(session, user_details)

    _db_cache.add(db_name)


def _create_user_node(session: Session, user_details: Dict[str, Any]):
    """
    Create user node in the form of 'Name_ID'.
    """
    if not user_details or user_details.get("id") is None:
        logger.warning(
            "Skipping user node creation due to missing user_details or user id."
        )
        return

    name_str = user_details.get("name") or ""

    safe_name = re.sub(r"[^a-zA-Z0-9]", "", name_str)

    parts = [p for p in [safe_name] if p]  # Filter empy parts
    parts.append(str(user_details["id"]))
    username = "_".join(parts)

    query = """
    MERGE (u:User {userId: $userId})
    ON CREATE SET
        u.googleId = $googleId,
        u.email = $email,
        u.name = $name,
        u.username = $username
    ON MATCH SET
        u.email = $email,
        u.name = $name,
        u.username = $username
    """
    params = {
        "userId": user_details["id"],
        "googleId": user_details.get("google_id"),
        "email": user_details.get("email"),
        "name": user_details.get("name"),
        "username": username,
    }

    session.run(query, params).consume()
    logger.info(f"User node ensured with display_name={username}")


def ensure_constraints(database: Optional[str] = None) -> None:
    """
    Ensure we have the required constraints for idempotent upserts:
    - For each label (except Counter): UNIQUE(n.id) and UNIQUE(n.source_id)
    - For :Counter: UNIQUE(c.name)

    This function is safe to call repeatedly (IF NOT EXISTS).
    """
    with get_neo4j(database=database) as s:
        cy_id = (
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE (n.id) IS UNIQUE"
        )
        cy_src = "CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE (n.source_id) IS UNIQUE"

        for label in [lbl for lbl in LABELS_WITH_UNIQUES if lbl != "Counter"]:
            s.run(cy_id.format(label=label)).consume()
            s.run(cy_src.format(label=label)).consume()

        # Counter keyed by `name`
        cy_name = (
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Counter) REQUIRE (c.name) IS UNIQUE"
        )
        s.run(cy_name).consume()

    logger.info(f"Constraints ensured for database '{database}'")


def get_request_db(request: Request) -> str:
    """
    Resolve the active tenant database from the current request.
    Falls back to deriving from JWT claims if middleware has not pre-populated state.
    """
    ui = getattr(request.app.state, "user_info", None) or {}
    user_id = ui.get("user_id") or ui.get("sub") or "anonymous"
    db_name = slug_db_name(user_id)
    ensure_user_database(db_name)
    return db_name


@contextmanager
def get_neo4j(
    database: Optional[str] = None, request: Optional[Request] = None
) -> Iterator[Session]:
    """
    Context manager yielding a Neo4j `Session` bound to the correct database.

    Parameters
    ----------
    database : str, optional
        Explicit database name. If omitted and `request` is provided, it is
        computed from the request JWT; otherwise falls back to Settings.NEO4J_DATABASE.
    request : Request, optional
        FastAPI request used to compute a tenant database when `database` is not given.

    Usage
    -----
    with get_neo4j(request=request) as session:
        session.run("MATCH (n) RETURN count(n)")
    """
    drv = _get_driver()
    if request and not database:
        database = get_request_db(request)
    database = database or settings.NEO4J_DATABASE
    session: Session = drv.session(database=database)
    try:
        yield session
    finally:
        session.close()


def close_neo4j():
    """
    Close the shared driver. Typically called during application shutdown.
    """
    global _driver
    if _driver:
        _driver.close()
        _driver = None

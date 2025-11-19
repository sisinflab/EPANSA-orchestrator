from __future__ import annotations
from typing import Dict, Any
from fastapi import Depends, Request, HTTPException, status

from backend.core.jwt_auth import JWTAuthFilter
from backend.core.deps import slug_db_name, ensure_user_database

# Singleton validator used by the dependency to avoid re-instantiation on every request.
_jwt = JWTAuthFilter()


def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Dependency-style function that returns the authenticated user's JWT claims.

    Behavior
    --------
    - If middleware has already validated the token, reuse `request.app.state.user_info`
      and `request.app.state.db_name` (fast path).
    - Otherwise, validate the token here and derive the tenant database.
    - Ensures the per-user database exists and attaches it to `request.app.state.db_name`.

    Returns
    -------
    dict
        The decoded JWT payload to be injected into route handlers.

    Raises
    ------
    HTTPException
        401 if the JWT is missing/invalid.
    """
    ui = getattr(request.app.state, "user_info", None)
    db_name = getattr(request.app.state, "db_name", None)

    if ui and db_name:
        return ui

    try:
        ui = _jwt.validate_request(request)
    except HTTPException as e:
        # Preserve the original error code/details for correct client handling.
        raise e
    except Exception:
        # Maintain a uniform 401 on unexpected validation errors.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing JWT token")

    user_id = ui.get("user_id") or ui.get("sub") or "anonymous"
    db_name = slug_db_name(user_id)
    ensure_user_database(db_name)

    request.app.state.user_info = ui
    request.app.state.db_name = db_name
    return ui


def jwt_dependency(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Simple wrapper to use with FastAPI routes:

        @app.get("/me")
        def whoami(user = Depends(jwt_dependency)):
            return {"sub": user.get("sub")}

    Keeping this alias makes endpoint signatures succinct and uniform.
    """
    return user

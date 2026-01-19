from __future__ import annotations
from fastapi import status
import jwt
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time

from backend.core.deps import slug_db_name, ensure_user_database
from backend.services.user_service import user_service

logger = logging.getLogger(__name__)

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "insecure-default-key")


class JWTService:
    def create_access_token(self, subject: str) -> str:
        """Create a JWT for a specific user."""
        payload = {
            "sub": subject,
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(days=7),
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
        return token


jwt_service = JWTService()


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware responsible for:
      1) Validating the inbound Bearer JWT (via JWTAuthFilter).
      2) Deriving a per-user Neo4j database name (multi-tenant isolation).
      3) Ensuring that database exists and constraints are applied on first access.
      4) Storing `user_info` and `db_name` on `request.app.state` for downstream routes.

    Notes
    -----
    - Paths listed in `excluded_paths` are bypassed for auth to allow health checks,
      docs, and internal sync endpoints to function without tokens.
    - Any exception during validation returns a 401 with a short JSON body; no stack
      trace is leaked to clients.
    """

    def __init__(self, app, excluded_paths=None):
        super().__init__(app)
        self.jwt_auth = JWTAuthFilter()
        # Keep this list narrowly scoped to endpoints that must be publicly reachable.
        self.excluded_paths = excluded_paths or [
            "/api/health",
            "/api/connections/count",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
            "/healthz",
            "/sync/internal",
            "/auth/google/exchange_code",
            "/auth/google/authorize",
        ]
        logger.info(
            f"JWT Auth Middleware initialized with excluded paths: {self.excluded_paths}"
        )

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Skip authentication for explicitly whitelisted routes or any `/public/*` assets.
        if (
            any(request.url.path.startswith(path) for path in self.excluded_paths)
            or "/public" in request.url.path
        ):
            logger.debug(f"Skipping authentication for: {request.url.path}")
            return await call_next(request)

        try:
            # Decode/validate the JWT and expose the claims to the request state.
            user_info = self.jwt_auth.validate_request(request)
            request.app.state.user_info = user_info

            user_id_str = user_info.get("sub")
            if not user_id_str:
                raise HTTPException(
                    status_code=401, detail="User ID (sub) missing from token"
                )

            # Fetch user record from the database to ensure they exist.
            try:
                user_id_int = int(user_id_str)
                user_record = user_service.get_user_by_id(user_id_int)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=401, detail="Invalid User ID format in token"
                )

            if not user_record:
                # valid JWT but no corresponding user in our system
                logger.warning(
                    f"JWT validation successful but no user found for id={user_id_str}"
                )
                raise HTTPException(status_code=401, detail="User not found")

            # create user_details dict for database initialization
            user_details = {
                "id": user_record.id,
                "google_id": user_record.google_id,
                "email": user_record.email,
                "name": user_record.name,
            }

            # calculate db name and ensure it exists
            db_name = slug_db_name(user_id_str)
            ensure_user_database(db_name, user_details=user_details)
            request.app.state.db_name = db_name

            username = user_record.name or "unknown"
            logger.info(f"Auth OK user={username} id={user_id_str} db={db_name}")

            response = await call_next(request)

            duration = time.time() - start_time
            logger.debug(
                f"{request.method} {request.url.path} completed in {duration:.3f}s"
            )
            return response

        except HTTPException as e:
            # Expected auth failures propagate as structured 4xx responses.
            logger.warning(
                f"JWT Authentication failed for {request.url.path}: {e.detail}"
            )
            return JSONResponse(
                status_code=e.status_code, content={"message": e.detail}
            )
        except Exception as e:
            # Do not leak internal errors; respond with a generic 401 to avoid oracle behavior.
            logger.error(
                f"Unexpected error during JWT authentication for {request.url.path}: {str(e)}"
            )
            return JSONResponse(
                status_code=401, content={"message": "Authentication failed"}
            )


# --- QUICKSTART ---
# 1) Install cryptographic dependencies:
#    pip install "pyjwt[crypto]"
# 2) Configure a signing secret (symmetric) or switch to asymmetric keys:
#    export JWT_SECRET="your-prod-grade-random-key"
#
# This module intentionally keeps validation minimal (alg + signature + exp).
# In production, you should validate additional claims such as:
#   - iss (issuer)     : trusted identity provider
#   - aud (audience)   : your API identifier
#   - iat/nbf/exp      : issue/start/expiry times
#   - jti              : token ID (optional replay defense via denylist)
# And prefer JWKS with RS256/ES256 to avoid secret sharing across services.


class JWTAuthFilter:
    """
    Thin wrapper around PyJWT to validate Bearer tokens from incoming requests.

    Attributes
    ----------
    secret_key : str
        Symmetric key for HS256 signature verification. In production, consider
        loading a JWKS and using RS256/ES256 public keys instead.
    algorithm : str
        The expected JWT algorithm (default: HS256).
    """

    def __init__(self):
        # In a real-world app, inject a JWKS loader or use your IdP library.
        self.secret_key = os.getenv("JWT_SECRET")
        if not self.secret_key:
            # Using a hard-coded default is unsafe in production. This warning is explicit.
            print(
                "Warning: JWT_SECRET environment variable not set. Using a default insecure key."
            )
            self.secret_key = "insecure-default-key"

        self.algorithm = "HS256"

    def validate_request(self, request: Request) -> dict:
        """
        Validate the Authorization header and decode the JWT.

        Returns
        -------
        dict
            The decoded JWT payload (claims).

        Raises
        ------
        HTTPException
            401 errors for missing/invalid headers, expired tokens, or signature failures.
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header is missing",
            )

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format",
            )

        token = parts[1]
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

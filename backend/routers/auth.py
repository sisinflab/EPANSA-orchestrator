from fastapi import APIRouter, Body, HTTPException, status
import logging

from backend.services import google_auth_service
from backend.services.user_service import user_service
from backend.core.jwt_auth import jwt_service
from backend.core.deps import slug_db_name, ensure_user_database
from backend.services.secure_store import credentials_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/google/exchange_code")
async def exchange_google_auth_code(
    auth_code: str = Body(..., embed=True, description="The serverAuthCode returned by Google Sign-In")
):
    """
    Exchange a Google authorization code, upsert the local user, persist the refresh
    token, and return an application JWT.

    Workflow
    --------
    1) Exchange the authorization code for Google credentials and fetch the user profile.
    2) Get or create the local user record using the Google subject (`sub`).
    3) Derive the per-user database name and ensure it exists.
    4) Store the Google refresh token bound to the local user id.
    5) Issue and return an application access token (JWT).
    """
    logger.info("Received Google code exchange request.")

    try:
        # Exchange code for tokens and retrieve the Google user profile.
        google_creds, user_profile = await google_auth_service.exchange_code_and_get_profile(auth_code)

        google_user_id = user_profile.get("sub")
        if not google_user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to extract user ID (sub) from Google profile."
            )

        # Get or create local user from Google identity.
        user = user_service.get_or_create_user_by_google_id(
            google_id=google_user_id,
            email=user_profile.get("email"),
            name=user_profile.get("name"),
        )

        # Local numeric user id (e.g., 1, 2, 3…).
        user_id_app = user.id

        # Build database name from local user id and ensure DB is provisioned.
        db_name = slug_db_name(str(user_id_app))
        user_details = {
            "id": user.id,           # Keep as integer for Neo4j node property.
            "google_id": user.google_id,
            "email": user.email,
            "name": user.name,
        }
        ensure_user_database(db_name, user_details=user_details)

        # 4) Persist the Google refresh token bound to the local user id.
        credentials_service.save_refresh_token(str(user_id_app), google_creds.refresh_token)

        # 5) Mint and return the application JWT for subsequent requests.
        app_jwt = jwt_service.create_access_token(subject=str(user_id_app))

        logger.info(f"Code exchange completed successfully for app_user_id={user_id_app}")
        return {"access_token": app_jwt, "token_type": "bearer"}

    except HTTPException:
        # Re-raise HTTP errors untouched so FastAPI preserves status and detail.
        raise
    except Exception as e:
        logger.error(f"Unexpected error during Google code exchange: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during authentication."
        )

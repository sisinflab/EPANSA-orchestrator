from fastapi import HTTPException
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
import requests
import logging

from backend.core.config import settings
from backend.services.secure_store import credentials_service
from google.auth.transport.requests import Request as GoogleRequest

logger = logging.getLogger(__name__)

# Centralized list of OAuth scopes used across Google integrations.
# Keeping these aligned avoids scope mismatch between consent and runtime refresh.
GOOGLE_OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/photoslibrary.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/contacts.readonly",
]


async def exchange_code_and_get_profile(auth_code: str) -> tuple[Credentials, dict]:
    """
    Exchange an OAuth2 authorization code for tokens and retrieve the user profile.

    Workflow
    --------
    1) Exchange the authorization `auth_code` for access and refresh tokens.
    2) Use the access token to fetch the user's profile via Google's UserInfo endpoint.
    3) Return both the `Credentials` object and the decoded user profile JSON.

    Parameters
    ----------
    auth_code : str
        The authorization code received from the Google OAuth consent screen.

    Returns
    -------
    tuple[Credentials, dict]
        A `(creds, profile)` pair with the authorized credentials and user information.
    """
    if not settings.GOOGLE_CLIENT_SECRETS_FILE:
        raise HTTPException(
            status_code=500, detail="Google client secrets file not configured."
        )

    try:
        flow = Flow.from_client_secrets_file(
            settings.GOOGLE_CLIENT_SECRETS_FILE,
            scopes=GOOGLE_OAUTH_SCOPES,
            redirect_uri=settings.GOOGLE_REDIRECT_URI,
        )
        flow.fetch_token(code=auth_code)
    except Exception as e:
        logger.error(f"Failed to fetch Google token: {e}")
        raise HTTPException(status_code=400, detail=f"Token exchange failed: {e}")

    creds = flow.credentials
    if not creds.refresh_token:
        raise HTTPException(
            status_code=400,
            detail=(
                "No refresh token received from Google. "
                "Ensure the user has granted offline access."
            ),
        )

    # Use the obtained credentials to query the OpenID Connect userinfo endpoint.
    try:
        userinfo_endpoint = "https://www.googleapis.com/oauth2/v3/userinfo"
        headers = {"Authorization": f"Bearer {creds.token}"}
        user_profile_response = requests.get(userinfo_endpoint, headers=headers)
        user_profile_response.raise_for_status()
        user_profile = user_profile_response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to retrieve Google user profile: {e}")
        raise HTTPException(
            status_code=500, detail="Unable to retrieve user profile from Google."
        )

    return creds, user_profile


# ----------------- Runtime credential retrieval & refresh ---------------------


def get_credentials_for_user(user_id: str) -> Credentials:
    """
    Build a valid `Credentials` object for a given user and refresh it if needed.

    Workflow
    --------
    - Retrieve the stored refresh token from the secure store.
    - Construct a transient `Credentials` object using client ID/secret.
    - Refresh the access token if expired or invalid.
    - Return the ready-to-use credentials for authorized API calls.

    Parameters
    ----------
    user_id : str
        Unique user identifier (as used in the refresh token storage).

    Returns
    -------
    Credentials
        Active and refreshed Google credentials for API access.

    Raises
    ------
    ValueError
        If no stored refresh token is found for the given user.
    RuntimeError
        If the refresh attempt fails (e.g., token revoked or expired).
    """
    refresh_token = credentials_service.get_refresh_token(user_id)
    if not refresh_token:
        raise ValueError(
            f"No stored credentials found for user {user_id}. "
            "The user must log in and grant offline access first."
        )

    info = {
        "refresh_token": refresh_token,
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    creds = Credentials.from_authorized_user_info(info, scopes=GOOGLE_OAUTH_SCOPES)

    try:
        if (creds.expired or not creds.valid) and creds.refresh_token:
            creds.refresh(GoogleRequest())
    except Exception as e:
        logger.error(f"Failed to refresh Google token for user {user_id}: {e}")
        raise RuntimeError(
            f"Could not refresh token for user {user_id}. It may have been revoked."
        )

    return creds

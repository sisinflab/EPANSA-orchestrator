import jwt
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Dev helper: generate a short-lived HS256 JWT that matches backend expectations.
# Security note: This script is for LOCAL TESTING only; never ship it in prod images.

# Load env (JWT_SECRET). Environment variables still have priority.
load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "insecure-default-key")
USER_ID_FOR_TEST = "test_user_2"  # Deterministic subject for easy debugging

# Minimal claims set; the API primarily relies on `sub` (subject)
payload = {
    "sub": USER_ID_FOR_TEST,
    "name": "Test User",
    "iat": datetime.now(timezone.utc),
    "exp": datetime.now(timezone.utc) + timedelta(hours=24),
}

# Generate HS256 token
token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

print("--- YOUR TEST JWT TOKEN ---")
print(token)
print("\nThis token is valid for 24 hours.")
print(f"The user id (sub) is: {USER_ID_FOR_TEST}")

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)


# In backend/core/config.py

import os
import json
import logging
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)

logger = logging.getLogger(__name__)

class Settings:
    """
    Centralized configuration holder.
    Always read settings via this class instead of calling os.getenv directly,
    so you keep a single source of truth and consistent defaults.
    """

    def __init__(self):
        # ---- App / HTTP ----
        self.PORT = int(os.getenv("PORT", "8000"))

        # ---- Neo4j connection ----
        self.NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
        self.NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

        # ---- Google OAuth / APIs ----
        self.GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
        self.GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
        self.GOOGLE_CLIENT_SECRETS_FILE = os.getenv("GOOGLE_CLIENT_SECRETS_FILE", "client_secret.json")
        self.GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost")

        # If ID/Secret are not set via ENV, load them from the JSON file.
        if not self.GOOGLE_CLIENT_ID and os.path.exists(self.GOOGLE_CLIENT_SECRETS_FILE):
            try:
                with open(self.GOOGLE_CLIENT_SECRETS_FILE, "r") as f:
                    data = json.load(f)
                    # Supports both 'web' and 'installed' app types
                    info = data.get("web") or data.get("installed", {})
                    self.GOOGLE_CLIENT_ID = info.get("client_id")
                    self.GOOGLE_CLIENT_SECRET = info.get("client_secret")
                    logger.info("Loaded Google credentials from client_secret.json")
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load Google credentials from {self.GOOGLE_CLIENT_SECRETS_FILE}: {e}")

        # ---- Storage paths ----
        self.STORAGE_BASE_PATH = os.getenv("STORAGE_BASE_PATH", "/app/data")
        self.TMP_DIR = os.getenv("TMP_DIR", f"{self.STORAGE_BASE_PATH}/tmp")
        self.SYNC_DIR = os.getenv("SYNC_DIR", f"{self.STORAGE_BASE_PATH}/sync")

        # ---- Secrets / secure storage ----
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        self.ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
        self.ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000")

        # ---- Recommender System Paths ----
        self.RECOMMENDER_DATA_PATH = os.getenv("RECOMMENDER_DATA_PATH", "/app/data/recommender/processed")
        self.VENUE_FEATURES = os.path.join(self.RECOMMENDER_DATA_PATH, "venue_features.parquet")
        self.VENUE_EMBEDDINGS = os.path.join(self.RECOMMENDER_DATA_PATH, "venue_embeddings.parquet")


# Instantiate the class to create the settings object
settings = Settings()

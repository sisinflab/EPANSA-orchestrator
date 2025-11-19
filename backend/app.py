from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Core config & security ---
from backend.core.config import settings
from backend.core.jwt_auth import JWTAuthMiddleware
from backend.core.deps import close_neo4j

# --- Routers ---
# Auth endpoints: Google auth code exchange + storage of refresh tokens
from backend.routers import auth as auth_router
# Functional endpoints: contacts/events/notes/alarms/phone calls + photo upload (client)
from backend.routers import routers as functional_router
# Runtime sync endpoints: run sync now for me / a user / all users
from backend.routers import sync_runtime as sync_runtime_router
from backend.services.scheduler import start_api_scheduler, stop_api_scheduler
from backend.services.secure_store import create_credentials_table
from backend.services.user_service import create_user_table
from libs.llm_graph_builder.src.shared.common_fn import load_embedding_model
from backend.services.recommender.build_poi_data import build_poi_data_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    This replaces the deprecated `on_event("startup")` and `on_event("shutdown")`.
    """
    # --- Code to run on startup ---
    logger = logging.getLogger(__name__)
    try:
        logger.info("--- Checking for recommender POI data before startup ---")
        build_poi_data_pipeline()
        logger.info("--- POI data check complete. ---")
    except FileNotFoundError as e:
        logger.error(f"FATAL: Could not start application due to missing recommender data. {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during POI data build: {e}", exc_info=True)
    try:
        create_credentials_table()
        create_user_table()
        start_api_scheduler()
        logger.info("Application startup complete. In-process scheduler started.")
    except Exception as e:
        logger.warning(f"Scheduler not started during startup: {e}")

    yield


    try:
        stop_api_scheduler()
        logger.info("In-process scheduler stopped.")
        close_neo4j()
        logger.info("Neo4j driver closed.")
    except Exception as e:
        logger.error(f"Error during application shutdown: {e}")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application for EPANSA.

    Includes:
    - Lifespan manager for startup/shutdown tasks;
    - JWT auth middleware (populates request.app.state.user_info / db_name);
    - CORS (tunable via env);
    - Routers: auth, functional, runtime sync;
    - Optional background scheduler to trigger Google sync periodically.
    """
    app = FastAPI(
        title="EPANSA Orchestrator",
        version="1.7.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan  # Register the new lifespan manager
    )

    # --- Logging baseline ---
    logging.basicConfig(
        level=getattr(logging, ("INFO").upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    # --- CORS (relax/tighten via env) ---
    origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- JWT middleware ---
    app.add_middleware(JWTAuthMiddleware)

    # --- Routers ---
    app.include_router(auth_router.router, prefix="", tags=["Auth"])
    app.include_router(functional_router.router, prefix="", tags=["Functional"])
    app.include_router(sync_runtime_router.router, prefix="", tags=["Sync"])
    app.state.embedding_model, app.state.embedding_dimension = load_embedding_model()

    # --- Healthcheck ---
    @app.get("/healthz", tags=["Meta"])
    def healthz():
        return {"ok": True, "service": "epansa-api", "version": "1.7.0"}

    return app


# Uvicorn entrypoint expects an `app` object at module level
app = create_app()
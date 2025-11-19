from __future__ import annotations
import logging
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler

from backend.services.secure_store import credentials_service
from backend.routers.sync_runtime import _run_all
from backend.core.deps import slug_db_name
from libs.llm_graph_builder.src.shared.common_fn import load_embedding_model

logger = logging.getLogger(__name__)
_scheduler: BackgroundScheduler | None = None


class FakeRequest:
    def __init__(self, uid: str):
        self.app = type("App", (), {})()
        self.app.state = type("State", (), {})()

        self.app.state.user_info = {"sub": uid}
        self.app.state.db_name = slug_db_name(uid)

        emb_model, emb_dim = load_embedding_model()
        self.app.state.embedding_model = emb_model
        self.app.state.embedding_dimension = emb_dim

def start_api_scheduler():
    global _scheduler
    if _scheduler:
        return _scheduler

    def job():
        try:
            user_ids = credentials_service.get_all_user_ids()
            if not user_ids:
                logger.info("[SCHED] No users with credentials found. Skipping sync job.")
                return

            logger.info(f"[SCHED] Running periodic Google sync for {len(user_ids)} users")
            for uid in user_ids:
                try:
                    logger.info(f"[SCHED] Starting sync for user: {uid}")
                    result = asyncio.run(_run_all(FakeRequest(uid), backfill=False))
                    logger.info(f"[SCHED] Sync result for user {uid}: {result}")
                    logger.info(f"[SCHED] Sync completed for user: {uid}")
                except Exception as user_sync_error:
                    logger.error(f"[SCHED] Error during sync for user {uid}: {user_sync_error}", exc_info=True)

        except Exception as e:
            logger.exception(f"[SCHED] Sync job failed: {e}")

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(job, "interval", seconds=100, id="periodic_google_sync", replace_existing=True)
    _scheduler.start()
    logger.info("[SCHED] BackgroundScheduler started (every 10m)")
    return _scheduler

def stop_api_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
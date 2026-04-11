"""
mall-ai: FastAPI-based AI Recommendation Service
Runs on port 8001 by default.

Start:
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload

Or:
    python main.py
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.engine import engine
from core.text_moderator import text_moderator
from config import settings, MODERATION_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    logger.info("Loading recommendation model...")
    engine.load()
    logger.info("Loading moderation model...")
    text_moderator.load(MODERATION_MODEL_PATH)
    yield
    # Shutdown
    logger.info("Shutting down mall-ai service.")


app = FastAPI(
    title="Mall AI Recommendation Service",
    description="Product recommendation engine using Collaborative Filtering + Content-Based approach",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)

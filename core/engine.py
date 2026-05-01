"""
Recommendation engine: loads trained model and serves predictions.
"""
import logging
import joblib
import numpy as np
from pathlib import Path

from config import settings, MODEL_PATH
from core.features import get_similar_products

logger = logging.getLogger(__name__)


class RecommendationEngine:
    def __init__(self):
        self._model_data: dict | None = None

    def load(self):
        if MODEL_PATH.exists():
            try:
                self._model_data = joblib.load(MODEL_PATH)
                logger.info("Recommendation model loaded from %s", MODEL_PATH)
            except Exception as exc:
                logger.error("Failed to load model: %s", exc)
                self._model_data = None
        else:
            logger.warning("Model file not found at %s. Run /retrain first.", MODEL_PATH)

    @property
    def is_ready(self) -> bool:
        return self._model_data is not None

    def recommend_for_user(self, user_id: str, limit: int = 12) -> list[str]:
        """
        Return top-K product IDs for a user.
        Uses SVD if available, falls back to content-based popularity.
        """
        if not self.is_ready:
            return []

        data = self._model_data
        cf_model = data.get("cf_model")
        product_ids: list[str] = data.get("product_ids", [])
        user_history: dict = data.get("user_history", {})

        viewed = set(user_history.get(user_id, []))
        candidates = [pid for pid in product_ids if pid not in viewed]

        # Only run SVD for users whose history was in the training data.
        # Cold-start users would otherwise get global-mean scores for every
        # candidate, making the sort order essentially random.
        is_known_user = user_id in user_history

        if cf_model is not None and hasattr(cf_model, "predict") and is_known_user:
            predictions = []
            for pid in candidates:
                try:
                    est = cf_model.predict(user_id, pid).est
                    predictions.append((pid, est))
                except Exception:
                    pass
            predictions.sort(key=lambda x: x[1], reverse=True)
            return [pid for pid, _ in predictions[:limit]]

        # Fallback: popularity order (product_ids is sorted by ratingAverage DESC)
        return candidates[:limit]

    def similar_products(
        self, product_id: str, limit: int = 8, exclude_ids: set[str] | None = None
    ) -> list[str]:
        """Return top-K similar product IDs using cosine similarity."""
        if not self.is_ready:
            return []

        data = self._model_data
        return get_similar_products(
            product_id=product_id,
            product_ids=data["product_ids"],
            similarity_matrix=data["similarity_matrix"],
            top_k=limit,
            exclude_ids=exclude_ids,
        )

    def reload(self):
        """Reload model from disk (called after retraining)."""
        self.load()


# Singleton engine instance
engine = RecommendationEngine()

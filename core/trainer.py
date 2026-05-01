"""
Training script: builds SVD collaborative filtering model,
content-based similarity matrix, co-purchase basket matrix,
and popularity scores. Saves all to disk as a single .pkl file.

Usage:
    python -m core.trainer
"""
import joblib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import cross_validate
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    logging.warning("scikit-surprise not installed. Falling back to basic collaborative filtering.")

from core.association import build_co_purchase_matrix
from core.data_loader import get_engine, load_interactions, load_product_features
from core.features import build_product_vectors, compute_similarity_matrix
from config import settings, MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_cf_model(interactions: pd.DataFrame):
    """Train SVD collaborative filtering model."""
    if not SURPRISE_AVAILABLE:
        logger.warning("Surprise not available — skipping CF model.")
        return None

    if len(interactions) < settings.min_interactions:
        logger.warning(
            "Not enough interactions to train CF model (%d rows, need %d).",
            len(interactions),
            settings.min_interactions,
        )
        return None

    reader = Reader(rating_scale=(interactions["score"].min(), interactions["score"].max()))
    data = Dataset.load_from_df(interactions[["userId", "productId", "score"]], reader)

    algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)

    # Quick cross-validate to log RMSE
    try:
        cv = cross_validate(algo, data, measures=["RMSE"], cv=2, verbose=False)
        logger.info("CF model CV RMSE: %.4f", np.mean(cv["test_rmse"]))
    except Exception as exc:
        logger.warning("Cross-validation failed: %s", exc)

    trainset = data.build_full_trainset()
    algo.fit(trainset)
    return algo


def train_and_save():
    logger.info("=" * 50)
    logger.info("Starting recommendation model training...")
    logger.info("=" * 50)

    # ── 1. Load interaction data ───────────────────────────────────────────
    logger.info("[1/5] Loading interactions from database...")
    interactions = load_interactions()
    logger.info("      Loaded %d (userId, productId) pairs.", len(interactions))

    # ── 2. Load product features ───────────────────────────────────────────
    logger.info("[2/5] Loading product features...")
    products = load_product_features()
    logger.info("      Loaded %d active products.", len(products))

    # ── 3. Collaborative Filtering (SVD) ───────────────────────────────────
    logger.info("[3/5] Training SVD collaborative filtering model...")
    cf_model = train_cf_model(interactions)
    if cf_model:
        logger.info("      SVD model trained successfully.")
    else:
        logger.warning("      SVD skipped — will use popularity fallback.")

    # ── 4. Content-Based: product vectors + similarity matrix ──────────────
    logger.info("[4/5] Building content-based similarity matrix...")
    product_matrix, product_ids, cat_ohe, brand_ohe, scaler = build_product_vectors(products)
    sim_matrix = compute_similarity_matrix(product_matrix)
    logger.info("      Similarity matrix: %s.", str(sim_matrix.shape))

    # ── 5. Market Basket: co-purchase matrix + popularity ──────────────────
    logger.info("[5/5] Building market basket & popularity index...")
    engine = get_engine()
    with engine.connect() as conn:
        co_purchase_matrix = build_co_purchase_matrix(conn)
    logger.info("      Co-purchase matrix: %d products have basket data.", len(co_purchase_matrix))

    # Popularity score = tổng score của mỗi sản phẩm trên toàn bộ users
    popularity_scores: dict[str, float] = (
        interactions.groupby("productId")["score"].sum().to_dict()
    )
    logger.info("      Popularity scores computed for %d products.", len(popularity_scores))

    # Build user history index với ĐIỂM SỐ (dùng cho trọng số Hybrid)
    # user_history = { userId: { productId: score, ... } }
    user_history: dict[str, dict[str, float]] = {}
    for uid, group in interactions.groupby("userId"):
        user_history[uid] = group.set_index("productId")["score"].to_dict()

    # Build product metadata index (dùng cho Hybrid re-ranking)
    # {productId: {"categoryId": "...", "brand": "..."}}
    product_meta: dict[str, dict] = {
        row["productId"]: {
            "categoryId": row.get("categoryId") or "",
            "brand":      (row.get("brand") or "").strip().lower(),
        }
        for _, row in products.iterrows()
    }

    # ── Lưu tất cả vào model_data ─────────────────────────────────────────
    model_data = {
        "cf_model":           cf_model,
        "product_matrix":     product_matrix,
        "product_ids":        product_ids,
        "similarity_matrix":  sim_matrix,
        "user_history":       user_history,
        "product_meta":       product_meta,        # ← MỚI: Hybrid re-ranking
        "co_purchase_matrix": co_purchase_matrix,
        "popularity_scores":  popularity_scores,
        "all_product_ids":    interactions["productId"].unique().tolist(),
        "encoders": {
            "category": cat_ohe,
            "brand":    brand_ohe,
            "scaler":   scaler,
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, MODEL_PATH)
    logger.info("=" * 50)
    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("=" * 50)
    return model_data


if __name__ == "__main__":
    train_and_save()

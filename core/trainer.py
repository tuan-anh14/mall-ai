"""
Training script: builds SVD collaborative filtering model
and content-based similarity matrix, then saves to disk.

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

from core.data_loader import load_interactions, load_product_features
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
        logger.warning("Not enough interactions to train CF model (%d rows).", len(interactions))
        return None

    reader = Reader(rating_scale=(interactions["score"].min(), interactions["score"].max()))
    data = Dataset.load_from_df(interactions[["userId", "productId", "score"]], reader)

    algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    # Quick cross-validate to log RMSE
    try:
        cv = cross_validate(algo, data, measures=["RMSE"], cv=2, verbose=False)
        logger.info("CV RMSE: %.4f", np.mean(cv["test_rmse"]))
    except Exception as exc:
        logger.warning("Cross-validation failed: %s", exc)

    trainset = data.build_full_trainset()
    algo.fit(trainset)
    return algo


def train_and_save():
    logger.info("Loading interactions from database...")
    interactions = load_interactions()
    logger.info("Loaded %d interaction rows.", len(interactions))

    logger.info("Loading product features...")
    products = load_product_features()
    logger.info("Loaded %d products.", len(products))

    # --- Collaborative Filtering ---
    cf_model = train_cf_model(interactions)

    # --- Content-Based ---
    product_matrix, product_ids, cat_enc, brand_enc, scaler = build_product_vectors(products)
    sim_matrix = compute_similarity_matrix(product_matrix)

    # Build user history index for fast recommendation
    user_history: dict[str, list[str]] = (
        interactions.groupby("userId")["productId"].apply(list).to_dict()
    )

    model_data = {
        "cf_model": cf_model,
        "product_matrix": product_matrix,
        "product_ids": product_ids,
        "similarity_matrix": sim_matrix,
        "user_history": user_history,
        "all_product_ids": interactions["productId"].unique().tolist(),
        "encoders": {"category": cat_enc, "brand": brand_enc, "scaler": scaler},
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    return model_data


if __name__ == "__main__":
    train_and_save()

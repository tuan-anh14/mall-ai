"""
Feature Engineering: transform product metadata into vectors
for content-based similarity search.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def build_product_vectors(products_df: pd.DataFrame):
    """
    Encode product features into a numeric matrix.
    Returns (matrix, product_ids).
    """
    df = products_df.copy().fillna({"brand": "unknown", "categoryName": "unknown"})

    cat_enc = LabelEncoder()
    brand_enc = LabelEncoder()

    df["cat_encoded"] = cat_enc.fit_transform(df["categoryName"].astype(str))
    df["brand_encoded"] = brand_enc.fit_transform(df["brand"].astype(str))

    scaler = MinMaxScaler()
    numeric = scaler.fit_transform(df[["price", "ratingAverage"]].fillna(0))

    matrix = np.column_stack([
        df["cat_encoded"].values,
        df["brand_encoded"].values,
        numeric,
    ])

    return matrix, df["productId"].tolist(), cat_enc, brand_enc, scaler


def compute_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all product pairs."""
    return cosine_similarity(matrix)


def get_similar_products(
    product_id: str,
    product_ids: list[str],
    similarity_matrix: np.ndarray,
    top_k: int = 8,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """Return top-K similar product IDs for a given product."""
    if product_id not in product_ids:
        return []

    idx = product_ids.index(product_id)
    scores = list(enumerate(similarity_matrix[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, _ in scores:
        pid = product_ids[i]
        if pid == product_id:
            continue
        if exclude_ids and pid in exclude_ids:
            continue
        results.append(pid)
        if len(results) >= top_k:
            break

    return results

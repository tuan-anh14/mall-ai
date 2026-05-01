"""
Feature Engineering: transform product metadata into vectors
for content-based similarity search.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def build_product_vectors(products_df: pd.DataFrame):
    """
    Encode product features into a numeric matrix.
    Uses one-hot encoding for categorical features (category, brand) so that
    cosine similarity is computed over semantically meaningful dimensions.
    Returns (matrix, product_ids, None, None, scaler).
    """
    df = products_df.copy().fillna({"brand": "unknown", "categoryName": "unknown"})

    # One-hot encode nominal categoricals — avoids spurious ordinal distances
    # that LabelEncoder would introduce (e.g. cat 3 ≈ cat 4 ≠ cat 1).
    cat_dummies = pd.get_dummies(df["categoryName"].astype(str), prefix="cat")
    brand_dummies = pd.get_dummies(df["brand"].astype(str), prefix="brand")

    scaler = MinMaxScaler()
    numeric = scaler.fit_transform(df[["price", "ratingAverage"]].fillna(0))

    matrix = np.hstack([
        cat_dummies.values.astype(float),
        brand_dummies.values.astype(float),
        numeric,
    ])

    return matrix, df["productId"].tolist(), None, None, scaler


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

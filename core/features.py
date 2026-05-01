"""
Feature Engineering: transform product metadata into vectors
for content-based similarity search.

Thay đổi so với v1:
- Dùng OneHotEncoder thay LabelEncoder cho categoryName và brand.
  LabelEncoder tạo số nguyên tuần tự (0,1,2...) khiến cosine similarity
  hiểu nhầm "Category 1 gần Category 2". OneHotEncoder tạo binary vector
  độc lập, đảm bảo không có quan hệ thứ tự giả tạo.

- Thêm trọng số (weight) theo mức độ quan trọng:
    Category × 3.0  — ngành hàng là yếu tố quyết định nhất
    Brand    × 2.0  — thương hiệu là yếu tố thứ hai
    Price    × 1.0  — giá tiền ít quan trọng hơn trong việc tìm SP tương tự
    Rating   × 1.0  — rating bổ sung thêm ngữ cảnh
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

logger = logging.getLogger(__name__)

# Trọng số cho từng nhóm feature
WEIGHT_CATEGORY = 3.0
WEIGHT_BRAND = 2.0
WEIGHT_NUMERIC = 1.0


def build_product_vectors(products_df: pd.DataFrame):
    """
    Encode product features into a weighted numeric matrix.

    Returns:
        matrix       — numpy array shape (n_products, n_features)
        product_ids  — list of productId theo thứ tự hàng trong matrix
        cat_ohe      — fitted OneHotEncoder cho categoryName
        brand_ohe    — fitted OneHotEncoder cho brand
        scaler       — fitted MinMaxScaler cho price, ratingAverage
    """
    df = products_df.copy().fillna({
        "brand": "unknown",
        "categoryName": "unknown",
        "price": 0.0,
        "ratingAverage": 0.0,
    })

    # ── Category: One-Hot Encoding ─────────────────────────────────────────
    cat_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_matrix = cat_ohe.fit_transform(df[["categoryName"]].astype(str))

    # ── Brand: One-Hot Encoding ────────────────────────────────────────────
    brand_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    brand_matrix = brand_ohe.fit_transform(df[["brand"]].astype(str))

    # ── Price + Rating: Normalize [0, 1] ───────────────────────────────────
    scaler = MinMaxScaler()
    numeric_matrix = scaler.fit_transform(
        df[["price", "ratingAverage"]].astype(float)
    )

    # ── Ghép và áp dụng trọng số ───────────────────────────────────────────
    matrix = np.hstack([
        cat_matrix   * WEIGHT_CATEGORY,
        brand_matrix * WEIGHT_BRAND,
        numeric_matrix * WEIGHT_NUMERIC,
    ])

    logger.info(
        "Product vectors built: %d products, %d features (cat=%d, brand=%d, numeric=2).",
        len(df),
        matrix.shape[1],
        cat_matrix.shape[1],
        brand_matrix.shape[1],
    )

    return matrix, df["productId"].tolist(), cat_ohe, brand_ohe, scaler


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
    """
    Return top-K similar product IDs for a given product.

    Args:
        product_id: ID sản phẩm cần tìm tương tự
        product_ids: Danh sách tất cả product IDs (thứ tự tương ứng matrix)
        similarity_matrix: Ma trận cosine similarity (n x n)
        top_k: Số sản phẩm tương tự cần trả về
        exclude_ids: Tập ID cần loại trừ khỏi kết quả

    Returns:
        Danh sách productId sorted theo độ tương đồng giảm dần
    """
    if product_id not in product_ids:
        return []

    exclude = exclude_ids or set()
    idx = product_ids.index(product_id)
    scores = list(enumerate(similarity_matrix[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, _ in scores:
        pid = product_ids[i]
        if pid == product_id:
            continue
        if pid in exclude:
            continue
        results.append(pid)
        if len(results) >= top_k:
            break

    return results

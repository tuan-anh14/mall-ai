"""
Load interaction data from PostgreSQL for model training.

Scoring logic (tín hiệu mạnh → điểm cao):
  - VIEW        → 0.5 per view, tối đa 5.0 (passive, ít tin cậy)
  - WISHLIST    → 4.0 (muốn sở hữu)
  - ORDER(DELIVERED) → 10.0 per lần mua (đã chi tiền thật)
  - REVIEW 5★   → 15.0 (cực kỳ hài lòng)
  - REVIEW 4★   → 8.0
  - REVIEW 3★   → 2.0 (trung lập)
  - REVIEW ≤ 2★ → -10.0 (không hài lòng → trừ điểm)

Lưu ý: Chỉ dùng đơn status='DELIVERED'. Đơn CANCELLED/RETURNED
không phản ánh ý muốn thực sự của user, dùng để train sẽ gây nhiễu.
"""
import logging

import pandas as pd
from sqlalchemy import create_engine, text

from config import settings

logger = logging.getLogger(__name__)


def get_engine():
    return create_engine(settings.database_url)


def load_interactions() -> pd.DataFrame:
    """
    Build a User-Item interaction matrix from DB.
    Returns DataFrame with columns: [userId, productId, score]

    Chỉ tổng hợp đơn hàng DELIVERED để đảm bảo tín hiệu sạch.
    Tích hợp Review rating làm Explicit Feedback.
    """
    engine = get_engine()

    with engine.connect() as conn:
        # ── 1. Views (passive signal) ──────────────────────────────────────
        views = pd.read_sql(
            text("""
                SELECT
                    "userId",
                    "productId",
                    LEAST("viewCount", 10) * 0.5 AS score
                FROM product_view_histories
            """),
            conn,
        )
        logger.info("Loaded %d view records.", len(views))

        # ── 2. Wishlist (intent to buy) ────────────────────────────────────
        wishlist = pd.read_sql(
            text("""
                SELECT
                    "userId",
                    "productId",
                    4.0 AS score
                FROM wishlist_items
            """),
            conn,
        )
        logger.info("Loaded %d wishlist records.", len(wishlist))

        # ── 3. Orders — CHỈ lấy DELIVERED (tiền thật đã chi) ───────────────
        orders = pd.read_sql(
            text("""
                SELECT
                    o."userId",
                    oi."productId",
                    COUNT(*) * 10.0 AS score
                FROM order_items oi
                JOIN orders o ON o.id = oi."orderId"
                WHERE o.status = 'DELIVERED'
                GROUP BY o."userId", oi."productId"
            """),
            conn,
        )
        logger.info("Loaded %d delivered-order records.", len(orders))

        # ── 4. Reviews — Explicit Feedback (có thể âm) ────────────────────
        reviews = pd.read_sql(
            text("""
                SELECT
                    "userId",
                    "productId",
                    CASE
                        WHEN rating = 5 THEN 15.0
                        WHEN rating = 4 THEN 8.0
                        WHEN rating = 3 THEN 2.0
                        WHEN rating <= 2 THEN -10.0
                    END AS score
                FROM reviews
            """),
            conn,
        )
        logger.info("Loaded %d review records.", len(reviews))

    # ── Gộp tất cả signals và tổng hợp theo (userId, productId) ───────────
    combined = pd.concat([views, wishlist, orders, reviews], ignore_index=True)

    # Loại bỏ các dòng NULL score (trường hợp rating ngoài 1-5)
    combined = combined.dropna(subset=["score"])

    aggregated = (
        combined
        .groupby(["userId", "productId"], as_index=False)["score"]
        .sum()
    )

    # Loại bỏ các cặp có tổng điểm ≤ 0 (User ghét SP đó)
    aggregated = aggregated[aggregated["score"] > 0]

    logger.info(
        "Final interaction matrix: %d (userId, productId) pairs.", len(aggregated)
    )
    return aggregated


def load_product_features() -> pd.DataFrame:
    """
    Load product metadata for content-based features.
    Chỉ lấy sản phẩm ACTIVE để không gợi ý hàng đã ngừng bán.
    """
    engine = get_engine()
    with engine.connect() as conn:
        products = pd.read_sql(
            text("""
                SELECT
                    p.id          AS "productId",
                    p."categoryId",
                    p.brand,
                    p.price,
                    p."ratingAverage",
                    c.name        AS "categoryName"
                FROM products p
                LEFT JOIN categories c ON c.id = p."categoryId"
                WHERE p.status = 'ACTIVE'
            """),
            conn,
        )
    logger.info("Loaded %d active products for content-based features.", len(products))
    return products


def load_basket_data(conn):
    """
    Trả SQLAlchemy connection để build_co_purchase_matrix() dùng.
    Hàm wrapper này giữ cho interface nhất quán.
    """
    return conn

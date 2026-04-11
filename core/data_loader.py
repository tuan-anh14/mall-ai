"""
Load interaction data from PostgreSQL for model training.
Scoring:
  - VIEW        → 1.0 (capped at 10 views)
  - WISHLIST    → 3.0
  - ORDER       → 5.0
"""
import pandas as pd
from sqlalchemy import create_engine, text
from config import settings


def get_engine():
    return create_engine(settings.database_url)


def load_interactions() -> pd.DataFrame:
    """
    Build a User-Item interaction matrix from DB.
    Returns DataFrame with columns: [userId, productId, score]
    """
    engine = get_engine()

    with engine.connect() as conn:
        # Views (Prisma uses camelCase column names)
        views = pd.read_sql(
            text("""
                SELECT "userId", "productId",
                       LEAST("viewCount", 10) * 1.0 AS score
                FROM product_view_histories
            """),
            conn,
        )

        # Wishlist
        wishlist = pd.read_sql(
            text("""
                SELECT "userId", "productId",
                       3.0 AS score
                FROM wishlist_items
            """),
            conn,
        )

        # Orders (count per user-product pair)
        orders = pd.read_sql(
            text("""
                SELECT o."userId", oi."productId",
                       COUNT(*) * 5.0 AS score
                FROM order_items oi
                JOIN orders o ON o.id = oi."orderId"
                GROUP BY o."userId", oi."productId"
            """),
            conn,
        )

    # Combine and aggregate scores
    combined = pd.concat([views, wishlist, orders], ignore_index=True)
    aggregated = (
        combined.groupby(["userId", "productId"], as_index=False)["score"]
        .sum()
    )

    return aggregated


def load_product_features() -> pd.DataFrame:
    """Load product metadata for content-based features."""
    engine = get_engine()
    with engine.connect() as conn:
        products = pd.read_sql(
            text("""
                SELECT p.id AS "productId",
                       p."categoryId",
                       p.brand,
                       p.price,
                       p."ratingAverage",
                       c.name AS "categoryName"
                FROM products p
                LEFT JOIN categories c ON c.id = p."categoryId"
                WHERE p.status = 'ACTIVE'
            """),
            conn,
        )
    return products

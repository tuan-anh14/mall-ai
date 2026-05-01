"""
Market Basket Analysis (Co-Purchase Mining).

Tìm các cặp sản phẩm thường được mua cùng nhau trong một đơn hàng.
Chỉ dùng đơn có status = 'DELIVERED' (đã giao thành công) để đảm bảo
tín hiệu sạch — đơn bị hủy/hoàn trả không phản ánh ý muốn thực sự.

Thuật toán: Co-occurrence counting (đơn giản, hiệu quả với dataset vừa)
Ví dụ kết quả:
    "iPhone" → [("Ốp lưng", 42), ("AirPods", 35), ("Sạc nhanh", 28)]
"""

import logging
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Số lần xuất hiện tối thiểu để tính là "thường mua kèm"
MIN_CO_COUNT = 2


def build_co_purchase_matrix(conn) -> dict[str, list[tuple[str, int]]]:
    """
    Xây dựng ma trận đồng mua (co-purchase) từ dữ liệu đơn hàng DELIVERED.

    Trả về dict:
        { productId: [(related_productId, co_count), ...] }
    Đã được sort theo co_count giảm dần.

    Args:
        conn: SQLAlchemy connection object

    Returns:
        dict mapping mỗi productId → danh sách (related_pid, count)
        đã sắp xếp theo số lần mua kèm giảm dần.
    """
    try:
        # Self-join order_items: tìm mọi cặp (A, B) cùng xuất hiện trong 1 đơn
        # Dùng oi1.productId < oi2.productId để tránh trùng lặp (A,B) và (B,A)
        basket_df = pd.read_sql(
            text("""
                SELECT
                    oi1."productId" AS product_a,
                    oi2."productId" AS product_b,
                    COUNT(*) AS co_count
                FROM order_items oi1
                JOIN order_items oi2
                    ON oi1."orderId" = oi2."orderId"
                    AND oi1."productId" < oi2."productId"
                JOIN orders o ON o.id = oi1."orderId"
                WHERE o.status = 'DELIVERED'
                GROUP BY oi1."productId", oi2."productId"
                HAVING COUNT(*) >= :min_count
                ORDER BY co_count DESC
            """),
            conn,
            params={"min_count": MIN_CO_COUNT},
        )
        logger.info("Loaded %d co-purchase pairs.", len(basket_df))
    except Exception as exc:
        logger.error("Failed to load co-purchase data: %s", exc)
        return {}

    if basket_df.empty:
        logger.warning(
            "No co-purchase pairs found (need >= %d co-occurrences). "
            "This is normal for a new store with few orders.",
            MIN_CO_COUNT,
        )
        return {}

    # Xây dựng lookup dict (hai chiều: A→B và B→A)
    co_matrix: dict[str, list[tuple[str, int]]] = {}
    for _, row in basket_df.iterrows():
        a = row["product_a"]
        b = row["product_b"]
        count = int(row["co_count"])
        co_matrix.setdefault(a, []).append((b, count))
        co_matrix.setdefault(b, []).append((a, count))

    # Sort mỗi danh sách theo co_count giảm dần
    for pid in co_matrix:
        co_matrix[pid] = sorted(co_matrix[pid], key=lambda x: x[1], reverse=True)

    logger.info("Co-purchase matrix built for %d products.", len(co_matrix))
    return co_matrix


def get_basket_recommendations(
    product_id: str,
    co_matrix: dict[str, list[tuple[str, int]]],
    top_k: int = 6,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """
    Trả danh sách top-K sản phẩm thường mua kèm với product_id.

    Args:
        product_id: ID sản phẩm đang xem / trong giỏ hàng
        co_matrix: Ma trận đồng mua từ build_co_purchase_matrix()
        top_k: Số sản phẩm gợi ý tối đa
        exclude_ids: Tập ID cần loại trừ (vd: sản phẩm đang trong giỏ)

    Returns:
        Danh sách productId sorted theo mức độ "thường mua kèm"
    """
    if product_id not in co_matrix:
        return []

    exclude = exclude_ids or set()
    results = []
    for related_pid, _ in co_matrix[product_id]:
        if related_pid in exclude:
            continue
        results.append(related_pid)
        if len(results) >= top_k:
            break

    return results

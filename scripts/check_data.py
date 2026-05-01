"""
Script kiểm tra nhanh dữ liệu trong DB để lấy sample IDs dùng cho test.
Chạy: python scripts/check_data.py
"""
from core.data_loader import get_engine
from sqlalchemy import text

engine = get_engine()

with engine.connect() as conn:
    print("=" * 60)
    print("📊 KIỂM TRA DỮ LIỆU TRONG DATABASE")
    print("=" * 60)

    # Đếm số lượng từng bảng
    tables = {
        "Users":                    'SELECT COUNT(*) FROM users',
        "Products (ACTIVE)":        "SELECT COUNT(*) FROM products WHERE status = 'ACTIVE'",
        "Orders (DELIVERED)":       "SELECT COUNT(*) FROM orders WHERE status = 'DELIVERED'",
        "Wishlist items":           'SELECT COUNT(*) FROM wishlist_items',
        "Product view histories":   'SELECT COUNT(*) FROM product_view_histories',
        "Reviews":                  'SELECT COUNT(*) FROM reviews',
    }

    for name, query in tables.items():
        count = conn.execute(text(query)).scalar()
        status = "✅" if count > 0 else "⚠️  RỖNG"
        print(f"  {status}  {name}: {count} bản ghi")

    print()
    print("─" * 60)
    print("🔍 SAMPLE USER IDs (có đơn hàng DELIVERED)")
    print("─" * 60)

    users = conn.execute(text("""
        SELECT DISTINCT o."userId", COUNT(*) as order_count
        FROM orders o
        WHERE o.status = 'DELIVERED'
        GROUP BY o."userId"
        ORDER BY order_count DESC
        LIMIT 5
    """)).fetchall()

    if users:
        for row in users:
            print(f"  userId: {row[0]}  ({row[1]} đơn đã giao)")
    else:
        print("  ⚠️  Chưa có đơn hàng DELIVERED nào!")

    print()
    print("─" * 60)
    print("🔍 SAMPLE PRODUCT IDs (có trong đơn DELIVERED)")
    print("─" * 60)

    products = conn.execute(text("""
        SELECT DISTINCT oi."productId", p.name, COUNT(*) as sold_count
        FROM order_items oi
        JOIN orders o ON o.id = oi."orderId"
        JOIN products p ON p.id = oi."productId"
        WHERE o.status = 'DELIVERED'
        GROUP BY oi."productId", p.name
        ORDER BY sold_count DESC
        LIMIT 5
    """)).fetchall()

    if products:
        for row in products:
            print(f"  productId: {row[0]}")
            print(f"  name:      {row[1]}")
            print(f"  sold:      {row[2]} lần")
            print()
    else:
        print("  ⚠️  Chưa có sản phẩm nào được order DELIVERED!")

    print()
    print("─" * 60)
    print("🔍 SAMPLE CO-PURCHASE (sản phẩm thường mua kèm)")
    print("─" * 60)

    pairs = conn.execute(text("""
        SELECT
            oi1."productId" AS product_a,
            p1.name AS name_a,
            oi2."productId" AS product_b,
            p2.name AS name_b,
            COUNT(*) AS co_count
        FROM order_items oi1
        JOIN order_items oi2
            ON oi1."orderId" = oi2."orderId"
            AND oi1."productId" < oi2."productId"
        JOIN orders o ON o.id = oi1."orderId"
        JOIN products p1 ON p1.id = oi1."productId"
        JOIN products p2 ON p2.id = oi2."productId"
        WHERE o.status = 'DELIVERED'
        GROUP BY oi1."productId", p1.name, oi2."productId", p2.name
        ORDER BY co_count DESC
        LIMIT 5
    """)).fetchall()

    if pairs:
        for row in pairs:
            print(f"  [{row[1]}] + [{row[3]}] → cùng mua {row[4]} lần")
    else:
        print("  ⚠️  Chưa có đơn nào có ≥ 2 sản phẩm!")

    print()
    print("=" * 60)
    print("✅ Xong! Dùng các ID trên để test API.")
    print("=" * 60)

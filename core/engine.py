"""
Recommendation engine: loads trained model and serves predictions.

3 loại gợi ý:
1. recommend_for_user()   — "Gợi ý cho bạn" (Trang chủ)
   Dùng SVD Collaborative Filtering. Nếu user mới/không đủ data
   → fallback về sản phẩm phổ biến nhất (popularity_scores).

2. similar_products()     — "Sản phẩm tương tự" (Trang chi tiết)
   Dùng Cosine Similarity trên content features (category, brand, price).

3. basket_recommendations() — "Thường mua kèm" (Trang chi tiết + Giỏ hàng)
   Dùng Co-purchase matrix (Market Basket Analysis).
"""
import logging
import joblib
import numpy as np
from pathlib import Path

from config import settings, MODEL_PATH
from core.features import get_similar_products
from core.association import get_basket_recommendations

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
            logger.warning(
                "Model file not found at %s. Run POST /retrain to train first.",
                MODEL_PATH,
            )

    @property
    def is_ready(self) -> bool:
        return self._model_data is not None

    # ──────────────────────────────────────────────────────────────────────
    # 1. RECOMMEND FOR USER — "Gợi ý cho bạn"
    # ──────────────────────────────────────────────────────────────────────
    def recommend_for_user(
        self, user_id: str, limit: int = 12
    ) -> tuple[list[str], bool]:
        """
        Return (product_ids, is_personalized) cho user.

        HYBRID APPROACH:
        1. SVD predict score cho tất cả candidates.
        2. Re-rank bằng cách BOOST điểm cho sản phẩm cùng category/brand
           với những gì user đã xem/mua/yêu thích.
        3. Kết quả: ưu tiên đúng danh mục user quan tâm, không bị "nhiễu"
           bởi hành vi của những user khác.

        is_personalized=True  → Dùng SVD + Hybrid boost (user có lịch sử).
        is_personalized=False → Không đủ data, trả sản phẩm phổ biến (Trending).
        """
        if not self.is_ready:
            return [], False

        data = self._model_data
        cf_model        = data.get("cf_model")
        product_ids     = data.get("product_ids", [])
        user_history    = data.get("user_history", {})
        product_meta    = data.get("product_meta", {})   # {pid: {categoryId, brand}}

        # user_history hiện tại là {pid: score}
        user_interactions: dict[str, float] = user_history.get(user_id, {})
        interacted = set(user_interactions.keys())

        # User mới: không có lịch sử → popularity fallback
        if not interacted:
            logger.debug("User %s: no history → popularity fallback.", user_id)
            return self._popularity_fallback(limit), False

        # ── Bước 1: Xây dựng "profile sở thích" của user ──────────────────
        # Đếm điểm số tương tác với mỗi category & brand (Dùng score thật!)
        category_weight: dict[str, float] = {}
        brand_weight: dict[str, float] = {}
        for pid, score in user_interactions.items():
            meta = product_meta.get(pid, {})
            cat = meta.get("categoryId", "")
            brand = meta.get("brand", "")
            if cat:
                # Cộng dồn bằng score (Order=10, View=0.5...) chứ không cộng 1.0
                category_weight[cat] = category_weight.get(cat, 0) + score
            if brand:
                brand_weight[brand] = brand_weight.get(brand, 0) + score

        # Normalize weights (để boost không quá lấn át CF score)
        max_cat   = max(category_weight.values(), default=1)
        max_brand = max(brand_weight.values(), default=1)

        # ── Bước 2: CF Scoring với SVD ─────────────────────────────────────
        candidates = [pid for pid in product_ids if pid not in interacted]
        if not candidates:
            return self._popularity_fallback(limit, exclude_ids=interacted), False

        cf_predictions: list[tuple[str, float]] = []
        if cf_model is not None and hasattr(cf_model, "predict"):
            for pid in candidates:
                try:
                    est = cf_model.predict(user_id, pid).est
                    cf_predictions.append((pid, est))
                except Exception:
                    pass

        # Nếu SVD không predict được → dùng popularity làm base score
        if not cf_predictions:
            popularity = self._model_data.get("popularity_scores", {})
            cf_predictions = [(pid, popularity.get(pid, 0.0)) for pid in candidates]

        # ── Bước 3: Hybrid Re-ranking ──────────────────────────────────────
        # TĂNG TRỌNG SỐ: Ưu tiên cực mạnh danh mục user đã xem
        CATEGORY_BOOST = 5.0  # Tăng từ 1.5 -> 5.0
        BRAND_BOOST    = 2.0  # Tăng từ 1.0 -> 2.0
        STRANGER_PENALTY = -2.0 # Trừ điểm nếu là danh mục user chưa từng xem

        scored: list[tuple[str, float]] = []
        for pid, cf_score in cf_predictions:
            meta  = product_meta.get(pid, {})
            cat   = meta.get("categoryId", "")
            brand = meta.get("brand", "")

            # Tính toán mức độ liên quan
            is_relevant_cat = cat in category_weight
            
            # Boost tỉ lệ với mức độ user quan tâm
            cat_boost   = (category_weight.get(cat,   0) / max_cat)   * CATEGORY_BOOST if is_relevant_cat else STRANGER_PENALTY
            brand_boost = (brand_weight.get(brand, 0) / max_brand) * BRAND_BOOST

            hybrid_score = cf_score + cat_boost + brand_boost
            scored.append((pid, hybrid_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        result = [pid for pid, _ in scored[:limit]]

        # ── Bước 4: Thống kê chi tiết ra Terminal (Dành cho Debug) ──────────
        print(f"\n{'='*60}")
        print(f" DEBUG RECOMMENDATION FOR USER: {user_id}")
        print(f"{'='*60}")
        print(f" - Profile: Top Categories {sorted(category_weight, key=category_weight.get, reverse=True)[:3]}")
        print(f" - Profile: Top Brands {sorted(brand_weight, key=brand_weight.get, reverse=True)[:3]}")
        print(f"\n [TOP 5 SCORING BREAKDOWN]:")
        
        for i, (pid, total_score) in enumerate(scored[:5]):
            meta = product_meta.get(pid, {})
            cat = meta.get("categoryId", "N/A")
            cat_boost = (category_weight.get(cat, 0) / max_cat) * CATEGORY_BOOST if cat in category_weight else STRANGER_PENALTY
            # Tìm SVD score gốc
            base_score = next((s for p, s in cf_predictions if p == pid), 0.0)
            
            print(f" {i+1}. PID: {pid[:10]}... | Total: {total_score:5.2f} | (Base: {base_score:4.2f}, CatBoost: {cat_boost:4.2f})")
        print(f"{'='*60}\n")

        return result, True

    def _popularity_fallback(
        self,
        limit: int,
        exclude_ids: set[str] | None = None,
    ) -> list[str]:
        """
        Trả sản phẩm phổ biến nhất (tổng score cao nhất trên toàn hệ thống).
        Dùng cho: user mới, CF không đủ data.
        """
        if not self.is_ready:
            return []

        popularity: dict[str, float] = self._model_data.get("popularity_scores", {})
        exclude = exclude_ids or set()

        sorted_popular = sorted(popularity, key=lambda x: popularity[x], reverse=True)
        return [pid for pid in sorted_popular if pid not in exclude][:limit]

    # ──────────────────────────────────────────────────────────────────────
    # 2. SIMILAR PRODUCTS — "Sản phẩm tương tự"
    # ──────────────────────────────────────────────────────────────────────
    def similar_products(
        self,
        product_id: str,
        limit: int = 8,
        exclude_ids: set[str] | None = None,
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

    # ──────────────────────────────────────────────────────────────────────
    # 3. BASKET RECOMMENDATIONS — "Thường mua kèm"
    # ──────────────────────────────────────────────────────────────────────
    def basket_recommendations(
        self,
        product_id: str,
        limit: int = 6,
        exclude_ids: set[str] | None = None,
    ) -> list[str]:
        """
        Return top-K products thường được mua kèm với product_id.
        Dựa trên Market Basket Analysis (co-purchase matrix).
        Dùng cho: Trang chi tiết sản phẩm, Giỏ hàng.
        """
        if not self.is_ready:
            return []

        co_matrix: dict = self._model_data.get("co_purchase_matrix", {})
        return get_basket_recommendations(
            product_id=product_id,
            co_matrix=co_matrix,
            top_k=limit,
            exclude_ids=exclude_ids,
        )

    def reload(self):
        """Reload model from disk (called after retraining)."""
        self.load()


# Singleton engine instance
engine = RecommendationEngine()

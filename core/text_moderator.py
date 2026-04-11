"""
TextModerator — Singleton load/predict/reload.
Fallback sang regex blacklist nếu model chưa được train.
"""
import logging
import re
from pathlib import Path
from typing import Optional, Union

import joblib

logger = logging.getLogger(__name__)

# Regex blacklist cứng: khớp 1 trong những từ/pattern này → TOXIC/SPAM
_TOXIC_PATTERNS = re.compile(
    r"\b(đmm|đm|dm|vcl|vl|đéo|deo|cmm|cc|cứt|cut|chó|cho|ngu|khốn|khon"
    r"|lừa đảo|lua dao|ăn cứt|hack|phế|phe|rác|rac|thứ rác|thứ chó"
    r"|mất dạy|mat day|vô học|vo hoc|địt|dit|cặc|cac|lồn|lon)\b",
    re.IGNORECASE | re.UNICODE,
)

_SPAM_PATTERNS = re.compile(
    r"((\+84|0)[0-9]{8,10})"          # Số điện thoại
    r"|https?://\S+"                    # URL
    r"|www\.\S+"                        # www
    r"|(\w{30,})"                       # Chuỗi ký tự vô nghĩa dài ≥30
    r"|(zalo|telegram|whatsapp|inbox|ib)\s*(mình|tôi|t\b)",  # SPAM social
    re.IGNORECASE | re.UNICODE,
)

# Tách riêng: ký tự lặp >=11 lần liên tiếp (vd: aaaaaaaaaaaaa)
_REPEAT_PATTERN = re.compile(r"(.)\1{10,}", re.UNICODE)

LABEL_NAMES = {0: "SAFE", 1: "TOXIC", 2: "SPAM"}


class TextModerator:
    def __init__(self):
        self._model_data: Optional[dict] = None

    def load(self, model_path: Union[str, Path] = "./models/moderation_model.pkl"):
        model_path = Path(model_path)
        if model_path.exists():
            try:
                self._model_data = joblib.load(model_path)
                logger.info("Moderation model loaded from %s", model_path)
            except Exception as exc:
                logger.error("Failed to load moderation model: %s", exc)
                self._model_data = None
        else:
            logger.warning(
                "Moderation model not found at %s. "
                "Using regex fallback. Run POST /moderate/retrain to train.",
                model_path,
            )

    @property
    def is_ready(self) -> bool:
        return self._model_data is not None

    def predict(self, text: str, threshold: float = 0.7) -> dict:
        """
        Trả về:
            {
                "allowed": bool,
                "label": "SAFE" | "TOXIC" | "SPAM",
                "score": float,   # confidence của label được chọn
            }
        """
        text = (text or "").strip()
        if not text:
            return {"allowed": True, "label": "SAFE", "score": 1.0}

        if self._model_data is not None:
            return self._predict_ml(text, threshold)
        else:
            return self._predict_regex(text)

    def _predict_ml(self, text: str, threshold: float) -> dict:
        # Regex pre-filter: bắt các trường hợp rõ ràng trước khi dùng ML
        regex_result = self._predict_regex(text)
        if not regex_result["allowed"]:
            return regex_result

        pipeline = self._model_data["pipeline"]
        label_names: dict = self._model_data.get("label_names", LABEL_NAMES)

        proba = pipeline.predict_proba([text])[0]
        label_idx = int(proba.argmax())
        score = float(proba[label_idx])
        label = label_names.get(label_idx, "SAFE")

        # Chỉ chặn nếu confident đủ ngưỡng VÀ không phải SAFE
        allowed = label == "SAFE" or score < threshold

        return {"allowed": allowed, "label": label, "score": round(score, 4)}

    def _predict_regex(self, text: str) -> dict:
        """Fallback regex khi chưa có model."""
        if _SPAM_PATTERNS.search(text) or _REPEAT_PATTERN.search(text):
            return {"allowed": False, "label": "SPAM", "score": 1.0}
        if _TOXIC_PATTERNS.search(text):
            return {"allowed": False, "label": "TOXIC", "score": 1.0}
        return {"allowed": True, "label": "SAFE", "score": 1.0}

    def reload(self, model_path: Union[str, Path] = "./models/moderation_model.pkl"):
        """Reload sau khi retrain."""
        self.load(model_path)


# Singleton
text_moderator = TextModerator()

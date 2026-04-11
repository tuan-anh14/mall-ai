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

# ─── Normalization ────────────────────────────────────────────────────────────

def _normalize_obfuscation(text: str) -> str:
    """
    Chuẩn hóa để bắt các biến thể né kiểm duyệt trước khi check regex:
      đ.m.m  → đmm
      v*c*l  → vcl
      d-i-t  → dit
      d1t    → dit  (l33tspeak)
      c4c    → cac
      đ m m  → đmm  (khoảng trắng giữa chữ đơn)
    """
    t = text.lower()
    # 0. Khôi phục từ tục tiếng Anh bị che dấu (*) — PHẢI chạy TRƯỚC khi xóa dấu *
    #    Vì nếu xóa * trước, f*** → f (mất context) → không nhận ra
    t = re.sub(r'\bf[\*@#!]{2,}(?:ing|ed|er)?(?=\s|$|[^a-zA-ZÀ-ỹ])', 'fucking', t)  # f***
    t = re.sub(r'\bf[\*@#!]?[uc][\*@#!]?k(?:ing|ed|er)?', 'fucking', t)            # f*ck
    t = re.sub(r'\bsh[\*@#!]{1,2}t\b', 'shit', t)                 # sh*t sh@t
    t = re.sub(r'\bb[\*@#!]{1,2}tch\b', 'bitch', t)               # b*tch b@tch
    t = re.sub(r'\ba[\*@#!]{1,2}s(?:hole)?\b', 'asshole', t)      # a**hole
    # 1. L33tspeak: thay số / ký tự phổ biến bằng chữ cái tương ứng
    for char, replacement in [('4','a'), ('1','i'), ('3','e'), ('0','o'), ('@','a'), ('$','s')]:
        t = t.replace(char, replacement)
    # 2. Xóa ký tự ngăn cách GIỮA hai ký tự chữ/số (đ.m → đm, v*c*l → vcl)
    t = re.sub(r'(?<=\w)[.\-*_+!](?=\w)', '', t)
    # 3. Thu gọn khoảng trắng giữa các chữ đơn lẻ liên tiếp (đ m m → đmm)
    t = re.sub(r'\b(\w)\s+(?=\w\b)', r'\1', t)
    return t


# ─── Regex patterns ───────────────────────────────────────────────────────────
# Chạy trên TEXT ĐÃ NORMALIZE → bắt được cả biến thể che giấu

_TOXIC_PATTERNS = re.compile(
    r"(?:"
    # ── Từ chửi thề tiếng Việt rõ ràng ───────────────────────────────────────
    r"đmm|dmm"                              # đmm / dmm
    r"|\bđm\b|\bdm\b"                       # đm / dm (đứng riêng)
    r"|\bvcl\b"                             # vcl
    r"|\bđéo\b"                             # đéo
    r"|\bcmm\b"                             # cmm
    r"|\bcứt\b"                             # cứt
    r"|\bđịt\b|\bdit\b"                     # địt / dit
    r"|\bcặc\b"                             # cặc (không dùng 'cac' để tránh false positive)
    r"|\blồn\b"                             # lồn (không dùng 'lon' - lon bia)
    r"|\bdái\b"                             # dái
    # ── Chửi rủa / xúc phạm ─────────────────────────────────────────────────
    r"|\bngu\b"                             # ngu (đứng độc lập)
    r"|\bkhốn\b|\bkhon\b"                   # khốn
    r"|mất\s*dạy|mat\s*day"
    r"|vô\s*học|vo\s*hoc"
    r"|lừa\s*đảo|lua\s*dao"
    r"|mẹ\s*kiếp|má\s*kiếp|me\s*kiep"
    r"|thằng\s+(?:ngu|chó|điên|khốn|khùng|đần|hèn|mất\s*dạy)"
    r"|con\s+(?:chó|lợn|heo|bò|đĩ|điên)"
    r"|đồ\s+(?:ngu|điên|khốn|hèn)"
    r"|chó\s*(?:chết|đẻ|cút|khốn)"         # chó trong cụm chửi
    # ── Tiếng Anh ─────────────────────────────────────────────────────────────
    r"|\bfuck(?:ing|er|ed)?\b|\bfucking\b"
    r"|\bbitch\b"
    r"|\basshole\b"
    r"|\bastard\b"
    r"|\bshit\b"
    r"|\bwtf\b|\bstfu\b|\bgtfo\b"
    r"|\bidiot\b"
    r"|\bmoron\b"
    r"|\bretard\b"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Spam: chạy trên TEXT GỐC (không normalize để không phá phone pattern)
_SPAM_PATTERNS = re.compile(
    # Số điện thoại: 0xxx xxx xxx hoặc +84 xxx xxx xxx (có hoặc không có dấu cách/gạch)
    r"(\+84|0)[\s\-.]?[0-9]{2,4}[\s\-.]?[0-9]{3,4}[\s\-.]?[0-9]{3,4}"
    r"|https?://\S+"                        # URL http/https
    r"|www\.\S+"                            # www.xxx
    r"|[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"  # Email
    r"|t\.me/\S+"                           # Telegram link
    r"|fb\.me/\S+|fb\.com/\S+"             # Facebook link
    r"|zalo\.me/\S+"                        # Zalo link
    r"|bit\.ly/\S+|tinyurl\.com/\S+"       # Short URL
    r"|(\w{30,})"                           # Chuỗi ký tự vô nghĩa dài
    r"|(zalo|telegram|whatsapp|viber|signal|line)\s*[:：]?\s*(mình|tôi|t\b|tui|mk)"
    r"|(inbox|ib|nhắn tin|liên hệ)\s+(mình|tôi|t\b|tui|mk)\b",
    re.IGNORECASE | re.UNICODE,
)

# Ký tự lặp liên tiếp ≥ 8 lần (giảm từ 11 xuống 8 để bắt thêm)
_REPEAT_PATTERN = re.compile(r"(.)\1{7,}", re.UNICODE)

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
            { "allowed": bool, "label": "SAFE"|"TOXIC"|"SPAM", "score": float }
        """
        text = (text or "").strip()
        if not text:
            return {"allowed": True, "label": "SAFE", "score": 1.0}

        if self._model_data is not None:
            return self._predict_ml(text, threshold)
        return self._predict_regex(text)

    # ── Predict paths ─────────────────────────────────────────────────────────

    def _predict_ml(self, text: str, threshold: float) -> dict:
        """Regex pre-filter → ML model."""
        # Bước 1: Regex bắt các trường hợp rõ ràng (bao gồm obfuscated)
        regex_result = self._predict_regex(text)
        if not regex_result["allowed"]:
            return regex_result

        # Bước 2: ML model cho các trường hợp tinh tế hơn
        pipeline = self._model_data["pipeline"]
        label_names: dict = self._model_data.get("label_names", LABEL_NAMES)

        proba = pipeline.predict_proba([text])[0]
        label_idx = int(proba.argmax())
        score = float(proba[label_idx])
        label = label_names.get(label_idx, "SAFE")

        # Chặn khi label KHÔNG phải SAFE VÀ model đủ confident (score >= threshold)
        allowed = (label == "SAFE") or (score < threshold)

        return {"allowed": allowed, "label": label, "score": round(score, 4)}

    def _predict_regex(self, text: str) -> dict:
        """
        Kiểm tra regex trên 2 lớp:
          - Spam: text gốc (tránh phá vỡ số điện thoại khi normalize)
          - Toxic: text đã normalize (bắt được đ.m, v*c*l, d1t...)
        """
        # SPAM check: text gốc
        if _SPAM_PATTERNS.search(text) or _REPEAT_PATTERN.search(text):
            return {"allowed": False, "label": "SPAM", "score": 1.0}

        # TOXIC check: normalize trước để bắt obfuscation
        normalized = _normalize_obfuscation(text)
        if _TOXIC_PATTERNS.search(normalized):
            return {"allowed": False, "label": "TOXIC", "score": 1.0}

        return {"allowed": True, "label": "SAFE", "score": 1.0}

    def reload(self, model_path: Union[str, Path] = "./models/moderation_model.pkl"):
        self.load(model_path)


# Singleton
text_moderator = TextModerator()

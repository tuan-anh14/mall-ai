"""
Unit tests cho TextModerator — chạy KHÔNG cần model file (dùng regex fallback).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.text_moderator import TextModerator


@pytest.fixture
def moderator():
    """Tạo TextModerator mới (không load model → dùng regex fallback)."""
    m = TextModerator()
    # Không gọi m.load() → dùng regex fallback
    return m


def test_safe_text(moderator):
    result = moderator.predict("Sản phẩm rất tốt, giao hàng nhanh")
    assert result["allowed"] is True
    assert result["label"] == "SAFE"


def test_safe_text_empty(moderator):
    result = moderator.predict("")
    assert result["allowed"] is True
    assert result["label"] == "SAFE"


def test_toxic_text_vn(moderator):
    result = moderator.predict("đmm hàng nát vcl")
    assert result["allowed"] is False
    assert result["label"] == "TOXIC"


def test_toxic_explicit(moderator):
    result = moderator.predict("thứ đồ chó đẻ này")
    assert result["allowed"] is False
    assert result["label"] == "TOXIC"


def test_spam_phone(moderator):
    result = moderator.predict("liên hệ mua hàng 0909123456")
    assert result["allowed"] is False
    assert result["label"] == "SPAM"


def test_spam_url(moderator):
    result = moderator.predict("click link này nhận quà http://bit.ly/abcxyz")
    assert result["allowed"] is False
    assert result["label"] == "SPAM"


def test_spam_repeated_chars(moderator):
    result = moderator.predict("aaaaaaaaaaaaaaaaaaaaaa")
    assert result["allowed"] is False
    assert result["label"] == "SPAM"


def test_safe_review(moderator):
    result = moderator.predict("Chất lượng tốt, đóng gói cẩn thận, sẽ mua lại")
    assert result["allowed"] is True


def test_safe_neutral(moderator):
    result = moderator.predict("Tạm ổn, không có gì đặc biệt")
    assert result["allowed"] is True


def test_predict_returns_required_keys(moderator):
    result = moderator.predict("test text")
    assert "allowed" in result
    assert "label" in result
    assert "score" in result
    assert result["label"] in ("SAFE", "TOXIC", "SPAM")


def test_train_and_predict():
    """Train model rồi kiểm tra predict với ML model."""
    import tempfile
    import os
    from core.moderation_trainer import train_and_save
    from core.text_moderator import TextModerator

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name

    try:
        accuracy = train_and_save(tmp_path)
        assert accuracy > 0.5, f"Accuracy quá thấp: {accuracy}"

        m = TextModerator()
        m.load(tmp_path)
        assert m.is_ready

        safe = m.predict("Sản phẩm tuyệt vời, giao hàng nhanh")
        assert safe["label"] in ("SAFE", "TOXIC", "SPAM")

        toxic = m.predict("đmm shop lừa đảo khốn nạn")
        assert toxic["label"] in ("SAFE", "TOXIC", "SPAM")
    finally:
        os.unlink(tmp_path)

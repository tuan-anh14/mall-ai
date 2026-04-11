"""
Train TF-IDF + Logistic Regression pipeline cho text moderation.
Labels: 0=SAFE, 1=TOXIC, 2=SPAM
"""
import logging
import re
from pathlib import Path
from typing import List, Optional, Union

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from core.seed_data import get_texts_and_labels

logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "SAFE", 1: "TOXIC", 2: "SPAM"}


def _normalize(text: str) -> str:
    """Chuẩn hóa văn bản đầu vào trước khi vectorize."""
    text = text.lower()
    # Bỏ URL
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    # Bỏ số điện thoại (0xxxxxxxxx hoặc +84xxxxxxxxx)
    text = re.sub(r"(\+84|0)[0-9]{8,10}", " phone ", text)
    # Chuẩn hóa dấu cách
    text = re.sub(r"\s+", " ", text).strip()
    return text


class NormalizingTfidfVectorizer(TfidfVectorizer):
    """TfidfVectorizer kèm bước normalize text trước."""

    def build_analyzer(self):
        base_analyzer = super().build_analyzer()

        def analyzer(text: str):
            return base_analyzer(_normalize(text))

        return analyzer


def train_and_save(model_path: Union[str, Path], extra_texts: Optional[List[str]] = None,
                   extra_labels: Optional[List[int]] = None) -> float:
    """
    Train pipeline và lưu vào model_path.
    Có thể thêm extra_texts/extra_labels (từ DB thực tế) để augment seed data.

    Trả về accuracy trên tập test.
    """
    texts, labels = get_texts_and_labels()

    if extra_texts and extra_labels:
        texts = texts + extra_texts
        labels = labels + extra_labels
        logger.info("Augmented dataset: seed=%d + extra=%d = %d mẫu",
                    len(get_texts_and_labels()[0]), len(extra_texts), len(texts))
    else:
        logger.info("Training trên seed dataset: %d mẫu", len(texts))

    # Split để đánh giá
    if len(texts) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.15, random_state=42, stratify=labels
        )
    else:
        X_train, X_test, y_train, y_test = texts, texts, labels, labels

    pipeline = Pipeline([
        ("tfidf", NormalizingTfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10_000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("Moderation model accuracy: %.4f", accuracy)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "label_names": LABEL_NAMES}, model_path)
    logger.info("Moderation model saved to %s", model_path)

    return accuracy

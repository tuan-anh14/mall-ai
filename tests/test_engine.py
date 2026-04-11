"""
Unit tests for recommendation engine (no DB required).
"""
import numpy as np
import pytest
from core.features import get_similar_products, build_product_vectors
import pandas as pd


PRODUCT_IDS = ["p1", "p2", "p3", "p4", "p5"]

# Simple identity-like similarity matrix
SIM_MATRIX = np.array([
    [1.0, 0.9, 0.3, 0.1, 0.2],
    [0.9, 1.0, 0.4, 0.2, 0.1],
    [0.3, 0.4, 1.0, 0.8, 0.7],
    [0.1, 0.2, 0.8, 1.0, 0.9],
    [0.2, 0.1, 0.7, 0.9, 1.0],
])


def test_get_similar_basic():
    result = get_similar_products("p1", PRODUCT_IDS, SIM_MATRIX, top_k=2)
    assert result == ["p2", "p3"], f"Expected ['p2', 'p3'], got {result}"


def test_get_similar_excludes_self():
    result = get_similar_products("p1", PRODUCT_IDS, SIM_MATRIX, top_k=5)
    assert "p1" not in result


def test_get_similar_exclude_ids():
    result = get_similar_products("p1", PRODUCT_IDS, SIM_MATRIX, top_k=3, exclude_ids={"p2"})
    assert "p2" not in result
    assert len(result) <= 3


def test_get_similar_unknown_product():
    result = get_similar_products("unknown", PRODUCT_IDS, SIM_MATRIX, top_k=3)
    assert result == []


def test_build_product_vectors():
    df = pd.DataFrame({
        "productId": ["p1", "p2", "p3"],
        "categoryId": ["c1", "c1", "c2"],
        "categoryName": ["Electronics", "Electronics", "Fashion"],
        "brand": ["Apple", "Samsung", "Nike"],
        "price": [999.0, 799.0, 99.0],
        "ratingAverage": [4.5, 4.2, 3.8],
    })
    matrix, product_ids, _, _, _ = build_product_vectors(df)
    assert matrix.shape == (3, 4)
    assert product_ids == ["p1", "p2", "p3"]

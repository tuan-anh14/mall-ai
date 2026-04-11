from fastapi import APIRouter
from pydantic import BaseModel

from core.engine import engine

router = APIRouter()


class RecommendRequest(BaseModel):
    userId: str
    limit: int = 12


class SimilarRequest(BaseModel):
    productId: str
    limit: int = 8


class RetrainResponse(BaseModel):
    success: bool
    message: str


@router.get("/health")
def health():
    return {"status": "ok", "model_ready": engine.is_ready}


@router.post("/recommend")
def recommend(req: RecommendRequest):
    product_ids = engine.recommend_for_user(req.userId, req.limit)
    return {"productIds": product_ids}


@router.post("/similar")
def similar(req: SimilarRequest):
    product_ids = engine.similar_products(req.productId, req.limit)
    return {"productIds": product_ids}


@router.post("/retrain")
def retrain():
    try:
        from core.trainer import train_and_save
        train_and_save()
        engine.reload()
        return RetrainResponse(success=True, message="Model retrained and reloaded successfully.")
    except Exception as exc:
        return RetrainResponse(success=False, message=str(exc))

from fastapi import APIRouter
from pydantic import BaseModel

from core.engine import engine
from core.text_moderator import text_moderator
from config import settings

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


# ─── Moderation ───────────────────────────────────────────────────────────────

class ModerateTextRequest(BaseModel):
    text: str


class ModerateTextResponse(BaseModel):
    allowed: bool
    label: str
    score: float


@router.post("/moderate/text", response_model=ModerateTextResponse)
def moderate_text(req: ModerateTextRequest):
    result = text_moderator.predict(req.text, threshold=settings.moderation_threshold)
    return ModerateTextResponse(**result)


@router.post("/moderate/retrain")
def moderate_retrain():
    try:
        from core.moderation_trainer import train_and_save as moderation_train_and_save
        accuracy = moderation_train_and_save(settings.moderation_model_path)
        text_moderator.reload(settings.moderation_model_path)
        return RetrainResponse(
            success=True,
            message=f"Moderation model retrained. Accuracy: {accuracy:.4f}",
        )
    except Exception as exc:
        return RetrainResponse(success=False, message=str(exc))

from fastapi import APIRouter
import torch

router = APIRouter(prefix="/api/v1", tags=["Health Check"])

@router.get("/health")
def health_check():
    from backend.services.advanced_predictor import predictor_ready
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "service": "Extractive Summarizer FastAPI Server",
        "version": "2.0.0",
        "vietnamese_model_ready": predictor_ready(),
        "vietnamese_method": "SBERT pretrained + Logistic Regression (v2)",
        "english_method": "SBERT + K-Means (legacy)"
    }

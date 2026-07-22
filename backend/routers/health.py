from fastapi import APIRouter
import torch

router = APIRouter(prefix="/api/v1", tags=["Health Check"])

@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "service": "Extractive Summarizer FastAPI Server"
    }

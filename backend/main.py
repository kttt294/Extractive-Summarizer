import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import health, summarize

app = FastAPI(
    title="Extractive Summarizer REST API",
    description="Tiếng Việt: SBERT pretrained + Logistic v2; tiếng Anh: SBERT + K-Means kế thừa.",
    version="2.0.0"
)

# Configure CORS Middleware to allow requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router)
app.include_router(summarize.router)

@app.on_event("startup")
def startup_event():
    if os.getenv('PRELOAD_MODELS', 'true').lower() == 'true':
        from backend.services.advanced_predictor import get_predictor
        print('Loading advanced ML v2 Vietnamese model...')
        get_predictor()
        print('Vietnamese Logistic v2 ready.')

@app.get("/api")
def api_root():
    return {
        "message": "Extractive Summarizer API",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }

# Mount Frontend static build if exists (for all-in-one deployment)
import os
from fastapi.staticfiles import StaticFiles
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "message": "Extractive Summarizer API",
            "docs_url": "/docs",
            "health_check": "/api/v1/health"
        }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

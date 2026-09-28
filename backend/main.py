import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import health, summarize

app = FastAPI(
    title="Extractive Summarizer REST API",
    description="FastAPI Backend Server for SBERT + K-Means Extractive Summarization",
    version="1.0.0"
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
    print("Pre-warming SBERT Embedding Engine...")
    try:
        from src.embedding import embed_sentences
        embed_sentences([(0, "Khởi động hệ thống tóm tắt.")], lang='vi')
        print("SBERT Engine ready!")
    except Exception as e:
        print(f"Pre-warming note: {e}")

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

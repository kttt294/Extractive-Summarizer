import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import health, summarize

app = FastAPI(
    title="Extractive Summarizer REST API",
    description="FastAPI Backend Server for SBERT + K-Means Extractive Summarization",
    version="1.0.0"
)

# Configure CORS Middleware to allow React Frontend requests
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

@app.get("/")
def root():
    return {
        "message": "Welcome to Extractive Summarizer API!",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

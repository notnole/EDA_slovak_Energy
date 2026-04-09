
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import data

app = FastAPI(title="Energy Market Dashboard")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/api")

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Energy Dashboard Backend"}

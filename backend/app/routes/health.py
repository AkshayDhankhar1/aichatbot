"""Health check endpoint for Railway/Render uptime probes."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}

"""POST /chat — streams a grounded answer as Server-Sent Events (SSE).

Request body:
    { "message": "your question", "history": [{"role": "...", "content": "..."}] }

Response: text/event-stream. Each event is `data: <json>\n\n` where json is one of
the event dicts produced by rag_chain.answer_stream (token / done / error).

Session-only: history is supplied by the client each request; nothing is stored
server-side (no DB, per project constraints).
"""
from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..rag_chain import answer_stream

router = APIRouter()


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = []


@router.post("/chat")
async def chat(req: ChatRequest):
    history = [t.model_dump() for t in req.history]

    async def event_generator():
        try:
            async for event in answer_stream(req.message, history):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # last-resort guard
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx/render)
        },
    )

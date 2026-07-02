"""
SHL Assessment Recommender — FastAPI Service

Endpoints:
  GET  /health          → {"status": "ok"}
  POST /chat            → {"reply": "...", "recommendations": [...], "end_of_conversation": bool}

The service is fully stateless. All conversation state is carried in the request body.
"""
import os
import sys
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the app directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

from agent import agent_reply, AgentResponse, Recommendation

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    recommendations: list[dict] | None = Field(default=None, description="Assistant's recommendations from prior turns (for shortlist tracking)")


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., description="Full conversation history")


class RecommendationResponse(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationResponse]
    end_of_conversation: bool


# ── Health ──────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Chat ───────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Schema: messages is required and must be non-empty
    if not req.messages:
        raise HTTPException(status_code=422, detail="messages cannot be empty")

    # Check turn cap
    user_turns = sum(1 for m in req.messages if m.role == "user")
    if user_turns > 8:
        raise HTTPException(
            status_code=422,
            detail="Conversation exceeds maximum of 8 user turns"
        )

    # Build raw messages list for agent — preserve recommendations for shortlist tracking
    raw_messages = []
    for m in req.messages:
        msg_dict = {"role": m.role, "content": m.content}
        if m.recommendations:
            msg_dict["recommendations"] = m.recommendations
        raw_messages.append(msg_dict)

    try:
        response: AgentResponse = await agent_reply(raw_messages)
    except Exception as e:
        # Log and return graceful error
        print(f"[ERROR] agent_reply failed: {e}", flush=True)
        raise HTTPException(
            status_code=503,
            detail="Agent service temporarily unavailable. Please retry."
        )

    # Validate response schema
    if not isinstance(response.reply, str):
        raise HTTPException(status_code=500, detail="Invalid reply type from agent")

    if not isinstance(response.recommendations, list):
        raise HTTPException(status_code=500, detail="Invalid recommendations type from agent")

    if len(response.recommendations) > 10:
        response.recommendations = response.recommendations[:10]

    if not isinstance(response.end_of_conversation, bool):
        raise HTTPException(status_code=500, detail="Invalid end_of_conversation type from agent")

    # Validate each recommendation
    validated_recs = []
    for r in response.recommendations:
        if isinstance(r, Recommendation):
            validated_recs.append(r.to_dict())
        elif isinstance(r, dict):
            validated_recs.append({
                "name": str(r.get("name", "")),
                "url": str(r.get("url", "")),
                "test_type": str(r.get("test_type", "K")),
            })
        else:
            # Skip invalid entries
            pass

    return ChatResponse(
        reply=response.reply,
        recommendations=validated_recs,
        end_of_conversation=response.end_of_conversation,
    )


# ── Run locally ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

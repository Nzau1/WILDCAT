#!/usr/bin/env python3
"""
WildCat HTTP API — wraps wildcat.py as a web service.
Deploy to Render.com (or Railway/Fly.io). The frontend calls these endpoints.
"""
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from wildcat import WildCat


app = FastAPI(title="WildCat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CAT = WildCat(use_llm=True, stream=False, verbose=True)


class ChatRequest(BaseModel):
    message: str
    mode: str = "active"


@app.get("/")
def root():
    return {
        "name": "WildCat API",
        "status": "ok",
        "provider": CAT.llm.provider if CAT.llm else "none",
        "model": CAT.llm.model if CAT.llm else "n/a",
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/stats")
def stats():
    return CAT.stats()


@app.post("/chat")
def chat(req: ChatRequest):
    if req.mode not in ("training", "testing", "active"):
        raise HTTPException(400, "mode must be training | testing | active")
    try:
        reply = CAT.respond(req.message, mode=req.mode)
        CAT.save()
        return {"reply": reply, "stats": CAT.stats(), "mode": req.mode}
    except Exception as e:
        raise HTTPException(500, f"WildCat error: {e}")


@app.post("/save")
def save():
    CAT.save()
    return {"saved": True, "stats": CAT.stats()}


@app.post("/reset")
def reset():
    CAT.reset()
    return {"reset": True, "stats": CAT.stats()}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
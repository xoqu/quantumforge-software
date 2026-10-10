"""FastAPI interface for the containerized RAG bot."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from secure_rag_bot import ProtectionMode, SecureRAGBot


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    protection: ProtectionMode = ProtectionMode.ALL


class Source(BaseModel):
    source: str
    chunk_id: str
    relevance: float


class AskResponse(BaseModel):
    chain_of_thoughts: dict[str, str]
    answer: str
    protection: ProtectionMode
    model_called: bool
    output_blocked: bool
    protection_events: list[str]
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bot = await asyncio.to_thread(SecureRAGBot.from_environment)
    yield


app = FastAPI(
    title="Secure RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/protection-modes")
def protection_modes() -> dict[str, list[str]]:
    return {"modes": [mode.value for mode in ProtectionMode]}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    try:
        result = await asyncio.to_thread(
            request.app.state.bot.ask,
            payload.question,
            payload.protection,
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"RAG request failed: {type(error).__name__}",
        ) from error

    return AskResponse(
        chain_of_thoughts=result.chain_of_thoughts,
        answer=result.answer,
        protection=result.protection_mode,
        model_called=result.model_called,
        output_blocked=result.output_blocked,
        protection_events=list(result.protection_events),
        sources=[
            Source(
                source=str(chunk.metadata.get("source", "unknown")),
                chunk_id=str(chunk.metadata.get("chunk_id", "unknown")),
                relevance=chunk.relevance,
            )
            for chunk in result.chunks
        ],
    )

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict


class LLMQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    include_raw_data: bool = False


def create_llm_router() -> APIRouter:
    router = APIRouter()

    @router.post("/llm/query")
    def llm_query(payload: LLMQueryRequest, request: Request) -> dict[str, Any]:
        agent = request.app.state.llm_agent
        result = agent.answer_query(payload.query, include_raw_data=payload.include_raw_data)
        out: dict[str, Any] = {
            "answer": result.answer,
            "tools_used": result.tools_used,
        }
        if payload.include_raw_data:
            out["raw_data"] = result.raw_data
        return out

    return router

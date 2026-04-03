from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from churnxgb.api.dependencies import build_app_state, repo_root_from_app_file
from churnxgb.api.routers.customers import router as customers_router
from churnxgb.api.routers.llm import router as llm_router
from churnxgb.api.routers.monitoring import router as monitoring_router
from churnxgb.api.routers.policy import router as policy_router
from churnxgb.api.routers.summary import router as summary_router


def create_app(repo_root: Path | None = None) -> FastAPI:
    repo_root = repo_root or repo_root_from_app_file()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for key, value in build_app_state(repo_root).items():
            setattr(app.state, key, value)
        yield

    app = FastAPI(
        title="ChurnXGB API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(summary_router)
    app.include_router(policy_router)
    app.include_router(customers_router)
    app.include_router(monitoring_router)
    app.include_router(llm_router)
    return app


app = create_app()

from __future__ import annotations

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - optional dependency guard
    FastAPI = None

from ..pipeline import run_training_pipeline


def create_app() -> "FastAPI":
    """Create the FastAPI application instance."""
    if FastAPI is None:
        raise ImportError("FastAPI is required to run the API. Install the project dependencies first.")

    app = FastAPI(title="Loan Approval Underwriting Assistant")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/pipeline")
    def pipeline_status() -> dict:
        return run_training_pipeline()

    return app


app = create_app() if FastAPI is not None else None

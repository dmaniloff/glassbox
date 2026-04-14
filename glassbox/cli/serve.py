"""Serve a trained glassbox probe as a REST endpoint.

Wraps a joblib model (from ``glassbox-train``) in a minimal FastAPI app
for integration with external systems like semantic-router.

Usage:
    glassbox-serve model.joblib
    glassbox-serve model.joblib --port 8080 --threshold 0.7
"""

import sys

import click


def _build_app(model_path: str, threshold_override: float | None = None):
    """Build the FastAPI app. Separated from CLI for testability."""
    import joblib
    import numpy as np
    from fastapi import FastAPI
    from pydantic import BaseModel

    model_dict = joblib.load(model_path)
    feat_cols = model_dict["feature_columns"]
    default_threshold = threshold_override or model_dict.get("threshold", 0.5)

    app = FastAPI(title="Glassbox Hallucination Detector")

    class ClassifyRequest(BaseModel):
        features: dict[str, float]
        threshold: float | None = None

    class ClassifyResponse(BaseModel):
        hallucination_probability: float
        is_hallucination: bool
        threshold: float

    class HealthResponse(BaseModel):
        status: str
        signal: str
        n_features: int
        train_auroc: float | None
        test_auroc: float | None

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            signal=model_dict.get("signal", "unknown"),
            n_features=len(feat_cols),
            train_auroc=model_dict.get("train_auroc"),
            test_auroc=model_dict.get("test_auroc"),
        )

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest):
        t = req.threshold if req.threshold is not None else default_threshold

        # Build feature vector in column order
        vec = np.array([req.features.get(col, 0.0) for col in feat_cols], dtype=np.float64)
        X = vec.reshape(1, -1)

        if model_dict.get("pca") is not None:
            X = model_dict["pca"].transform(X)

        proba = float(model_dict["model"].predict_proba(X)[0, 1])

        return ClassifyResponse(
            hallucination_probability=proba,
            is_hallucination=proba >= t,
            threshold=t,
        )

    return app


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host.")
@click.option("--port", default=8080, type=int, show_default=True, help="Bind port.")
@click.option("--threshold", default=None, type=float, help="Override model threshold.")
def main(model_path: str, host: str, port: int, threshold: float | None) -> None:
    """Serve a trained hallucination detection model as a REST API."""
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn not installed. Install with: pip install 'glassbox[serve]'", err=True)
        sys.exit(1)

    app = _build_app(model_path, threshold_override=threshold)
    click.echo(f"Serving model from {model_path} on {host}:{port}")
    click.echo(f"  POST /classify  — run classification")
    click.echo(f"  GET  /health    — model info")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

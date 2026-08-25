from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.common import FREQ_COLS, clean_spectrum, validate_spectrum
from src.semi_supervised import SemiSupervisedDiagnosticPipeline

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "svc_model.pkl"
app = FastAPI(title="Aesteel Engine Diagnostics API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
_pipeline: SemiSupervisedDiagnosticPipeline | None = None
_pipeline_lock = threading.Lock()


def _build_model() -> SemiSupervisedDiagnosticPipeline:
    train = clean_spectrum(pd.read_csv(DATA_DIR / "train.csv"))
    val = clean_spectrum(pd.read_csv(DATA_DIR / "val.csv"))
    pipeline = SemiSupervisedDiagnosticPipeline(feature_count=20)
    pipeline.fit(train, val)
    pipeline.save(MODEL_PATH)
    return pipeline


def _model_is_current() -> bool:
    """Return False when the cached model predates its training/reference data."""
    if not MODEL_PATH.exists():
        return False
    try:
        model_mtime = MODEL_PATH.stat().st_mtime
        data_mtime = max(
            (DATA_DIR / "train.csv").stat().st_mtime,
            (DATA_DIR / "val.csv").stat().st_mtime,
        )
        return model_mtime >= data_mtime
    except OSError:
        return False


def get_pipeline() -> SemiSupervisedDiagnosticPipeline:
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        try:
            if _model_is_current():
                loaded = SemiSupervisedDiagnosticPipeline.load(MODEL_PATH)
                if (
                    getattr(loaded, "selected_features", None)
                    and getattr(loaded, "model", None) is not None
                    and getattr(loaded, "isolation_forest", None) is not None
                ):
                    _pipeline = loaded
                else:
                    _pipeline = _build_model()
            else:
                _pipeline = _build_model()
        except Exception:
            logger.exception("Could not load/build diagnostic model")
            raise
    return _pipeline


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "api": "online",
        "model": "online" if _model_is_current() else "ready-on-startup",
        "model_type": "RBF SVC",
        "anomaly_detector": "IsolationForest",
    }


@app.get("/api/model")
def model_info() -> dict:
    pipeline = get_pipeline()
    return {
        "model": "RBF SVC",
        "anomaly_detector": "IsolationForest",
        "training_data": "train.csv only",
        "reference_data": "val.csv for semantic pseudo-labeling only",
        "features": len(pipeline.selected_features),
        "features_before_selection": len(pipeline.feature_names),
        "selected_features": pipeline.selected_features,
        "hyperparameters": pipeline.params,
        "training_metrics": pipeline.training_metrics,
        "feature_importance": pipeline.feature_importance(12),
        "severity_feature_importance": pipeline.severity_feature_importance(12),
    }


@app.get("/api/features")
def feature_analysis() -> dict:
    pipeline = get_pipeline()
    return {
        "features_before_selection": len(pipeline.feature_names),
        "features_after_selection": len(pipeline.selected_features),
        "selected_features": pipeline.selected_features,
        "importance": pipeline.feature_analysis,
        "severity_importance": pipeline.severity_feature_analysis,
    }


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["engine_id", "cylinder", *FREQ_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")
    cleaned = clean_spectrum(df)
    valid, message = validate_spectrum(cleaned)
    if not valid:
        raise HTTPException(status_code=422, detail=message)
    return cleaned


@app.post("/api/diagnose")
async def diagnose(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Only CSV measurements are supported")
    started = time.perf_counter()
    try:
        payload = await file.read()
        if len(payload) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File is too large (max 20 MB)")
        df = _validate_df(pd.read_csv(pd.io.common.BytesIO(payload)))
        pipeline = get_pipeline()
        results = pipeline.predict(df)
        elapsed_ms = (time.perf_counter() - started) * 1000
        rows = results.to_dict(orient="records")
        label_counts = results["label"].value_counts().to_dict() if not results.empty else {}
        severity_counts = results["severity"].value_counts().to_dict() if not results.empty else {}
        confidence = results["confidence"].to_numpy() if not results.empty else []
        anomaly = results["anomaly_score"].to_numpy() if not results.empty else []
        return JSONResponse(
            {
                "status": "ok",
                "filename": file.filename,
                "row_count": len(rows),
                "analysis_time_ms": round(elapsed_ms, 1),
                "rows": rows,
                "summary": {
                    "label_counts": label_counts,
                    "severity_counts": severity_counts,
                    "mean_confidence": round(float(confidence.mean()), 4) if len(confidence) else 0.0,
                    "min_confidence": round(float(confidence.min()), 4) if len(confidence) else 0.0,
                    "mean_anomaly": round(float(anomaly.mean()), 4) if len(anomaly) else 0.0,
                    "high_anomaly_rows": int((anomaly >= 0.65).sum()) if len(anomaly) else 0,
                },
                "feature_importance": pipeline.feature_importance(12),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Diagnostic request failed")
        raise HTTPException(status_code=400, detail="Measurement could not be processed") from exc

"""Legacy Streamlit diagnostic UI.

The primary UI is frontend/index.html + FastAPI. This module is retained for
backward compatibility, but now loads the same production SVC pipeline as the
FastAPI backend instead of referencing removed legacy model artifacts.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.common import FREQ_COLS, clean_spectrum, load_csv
from src.semi_supervised import SemiSupervisedDiagnosticPipeline

DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_MODEL = ROOT / "models" / "svc_model.pkl"

st.set_page_config(
    page_title="Aesteel Engine Diagnostics",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def create_spectrum_chart(
    df: pd.DataFrame,
    cylinder_idx: int,
    engine_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    row = df.iloc[cylinder_idx]
    spectrum = row[FREQ_COLS].to_numpy(dtype=float)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Widmo akustyczne", "Analiza odchylenia"),
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(21)),
            y=spectrum,
            mode="lines+markers",
            name=f"Cylinder {row['cylinder']}",
        ),
        row=1,
        col=1,
    )
    if engine_df is not None and len(engine_df) > 1:
        other = engine_df[engine_df["cylinder"] != row["cylinder"]]
        if len(other) > 0:
            median = other[FREQ_COLS].median().to_numpy(dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=list(range(21)),
                    y=median,
                    mode="lines",
                    name="Mediana innych cylindrów",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=list(range(21)),
                    y=spectrum - median,
                    name="Odchylenie od mediany",
                ),
                row=2,
                col=1,
            )
    fig.update_layout(
        height=600,
        title_text=f"Widmo cylindra {row['cylinder']} (silnik {row['engine_id']})",
    )
    fig.update_xaxes(title_text="Częstotliwość [kHz]", row=2, col=1)
    fig.update_yaxes(title_text="Amplituda [mV]", row=1, col=1)
    fig.update_yaxes(title_text="Odchylenie [mV]", row=2, col=1)
    return fig


def load_pipeline(model_path: Path, data_dir: Path) -> SemiSupervisedDiagnosticPipeline:
    if model_path.exists():
        return SemiSupervisedDiagnosticPipeline.load(model_path)
    train = clean_spectrum(pd.read_csv(data_dir / "train.csv"))
    val = clean_spectrum(pd.read_csv(data_dir / "val.csv"))
    pipeline = SemiSupervisedDiagnosticPipeline(feature_count=20)
    pipeline.fit(train, val)
    pipeline.save(model_path)
    return pipeline


def run_app(data_dir: Path, model_path: Path) -> None:
    st.title("Aesteel Engine Diagnostics System")
    st.caption("Legacy Streamlit interface — primary interface: frontend/index.html")
    st.sidebar.header("Konfiguracja")
    work_mode = st.sidebar.radio(
        "Tryb pracy:",
        ["Przeglądaj dane testowe", "Upload własnych danych"],
    )

    df = None
    if work_mode == "Upload własnych danych":
        uploaded_file = st.sidebar.file_uploader(
            "Wgraj plik CSV z widmem akustycznym", type=["csv"]
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(io.StringIO(uploaded_file.read().decode("utf-8")))
                required = ["engine_id", "cylinder", *FREQ_COLS]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    st.error(f"Brak wymaganych kolumn: {missing}")
                    return
                df = clean_spectrum(df)
                st.sidebar.success(f"Wczytano {len(df)} próbek")
            except Exception:
                st.error("Nie udało się wczytać pliku CSV.")
                return
    else:
        test_path = data_dir / "test.csv"
        val_path = data_dir / "val.csv"
        if test_path.exists():
            df = clean_spectrum(load_csv(str(test_path)))
        elif val_path.exists():
            df = clean_spectrum(load_csv(str(val_path)))
        else:
            st.error("Nie znaleziono plików test.csv ani val.csv!")
            return

    if df is None or df.empty:
        return

    engines = sorted(df["engine_id"].unique())
    selected_engine = st.sidebar.selectbox("Wybierz silnik:", engines)
    engine_data = df[df["engine_id"] == selected_engine].reset_index(drop=True)
    cylinders = engine_data["cylinder"].tolist()
    selected_cylinder = st.sidebar.selectbox("Wybierz cylinder:", cylinders)
    cylinder_idx = cylinders.index(selected_cylinder)

    prediction = None
    try:
        pipeline = load_pipeline(model_path, data_dir)
        prediction = pipeline.predict(engine_data.iloc[[cylinder_idx]]).iloc[0]
        st.sidebar.success("Model załadowany")
    except Exception as exc:
        st.sidebar.error(f"Model nie może zostać załadowany: {exc}")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(
            create_spectrum_chart(engine_data, cylinder_idx, engine_data),
            use_container_width=True,
        )
    with col2:
        st.subheader("Wynik diagnostyki")
        if prediction is not None:
            st.metric("Label", str(prediction["label"]))
            st.metric("Severity", str(prediction["severity"]))
            st.metric("Confidence", f"{float(prediction['confidence']) * 100:.1f}%")
            st.metric("Anomaly score", f"{float(prediction['anomaly_score']):.3f}")
        else:
            spectrum = engine_data.iloc[cylinder_idx][FREQ_COLS].to_numpy(dtype=float)
            st.metric("Średnia", f"{spectrum.mean():.2f} mV")
            st.metric("Peak", f"{spectrum.max():.2f} mV @ {int(np.argmax(spectrum))} kHz")

    st.subheader(f"Podsumowanie silnika {selected_engine}")
    st.dataframe(engine_data[["cylinder", *FREQ_COLS]], use_container_width=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aesteel Engine Diagnostic App")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    run_app(args.data_dir, args.model_path)

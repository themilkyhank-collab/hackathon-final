"""
ENGIN - Legacy Streamlit diagnostic UI.

The primary UI is now frontend/index.html + FastAPI. This module remains for
backward compatibility with the existing Streamlit workflow.
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import sys
from pathlib import Path as PathLib
src_path = PathLib(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from common import FREQ_COLS, LABELS, FAULT_LABELS, SEVERITIES, load_csv, clean_spectrum
from pipeline import EngineDiagnosticPipeline

st.set_page_config(page_title="Aesteel Engine Diagnostics", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")


def create_spectrum_chart(df: pd.DataFrame, cylinder_idx: int, engine_df: pd.DataFrame = None) -> go.Figure:
    row = df.iloc[cylinder_idx]
    spectrum = row[FREQ_COLS].values.astype(float)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Widmo akustyczne", "Analiza odchylenia"))
    fig.add_trace(go.Scatter(x=list(range(21)), y=spectrum, mode="lines+markers", name=f"Cylinder {row['cylinder']}"), row=1, col=1)
    if engine_df is not None and len(engine_df) > 1:
        other = engine_df[engine_df["cylinder"] != row["cylinder"]]
        if len(other) > 0:
            median = other[FREQ_COLS].median().values.astype(float)
            fig.add_trace(go.Scatter(x=list(range(21)), y=median, mode="lines", name="Mediana innych cylindrów"), row=1, col=1)
            deviation = spectrum - median
            fig.add_trace(go.Bar(x=list(range(21)), y=deviation, name="Odchylenie od mediany"), row=2, col=1)
    fig.update_layout(height=600, title_text=f"Widmo cylindra {row['cylinder']} (silnik {row['engine_id']})")
    fig.update_xaxes(title_text="Częstotliwość [kHz]", row=2, col=1)
    fig.update_yaxes(title_text="Amplituda [mV]", row=1, col=1)
    fig.update_yaxes(title_text="Odchylenie [mV]", row=2, col=1)
    return fig


def run_app(data_dir: Path, model_path: Optional[Path] = None):
    st.title("Aesteel Engine Diagnostics System")
    st.caption("Legacy Streamlit interface — primary interface: frontend/index.html")
    st.sidebar.header("Konfiguracja")
    work_mode = st.sidebar.radio("Tryb pracy:", ["Przeglądaj dane testowe", "Upload własnych danych"])
    df = None
    uploaded_file = None

    if work_mode == "Upload własnych danych":
        uploaded_file = st.sidebar.file_uploader("Wgraj plik CSV z widmem akustycznym", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(io.StringIO(uploaded_file.read().decode("utf-8")))
                required = ["engine_id", "cylinder", *FREQ_COLS]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    st.error(f"Brak wymaganych kolumn: {missing}")
                    df = None
                else:
                    df = clean_spectrum(df)
                    st.sidebar.success(f"Wczytano {len(df)} próbek")
            except Exception as exc:
                st.error("Nie udało się wczytać pliku CSV.")
                logging.getLogger(__name__).exception("CSV upload failed: %s", exc)
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

    if df is None:
        return

    engines = sorted(df["engine_id"].unique())
    selected_engine = st.sidebar.selectbox("Wybierz silnik:", engines)
    engine_data = df[df["engine_id"] == selected_engine].reset_index(drop=True)
    cylinders = engine_data["cylinder"].tolist()
    selected_cylinder = st.sidebar.selectbox("Wybierz cylinder:", cylinders)
    cylinder_idx = cylinders.index(selected_cylinder)

    pipeline = None
    prediction = None
    if model_path and model_path.exists():
        try:
            pipeline = EngineDiagnosticPipeline.load(str(model_path))
            prediction = pipeline.predict(engine_data.iloc[[cylinder_idx]]).iloc[0]
            st.sidebar.success("Model załadowany")
        except Exception as exc:
            st.sidebar.warning("Model nie może zostać załadowany")
            logging.getLogger(__name__).exception("Model load failed: %s", exc)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(create_spectrum_chart(engine_data, cylinder_idx, engine_data), use_container_width=True)
    with col2:
        st.subheader("Wynik diagnostyki")
        if prediction is not None:
            st.metric("Label", str(prediction["label"]))
            st.metric("Severity", str(prediction["severity"]))
            st.metric("Confidence", f"{float(prediction['confidence'])*100:.1f}%")
            st.metric("Anomaly score", f"{float(prediction['anomaly_score']):.3f}")
        else:
            spectrum = engine_data.iloc[cylinder_idx][FREQ_COLS].values.astype(float)
            st.metric("Średnia", f"{spectrum.mean():.2f} mV")
            st.metric("Peak", f"{spectrum.max():.2f} mV @ {int(np.argmax(spectrum))} kHz")

    st.subheader(f"Podsumowanie silnika {selected_engine}")
    st.dataframe(engine_data[["cylinder", *FREQ_COLS]], use_container_width=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aesteel Engine Diagnostic App")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=Path("models/engin_pipeline.pkl"))
    args = parser.parse_args()
    run_app(args.data_dir, args.model_path)

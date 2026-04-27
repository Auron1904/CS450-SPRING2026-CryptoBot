from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.ohlcv import fetch_and_save_ohlcv
from src.features.indicators import build_feature_dataset
from src.model.rolling_year_eval import rolling_year_evaluation
from src.model.train import (
    build_feature_importance_table,
    load_training_data,
    train_xgb_with_timeseries_cv,
)


def render_pipeline() -> None:
    """
    Render the ML pipeline tab: Fetch → Build → Train → Evaluate.
    """
    st.header("ML Pipeline")

    # Initialize session state
    if "pipeline_raw_df" not in st.session_state:
        st.session_state["pipeline_raw_df"] = None
    if "pipeline_features_df" not in st.session_state:
        st.session_state["pipeline_features_df"] = None
    if "pipeline_train_results" not in st.session_state:
        st.session_state["pipeline_train_results"] = None
    if "pipeline_eval_summary" not in st.session_state:
        st.session_state["pipeline_eval_summary"] = None
    if "pipeline_predictions" not in st.session_state:
        st.session_state["pipeline_predictions"] = None

    project_root = Path(__file__).resolve().parents[2]

    # ========== SECTION 1: FETCH OHLCV DATA ==========
    st.subheader("1️⃣ Fetch OHLCV Data")
    if st.button("Fetch OHLCV Data", key="fetch_ohlcv_btn"):
        with st.spinner("Fetching OHLCV data from Yahoo Finance..."):
            try:
                raw_output_dir = project_root / "data" / "raw"
                raw_output_dir.mkdir(parents=True, exist_ok=True)

                raw_df, output_path = fetch_and_save_ohlcv(
                    symbol="BTC-USD",
                    timeframe="1d",
                    output_dir=raw_output_dir,
                )
                st.session_state["pipeline_raw_df"] = raw_df

                st.success(f"✅ OHLCV data fetched and saved to {output_path}")
                st.info(
                    f"📊 Rows: {len(raw_df)} | Date range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}"
                )

                with st.expander("Show first 5 rows"):
                    st.dataframe(raw_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Failed to fetch OHLCV data: {e}")

    # ========== SECTION 2: BUILD FEATURES ==========
    st.subheader("2️⃣ Build Features")
    if st.button("Build Features", key="build_features_btn"):
        with st.spinner("Building feature dataset..."):
            try:
                if st.session_state["pipeline_raw_df"] is None:
                    st.warning("⚠️ Please fetch OHLCV data first.")
                else:
                    raw_df = st.session_state["pipeline_raw_df"]
                    feature_df = build_feature_dataset(raw_df)

                    processed_path = project_root / "data" / "processed"
                    processed_path.mkdir(parents=True, exist_ok=True)
                    feature_csv = processed_path / "btc_features.csv"
                    feature_df.to_csv(feature_csv, index=False)

                    st.session_state["pipeline_features_df"] = feature_df

                    st.success(f"✅ Features built and saved to {feature_csv}")
                    st.info(
                        f"📊 Rows: {len(feature_df)} | Columns: {len(feature_df.columns)}"
                    )

                    with st.expander("Show first 5 rows"):
                        st.dataframe(feature_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Failed to build features: {e}")

    # ========== SECTION 3: TRAIN MODEL ==========
    st.subheader("3️⃣ Train Model")
    if st.button("Train Model", key="train_model_btn"):
        with st.spinner("Training XGBoost model..."):
            try:
                feature_path = project_root / "data" / "processed" / "btc_features.csv"
                if not feature_path.exists():
                    st.warning("⚠️ Please build features first.")
                else:
                    X, y = load_training_data(feature_path)
                    results = train_xgb_with_timeseries_cv(X, y, n_splits=5)

                    model = results["model"]
                    model_path = project_root / "models"
                    model_path.mkdir(parents=True, exist_ok=True)
                    model.save_model(str(model_path / "btc_model.json"))

                    st.session_state["pipeline_train_results"] = results

                    st.success("✅ Model trained and saved")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                    col2.metric("Precision", f"{results['precision']:.4f}")
                    col3.metric("Recall", f"{results['recall']:.4f}")
                    col4.metric("Rows Used", len(X))

                    st.caption(f"Fold Number: {results['fold_number']}")

                    with st.expander("Show Classification Report"):
                        st.text(results["classification_report"])

                    with st.expander("Show Feature Importance (Top 10)"):
                        importance_df = build_feature_importance_table(results["model"])
                        st.dataframe(
                            importance_df.head(10), use_container_width=True
                        )
            except Exception as e:
                st.error(f"❌ Failed to train model: {e}")

    # ========== SECTION 4: EVALUATE MODEL ==========
    st.subheader("4️⃣ Evaluate Model")
    if st.button("Evaluate Model", key="evaluate_model_btn"):
        with st.spinner("Running walk-forward evaluation..."):
            try:
                feature_path = project_root / "data" / "processed" / "btc_features.csv"
                if not feature_path.exists():
                    st.warning("⚠️ Please build features first.")
                else:
                    feature_df = pd.read_csv(feature_path)
                    # Convert timestamp to datetime for year extraction in rolling_year_evaluation
                    feature_df["timestamp"] = pd.to_datetime(
                        feature_df["timestamp"], errors="coerce"
                    )
                    summary_df, predictions_df = rolling_year_evaluation(
                        feature_df, train_years=2
                    )

                    # Save to outputs/ so backtest tab can load them
                    outputs_path = project_root / "outputs"
                    outputs_path.mkdir(parents=True, exist_ok=True)

                    summary_df.to_csv(outputs_path / "rolling_summary.csv", index=False)
                    predictions_df.to_csv(outputs_path / "rolling_predictions.csv", index=False)

                    st.session_state["pipeline_eval_summary"] = summary_df
                    st.session_state["pipeline_predictions"] = predictions_df

                    st.success("✅ Evaluation complete and saved to outputs/")

                    with st.expander("Show Evaluation Summary"):
                        st.dataframe(summary_df, use_container_width=True)

                    # Display accuracy line chart
                    if "test_year" in summary_df.columns and "accuracy" in summary_df.columns:
                        st.line_chart(
                            summary_df.set_index("test_year")[["accuracy"]],
                        )

                    with st.expander("Show Download Links"):
                        col1, col2 = st.columns(2)
                        col1.download_button(
                            "📥 Download Summary CSV",
                            summary_df.to_csv(index=False),
                            "eval_summary.csv",
                            "text/csv",
                        )
                        col2.download_button(
                            "📥 Download Predictions CSV",
                            predictions_df.to_csv(index=False),
                            "eval_predictions.csv",
                            "text/csv",
                        )
            except Exception as e:
                st.error(f"❌ Failed to evaluate model: {e}")

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.backtest.ai_engine import (
    load_predictions,
    load_price_data,
    prepare_backtest_frame,
    run_single_year_backtest,
)


def calculate_sharpe_ratio(equity_series: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio from equity curve.
    """
    if equity_series.empty or len(equity_series) < 2:
        return 0.0
    returns = equity_series.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252)
    return float((excess_returns.mean() / excess_returns.std()) * np.sqrt(252))


def render_backtest() -> None:
    """
    Render the backtest tab: evaluation results and performance metrics.
    """
    st.header("Backtest & Evaluation")

    # Initialize session state
    if "backtest_summary" not in st.session_state:
        st.session_state["backtest_summary"] = None
    if "backtest_equity_curve" not in st.session_state:
        st.session_state["backtest_equity_curve"] = None
    if "backtest_predictions" not in st.session_state:
        st.session_state["backtest_predictions"] = None

    project_root = Path(__file__).resolve().parents[2]

    # ========== SECTION 1: ROLLING YEAR EVALUATION RESULTS ==========
    st.subheader("1️⃣ Rolling Year Evaluation Results")

    try:
        # Try to load from session state first
        eval_summary = st.session_state.get("pipeline_eval_summary")
        eval_predictions = st.session_state.get("pipeline_predictions")

        # If not in session, try to load from files
        if eval_summary is None:
            eval_summary_path = (
                project_root / "outputs" / "rolling_summary.csv"
            )
            if eval_summary_path.exists():
                eval_summary = pd.read_csv(eval_summary_path)
                st.session_state["pipeline_eval_summary"] = eval_summary

        if eval_predictions is None:
            eval_predictions_path = (
                project_root / "outputs" / "rolling_predictions.csv"
            )
            if eval_predictions_path.exists():
                eval_predictions = pd.read_csv(eval_predictions_path)
                st.session_state["pipeline_predictions"] = eval_predictions

        if eval_summary is not None:
            st.success("✅ Evaluation results loaded")

            with st.expander("Show Summary Table", expanded=True):
                st.dataframe(eval_summary, use_container_width=True)

            # Accuracy chart
            if "test_year" in eval_summary.columns and "accuracy" in eval_summary.columns:
                st.line_chart(
                    eval_summary.set_index("test_year")[["accuracy"]],
                    y_label="Accuracy",
                    use_container_width=True,
                )
        else:
            st.info("📊 No evaluation results found. Run the Pipeline tab first.")

    except Exception as e:
        st.error(f"❌ Failed to load evaluation results: {e}")

    # ========== SECTION 2: PERFORMANCE METRICS ==========
    st.subheader("2️⃣ Performance Metrics")

    try:
        eval_summary = st.session_state.get("pipeline_eval_summary")
        if eval_summary is not None and not eval_summary.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Avg Accuracy",
                f"{eval_summary['accuracy'].mean():.4f}",
            )
            col2.metric(
                "Avg Precision",
                f"{eval_summary['precision'].mean():.4f}",
            )
            col3.metric(
                "Avg Recall",
                f"{eval_summary['recall'].mean():.4f}",
            )
        else:
            st.info("📊 Metrics unavailable. Run the Pipeline tab first.")
    except Exception as e:
        st.error(f"❌ Failed to calculate metrics: {e}")

    # ========== SECTION 3: BACKTEST RESULTS & EQUITY CURVE ==========
    st.subheader("3️⃣ AI Backtest Results")

    col_run, col_clear = st.columns(2)
    with col_run:
        run_backtest = st.button("Run AI Backtest", key="run_backtest_btn")
    with col_clear:
        if st.button("Clear Backtest", key="clear_backtest_btn"):
            st.session_state["backtest_summary"] = None
            st.session_state["backtest_equity_curve"] = None
            st.session_state["backtest_predictions"] = None
            st.rerun()

    if run_backtest:
        with st.spinner("Running AI backtest..."):
            try:
                predictions_path = (
                    project_root / "outputs" / "rolling_predictions.csv"
                )
                feature_path = (
                    project_root / "data" / "processed" / "btc_features.csv"
                )

                if not predictions_path.exists():
                    st.warning(
                        "⚠️ Predictions not found at outputs/rolling_predictions.csv"
                    )
                    st.info("Run the Pipeline tab and click 'Evaluate Model' first")
                    return
                elif not feature_path.exists():
                    st.warning(
                        "⚠️ Feature data not found. Please build features first."
                    )
                else:
                    predictions_df = load_predictions(predictions_path)
                    price_df = load_price_data(feature_path)
                    merged_df = prepare_backtest_frame(predictions_df, price_df)

                    # Run backtest on all years
                    all_equity_curves = []
                    all_summary_stats = []

                    for test_year in sorted(merged_df["test_year"].unique()):
                        year_df = merged_df[
                            merged_df["test_year"] == test_year
                        ].copy()
                        if year_df.empty:
                            continue

                        equity_curve, trades, summary = run_single_year_backtest(
                            year_df,
                            initial_cash=10000,
                            fee_rate=0.001,
                            buy_threshold=0.52,
                            sell_threshold=0.48,
                            probabilistic_trading=True,
                        )

                        all_equity_curves.append(
                            equity_curve.assign(test_year=test_year)
                        )
                        all_summary_stats.append(summary)

                    if all_equity_curves:
                        combined_equity = pd.concat(all_equity_curves, ignore_index=True)
                        summary_stats_df = pd.DataFrame(all_summary_stats)

                        st.session_state["backtest_equity_curve"] = combined_equity
                        st.session_state["backtest_summary"] = summary_stats_df

                        st.success("✅ Backtest complete")

                        # Calculate Sharpe ratio
                        sharpe = calculate_sharpe_ratio(combined_equity["equity"])

                        # Display metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric(
                            "Avg Return %",
                            f"{summary_stats_df['total_return_pct'].mean():.2f}%",
                        )
                        col2.metric(
                            "Avg Win Rate %",
                            f"{summary_stats_df['win_rate_pct'].mean():.2f}%",
                        )
                        col3.metric(
                            "Avg Max DD %",
                            f"{summary_stats_df['max_drawdown_pct'].mean():.2f}%",
                        )
                        col4.metric(
                            "Total Trades",
                            int(summary_stats_df["closed_trades"].sum()),
                        )
                        col5.metric(
                            "Avg Sharpe",
                            f"{sharpe:.4f}",
                        )

                        # Equity curve chart
                        st.line_chart(
                            combined_equity.set_index("timestamp")["equity"],
                            y_label="Portfolio Equity ($)",
                            use_container_width=True,
                        )

                        with st.expander("Show Backtest Summary by Year"):
                            st.dataframe(summary_stats_df, use_container_width=True)
                    else:
                        st.warning("⚠️ No backtest results generated.")

            except Exception as e:
                st.error(f"❌ Backtest failed: {e}")

    # Display stored backtest results if available
    if st.session_state["backtest_summary"] is not None:
        st.subheader("📈 Stored Backtest Results")

        summary_df = st.session_state["backtest_summary"]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(
            "Avg Return %",
            f"{summary_df['total_return_pct'].mean():.2f}%",
        )
        col2.metric(
            "Avg Win Rate %",
            f"{summary_df['win_rate_pct'].mean():.2f}%",
        )
        col3.metric(
            "Avg Max DD %",
            f"{summary_df['max_drawdown_pct'].mean():.2f}%",
        )
        col4.metric(
            "Total Trades",
            int(summary_df["closed_trades"].sum()),
        )

        if st.session_state["backtest_equity_curve"] is not None:
            sharpe = calculate_sharpe_ratio(
                st.session_state["backtest_equity_curve"]["equity"]
            )
            col5.metric(
                "Avg Sharpe",
                f"{sharpe:.4f}",
            )

        if st.session_state["backtest_equity_curve"] is not None:
            st.line_chart(
                st.session_state["backtest_equity_curve"].set_index("timestamp")[
                    "equity"
                ],
                y_label="Portfolio Equity ($)",
                use_container_width=True,
            )

        with st.expander("Show Download Links"):
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Summary CSV",
                    summary_df.to_csv(index=False),
                    "backtest_summary.csv",
                    "text/csv",
                )
            with col2:
                if st.session_state["backtest_equity_curve"] is not None:
                    st.download_button(
                        "📥 Download Equity Curve CSV",
                        st.session_state["backtest_equity_curve"].to_csv(index=False),
                        "backtest_equity_curve.csv",
                        "text/csv",
                    )

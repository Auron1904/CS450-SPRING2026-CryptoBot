import streamlit as st


def render_manual_controls(client) -> None:
    try:
        if "latest_order" not in st.session_state:
            st.session_state["latest_order"] = None

        st.sidebar.header("Manual Trade")
        trade_notional = st.sidebar.number_input(
            "Order Notional (USD)", min_value=1.0, value=50.0, step=1.0
        )
        confirm_trade = st.sidebar.checkbox("I confirm this paper trade")

        st.sidebar.subheader("Developer Override")
        force_signal = st.sidebar.selectbox(
            "Force Signal",
            ("None", "BUY", "SELL"),
            key="force_signal",
        )

        if st.sidebar.button("Execute Paper Trade"):
            if client is None:
                st.sidebar.error(
                    "Alpaca client is not configured. Check your .env keys."
                )
                return

            if not confirm_trade:
                st.sidebar.warning("Please confirm before placing an order.")
                return

            latest_signal = st.session_state.get("latest_signal")
            if not latest_signal and force_signal == "None":
                st.sidebar.warning("Generate a trade signal first.")
                return

            signal_value = "HOLD"
            if latest_signal:
                signal_value = latest_signal.get("signal", "HOLD")
            if force_signal in {"BUY", "SELL"}:
                signal_value = force_signal

            if signal_value == "HOLD":
                st.sidebar.warning("Current signal is HOLD. No order sent.")
                return
            if signal_value not in {"BUY", "SELL"}:
                st.sidebar.error("Invalid signal. Please generate a new signal.")
                return

            positions = client.get_open_positions()
            if "error" in positions:
                st.sidebar.error(positions["error"])
                return

            if signal_value == "SELL" and not positions.get("has_btc_position", False):
                st.sidebar.warning("No BTC position found. SELL order blocked.")
                return

            order_result = client.execute_market_order(
                symbol="BTC/USD",
                notional=float(trade_notional),
                side=signal_value,
            )
            st.session_state["latest_order"] = order_result

            if "error" in order_result:
                st.sidebar.error(order_result["error"])
            else:
                st.sidebar.success(
                    f"Order sent: {signal_value} ({order_result['status']})"
                )
                st.toast(f"Forced/manual order sent: {signal_value}")

            
    except Exception as exc:
        st.sidebar.error(f"Controls component failed: {exc}")

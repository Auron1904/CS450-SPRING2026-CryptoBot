import streamlit as st


def render_account_metrics(client) -> None:
    try:
        st.subheader("Account Summary")

        if client is None:
            st.error("Alpaca client is not configured. Check your .env keys.")
            return

        account_details = client.get_account_details()
        if "error" in account_details:
            st.error(account_details["error"])
            return

        positions = client.get_open_positions()
        if "error" in positions:
            st.error(positions["error"])
            return

        snapshot = {
            "equity": float(account_details.get("equity", "0")),
            "buying_power": float(account_details.get("buying_power", "0")),
            "position_state": positions.get("position_state", "Flat"),
            "unrealized_pl": float(positions.get("unrealized_pl", "0")),
        }

        previous_snapshot = st.session_state.get("previous_account_snapshot")
        st.session_state["previous_account_snapshot"] = snapshot

        equity_delta = None
        buying_power_delta = None
        if isinstance(previous_snapshot, dict):
            equity_delta = snapshot["equity"] - float(
                previous_snapshot.get("equity", 0.0)
            )
            buying_power_delta = snapshot["buying_power"] - float(
                previous_snapshot.get("buying_power", 0.0)
            )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Total Equity",
            f"${snapshot['equity']:,.2f}",
            None if equity_delta is None else f"${equity_delta:,.2f}",
        )
        col2.metric(
            "Buying Power",
            f"${snapshot['buying_power']:,.2f}",
            None if buying_power_delta is None else f"${buying_power_delta:,.2f}",
        )
        col3.metric("Unrealized P/L", f"${snapshot['unrealized_pl']:,.2f}")
        st.caption(f"Current Position: {snapshot['position_state']}")
    except Exception as exc:
        st.error(f"Account component failed: {exc}")

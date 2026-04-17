import os

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from dotenv import load_dotenv


class AlpacaClient:
    def __init__(self) -> None:
        load_dotenv()

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL")

        if not api_key or not secret_key:
            raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY in .env")

        self.client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
            url_override=base_url if base_url else None,
        )

    def get_account_details(self) -> dict:
        try:
            account = self.client.get_account()
            return {
                "cash": str(account.cash),
                "buying_power": str(account.buying_power),
                "equity": str(account.equity),
            }
        except Exception as exc:
            return {"error": f"Failed to fetch account details: {exc}"}

    def execute_market_order(
        self, symbol: str, notional: float, side: str | OrderSide
    ) -> dict:
        if symbol != "BTC/USD":
            return {"error": "Only BTC/USD is supported"}

        if notional <= 0:
            return {"error": "Notional must be greater than 0"}

        parsed_side = self._parse_side(side)
        if parsed_side is None:
            return {"error": "Side must be BUY or SELL"}

        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=parsed_side,
                time_in_force=TimeInForce.GTC,
            )
            order = self.client.submit_order(order_data=order_request)
            return {"order_id": str(order.id), "status": str(order.status)}
        except Exception as exc:
            return {"error": f"Failed to execute market order: {exc}"}

    def get_open_positions(self) -> dict:
        try:
            positions = self.client.get_all_positions()
            btc_position = next(
                (
                    position
                    for position in positions
                    if position.symbol in {"BTCUSD", "BTC/USD"}
                    and float(position.qty) > 0
                ),
                None,
            )

            if btc_position is None:
                return {
                    "has_btc_position": False,
                    "qty": "0",
                    "position_state": "Flat",
                    "unrealized_pl": "0",
                }

            return {
                "has_btc_position": True,
                "qty": str(btc_position.qty),
                "symbol": str(btc_position.symbol),
                "position_state": "Long",
                "unrealized_pl": str(btc_position.unrealized_pl),
            }
        except Exception as exc:
            return {"error": f"Failed to fetch open positions: {exc}"}

    def get_recent_trades(self, limit: int = 10) -> list[dict] | dict:
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            orders = self.client.get_orders(filter=request)
            rows = []
            for order in orders:
                rows.append(
                    {
                        "timestamp": str(order.created_at),
                        "symbol": str(order.symbol),
                        "side": str(order.side),
                        "notional_usd": str(order.notional),
                        "status": str(order.status),
                    }
                )
            return rows
        except Exception as exc:
            return {"error": f"Failed to fetch recent trades: {exc}"}

    @staticmethod
    def _parse_side(side: str | OrderSide) -> OrderSide | None:
        if isinstance(side, OrderSide):
            if side in {OrderSide.BUY, OrderSide.SELL}:
                return side
            return None

        normalized = str(side).strip().upper()
        if normalized == "BUY":
            return OrderSide.BUY
        if normalized == "SELL":
            return OrderSide.SELL
        return None

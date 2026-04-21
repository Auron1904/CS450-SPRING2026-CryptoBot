from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import pandas as pd


@dataclass
class PerformanceSummary:
    total_return_pct: float
    win_rate_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    final_equity: float
    total_trades: int


class BacktestEngine:
    """
    Simple long-only backtesting engine for feature-engineered market data.

    Strategy rules for the first test pass:
    - BUY when RSI_14 < 30 and there is no open position.
    - SELL when RSI_14 > 70 and there is an open position.
    - HOLD otherwise.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        starting_capital: float = 10_000.0,
        fee_rate: float = 0.001,
    ) -> None:
        self.df = self._prepare_data(df)
        self.starting_capital = float(starting_capital)
        self.fee_rate = float(fee_rate)

        self.cash = self.starting_capital
        self.position_units = 0.0
        self.entry_price: float | None = None
        self.entry_fee: float = 0.0

        self.trade_history: list[dict] = []
        self.equity_curve: list[dict] = []
        self.closed_trade_returns: list[float] = []

    @staticmethod
    def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize the input data before running the backtest."""
        prepared_df = df.copy()

        required_columns = {"timestamp", "close", "RSI_14"}
        missing_columns = sorted(required_columns.difference(prepared_df.columns))
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Input dataframe is missing required columns: {missing}")

        prepared_df["timestamp"] = pd.to_datetime(prepared_df["timestamp"], errors="coerce")
        prepared_df["close"] = pd.to_numeric(prepared_df["close"], errors="coerce")
        prepared_df["RSI_14"] = pd.to_numeric(prepared_df["RSI_14"], errors="coerce")

        prepared_df = prepared_df.dropna(subset=["timestamp", "close", "RSI_14"])
        prepared_df = prepared_df.sort_values("timestamp").reset_index(drop=True)

        if prepared_df.empty:
            raise ValueError("Input dataframe has no usable rows after cleaning.")

        return prepared_df

    def _infer_periods_per_year(self) -> float:
        """
        Estimate how many bars occur in a year from timestamp spacing.

        This keeps Sharpe annualization aligned with the actual data frequency
        instead of assuming every backtest uses daily candles.
        """
        timestamp_deltas = self.df["timestamp"].diff().dropna()
        if timestamp_deltas.empty:
            return 365.25

        median_delta_seconds = timestamp_deltas.dt.total_seconds().median()
        if pd.isna(median_delta_seconds) or median_delta_seconds <= 0:
            return 365.25

        seconds_per_year = 365.25 * 24 * 60 * 60
        return seconds_per_year / float(median_delta_seconds)

    def _generate_signal(self, rsi_value: float) -> str:
        """Return the trading action for the current bar."""
        if self.position_units == 0 and rsi_value < 30:
            return "BUY"
        if self.position_units > 0 and rsi_value > 70:
            return "SELL"
        return "HOLD"

    def _buy(self, timestamp: pd.Timestamp, price: float, rsi_value: float) -> None:
        """
        Enter a full-size long position using all available cash.

        The fee is taken from available cash first, and the remainder is used
        to purchase the asset.
        """
        if self.cash <= 0:
            return

        fee_paid = self.cash * self.fee_rate
        notional = self.cash - fee_paid
        units_bought = notional / price

        self.position_units = units_bought
        self.cash = 0.0
        self.entry_price = price
        self.entry_fee = fee_paid

        self.trade_history.append(
            {
                "timestamp": timestamp,
                "action": "BUY",
                "price": price,
                "rsi_14": rsi_value,
                "units": units_bought,
                "gross_value": notional,
                "fee": fee_paid,
                "cash_after_trade": self.cash,
                "position_units_after_trade": self.position_units,
                "realized_pnl": pd.NA,
                "equity_after_trade": self.cash + self.position_units * price,
            }
        )

    def _sell(self, timestamp: pd.Timestamp, price: float, rsi_value: float) -> None:
        """Close the full long position and record realized PnL."""
        if self.position_units <= 0 or self.entry_price is None:
            return

        gross_value = self.position_units * price
        fee_paid = gross_value * self.fee_rate
        cash_received = gross_value - fee_paid

        entry_gross_cost = self.position_units * self.entry_price
        total_cost_basis = entry_gross_cost + self.entry_fee
        realized_pnl = cash_received - total_cost_basis

        if total_cost_basis > 0:
            self.closed_trade_returns.append(realized_pnl / total_cost_basis)

        self.cash = cash_received
        self.trade_history.append(
            {
                "timestamp": timestamp,
                "action": "SELL",
                "price": price,
                "rsi_14": rsi_value,
                "units": self.position_units,
                "gross_value": gross_value,
                "fee": fee_paid,
                "cash_after_trade": self.cash,
                "position_units_after_trade": 0.0,
                "realized_pnl": realized_pnl,
                "equity_after_trade": self.cash,
            }
        )

        self.position_units = 0.0
        self.entry_price = None
        self.entry_fee = 0.0

    def _record_equity_snapshot(
        self,
        timestamp: pd.Timestamp,
        price: float,
        signal: str,
    ) -> None:
        """Track mark-to-market account value for performance metrics."""
        equity = self.cash + self.position_units * price
        self.equity_curve.append(
            {
                "timestamp": timestamp,
                "close": price,
                "signal": signal,
                "cash": self.cash,
                "position_units": self.position_units,
                "equity": equity,
            }
        )

    def run(self) -> PerformanceSummary:
        """Execute the RSI strategy across the full dataframe."""
        first_row = self.df.iloc[0]
        self._record_equity_snapshot(
            timestamp=first_row["timestamp"],
            price=float(first_row["close"]),
            signal="START",
        )

        for row in self.df.itertuples(index=False):
            timestamp = row.timestamp
            close_price = float(row.close)
            rsi_value = float(row.RSI_14)

            signal = self._generate_signal(rsi_value)

            if signal == "BUY":
                self._buy(timestamp=timestamp, price=close_price, rsi_value=rsi_value)
            elif signal == "SELL":
                self._sell(timestamp=timestamp, price=close_price, rsi_value=rsi_value)

            self._record_equity_snapshot(
                timestamp=timestamp,
                price=close_price,
                signal=signal,
            )

        return self.calculate_performance_metrics()

    def calculate_performance_metrics(self) -> PerformanceSummary:
        """Calculate portfolio-level summary statistics from the backtest."""
        if not self.equity_curve:
            raise ValueError("Run the backtest before calculating metrics.")

        equity_df = pd.DataFrame(self.equity_curve)
        equity_series = equity_df["equity"]

        final_equity = float(equity_series.iloc[-1])
        total_return_pct = ((final_equity / self.starting_capital) - 1.0) * 100

        if self.closed_trade_returns:
            winning_trades = sum(trade_return > 0 for trade_return in self.closed_trade_returns)
            win_rate_pct = (winning_trades / len(self.closed_trade_returns)) * 100
        else:
            win_rate_pct = 0.0

        running_peak = equity_series.cummax()
        drawdown_series = (equity_series / running_peak) - 1.0
        max_drawdown_pct = float(drawdown_series.min()) * 100

        annualization_factor = self._infer_periods_per_year()
        period_returns = equity_series.pct_change().dropna()
        if period_returns.empty or float(period_returns.std()) == 0.0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = float(
                (period_returns.mean() / period_returns.std()) * sqrt(annualization_factor)
            )

        return PerformanceSummary(
            total_return_pct=total_return_pct,
            win_rate_pct=win_rate_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            final_equity=final_equity,
            total_trades=len(self.closed_trade_returns),
        )

    def get_trade_history_df(self) -> pd.DataFrame:
        """Return executed trades as a dataframe for inspection or saving."""
        return pd.DataFrame(self.trade_history)

    def save_trade_history(self, output_path: str | Path) -> Path:
        """Save executed trades to CSV."""
        trade_df = self.get_trade_history_df()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        trade_df.to_csv(destination, index=False)
        return destination

    @staticmethod
    def print_summary(summary: PerformanceSummary) -> None:
        """Print a readable performance report to the terminal."""
        print("Backtest Summary")
        print("----------------")
        print(f"Final Equity: ${summary.final_equity:,.2f}")
        print(f"Total Return: {summary.total_return_pct:.2f}%")
        print(f"Win Rate: {summary.win_rate_pct:.2f}%")
        print(f"Max Drawdown: {summary.max_drawdown_pct:.2f}%")
        print(f"Sharpe Ratio: {summary.sharpe_ratio:.4f}")
        print(f"Closed Trades: {summary.total_trades}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "btc_features.csv"
    output_path = project_root / "data" / "results" / "backtest_log.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at '{input_path}'. "
            "Run src/features/indicators.py first."
        )

    features_df = pd.read_csv(input_path)
    engine = BacktestEngine(features_df, starting_capital=10_000.0, fee_rate=0.001)
    summary = engine.run()
    saved_path = engine.save_trade_history(output_path)

    BacktestEngine.print_summary(summary)
    print(f"\nSaved trade history to: {saved_path}")

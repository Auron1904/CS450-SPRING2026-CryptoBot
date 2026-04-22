from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import random


PREDICTIONS_REQUIRED_COLUMNS = [
    "timestamp",
    "y_true",
    "y_pred",
    "prob_up",
    "train_years",
    "test_year",
]

PRICE_REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "close",
]


@dataclass
class Trade:
    test_year: int
    entry_signal_time: str
    entry_exec_time: str
    exit_signal_time: str
    exit_exec_time: str
    entry_price: float
    exit_price: float
    btc_size: float
    fees_paid: float
    pnl_dollars: float
    return_pct: float
    entry_prob_up: float
    exit_prob_up: float


def load_predictions(predictions_path: str | Path) -> pd.DataFrame:
    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in PREDICTIONS_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Predictions file is missing required columns: {', '.join(missing)}"
        )

    df = df[PREDICTIONS_REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["prob_up"] = pd.to_numeric(df["prob_up"], errors="coerce")
    df["test_year"] = pd.to_numeric(df["test_year"], errors="coerce")

    df = df.dropna().sort_values(["test_year", "timestamp"]).reset_index(drop=True)
    df["y_true"] = df["y_true"].astype(int)
    df["y_pred"] = df["y_pred"].astype(int)
    df["test_year"] = df["test_year"].astype(int)
    return df


def load_price_data(feature_data_path: str | Path) -> pd.DataFrame:
    path = Path(feature_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in PRICE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature dataset is missing required columns: {', '.join(missing)}"
        )

    df = df[PRICE_REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Execution happens on the next market bar's open.
    df["next_open"] = df["open"].shift(-1)
    df["next_timestamp"] = df["timestamp"].shift(-1)

    return df


def prepare_backtest_frame(
    predictions_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge out-of-sample predictions with market prices.

    Each prediction made at timestamp T is executed at timestamp T+1 open.
    """
    merged = predictions_df.merge(price_df, on="timestamp", how="inner")

    merged = merged.dropna(
        subset=["timestamp", "close", "next_open", "next_timestamp", "prob_up", "y_pred", "test_year"]
    ).copy()

    merged = merged.sort_values(["test_year", "timestamp"]).reset_index(drop=True)
    return merged


def compute_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    return float(drawdown.min() * 100.0)

def scale_probability(
    prob_up: float,
    buy_threshold: float,
    max_prob: float = 0.75,
) -> float:
    """
    Convert model probability into a trading probability.

    Below buy_threshold -> 0 chance
    At/above max_prob   -> 1 chance
    In between          -> linearly scaled chance
    """
    if prob_up < buy_threshold:
        return 0.0
    if prob_up >= max_prob:
        return 1.0

    width = max_prob - buy_threshold
    if width <= 0:
        return 1.0

    return (prob_up - buy_threshold) / width

def run_single_year_backtest(
    year_df: pd.DataFrame,
    *,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
    buy_threshold: float = 0.52,
    sell_threshold: float = 0.48,
    probabilistic_trading: bool = True,
    rng: random.Random | None = None,
    exit_patience: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Strategy modes:

    Deterministic:
      - BUY if flat and y_pred == 1 and prob_up >= buy_threshold
      - SELL if long and (y_pred == 0 or prob_up < sell_threshold)

    Probabilistic:
      - BUY with probability scaled from prob_up
      - SELL using the same deterministic exit rule

    Signals are generated from prediction row T and executed at next_open.
    """
    if year_df.empty:
        raise ValueError("year_df is empty")

    if rng is None:
        rng = random.Random(42)

    cash = float(initial_cash)
    btc = 0.0
    in_position = False

    entry_signal_time = None
    entry_exec_time = None
    entry_price = 0.0
    entry_fee = 0.0
    entry_prob_up = 0.0
    bad_signal_count = 0

    test_year = int(year_df["test_year"].iloc[0])

    equity_rows: list[dict] = []
    trades: list[Trade] = []

    for _, row in year_df.iterrows():
        signal_time = row["timestamp"]
        current_close = float(row["close"])
        next_open = float(row["next_open"])
        next_timestamp = row["next_timestamp"]
        prob_up = float(row["prob_up"])
        y_pred = int(row["y_pred"])

        equity_before_action = cash + btc * current_close
        action = "HOLD"

        enter_trade = False

        # Entry logic
        if not in_position and y_pred == 1:
            if probabilistic_trading:
                entry_chance = scale_probability(
                    prob_up=prob_up,
                    buy_threshold=buy_threshold,
                    max_prob=0.75,
                )
                enter_trade = rng.random() < entry_chance
            else:
                enter_trade = prob_up >= buy_threshold

        if enter_trade:
            fee = cash * fee_rate
            deployable_cash = cash - fee

            if deployable_cash > 0:
                btc = deployable_cash / next_open
                cash = 0.0
                in_position = True

                entry_signal_time = signal_time
                entry_exec_time = next_timestamp
                entry_price = next_open
                entry_fee = fee
                entry_prob_up = prob_up
                bad_signal_count = 0
                action = "BUY"

        # Exit logic with patience
        elif in_position:
            bad_signal = (y_pred == 0) or (prob_up < sell_threshold)

            if bad_signal:
                bad_signal_count += 1
            else:
                bad_signal_count = 0

            if bad_signal_count >= exit_patience:
                gross_value = btc * next_open
                exit_fee = gross_value * fee_rate
                net_value = gross_value - exit_fee

                cash = net_value
                total_fees = entry_fee + exit_fee

                pnl_dollars = (next_open - entry_price) * btc - total_fees
                cost_basis = entry_price * btc + entry_fee
                return_pct = (pnl_dollars / cost_basis) * 100 if cost_basis > 0 else 0.0

                trades.append(
                    Trade(
                        test_year=test_year,
                        entry_signal_time=str(entry_signal_time),
                        entry_exec_time=str(entry_exec_time),
                        exit_signal_time=str(signal_time),
                        exit_exec_time=str(next_timestamp),
                        entry_price=float(entry_price),
                        exit_price=float(next_open),
                        btc_size=float(btc),
                        fees_paid=float(total_fees),
                        pnl_dollars=float(pnl_dollars),
                        return_pct=float(return_pct),
                        entry_prob_up=float(entry_prob_up),
                        exit_prob_up=float(prob_up),
                    )
                )

                btc = 0.0
                in_position = False
                entry_signal_time = None
                entry_exec_time = None
                entry_price = 0.0
                entry_fee = 0.0
                entry_prob_up = 0.0
                bad_signal_count = 0
                action = "SELL"

        equity_after_action = cash + btc * current_close

        equity_rows.append(
            {
                "test_year": test_year,
                "timestamp": signal_time,
                "close": current_close,
                "cash": cash,
                "btc": btc,
                "equity_before_action": equity_before_action,
                "equity": equity_after_action,
                "prob_up": prob_up,
                "y_pred": y_pred,
                "action": action,
            }
        )

    final_close = float(year_df.iloc[-1]["close"])
    final_equity = cash + btc * final_close

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame([asdict(t) for t in trades])

    total_return_pct = ((final_equity / initial_cash) - 1.0) * 100.0
    max_drawdown_pct = compute_drawdown_pct(equity_df["equity"])

    if trades_df.empty:
        win_rate_pct = 0.0
        closed_trades = 0
    else:
        win_rate_pct = float((trades_df["pnl_dollars"] > 0).mean() * 100.0)
        closed_trades = int(len(trades_df))

    summary = {
        "test_year": test_year,
        "initial_cash": round(initial_cash, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "win_rate_pct": round(win_rate_pct, 2),
        "closed_trades": closed_trades,
    }

    return equity_df, trades_df, summary

def run_buy_and_hold_benchmark(
    year_df: pd.DataFrame,
    *,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    """
    Buy at the first available next_open in the test year and hold until the
    final close of that same test year.
    """
    if year_df.empty:
        raise ValueError("year_df is empty")

    test_year = int(year_df["test_year"].iloc[0])

    first_row = year_df.iloc[0]
    entry_exec_time = first_row["next_timestamp"]
    entry_price = float(first_row["next_open"])

    # Pay entry fee, invest the rest
    entry_fee = initial_cash * fee_rate
    deployable_cash = initial_cash - entry_fee
    btc = deployable_cash / entry_price

    equity_rows = []

    for _, row in year_df.iterrows():
        current_close = float(row["close"])
        equity = btc * current_close

        equity_rows.append(
            {
                "test_year": test_year,
                "timestamp": row["timestamp"],
                "close": current_close,
                "btc": btc,
                "equity": equity,
                "action": "HOLD",
            }
        )

    final_close = float(year_df.iloc[-1]["close"])
    gross_final_value = btc * final_close
    exit_fee = gross_final_value * fee_rate
    final_equity = gross_final_value - exit_fee

    equity_df = pd.DataFrame(equity_rows)
    max_drawdown_pct = compute_drawdown_pct(equity_df["equity"])
    total_return_pct = ((final_equity / initial_cash) - 1.0) * 100.0

    summary = {
        "test_year": test_year,
        "benchmark_entry_time": str(entry_exec_time),
        "benchmark_entry_price": round(entry_price, 4),
        "benchmark_final_close": round(final_close, 4),
        "benchmark_final_equity": round(final_equity, 2),
        "benchmark_total_return_pct": round(total_return_pct, 2),
        "benchmark_max_drawdown_pct": round(max_drawdown_pct, 2),
    }

    return equity_df, summary
def run_buy_and_hold_for_all_years(
    merged_df: pd.DataFrame,
    *,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_equity = []
    all_summary = []

    for test_year in sorted(merged_df["test_year"].unique()):
        year_df = merged_df[merged_df["test_year"] == test_year].copy()
        if year_df.empty:
            continue

        equity_df, summary = run_buy_and_hold_benchmark(
            year_df,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
        )

        all_equity.append(equity_df)
        all_summary.append(summary)

    if not all_summary:
        raise RuntimeError("No buy-and-hold benchmark results were produced.")

    equity_out = pd.concat(all_equity, ignore_index=True)
    summary_out = pd.DataFrame(all_summary)

    return equity_out, summary_out

def run_out_of_sample_backtests(
    merged_df: pd.DataFrame,
    *,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
    buy_threshold: float = 0.52,
    sell_threshold: float = 0.48,
    probabilistic_trading: bool = True,
    exit_patience: int = 3,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_equity = []
    all_trades = []
    all_summary = []

    rng = random.Random(random_seed)

    for test_year in sorted(merged_df["test_year"].unique()):
        year_df = merged_df[merged_df["test_year"] == test_year].copy()
        if year_df.empty:
            continue

        year_rng = random.Random(rng.randint(0, 10_000_000))

        equity_df, trades_df, summary = run_single_year_backtest(
            year_df,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            probabilistic_trading=probabilistic_trading,
            rng=year_rng,
            
        )

        all_equity.append(equity_df)    
        all_trades.append(trades_df)
        all_summary.append(summary)

    if not all_summary:
        raise RuntimeError("No yearly backtests were produced.")

    equity_out = pd.concat(all_equity, ignore_index=True)
    trades_out = (
        pd.concat(all_trades, ignore_index=True)
        if any(not df.empty for df in all_trades)
        else pd.DataFrame()
    )
    summary_out = pd.DataFrame(all_summary)

    return equity_out, trades_out, summary_out


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    predictions_path = project_root / "data" / "processed" / "rolling_year_predictions.csv"
    feature_data_path = project_root / "data" / "processed" / "btc_features.csv"

    equity_output_path = project_root / "data" / "backtests" / "ai_oos_equity_curve.csv"
    trades_output_path = project_root / "data" / "backtests" / "ai_oos_trade_history.csv"
    summary_output_path = project_root / "data" / "backtests" / "ai_oos_summary.csv"

    predictions_df = load_predictions(predictions_path)
    price_df = load_price_data(feature_data_path)
    merged_df = prepare_backtest_frame(predictions_df, price_df)

    equity_df, trades_df, summary_df = run_out_of_sample_backtests(
        merged_df,
        initial_cash=10_000.0,
        fee_rate=0.001,
        buy_threshold=0.52,
        sell_threshold=0.48,
        probabilistic_trading=True,
        random_seed=42,
        exit_patience=3,
    )

    benchmark_equity_df, benchmark_summary_df = run_buy_and_hold_for_all_years(
        merged_df,
        initial_cash=10_000.0,
        fee_rate=0.001,
    )

    summary_df = summary_df.merge(benchmark_summary_df, on="test_year", how="left")
    summary_df["alpha_vs_buy_hold_pct"] = (
        summary_df["total_return_pct"] - summary_df["benchmark_total_return_pct"]
    ).round(2)
    benchmark_equity_output_path = project_root / "data" / "backtests" / "buy_hold_equity_curve.csv"
    benchmark_equity_df.to_csv(benchmark_equity_output_path, index=False)
    print(f"Saved buy-and-hold equity curve to: {benchmark_equity_output_path}")

    equity_output_path.parent.mkdir(parents=True, exist_ok=True)
    equity_df.to_csv(equity_output_path, index=False)
    trades_df.to_csv(trades_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print("\n=== OUT-OF-SAMPLE AI BACKTEST SUMMARY ===")
    print(summary_df.to_string(index=False))

    print(f"\nSaved equity curve to: {equity_output_path}")
    print(f"Saved trade history to: {trades_output_path}")
    print(f"Saved yearly summary to: {summary_output_path}")


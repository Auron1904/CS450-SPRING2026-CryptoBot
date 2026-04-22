from __future__ import annotations

from pathlib import Path
import random
import json

import pandas as pd


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "SMA_20",
    "SMA_50",
    "EMA_9",
    "ATR_14",
    "ret_1",
    "ret_3",
    "ret_7",
    "close_to_sma20",
    "close_to_sma50",
    "sma20_minus_sma50",
    "ema9_minus_sma20",
    "macd_cross_gap",
    "atr_pct",
    "ret_1_std_7",
    "ret_1_std_14",
    "high_low_range_pct",
    "vol_ret_1",
    "vol_ratio_20",
    "range_pos_20",
]
TARGET_COLUMN = "Target"


def load_feature_dataset(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required_columns = ["timestamp"] + FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    numeric_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["timestamp"] + FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def sample_params(rng: random.Random) -> dict:
    return {
        "n_estimators": rng.choice([100, 150, 200, 300, 400]),
        "max_depth": rng.choice([2, 3, 4, 5, 6]),
        "learning_rate": rng.choice([0.01, 0.03, 0.05, 0.08, 0.1]),
        "subsample": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "min_child_weight": rng.choice([1, 2, 4, 6]),
        "gamma": rng.choice([0.0, 0.1, 0.3, 0.5, 1.0]),
        "reg_alpha": rng.choice([0.0, 0.01, 0.1, 1.0]),
        "reg_lambda": rng.choice([0.5, 1.0, 2.0, 5.0]),
        "buy_threshold": rng.choice([0.50, 0.51, 0.52, 0.53, 0.55, 0.58]),
        "sell_threshold": rng.choice([0.45, 0.47, 0.48, 0.49, 0.50]),
    }


def compute_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    return float(drawdown.min() * 100.0)


def run_single_year_backtest(
    year_df: pd.DataFrame,
    *,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
    buy_threshold: float = 0.52,
    sell_threshold: float = 0.48,
) -> dict:
    cash = float(initial_cash)
    btc = 0.0
    in_position = False
    entry_price = 0.0
    entry_fee = 0.0
    closed_trade_pnls: list[float] = []

    equity_rows: list[float] = []

    for _, row in year_df.iterrows():
        current_close = float(row["close"])
        next_open = float(row["next_open"])
        prob_up = float(row["prob_up"])
        y_pred = int(row["y_pred"])

        if (not in_position) and (y_pred == 1) and (prob_up >= buy_threshold):
            fee = cash * fee_rate
            deployable_cash = cash - fee
            if deployable_cash > 0:
                btc = deployable_cash / next_open
                cash = 0.0
                in_position = True
                entry_price = next_open
                entry_fee = fee

        elif in_position and ((y_pred == 0) or (prob_up < sell_threshold)):
            gross_value = btc * next_open
            exit_fee = gross_value * fee_rate
            net_value = gross_value - exit_fee
            cash = net_value

            pnl_dollars = (next_open - entry_price) * btc - (entry_fee + exit_fee)
            closed_trade_pnls.append(float(pnl_dollars))

            btc = 0.0
            in_position = False
            entry_price = 0.0
            entry_fee = 0.0

        equity_rows.append(cash + btc * current_close)

    final_close = float(year_df.iloc[-1]["close"])
    final_equity = cash + btc * final_close
    total_return_pct = ((final_equity / initial_cash) - 1.0) * 100.0
    max_drawdown_pct = compute_drawdown_pct(pd.Series(equity_rows))

    win_rate_pct = 0.0 if not closed_trade_pnls else 100.0 * sum(p > 0 for p in closed_trade_pnls) / len(closed_trade_pnls)

    return {
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate_pct": win_rate_pct,
        "closed_trades": len(closed_trade_pnls),
    }


def run_buy_and_hold_benchmark(
    year_df: pd.DataFrame,
    *,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
) -> dict:
    first_row = year_df.iloc[0]
    entry_price = float(first_row["next_open"])
    entry_fee = initial_cash * fee_rate
    btc = (initial_cash - entry_fee) / entry_price

    equity_rows = [btc * float(row["close"]) for _, row in year_df.iterrows()]

    final_close = float(year_df.iloc[-1]["close"])
    gross_final_value = btc * final_close
    exit_fee = gross_final_value * fee_rate
    final_equity = gross_final_value - exit_fee
    total_return_pct = ((final_equity / initial_cash) - 1.0) * 100.0
    max_drawdown_pct = compute_drawdown_pct(pd.Series(equity_rows))

    return {
        "benchmark_final_equity": final_equity,
        "benchmark_total_return_pct": total_return_pct,
        "benchmark_max_drawdown_pct": max_drawdown_pct,
    }


def evaluate_one_trial(
    df: pd.DataFrame,
    params: dict,
    train_years: int = 2,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from xgboost import XGBClassifier

    available_years = sorted(df["timestamp"].dt.year.unique())
    if len(available_years) < train_years + 1:
        raise ValueError("Not enough distinct years for rolling evaluation")

    yearly_rows = []
    accuracy_vals = []
    precision_vals = []
    recall_vals = []

    for i in range(train_years, len(available_years)):
        train_year_block = available_years[i - train_years:i]
        test_year = available_years[i]

        train_df = df[df["timestamp"].dt.year.isin(train_year_block)].copy()
        test_df = df[df["timestamp"].dt.year == test_year].copy()
        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[TARGET_COLUMN]

        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            gamma=params["gamma"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        prob_up = model.predict_proba(X_test)[:, 1]

        accuracy_vals.append(float(accuracy_score(y_test, y_pred)))
        precision_vals.append(float(precision_score(y_test, y_pred, zero_division=0)))
        recall_vals.append(float(recall_score(y_test, y_pred, zero_division=0)))

        test_copy = test_df[["timestamp", "open", "close"]].copy()
        test_copy["y_pred"] = y_pred
        test_copy["prob_up"] = prob_up
        test_copy["next_open"] = test_copy["open"].shift(-1)
        test_copy["next_timestamp"] = test_copy["timestamp"].shift(-1)
        test_copy = test_copy.dropna(subset=["next_open", "next_timestamp"]).reset_index(drop=True)

        if test_copy.empty:
            continue

        ai_stats = run_single_year_backtest(
            test_copy,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            buy_threshold=params["buy_threshold"],
            sell_threshold=params["sell_threshold"],
        )
        bh_stats = run_buy_and_hold_benchmark(
            test_copy,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
        )

        yearly_rows.append({
            "test_year": test_year,
            **ai_stats,
            **bh_stats,
            "alpha_vs_buy_hold_pct": ai_stats["total_return_pct"] - bh_stats["benchmark_total_return_pct"],
        })

    if not yearly_rows:
        raise RuntimeError("No valid yearly rows produced")

    yearly_df = pd.DataFrame(yearly_rows)

    result = {
        "mean_accuracy": yearly_df["test_year"].count() and sum(accuracy_vals) / len(accuracy_vals),
        "mean_precision": yearly_df["test_year"].count() and sum(precision_vals) / len(precision_vals),
        "mean_recall": yearly_df["test_year"].count() and sum(recall_vals) / len(recall_vals),
        "mean_return_pct": yearly_df["total_return_pct"].mean(),
        "mean_drawdown_pct": yearly_df["max_drawdown_pct"].mean(),
        "mean_win_rate_pct": yearly_df["win_rate_pct"].mean(),
        "mean_alpha_vs_buy_hold_pct": yearly_df["alpha_vs_buy_hold_pct"].mean(),
        "total_closed_trades": yearly_df["closed_trades"].sum(),
        "years_tested": len(yearly_df),
        "params_json": json.dumps(params, sort_keys=True),
    }
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "btc_features.csv"
    output_path = project_root / "data" / "model_search_results.csv"

    df = load_feature_dataset(input_path)

    rng = random.Random(42)
    n_trials = 100

    results = []
    for trial in range(1, n_trials + 1):
        params = sample_params(rng)
        try:
            row = evaluate_one_trial(df, params, train_years=2)
            row["trial"] = trial
            results.append(row)
            print(
                f"Trial {trial:03d} | "
                f"alpha={row['mean_alpha_vs_buy_hold_pct']:.2f} | "
                f"return={row['mean_return_pct']:.2f} | "
                f"drawdown={row['mean_drawdown_pct']:.2f}"
            )
        except Exception as exc:
            print(f"Trial {trial:03d} failed: {exc}")

    if not results:
        raise RuntimeError("No successful trials")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["mean_alpha_vs_buy_hold_pct", "mean_return_pct"],
        ascending=[False, False],
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\nTop 10 models")
    print(results_df.head(10).to_string(index=False))
    print(f"\nSaved search results to: {output_path}")


if __name__ == "__main__":
    main()
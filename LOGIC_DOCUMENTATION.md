# Technical Logic Documentation

## 1. Project Purpose

This project is a Bitcoin trading research and dashboard system built around four connected workflows:

1. Market data ingestion from Yahoo Finance
2. Feature engineering for technical-analysis signals
3. Supervised machine-learning prediction of next-day direction
4. Backtesting and dashboard display of both strategy logic and AI forecasts

The core pipeline is:

`Yahoo Finance -> raw OHLCV CSV -> engineered feature CSV -> XGBoost model -> prediction/backtest/dashboard`

---

## 2. Data Flow

### 2.1 Raw Market Data Ingestion

File: [`src/data/ohlcv.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/data/ohlcv.py:1)

This module is responsible for pulling Bitcoin OHLCV data from Yahoo Finance using `yfinance`.

Main logic:

- `fetch_ohlcv(...)`
  - normalizes the requested interval
  - downloads candles from Yahoo Finance
  - standardizes columns to:
    - `timestamp`
    - `open`
    - `high`
    - `low`
    - `close`
    - `volume`
  - converts timestamps and numeric columns into clean types
  - removes duplicate or invalid rows
  - appends metadata:
    - `symbol`
    - `timeframe`

- `save_ohlcv(...)`
  - writes the cleaned dataframe to `data/raw`

- `fetch_and_save_ohlcv(...)`
  - convenience wrapper that fetches and saves in one step

For the BTC daily pipeline, the raw file is saved as:

- `data/raw/BTC-USD_1d.csv`

### 2.2 Feature Engineering

File: [`src/features/indicators.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/features/indicators.py:1)

This module converts raw OHLCV candles into model-ready technical features.

Flow:

1. Start with the raw OHLCV dataframe
2. Add technical indicators
3. Create the forward-looking `Target` label
4. Remove rows with incomplete feature values
5. Preserve the newest row even if its `Target` is still unknown

The processed feature dataset is saved as:

- `data/processed/btc_features.csv`

### 2.3 Model Training

File: [`src/model/train.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/model/train.py:1)

Training reads `btc_features.csv`, keeps only the model feature columns plus `Target`, drops any rows with missing values, and trains an XGBoost classifier using chronological cross-validation.

The trained model is saved as:

- `models/btc_model.json`

### 2.4 Prediction

File: [`src/model/predict.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/model/predict.py:1)

Prediction loads the most recent scorable feature row from `btc_features.csv`, loads the saved model, and predicts whether the next day will be `UP` or `DOWN`.

### 2.5 Dashboard Integration

Files:

- [`app.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/app.py:1)
- [`src/dashboard/forecast.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/dashboard/forecast.py:1)

The Streamlit dashboard renders the forecast card below the main price chart. It can also trigger the live data pipeline before prediction.

### 2.6 Backtesting

File: [`src/backtest/engine.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/backtest/engine.py:1)

Backtesting uses the processed feature dataset and simulates an RSI-based long-only strategy over historical data.

---

## 3. Feature Engineering Logic

### 3.1 Technical Indicators Used

Defined in [`src/features/indicators.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/features/indicators.py:67).

The model uses these eight indicators:

1. `RSI_14`
   - 14-period Relative Strength Index
   - momentum / overbought-oversold signal

2. `MACD`
   - Moving Average Convergence Divergence main line

3. `MACD_signal`
   - MACD signal line

4. `MACD_hist`
   - MACD histogram

5. `SMA_20`
   - 20-period simple moving average

6. `SMA_50`
   - 50-period simple moving average

7. `EMA_9`
   - 9-period exponential moving average

8. `ATR_14`
   - 14-period Average True Range
   - volatility measure

These features are the exact feature set defined in [`src/model/train.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/model/train.py:7).

### 3.2 T+1 Target Labeling Logic

Defined in [`src/features/indicators.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/features/indicators.py:119).

The target is engineered as:

- `next_close = close.shift(-1)`
- `Target = 1` if `next_close > current_close`
- `Target = 0` otherwise

Interpretation:

- a row at day `T` contains features computed from information available at day `T`
- its label answers whether the price at day `T+1` closes above day `T`

This is a strict forward-looking setup:

- features come from the current row
- label comes from the next row

That prevents direct target leakage from the future into the feature vector.

### 3.3 The Dropna Fix We Implemented

This was the key improvement needed to make the system a true forward-looking forecaster.

Previously:

- the script built indicators
- added `Target`
- then ran a blanket `dropna()`

That removed the final row because the last row naturally has no future close yet, so its `Target` is `NaN`.

Effect of the old logic:

- training was clean
- but inference lost the newest market day
- the model predicted from the last labeled day instead of the current day

Current fixed logic:

File: [`src/features/indicators.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/features/indicators.py:145)

- `build_feature_dataset(...)` now drops rows only when required feature columns are missing
- it does **not** require `Target` to be non-null
- therefore the newest row is preserved even when `Target` is missing

Result:

- `train.py` still gets clean labeled data because it drops rows with missing `Target` on load
- `btc_features.csv` keeps the newest unlabeled row for forecasting
- `predict.py` can now use the current day to forecast tomorrow

---

## 4. Model Architecture and Training Logic

### 4.1 Model Type

File: [`src/model/train.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/model/train.py:98)

The classifier is:

- `XGBClassifier` from XGBoost

Configured hyperparameters:

- `n_estimators=200`
- `max_depth=4`
- `learning_rate=0.05`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `random_state=42`
- `eval_metric="logloss"`

This is a binary classification model:

- class `1` means `UP`
- class `0` means `DOWN`

### 4.2 Training Dataset Construction

File: [`src/model/train.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/model/train.py:14)

`load_training_data(...)`:

- loads `btc_features.csv`
- keeps only:
  - the eight feature columns
  - `Target`
- converts all columns to numeric
- runs `dropna()`

This ensures:

- every training row has complete features
- every training row has a known forward label
- the newest unlabeled inference row is excluded automatically

### 4.3 Time-Series Cross-Validation

File: [`src/model/train.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/model/train.py:70)

Training uses:

- `TimeSeriesSplit(n_splits=5)`

This is important because it preserves chronology:

- earlier data is used for training
- later data is used for testing
- there is no random shuffling

That makes the validation more realistic for financial time-series data.

### 4.4 Current Model Metrics

Measured from the current repository state by running the training logic on `data/processed/btc_features.csv`.

Training rows used:

- `4181`

Final `TimeSeriesSplit` evaluation fold:

- `fold_number = 5`

Current metrics:

- Accuracy: `0.5000`
- Precision: `0.4974`
- Recall: `0.2709`

Classification report:

```text
              precision    recall  f1-score   support

           0       0.50      0.73      0.59       349
           1       0.50      0.27      0.35       347

    accuracy                           0.50       696
   macro avg       0.50      0.50      0.47       696
weighted avg       0.50      0.50      0.47       696
```

Interpretation:

- the current model is roughly at chance overall on the final fold
- it is better at identifying class `0` than class `1`
- recall for `UP` predictions is relatively low

### 4.5 Current Feature Importance Ranking

Measured from the fitted model returned on the final training fold.

Current ranking:

```text
    feature  importance
  MACD_hist    0.137522
     SMA_50    0.129775
       MACD    0.127814
     ATR_14    0.125806
     RSI_14    0.125756
      EMA_9    0.125200
     SMA_20    0.114400
MACD_signal    0.113728
```

The feature weights are fairly spread out, which suggests the model is using a mix of trend, momentum, and volatility signals rather than depending on a single indicator.

---

## 5. Prediction Logic

### 5.1 Which Row Is Used for Inference

File: [`src/model/predict.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/model/predict.py:13)

`load_latest_feature_row(...)`:

- loads the processed feature dataset
- sorts by timestamp
- checks which rows have complete feature values
- selects the **last scorable row**

Important detail:

- this row does **not** need a `Target`
- it only needs the feature columns to be valid

That is what allows the current-day row to be used for forecasting tomorrow.

### 5.2 What Day Is Being Predicted

File: [`src/model/predict.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/model/predict.py:101)

The prediction target date is explicitly computed as:

- `target_timestamp = latest_timestamp + 1 day`

So the system is predicting:

- the next day after the most recent feature row

In plain terms:

- latest feature row = today (`T`)
- prediction output = tomorrow (`T+1`)

### 5.3 Probability Logic

File: [`src/model/predict.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/model/predict.py:84)

The model returns:

- a class prediction (`UP` or `DOWN`)
- a probability of the `UP` class via `predict_proba`

The dashboard then converts that into a direction-aligned confidence:

- if prediction is `UP`, display `probability_up`
- if prediction is `DOWN`, display `1 - probability_up`

This means the shown probability always corresponds to the displayed direction.

---

## 6. Backtest Logic

### 6.1 Strategy Definition

File: [`src/backtest/engine.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/backtest/engine.py:17)

The current backtest engine is a simple long-only RSI strategy.

Rules:

- `BUY` when:
  - there is no open position
  - `RSI_14 < 30`

- `SELL` when:
  - there is an open position
  - `RSI_14 > 70`

- otherwise `HOLD`

### 6.2 Trading Assumptions

The engine assumes:

- starting capital = `$10,000`
- fee rate = `0.001` (0.1%)
- full allocation on each buy
- full exit on each sell

It tracks:

- trade history
- realized PnL
- equity curve
- closed-trade returns

### 6.3 Performance Metrics

Defined in [`src/backtest/engine.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/backtest/engine.py:201)

The engine reports:

- Final Equity
- Total Return %
- Win Rate %
- Max Drawdown %
- Sharpe Ratio
- Closed Trades

### 6.4 Current Backtest Results

Measured from the current repository state using `data/processed/btc_features.csv`.

Current summary:

- Final Equity: `$76,006.31`
- Total Return: `660.06%`
- Win Rate: `78.95%`
- Max Drawdown: `-64.53%`
- Sharpe Ratio: `0.6270`
- Closed Trades: `19`

Interpretation:

- the strategy generated very large cumulative returns on this dataset
- but it also experienced very deep drawdowns
- this means the strategy is aggressive and volatile, not low-risk

For presentation purposes, the clean comparison is:

- Return: very high
- Drawdown: also very high

That is the core backtest trade-off.

---

## 7. Web App Integration

### 7.1 Dashboard Structure

Main entry file:

- [`app.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/app.py:1)

Relevant flow:

1. Render price chart
2. Render AI forecast card
3. Render account metrics
4. Render trade signal information
5. Render trade history

The forecast widget is intentionally inserted below the main chart to make it highly visible without changing the rest of the layout.

### 7.2 Live Pipeline Trigger

File: [`src/dashboard/forecast.py`](/Users/quanmiki/Downloads/CS450-SPRING2026-CryptoBot-main/src/dashboard/forecast.py:12)

The dashboard can trigger a live refresh through the `Refresh Data & Predict` button.

When pressed:

1. Show spinner:
   - `Fetching live market data...`
2. Run raw data ingestion:
   - `fetch_and_save_ohlcv(symbol="BTC-USD", timeframe="1d")`
3. Run feature engineering:
   - `build_feature_dataset(raw_df)`
4. Save updated feature CSV:
   - `data/processed/btc_features.csv`
5. Run prediction:
   - `get_latest_prediction(...)`
6. Update the widget in the same session

### 7.3 Forecast Widget Output

The dashboard displays:

- `AI Forecast (Date: [Target Date])`
- `Prediction: UP` or `DOWN`
- `Probability: XX.XX%`
- a progress bar showing the confidence

Because the processed dataset now preserves the newest unlabeled row, the widget can act as a true next-day forecaster after the refresh pipeline runs.

---

## 8. End-to-End Summary

The current system now works as a coherent forward-looking pipeline:

1. Yahoo Finance supplies daily BTC candles
2. Raw OHLCV data is cleaned and stored
3. Technical indicators are engineered
4. A T+1 label is created using `shift(-1)`
5. The newest row is preserved even if `Target` is still unknown
6. Training uses only fully labeled rows
7. Prediction uses the latest fully featured row, even if unlabeled
8. The dashboard can refresh the live pipeline and immediately display tomorrow’s forecast

This design keeps the training set clean, preserves chronological integrity, and allows the deployed dashboard to function as a true next-day forecasting interface.

---

## 9. Notes and Limitations

- The training metrics are currently modest and indicate that the predictive model still needs improvement.
- The backtest results are strong in return terms but come with severe drawdowns, so strategy risk is still high.
- `indicators.py` uses a lazy `pandas_ta` import with a temporary `numba` patch to avoid cache-related runtime issues.
- The forecasting workflow is structurally forward-looking, but model quality should be validated further before using it for real trading decisions.

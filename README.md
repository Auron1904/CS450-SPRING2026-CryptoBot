# CS450-SPRING2026-CryptoBot
This is a group project for CS-450 at SDSU. The project is an Artificial Intelligence project with five collaborators. The main goal of the project is to train a bot using crypto data to make profitable trades automatically. 

## 🚀 Quick Start (Team Setup)
1. **Install uv:** - Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. **Sync Environment:**
   Run `uv sync` in the root folder. (This automatically installs Python 3.12 and all libraries).
3. **Setup Secrets**:   
   Add your API keys to the new `.env` file.
4. ** 🐳 Running with Docker**:
If you want to test the containerized version:
   Run `docker compose up --build`
   Access the dashboard at: `http://localhost:8501`

## 📋 Development Rules
- **Adding Libraries**: Run `uv add <package_name>`. 
- **Always** update dependencies[] and specify versions in dev[] in`pyproject.toml`
- **Sync uv environment** by running `uv sync` 

## Project Structure

```
src/
├── model/
│   ├── train.py              # XGBoost training pipeline
│   ├── predict.py            # Make predictions
│   └── rolling_year_eval.py  # Walk-forward validation
├── features/
│   └── indicators.py         # Build 28 technical indicators
├── data/
│   ├── ohlcv.py             # Fetch OHLCV from Yahoo Finance
│   └── downloader.py        # Fetch from CoinGecko
├── dashboard/
│   ├── chart.py             # Price chart (Plotly dark)
│   ├── forecast.py          # AI prediction display
│   ├── account.py           # Account metrics
│   ├── history.py           # Trade history table
│   ├── controls.py          # Manual trade controls
│   ├── auto_trade.py        # Auto-trade logic
│   ├── pipeline.py          # Pipeline tab (Fetch → Build → Train → Evaluate)
│   └── backtest.py          # Backtest tab (Results & Performance metrics)
├── bot/
│   ├── alpaca_client.py     # Trading API wrapper
│   └── logic.py             # MA_5 strategy
└── backtest/
    └── ai_engine.py         # Backtesting engine

app.py                        # Main dashboard (layout manager)
models/
└── btc_model.json           # Trained XGBoost model
data/
├── raw/
│   ├── bitcoin_dataset.csv  # CoinGecko data
│   └── BTC-USD_1d.csv       # OHLCV data
└── processed/
    └── btc_features.csv     # Features with indicators
```         

## 📊 Dashboard Tabs

The application has **3 main tabs**:

### 1️⃣ **Dashboard Tab** (Default)
- **Price Chart**: BTC price with 5-day moving average (Plotly)
- **AI Forecast**: ML model prediction (UP/DOWN) with probability
- **Account Summary**: Live trading account metrics (Equity, Buying Power, Position, P/L)
- **Trade Signal**: Latest MA_5 signal and current price
- **Trade History**: Recent trades from paper trading account

### 2️⃣ **Pipeline Tab**
Complete ML training workflow with 4 steps:

1. **Fetch OHLCV Data** - Downloads BTC-USD daily data from Yahoo Finance
   - Output: `data/raw/BTC-USD_1d.csv`
   
2. **Build Features** - Adds 28 technical indicators (RSI, MACD, SMA, etc.)
   - Output: `data/processed/btc_features.csv`
   
3. **Train Model** - Trains XGBoost classifier with TimeSeriesSplit CV
   - Config: 200 estimators, max_depth=4, learning_rate=0.05
   - Output: `models/btc_model.json`
   - Displays: Accuracy, Precision, Recall, Classification Report, Top 10 Features
   
4. **Evaluate Model** - Walk-forward validation (yearly rolling windows)
   - Input: `data/processed/btc_features.csv`
   - Output: `outputs/rolling_summary.csv`, `outputs/rolling_predictions.csv`
   - Shows: Summary table, accuracy trend chart, download buttons

### 3️⃣ **Backtest Tab**
View and analyze evaluation results:

1. **Evaluation Results** - Table from Pipeline evaluation
   - Columns: train_start_year, train_end_year, test_year, train_rows, test_rows, accuracy, precision, recall
   - Chart: Accuracy over time
   
2. **Performance Metrics** - Average stats across all test years
   - Average Accuracy, Precision, Recall
   
3. **AI Backtest** - Optional backtesting engine
   - Run backtest on predictions
   - Shows: Return %, Win Rate, Max Drawdown, Total Trades, Sharpe Ratio
   - Displays: Equity curve chart, yearly breakdown

## 🎯 Running the Application

```bash
# Install dependencies
uv sync

# Start the dashboard
streamlit run app.py
```

Then open your browser to: `http://localhost:8501`

## 📁 Data Flow

```
Yahoo Finance (OHLCV)
    ↓
[Pipeline Tab] Fetch OHLCV
    ↓ data/raw/BTC-USD_1d.csv
    ↓
[Pipeline Tab] Build Features
    ↓ data/processed/btc_features.csv
    ↓
[Pipeline Tab] Train Model
    ↓ models/btc_model.json + metrics
    ↓
[Pipeline Tab] Evaluate Model
    ↓ outputs/rolling_summary.csv
    ↓
[Backtest Tab] Display Results
```

## 🔧 Sidebar Controls

- **Data Settings**: Slider for days of history, fetch fresh data button
- **Trade Signal**: Generate MA_5 signal manually
- **Manual Trade**: Execute trades with custom notional and force signal
- **Auto-Trade (24hr)**: Enable automatic trading with cooldown and confirmation

## ⚙️ Configuration

### Environment Variables (`.env`)
```
CG_API_KEY=your_coingecko_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Model Features (28 total)
- **OHLCV**: open, high, low, close, volume
- **Momentum**: RSI_14, MACD, MACD_signal, MACD_hist
- **Moving Averages**: SMA_20, SMA_50, EMA_9
- **Volatility**: ATR_14
- **Returns**: ret_1, ret_3, ret_7
- **Derived**: close_to_sma20, close_to_sma50, sma20_minus_sma50, ema9_minus_sma20, macd_cross_gap, atr_pct, ret_1_std_7, ret_1_std_14, high_low_range_pct, vol_ret_1, vol_ratio_20, range_pos_20

## ✅ Testing

Run the test suite:
```bash
# Component tests (13 tests)
python -m pytest tests/test_components.py -v

# Integration tests (3 tests)
python tests/test_integration.py
```

## 📝 Known Issues & Troubleshooting

### Evaluation Step Fails
**Error**: "Can only use .dt accessor with datetimelike values"
**Solution**: Timestamp column must be datetime. Pipeline now auto-converts CSV timestamps.

### Model Not Found
**Error**: Model file not found at `models/btc_model.json`
**Solution**: Run Pipeline tab → Train Model button first

### No Data Available
**Error**: "No evaluation results found"
**Solution**: 
1. Click Pipeline tab → Fetch OHLCV Data
2. Click Pipeline tab → Build Features
3. Click Pipeline tab → Train Model
4. Click Pipeline tab → Evaluate Model         

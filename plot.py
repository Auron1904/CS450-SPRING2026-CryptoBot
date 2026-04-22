import pandas as pd
import matplotlib.pyplot as plt

# Load data
ai = pd.read_csv("data/backtests/ai_oos_equity_curve.csv")
bh = pd.read_csv("data/backtests/buy_hold_equity_curve.csv")

# Convert timestamp
ai["timestamp"] = pd.to_datetime(ai["timestamp"])
bh["timestamp"] = pd.to_datetime(bh["timestamp"])

# Plot
plt.figure()

for year in ai["test_year"].unique():
    plt.figure()

    ai_year = ai[ai["test_year"] == year]
    bh_year = bh[bh["test_year"] == year]

    plt.plot(ai_year["timestamp"], ai_year["equity"], label="AI")
    plt.plot(bh_year["timestamp"], bh_year["equity"], linestyle="--", label="Buy & Hold")

    plt.title(f"{year}: AI vs Buy & Hold")
    plt.legend()
    plt.show()
    
plt.title("AI vs Buy & Hold Equity Curve")
plt.xlabel("Time")
plt.ylabel("Equity ($)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
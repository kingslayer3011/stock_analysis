import os
import pickle
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Constants
MONTH_DAYS = [21, 42, 63, 105, 210, 315, 420]
CACHE_FILE = "c25_history.pkl"
ALERTS = []

# C25 tickers
C25_TICKERS = [
    "CARL-B.CO", "CHR.CO", "COLO-B.CO", "DSV.CO", "FLS.CO", "GN.CO",
    "JYSK.CO", "MAERSK-B.CO", "NETC.CO", "NOVO-B.CO", "NYK.CO", "ORSTED.CO",
    "PANDORA.CO", "ROCK-B.CO", "SAS-DKK.CO", "SIM.CO", "SYDB.CO", "TOP.CO",
    "TRYG.CO", "VWS.CO", "ZEAL.CO", "AMBU-B.CO", "DEMANT.CO", "GEN.CO", "NDA-DK.CO"
]

# Utility: fetch and cache
def fetch_history(tickers, period="2y", interval="1d", use_cache=True):
    if use_cache and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        if data.get("timestamp", datetime.min) > datetime.now() - timedelta(days=1):
            return data["prices"]

    prices = yf.download(tickers, period=period, interval=interval, group_by="ticker", progress=False)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"timestamp": datetime.now(), "prices": prices}, f)
    return prices

# Indicators suite
def compute_indicators(df):
    res = {}
    for days in MONTH_DAYS:
        res[f"MA_{days}d"] = df["Close"].rolling(window=days).mean().iloc[-1]

    ma20 = df["Close"].rolling(window=20).mean()
    std20 = df["Close"].rolling(window=20).std()
    res["BB_upper"] = (ma20 + 2 * std20).iloc[-1]
    res["BB_lower"] = (ma20 - 2 * std20).iloc[-1]

    tr = pd.concat([
        df["High"] - df["Low"],
        np.abs(df["High"] - df["Close"].shift()),
        np.abs(df["Low"] - df["Close"].shift())
    ], axis=1).max(axis=1)
    res["ATR_14"] = tr.rolling(14).mean().iloc[-1]

    direction = np.sign(df["Close"].diff()).fillna(0)
    res["OBV"] = (direction * df["Volume"]).cumsum().iloc[-1]

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    sto_k = 100 * (df["Close"] - low14) / (high14 - low14)
    res["Stoch_%K"] = sto_k.iloc[-1]
    res["Stoch_%D"] = sto_k.rolling(3).mean().iloc[-1]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    res["RSI_14"] = (100 - (100 / (1 + rs))).iloc[-1]

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    res["MACD"] = macd.iloc[-1]
    res["MACD_Signal"] = signal.iloc[-1]

    x = np.arange(len(df))
    y = df["Close"].values
    slope, _, _, _, _ = stats.linregress(x, y)
    res["Trend_Slope"] = slope

    res["Total_Return_%"] = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    res["Close"] = df["Close"].iloc[-1]
    res["Prev_Close"] = df["Close"].iloc[-2] if len(df) > 1 else np.nan
    return res

# Relative and correlation metrics
def compute_relative_and_corr(prices):
    close = prices.xs("Close", level=1, axis=1)
    index_return = close.mean(axis=1)
    rel_perf = {}
    corr = {}
    for ticker in close.columns:
        series = close[ticker]
        rel_perf[ticker] = (
            (series.iloc[-1] / series.iloc[0] - 1) * 100 -
            ((index_return.iloc[-1] / index_return.iloc[0] - 1) * 100)
        )
        corr[ticker] = series.rolling(window=60).corr(index_return).iloc[-1]
    return rel_perf, corr

# Alerting
def check_alerts(df_indicators):
    for idx, row in df_indicators.iterrows():
        if row["RSI_14"] < 30:
            ALERTS.append(f"{row['Ticker']}: RSI below 30 ({row['RSI_14']:.1f})")
        if row["Close"] > row["MA_210d"] and row["Prev_Close"] <= row["MA_210d"]:
            ALERTS.append(f"{row['Ticker']}: Price crossed above 200d MA")

# Graph worst performers
def plot_worst_performers(prices, df_final, output_folder):
    worst = df_final.nsmallest(10, "Total_Return_%")["Ticker"].tolist()

    close = prices.xs("Close", level=1, axis=1)

    since = close.index[-1] - pd.DateOffset(months=12)
    recent = close.loc[close.index >= since].ffill().bfill()

    valid = [t for t in worst if t in recent.columns]
    recent = recent[valid]
    normalized = recent.divide(recent.iloc[0]) * 100

    plt.figure(figsize=(12, 6))
    for ticker, series in normalized.items():
        # ✅ convert both to NumPy arrays
        plt.plot(series.index.to_numpy(), series.values, label=ticker)

    plt.title("Worst 10 Performing C25 Stocks — Last 12 Months")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base 100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "worst_10_performers.png"))




# Main
def main():
    os.chdir('/Users/adamwested/Desktop/Stock analysis/price_screener')
    output_folder = "results"
    os.makedirs(output_folder, exist_ok=True)

    prices = fetch_history(C25_TICKERS)
    rel_perf, corr = compute_relative_and_corr(prices)

    results = []
    def process(ticker):
        try:
            df = prices[ticker].dropna()
            if df.empty:
                raise ValueError(f"No data for {ticker}")
            indicators = compute_indicators(df)
            indicators["Ticker"] = ticker
            indicators["Rel_Perf_vs_Index_%"] = rel_perf[ticker]
            indicators["Corr_vs_Index"] = corr[ticker]
            return indicators
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=5) as exec:
        futures = [exec.submit(process, t) for t in C25_TICKERS]
        for f in futures:
            result = f.result()
            if result:
                results.append(result)

    df_final = pd.DataFrame(results).sort_values("Total_Return_%")
    df_final.to_csv(os.path.join(output_folder, "C25_full_screener.csv"), index=False)
    print(df_final)

    check_alerts(df_final)

    if ALERTS:
        print("\nAlerts:")
        for a in ALERTS:
            print("-", a)

    plot_worst_performers(prices, df_final, output_folder)

if __name__ == "__main__":
    main()

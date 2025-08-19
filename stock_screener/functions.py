import pandas as pd
import numpy as np
import yfinance as yf
import json
from data_fetcher import get_historical_data
import os

def get_share_price(df, ticker):
    share_price = pd.DataFrame(yf.Ticker(ticker).history(period="max", auto_adjust=True, actions=True)["Close"].reset_index())
    share_price.rename(columns={"Date": "date", "Close": "share_price"}, inplace=True)
    # Convert 'date' to date, not datetime
    share_price["date"] = pd.to_datetime(share_price["date"]).dt.date
    # Set Date as index
    share_price.set_index("date", inplace=True)
    # change index from datetime to date
    df.reset_index(inplace=True)
    df["date"] = pd.to_datetime(df["index"]).dt.date
    # Set Date as index
    df.set_index("date", inplace=True)
    # Get the last valid share price for the dates in df
    valid_dates = df.reset_index()["date"].unique()  
    share_prices = []
    for date in valid_dates:
        if date in share_price.index:
            share_prices.append(share_price.loc[date, "share_price"])
        else:
            share_prices.append(share_price[share_price.index <= date]["share_price"].iloc[-1] if not share_price[share_price.index <= date].empty else np.nan)
    return share_prices

# Define the row-wise CAGR function
def rolling_cagr(series):
    cagr_list = []
    for i in range(len(series)):
        sub_series = series.iloc[:i+1].dropna()
        if len(sub_series) < 2:
            cagr_list.append(np.nan)
            continue
        start, end = sub_series.iloc[0], sub_series.iloc[-1]
        n = len(sub_series) - 1
        if start <= 0 or end <= 0:
            cagr_list.append(np.nan)
        else:
            cagr = (end / start) ** (1 / n) - 1
            cagr_list.append(cagr)
    return pd.Series(cagr_list, index=series.index)


def combined_hist_data(comp):
    # Build a lookup for company metadata
    meta = {entry['ticker']: {
                'company_name': entry.get('company_name'),
                'company_sector': entry.get('company_sector'),
                'company_industry': entry.get('company_industry')
            } for entry in comp}

    # Extract tickers
    tickers = list(meta.keys())

    # Fetch and combine datasets
    dfs = []
    for t in tickers:
        try:
            df_t, val_info_t = get_historical_data(t)
            df_t = df_t.copy()
            df_t['ticker'] = t
            df_t['share_price'] = get_share_price(df_t, t)
            # Add company metadata
            df_t['company_name'] = meta[t]['company_name']
            df_t['company_sector'] = meta[t]['company_sector']
            df_t['company_industry'] = meta[t]['company_industry']
            dfs.append(df_t)
        except Exception as e:
            print(f"Failed for ticker: {t} ({e})")

    # Concatenate all dataframes into a single big dataset
    big_df = pd.concat(dfs, ignore_index=True)
    big_df = big_df.rename(columns={'index': 'date'})

    return big_df


def run_extraction():
    # Load JSON data from file
    with open('all_companies.json', 'r') as file:
        comp = json.load(file)

    big_df = combined_hist_data(comp)
    big_df = calculate_metrics(big_df)
    big_df.to_csv("big_df.csv", index=False)

    # Filter and save TTM-only data
    big_df_ttm = big_df[big_df["ttm"] == 1].copy()
    big_df_ttm.to_csv("big_df_ttm.csv", index=False)

    # Calculate descriptive statistics
    stats = big_df_ttm.describe(percentiles=[0.25, 0.5, 0.75]).T[
        ["min", "25%", "50%", "mean", "75%", "max"]
    ].rename(columns={
        "25%": "p25",
        "50%": "median",
        "75%": "p75"
    }).to_dict(orient="index")

    # Save stats to JSON
    with open("big_df_ttm_stats.json", "w") as f:
        json.dump(convert_timestamps(stats), f, indent=2)

    print("Extraction complete. Files saved:")
    print("- big_df.csv")
    print("- big_df_ttm.csv")
    print("- big_df_ttm_stats.json")


def load_screening_criteria(filename):
    with open(filename, 'r') as f:
        criteria = json.load(f)
    return criteria

def filter_by_criteria(df, criteria):
    mask = pd.Series(True, index=df.index)
    metrics = []

    # Sector filtering
    if "include_sectors" in criteria:
        included = criteria["include_sectors"]
        mask &= df["company_sector"].isin(included)

    if "exclude_sectors" in criteria:
        excluded = criteria["exclude_sectors"]
        mask &= ~df["company_sector"].isin(excluded)

    # Metric filtering
    sector_keys = {"include_sectors", "exclude_sectors"}
    for metric, bounds in criteria.items():
        if metric in sector_keys:
            continue
        metrics.append(metric)
        if "min" in bounds:
            mask &= df[metric] >= bounds["min"]
        if "max" in bounds:
            mask &= df[metric] <= bounds["max"]

    filtered_df = df[mask].reset_index(drop=True)
    return filtered_df, metrics


def run_screen(screen_json_filename):
    # Load screening criteria
    criteria = load_screening_criteria(screen_json_filename)

    # Load the big dataframe from CSV (adjust path if needed)
    big_df = pd.read_csv("big_df_ttm.csv")
    print(criteria)
    
    # Filter dataframe by criteria
    filtered_df, metrics = filter_by_criteria(big_df, criteria)
    print(filtered_df[["ticker", "company_sector"] + metrics])

    # Prepare output filename
    base_name = os.path.splitext(screen_json_filename)[0]  # remove extension
    output_csv = f"{base_name}_results.csv"

    # Save filtered dataframe as CSV
    filtered_df.to_csv(output_csv, index=False)

    print(f"Filtered results saved to: {output_csv}")

def convert_timestamps(obj):
    if isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(i) for i in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


def calculate_metrics(df):
    df["EPS"] = df["Net Income"] / df["Ordinary Shares Number"]
    df["EPS Growth Rate"] = df.groupby("ticker")["EPS"].pct_change() * 100
    df["PE Ratio"] = df["share_price"] / df["EPS"]
    df["PEG Ratio"] = df["PE Ratio"] / df["EPS Growth Rate"]
    df["book_value"] = df["Total Assets"] - df["Total Debt"]
    df["book_value_per_share"] = df["book_value"] / df["Ordinary Shares Number"]
    df["PB Ratio"] = df["share_price"] / df["book_value_per_share"]
    df["graham_number"] = np.sqrt(22.5 * df["EPS"] * df["book_value_per_share"])
    df["p_to_graham"] = df["share_price"] / df["graham_number"]
    df["dividends_per_share"] = df["Dividends"] / df["Ordinary Shares Number"]
    df["dividend_yield"] = df["dividends_per_share"] / df["share_price"]
    df["dividend_payout_ratio"] = df["dividends_per_share"] / df["EPS"]
    df["debt_to_equity"] = df["Total Debt"] / df["book_value"]
    df["current_ratio"] = df["Current Assets"] / df["Current Liabilities"]
    #df["quick_ratio"] = (df["Current Assets"] - df["Inventory"]) / df["Current Liabilities"]
    df["interest_coverage"] = df["EBIT"] / df["Interest Expense"]
    df["fcf_per_share"] = df["FCF"] / df["Ordinary Shares Number"]
    df["fcf_yield"] = df["fcf_per_share"] / df["share_price"]
    df["market_cap"] = df["share_price"] * df["Ordinary Shares Number"]

    df['revenue_growth_pct'] = df.groupby('ticker')['Revenue'].transform(lambda x: x.pct_change() * 100)
    df['ebit_growth_pct'] = df.groupby('ticker')['EBIT'].transform(lambda x: x.pct_change() * 100)
    df['net_income_growth_pct'] = df.groupby('ticker')['Net Income'].transform(lambda x: x.pct_change() * 100)
    df['fcf_growth_pct'] = df.groupby('ticker')['FCF'].transform(lambda x: x.pct_change() * 100)
    df['roce_trend_pct'] = df.groupby('ticker')['ROCE'].transform(lambda x: x.pct_change() * 100)

    # Apply row-wise CAGR within each group
    df['cagr_revenue'] = df.groupby('ticker')['Revenue'].transform(rolling_cagr)
    df['cagr_ebit'] = df.groupby('ticker')['EBIT'].transform(rolling_cagr)
    df['cagr_net_income'] = df.groupby('ticker')['Net Income'].transform(rolling_cagr)
    df['cagr_fcf'] = df.groupby('ticker')['FCF'].transform(rolling_cagr)
    df['cagr_eps'] = df.groupby('ticker')['EPS'].transform(rolling_cagr)

    return df
# data_fetcher.py

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple


def calculate_growth_stats(series: pd.Series) -> Tuple[float, float]:
    """
    Given a pd.Series of annual numbers indexed by year (e.g. Revenue),
    returns:
      - CAGR over the full span ([n_last / n_first]^(1/(N–1)) – 1)
      - Latest year‐over‐year change (last / prior – 1)

    If series has fewer than 2 points, returns (np.nan, np.nan).
    """
    try:
        n_first = series.iloc[0]
        n_last = series.iloc[-1]
        periods = len(series) - 1
        cagr = (n_last / n_first) ** (1 / periods) - 1
    except Exception:
        cagr = np.nan

    try:
        recent = series.iloc[-1] / series.iloc[-2] - 1
    except Exception:
        recent = np.nan

    return cagr, recent


# -------------------- Get Historical Data --------------------------------
def get_historical_data(ticker: str):
    """
    Fetches historical financial statement data for the given ticker.
    Returns a DataFrame (indexed by Year, oldest→newest) with columns:
      ["Revenue", "EBITDA", "Net Income", "FCF", "Total Debt", "Cash",
       "FCF/EBITDA", "EBITDA/Revenue", "FCF/Revenue", "P/FCF", "EV/EBITDA"]
    and a valuation_info dict with current summary stats.

    If any row (e.g. "Cash") is missing from a statement, its column will be filled with NaN.
    """
    t = yf.Ticker(ticker)
    try:
        # Attempt to pull each line item; if missing, we'll handle below.
        raw_income = t.financials
        raw_cashflow = t.cashflow
        raw_balance = t.balance_sheet

        # Define helper to safely extract a row from a DataFrame
        def safe_row(df: pd.DataFrame, row_name: str) -> pd.Series:
            if (df is None) or (row_name not in df.index):
                return None
            return df.loc[row_name]

        hist_revenue = safe_row(raw_income, "Total Revenue")
        hist_ebitda = safe_row(raw_income, "EBITDA")
        hist_income = safe_row(raw_income, "Net Income")
        hist_cashflow = safe_row(raw_cashflow, "Free Cash Flow")
        hist_debt = safe_row(raw_balance, "Total Debt")
        hist_cash = safe_row(raw_balance, "Cash")

        # Build list of all available indices (timestamps)
        indices_list = []
        for series in [hist_revenue, hist_ebitda, hist_income, hist_cashflow, hist_debt, hist_cash]:
            if series is not None:
                indices_list.append(series.index)

        if not indices_list:
            raise ValueError("No financial data available for any line items.")

        # Union all indices to get a complete timeline
        common_index = indices_list[0]
        for idx in indices_list[1:]:
            common_index = common_index.union(idx)
        common_index = common_index.sort_values()

        # Reindex each metric series to the common_index (fill missing with NaN)
        def reindexed_or_empty(series: pd.Series) -> pd.Series:
            if series is None:
                return pd.Series(data=np.nan, index=common_index)
            return series.reindex(common_index)

        s_revenue = reindexed_or_empty(hist_revenue)
        s_ebitda = reindexed_or_empty(hist_ebitda)
        s_income = reindexed_or_empty(hist_income)
        s_cashflow = reindexed_or_empty(hist_cashflow)
        s_debt = reindexed_or_empty(hist_debt)
        s_cash = reindexed_or_empty(hist_cash)

        # Combine into a single DataFrame (rows = dates; columns = metrics)
        df = pd.DataFrame({
            "Revenue": s_revenue,
            "EBITDA": s_ebitda,
            "Net Income": s_income,
            "FCF": s_cashflow,
            "Total Debt": s_debt,
            "Cash": s_cash
        })

        # Ensure the index is named "Year" (timestamps)
        df.index.name = "Year"

        # Sort ascending so oldest→newest
        df = df.sort_index(ascending=True)

        # Compute additional ratio columns
        df["FCF/EBITDA"] = df["FCF"] / df["EBITDA"]
        df["EBITDA/Revenue"] = df["EBITDA"] / df["Revenue"]
        df["FCF/Revenue"] = df["FCF"] / df["Revenue"]

        # For P/FCF and EV/EBITDA, we need current market/enterprise values
        share_price = t.info.get("currentPrice", np.nan)
        number_shares = t.info.get("sharesOutstanding", np.nan)
        if not np.isnan(share_price) and not np.isnan(number_shares):
            market_cap = share_price * number_shares
        else:
            market_cap = np.nan

        enterprise_value = t.info.get("enterpriseValue", np.nan)

        df["P/FCF"] = (market_cap / df["FCF"]).replace([np.inf, -np.inf], np.nan)
        df["EV/EBITDA"] = (enterprise_value / df["EBITDA"]).replace([np.inf, -np.inf], np.nan)

        # For valuation_info, take the latest (newest) "Total Debt" and "Cash"
        try:
            latest_debt = s_debt.dropna().iloc[-1]
        except Exception:
            latest_debt = np.nan

        try:
            latest_cash = s_cash.dropna().iloc[-1]
        except Exception:
            latest_cash = np.nan

        valuation_info = {
            "market_cap": market_cap,
            "share_price": share_price,
            "number_shares": number_shares,
            "enterprise_value": enterprise_value,
            "total_debt": latest_debt,
            "cash": latest_cash,
            "net_debt": (latest_debt - latest_cash) if (not np.isnan(latest_debt) and not np.isnan(latest_cash)) else np.nan
        }

        return df, valuation_info

    except Exception as e:
        raise ValueError(f"Could not retrieve data for {ticker}. Error: {e}")

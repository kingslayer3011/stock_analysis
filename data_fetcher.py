import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from functools import reduce
import logging

"""
import debugpy
debugpy.listen(("localhost", 5682))  # You can use any open port, e.g., 5678
print("Waiting for debugger attach...")
debugpy.wait_for_client()  # This will pause execution until you attach the debugger
"""

def safe_divide(numerator, denominator):
    # if either is a Series with one element, convert to scalar
    if isinstance(numerator, pd.Series) and len(numerator) == 1:
        numerator = numerator.iloc[0]
    if isinstance(denominator, pd.Series) and len(denominator) == 1:
        denominator = denominator.iloc[0]
    if pd.notna(numerator) and pd.notna(denominator) and denominator != 0:
        return numerator / denominator
    else:
        return np.nan

def calculate_fcff(
    df: pd.Series
) -> float:
    """
    Calculate Free Cash Flow to the Firm (FCFF) using the standard formula:
    FCFF = EBIT × (1 – effective tax rate) + D&A – CapEx – Change in NWC
    Expects a row (Series) with keys: 'EBITDA', 'Depreciation And Amortization', 'CapEx', 'Delta WC', 'Tax Provision', 'Pretax Income'.
    """
    ebit = df.get("EBITDA", np.nan)
    depreciation_amortization = df.get("Depreciation And Amortization", np.nan)
    capex = df.get("CapEx", np.nan)
    change_in_nwc = df.get("Delta WC", np.nan)
    # Use effective_tax_rate from df if present, else fallback to 0.25
    effective_tax_rate = df.get("effective_tax_rate", 0.25)
    if pd.isna(effective_tax_rate) or effective_tax_rate < 0 or effective_tax_rate > 1:
        effective_tax_rate = 0.25
    return ebit * (1 - effective_tax_rate) + depreciation_amortization - capex - change_in_nwc

def calculate_fcfe(
    df: pd.Series
) -> float:
    """
    Calculate Free Cash Flow to Equity (FCFE) using the formula:
    FCFE = FCFF + Change in Debt
    Expects a row (Series) with keys: 'EBITDA', 'Depreciation And Amortization', 'CapEx', 'Delta WC', 'Change in Debt', 'Tax Provision', 'Pretax Income'.
    """
    fcff = calculate_fcff(df)
    change_in_debt = df.get("Change in Debt", np.nan)
    return fcff + change_in_debt if pd.notna(change_in_debt) else fcff

def calculate_reinvestment_rate(
    capex: float,
    depreciation: float,
    delta_wc: float,
    ebit: float,
    tax_rate: float,
) -> float:
    """Calculate the reinvestment rate using the formula:
    Reinvestment Rate = (CapEx + Depreciation + ΔWC) / NOPAT
    where NOPAT = EBIT × (1 - tax rate)
    """
    nopat = ebit * (1 - tax_rate)
    return safe_divide(capex + depreciation + delta_wc, nopat)


def safe_value(df, row):
    if df is not None and row in df.index:
        val = df.loc[row]
        if isinstance(val, pd.Series):
            return val.iloc[0]
        return val
    return np.nan


def build_ttm_row(raw_income_ttm, raw_cashflow_ttm, raw_balance_ttm, safe_value, df):
    ttm_row = {
        "Revenue":     safe_value(raw_income_ttm, "Total Revenue"),
        "EBITDA":      safe_value(raw_income_ttm, "EBITDA"),
        "Net Income":  safe_value(raw_income_ttm, "Net Income"),
        "FCF":         safe_value(raw_cashflow_ttm, "Free Cash Flow"),
        "Depreciation And Amortization": safe_value(raw_cashflow_ttm, "Depreciation And Amortization"),
        "Total Debt":  safe_value(raw_balance_ttm, "Total Debt"),
        "Net Debt":    np.nan,  # will set below
        "Cash":        safe_value(raw_balance_ttm, "Cash And Cash Equivalents"),
        "CapEx":       safe_value(raw_cashflow_ttm, "Capital Expenditure"),
        "Operating Cash Flow": safe_value(raw_cashflow_ttm, "Operating Cash Flow"),
        "Current Assets": safe_value(raw_balance_ttm, "Current Assets"),
        "Current Liabilities": safe_value(raw_balance_ttm, "Current Liabilities"),
        "Delta WC":    safe_value(raw_cashflow_ttm, "Change In Working Capital"),
        "Tax Provision": safe_value(raw_income_ttm, "Tax Provision"),
        "Pretax Income": safe_value(raw_income_ttm, "Pretax Income")
    }

    # Set Net Debt: use value if not nan, else fallback to Total Debt - Cash
    net_debt_val = safe_value(raw_balance_ttm, "Net Debt")
    if pd.isna(net_debt_val):
        total_debt = ttm_row["Total Debt"]
        cash = ttm_row["Cash"]
        net_debt_val = total_debt - cash if pd.notna(total_debt) and pd.notna(cash) else np.nan
    ttm_row["Net Debt"] = net_debt_val

    # Calculate TTM change in debt
    if raw_balance_ttm is not None and not raw_balance_ttm.empty:
        debt_series = raw_balance_ttm.loc["Total Debt"] if "Total Debt" in raw_balance_ttm.index else pd.Series(dtype=float)
        non_nan_quarters = debt_series.dropna().index
        if len(non_nan_quarters) > 0:
            latest_qtr = non_nan_quarters[0]
            latest_qtr_date = pd.to_datetime(latest_qtr)
            prev_year_qtr_date = latest_qtr_date - pd.DateOffset(years=1)
            prev_year_qtr = raw_balance_ttm.columns[raw_balance_ttm.columns == prev_year_qtr_date.strftime("%Y-%m-%d")]
            if len(prev_year_qtr) == 0:
                prev_year_qtr = raw_balance_ttm.columns[raw_balance_ttm.columns <= prev_year_qtr_date.strftime("%Y-%m-%d")]
                if len(prev_year_qtr) > 0:
                    prev_year_qtr = prev_year_qtr[-1]
                else:
                    prev_year_qtr = None
            else:
                prev_year_qtr = prev_year_qtr[0]
            prev_year_qtr = prev_year_qtr.strftime("%Y-%m-%d")

            latest_debt = safe_value(raw_balance_ttm[latest_qtr], "Total Debt")
            prev_year_debt = safe_value(raw_balance_ttm[prev_year_qtr], "Total Debt") if prev_year_qtr is not None else np.nan
            change_in_debt_ttm = latest_debt - prev_year_debt if pd.notna(latest_debt) and pd.notna(prev_year_debt) else np.nan
        else:
            change_in_debt_ttm = np.nan
    else:
        change_in_debt_ttm = np.nan

    ttm_row["Change in Debt"] = change_in_debt_ttm

    return ttm_row

def get_historical_data(
    ticker: str,
    tax_rate: float = 0.0,
    fields: List[str] = None,
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch historical and TTM financial data and key metrics.
    Returns:
        - df: DataFrame with yearly and TTM rows
        - valuation_info: Dict of current market valuation metrics
    """
    if logger is None:
        logger = logging.getLogger("data_fetcher")
    if fields is None:
        fields = [
            "Revenue", "EBITDA", "Net Income", "FCF", "Depreciation And Amortization", "Total Debt", "Cash",
            "CapEx", "Operating Cash Flow", "Net Debt", "Current Assets", "Current Liabilities", "Delta WC",
            "Tax Provision", "Pretax Income"
        ]

    def build_row(raw_dict, index=None):
        row = {k: raw_dict.get(k, np.nan) for k in fields}
        return pd.Series(row, name=index if index is not None else 0)

    t = yf.Ticker(ticker)

    raw_income        = t.financials
    raw_cashflow      = t.cashflow
    raw_balance       = t.balance_sheet
    raw_income_ttm    = t.ttm_financials
    raw_cashflow_ttm  = t.ttm_cashflow
    raw_balance_ttm   = t.quarterly_balancesheet

    def safe_row(df, row_name):
        return df.loc[row_name] if df is not None and row_name in df.index else None

    # Historical lines (only extract what is actually used below)
    hist_revenue    = safe_row(raw_income, "Total Revenue")
    hist_ebitda     = safe_row(raw_income, "EBITDA")
    hist_income     = safe_row(raw_income, "Net Income")
    hist_fcf        = safe_row(raw_cashflow, "Free Cash Flow")
    hist_ocf        = safe_row(raw_cashflow, "Operating Cash Flow")
    hist_net_debt   = safe_row(raw_balance, "Net Debt")
    hist_cur_assets = safe_row(raw_balance, "Current Assets")
    hist_cur_liab   = safe_row(raw_balance, "Current Liabilities")
    hist_capex      = safe_row(raw_cashflow, "Capital Expenditure")
    hist_depr_amort = safe_row(raw_cashflow, "Depreciation And Amortization")
    hist_debt       = safe_row(raw_balance, "Total Debt")
    hist_cash       = safe_row(raw_balance, "Cash And Cash Equivalents")
    hist_delta_wc   = safe_row(raw_cashflow, "Change In Working Capital")
    hist_tax_provision = safe_row(raw_income, "Tax Provision")
    hist_pretax_income = safe_row(raw_income, "Pretax Income")

    # Create unified index, converting all indices to string dates (YYYY-MM-DD) if they are Timestamps
    indices = [s.index for s in [hist_revenue, hist_ebitda, hist_income, hist_fcf, hist_ocf, hist_net_debt, hist_capex, hist_depr_amort, hist_debt, hist_cash] if s is not None]
    def to_str_date(idx):
        if isinstance(idx, pd.Timestamp):
            return idx.strftime("%Y-%m-%d")
        return str(idx)
    if indices:
        all_indices = reduce(lambda x, y: x.union(y), indices)
        str_indices = [to_str_date(idx) for idx in all_indices]
        common_index = pd.Index(str_indices)
    else:
        common_index = pd.Index([])

    # Build historical DataFrame using build_row for each period
    df_rows = []
    for period in common_index:
        # Net Debt: use value if not nan, else fallback to Total Debt - Cash
        net_debt_val = hist_net_debt[period] if hist_net_debt is not None and period in hist_net_debt else np.nan
        if pd.isna(net_debt_val):
            total_debt = hist_debt[period] if hist_debt is not None and period in hist_debt else np.nan
            cash = hist_cash[period] if hist_cash is not None and period in hist_cash else np.nan
            net_debt_val = total_debt - cash if pd.notna(total_debt) and pd.notna(cash) else np.nan
        raw_dict = {
            "Revenue": hist_revenue[period] if hist_revenue is not None and period in hist_revenue else np.nan,
            "EBITDA": hist_ebitda[period] if hist_ebitda is not None and period in hist_ebitda else np.nan,
            "Net Income": hist_income[period] if hist_income is not None and period in hist_income else np.nan,
            "FCF": hist_fcf[period] if hist_fcf is not None and period in hist_fcf else np.nan,
            "Depreciation And Amortization": hist_depr_amort[period] if hist_depr_amort is not None and period in hist_depr_amort else np.nan,
            "Total Debt": hist_debt[period] if hist_debt is not None and period in hist_debt else np.nan,
            "Cash": hist_cash[period] if hist_cash is not None and period in hist_cash else np.nan,
            "CapEx": hist_capex[period] if hist_capex is not None and period in hist_capex else np.nan,
            "Operating Cash Flow": hist_ocf[period] if hist_ocf is not None and period in hist_ocf else np.nan,
            "Net Debt": net_debt_val,
            "Current Assets": hist_cur_assets[period] if hist_cur_assets is not None and period in hist_cur_assets else np.nan,
            "Current Liabilities": hist_cur_liab[period] if hist_cur_liab is not None and period in hist_cur_liab else np.nan,
            "Delta WC": hist_delta_wc[period] if hist_delta_wc is not None and period in hist_delta_wc else np.nan,
            "Tax Provision": hist_tax_provision[period] if hist_tax_provision is not None and period in hist_tax_provision else np.nan,
            "Pretax Income": hist_pretax_income[period] if hist_pretax_income is not None and period in hist_pretax_income else np.nan
        }
        df_rows.append(build_row(raw_dict, index=period))

    # TTM row using build_row
    def safe_value(df, row):
        if df is not None and row in df.index:
            val = df.loc[row]
            if isinstance(val, pd.Series):
                return val.iloc[0]
            return val
        return np.nan


    # Build TTM row but do NOT append to df_rows
    ttm_row = build_ttm_row(raw_income_ttm, raw_cashflow_ttm, raw_balance_ttm, safe_value, None)
    # Use today's date as the TTM index (timestamp)
    ttm_index = pd.Timestamp.today().normalize()
    df = pd.DataFrame(df_rows)

    # Set index and sort once
    df.index.name = "Year"
    # Convert all date-like indices to pandas Timestamps, keep non-date (e.g., 'TTM') as NaT
    df.index = pd.to_datetime(df.index, errors='coerce')
    # Only sort if all index values are the same type and not all NaT
    index_types = set(type(idx) for idx in df.index if pd.notna(idx))
    if len(index_types) == 1 and list(index_types)[0] is pd.Timestamp:
        df = df.sort_index()

    sp    = t.info.get("currentPrice", np.nan)
    sh    = t.info.get("sharesOutstanding", np.nan)
    mcap  = sp * sh if not np.isnan(sp) and not np.isnan(sh) else np.nan
    ev    = t.info.get("enterpriseValue", np.nan)

    latest_debt = df["Total Debt"].dropna().iloc[-1] if not df["Total Debt"].dropna().empty else np.nan
    latest_cash = df["Cash"].dropna().iloc[-1] if not df["Cash"].dropna().empty else np.nan
    valuation_info = {
        "market_cap":       mcap,
        "share_price":      sp,
        "number_shares":    sh,
        "enterprise_value": ev,
        "total_debt":       latest_debt,
        "cash":             latest_cash,
        "net_debt":         (latest_debt - latest_cash) if pd.notna(latest_debt) and pd.notna(latest_cash) else np.nan
    }



    # Add ttm indicator and build ttm_df from ttm_row
    df['ttm'] = 0
    # Set TTM index to today's date as a pandas Timestamp
    ttm_index = pd.Timestamp.today().normalize()
    ttm_df = pd.DataFrame([ttm_row], index=[ttm_index])
    ttm_df['ttm'] = 1
    # Add any missing columns to ttm_df (e.g., ReinvestmentRate, FCFF, FCFE)
    for col in df.columns:
        if col not in ttm_df.columns:
            ttm_df[col] = np.nan
    for col in ttm_df.columns:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder columns to match
    ttm_df = ttm_df[df.columns]
    full_df = pd.concat([df, ttm_df], axis=0)

    # Calculate effective tax rate for each row and add as a column
    def compute_effective_tax_rate(row):
        tax_provision = row.get("Tax Provision", np.nan)
        pretax_income = row.get("Pretax Income", np.nan)
        if pd.notna(tax_provision) and pd.notna(pretax_income) and pretax_income != 0:
            rate = tax_provision / pretax_income
            if 0 <= rate <= 1:
                return rate
        return 0.25

    full_df["effective_tax_rate"] = full_df.apply(compute_effective_tax_rate, axis=1)

    # Reinvestment metrics (row-wise application)
    full_df["ReinvestmentRate"] = full_df.apply(
        lambda row: calculate_reinvestment_rate(
            row.get("CapEx", np.nan),
            row.get("Depreciation And Amortization", np.nan),
            row.get("Delta WC", np.nan),
            row.get("EBITDA", np.nan),
            row.get("effective_tax_rate", 0.25)
        ), axis=1
    )

    # === FCFF and FCFE calculation as one of the last steps (including TTM row) ===
    # Calculate FCFF and FCFE for all rows (including TTM)
    full_df["FCFF"] = full_df.apply(lambda row: calculate_fcff(row), axis=1)
    full_df["FCFE"] = full_df.apply(lambda row: calculate_fcfe(row), axis=1)

    return full_df, valuation_info


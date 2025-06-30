import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import yfinance as yf
import matplotlib.pyplot as plt

import debugpy
debugpy.listen(("localhost", 5680))  # You can use any open port, e.g., 5678
print("Waiting for debugger attach...")
debugpy.wait_for_client()  # This will pause execution until you attach the debugger



def compute_terminal_value(last_fcf: float, g: float, r: float) -> float:
    if r <= g:
        raise ValueError(f"WACC (r={r:.2%}) must be greater than g={g:.2%} for terminal value.")
    return last_fcf * (1 + g) / (r - g)

def discount_cash_flows(cash_flows: List[float], discount_rate: float) -> List[float]:
    discounted = []
    for t, cf in enumerate(cash_flows, start=1):
        discounted.append(cf / ((1 + discount_rate) ** t))
    return discounted

def estimate_wacc(ticker_symbol,
                  cost_of_debt = 0.0253,   # e.g. 0.045 for 4.5%
                  cost_of_equity = 0.1  # e.g. 0.09 for 9.0%
                 ):
    """
    Estimate WACC by:
      1. Pulling average Debt & Equity from the last 2 years' balance sheet
      2. Taking cost_of_debt and cost_of_equity as given
      3. Computing weights and WACC, plus a breakdown chart
    """
    # 1) Fetch and prepare balance-sheet data
    tk = yf.Ticker(ticker_symbol)
    bs = tk.balance_sheet.transpose().iloc[:2]
    bs['Total_Debt'] = bs.get('Short Long Term Debt', 0) + bs.get('Long Term Debt', 0)
    bs['Equity']     = bs['Total Assets'] - bs['Total_Debt'] - bs.get('Minority Interest', 0)

    D_avg = bs['Total_Debt'].mean()
    E_avg = bs['Equity'].mean()
    V     = D_avg + E_avg
    w_d   = D_avg / V
    w_e   = E_avg / V

    # 2) Calculate WACC
    debt_contrib   = w_d * cost_of_debt
    equity_contrib = w_e * cost_of_equity
    wacc = debt_contrib + equity_contrib


    # 4) Return summary
    return {
        'Debt (avg)':       D_avg,
        'Equity (avg)':     E_avg,
        'w_d':              w_d,
        'w_e':              w_e,
        'r_d (input)':      cost_of_debt,
        'r_e (input)':      cost_of_equity,
        'Debt contrib':     debt_contrib,
        'Equity contrib':   equity_contrib,
        'WACC':             wacc
    }

def run_dcf(
    df_hist: pd.DataFrame,
    df_proj: pd.DataFrame,
    terminal_growth: float,
    wacc: float,
    forecast_horizon: int = None,
    FCF_col: str = "FCF",
    p_method: str = "cagr",
    number_shares: int = 1,
    share_price: float = 1.0,
) -> tuple:
    """
    Computes DCF valuation using provided historical and projected values for a specified forecast horizon.

    Args:
        df_hist: Historical DataFrame (columns: Year, Revenue, EBITDA, FCF, ...)
        df_proj: Projected DataFrame (columns: Year, Revenue, EBITDA, FCF, ...)
        terminal_growth: Terminal growth rate (as a decimal, e.g., 0.025 for 2.5%)
        wacc: Weighted Average Cost of Capital (as a decimal, e.g., 0.08 for 8%)
        forecast_horizon: Number of projected years to include in DCF (first x years). If None, uses all available projections.
        FCF_col: Name of Free Cash Flow column in df.
        p_method: Projection method to filter on (e.g., "cagr").
        number_shares: Number of shares outstanding.
        share_price: Current share price for upside calculation.

    Returns:
        full_df: Combined DataFrame with DCF calculations (historical + selected projections)
        valuation_summary: dict with valuation results
    """
    # Filter projections by method and variable
    proj_mask = (df_proj["method"] == p_method) & (df_proj["variable"] == FCF_col)
    df_proj_fcf = df_proj.loc[proj_mask, ["Year", "value"]].copy()
    df_proj_fcf.rename(columns={"value": FCF_col}, inplace=True)

    # Limit to forecast_horizon if specified
    if forecast_horizon is not None:
        df_proj_fcf = df_proj_fcf.iloc[:forecast_horizon]

    # Extract cash flows series
    cash_flows = df_proj_fcf[FCF_col].copy().reset_index(drop=True)

    # Calculate growth series and last projected FCF
    growth_series = cash_flows.pct_change().dropna()
    last_proj_fcf = cash_flows.iloc[-1]

    # Calculate terminal value
    if wacc > terminal_growth:
        terminal_val = last_proj_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    else:
        terminal_val = np.nan

    # Add terminal value to last projected FCF
    cash_flows.iloc[-1] += terminal_val

    # Discount cash flows
    discount_factors = 1 / (1 + wacc) ** np.arange(1, len(cash_flows) + 1)
    pv_vals = cash_flows * discount_factors

    # Build projected DataFrame with DCF metrics
    df_proj_sub = df_proj_fcf.copy()
    df_proj_sub["PV(FCF+TV)"] = pv_vals
    df_proj_sub["Discount Factor"] = discount_factors
    df_proj_sub["Type"] = "projected"

    # Build historical DataFrame for output
    df_hist_sub = df_hist.reset_index()
    df_hist_sub = df_hist_sub.rename(columns={"index": "Year"})
    df_hist_sub = df_hist_sub[["Year", FCF_col]].copy()
    df_hist_sub['Year'] = pd.to_datetime(df_hist_sub['Year']).dt.year
    df_hist_sub["PV(FCF+TV)"] = np.nan
    df_hist_sub["Discount Factor"] = np.nan
    df_hist_sub["Type"] = "historical"

    # Combine historical and projected
    full_df = pd.concat([df_hist_sub, df_proj_sub.rename(columns={FCF_col: "FCF"})], ignore_index=True)

    # Valuation summary calculations
    sum_pv = float(np.nansum(pv_vals))
    avg_growth = float(np.nanmean(growth_series)) if not growth_series.empty else np.nan
    per_share_value = sum_pv / number_shares
    upside = 100 * (per_share_value - share_price) / share_price

    valuation_summary = {
        "sum_pv": sum_pv,
        "terminal_value": float(terminal_val),
        "npv": sum_pv,
        "per_share_value": per_share_value,
        "upside": upside,
        "average_growth_rate": avg_growth,
    }
    debugpy.breakpoint()
    return full_df, valuation_summary


def perform_sensitivity_analysis(
    last_fcf: float,
    wacc: float,
    terminal_growth: float,
    projection_years: int,
    fcf_to_ebitda_ratio: float,
    market_cap: float,
) -> pd.DataFrame:
    """
    Sensitivity #1: vary FCF→EBITDA ratio (0.4 to 0.8 in 9 steps) vs. EBITDA growth (±2% around terminal_growth).
    Computes (PV CF+TV)/market_cap. Returns a DataFrame indexed by ratio, columns by growth_rate.
    """
    ratios = np.linspace(0.4, 0.8, 9)
    growth_rates = np.linspace(max(0.0, terminal_growth - 0.02), terminal_growth + 0.02, 9)

    heatmap_matrix = []
    for ratio in ratios:
        row = []
        for g in growth_rates:
            try:
                last_ebitda = last_fcf / ratio
                projected_ebitda = [
                    last_ebitda * ((1 + g) ** i)
                    for i in range(1, projection_years + 1)
                ]
                projected_fcf = [eb * ratio for eb in projected_ebitda]
                terminal_val = compute_terminal_value(projected_fcf[-1], terminal_growth, wacc)
                cf_with_term = projected_fcf.copy()
                cf_with_term[-1] += terminal_val
                pv_vals = discount_cash_flows(cf_with_term, wacc)
                iv_total = sum(pv_vals)
                row.append((iv_total / market_cap) * 100.0 if market_cap > 0 else np.nan)
            except ValueError:
                row.append(np.nan)
        heatmap_matrix.append(row)

    df_heat = pd.DataFrame(
        heatmap_matrix,
        index=np.round(ratios, 2),
        columns=np.round(growth_rates, 3),
    )
    df_heat.index.name = "FCF/EBITDA Ratio"
    df_heat.columns.name = "EBITDA Growth"
    return df_heat

def perform_additional_sensitivity(
    df_hist: pd.DataFrame,
    market_cap: float,
    wacc: float,
    terminal_growth: float,
    projection_years: int,
    fcf_to_ebitda_ratio: float,
) -> pd.DataFrame:
    """
    Sensitivity #2: vary WACC ±5% (11 steps) vs. terminal_growth ±2% (11 steps),
    compute (PV CF+TV)/market_cap. Returns a DataFrame indexed by WACC, columns by g.
    """
    wacc_range = np.linspace(max(0.0, wacc - 0.05), wacc + 0.05, 11)
    g_range = np.linspace(max(0.0, terminal_growth - 0.02), terminal_growth + 0.02, 11)
    last_ebitda = df_hist["EBITDA"].iloc[-1]

    rows = []
    for w in wacc_range:
        row = []
        for g in g_range:
            try:
                e_first = df_hist["EBITDA"].iloc[0]
                e_last = df_hist["EBITDA"].iloc[-1]
                periods = len(df_hist["EBITDA"]) - 1
                hist_eb_growth = (e_last / e_first) ** (1 / periods) - 1

                projected_ebitda = [
                    last_ebitda * ((1 + hist_eb_growth) ** i)
                    for i in range(1, projection_years + 1)
                ]
                projected_fcf = [eb * fcf_to_ebitda_ratio for eb in projected_ebitda]

                terminal_val = compute_terminal_value(projected_fcf[-1], g, w)
                cf_with_term = projected_fcf.copy()
                cf_with_term[-1] += terminal_val
                pv_vals = discount_cash_flows(cf_with_term, w)
                iv_total = sum(pv_vals)
                row.append((iv_total / market_cap) * 100.0 if market_cap > 0 else np.nan)
            except ValueError:
                row.append(np.nan)
        rows.append(row)

    df_heat2 = pd.DataFrame(
        rows,
        index=np.round(wacc_range, 3),
        columns=np.round(g_range, 3),
    )
    df_heat2.index.name = "WACC"
    df_heat2.columns.name = "Terminal Growth"
    return df_heat2
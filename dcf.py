import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import yfinance as yf
import matplotlib.pyplot as plt
import debugpy
from datetime import datetime, timedelta
import os  

"""
import debugpy
debugpy.listen(("localhost", 5682))  # You can use any open port, e.g., 5678
print("Waiting for debugger attach...")
debugpy.wait_for_client()  # This will pause execution until you attach the debugger
"""


# --- New functions for beta estimation ---
def get_country_index(ticker):
    """
    Returns a recommended major index ticker for beta calculation based on the company's country.
    Uses yfinance to fetch the country from ticker.info.
    """
    country_index_map = {
        "United States": "^GSPC",      # S&P 500
        "Denmark": "^OMXC25",          # OMX Copenhagen 25
        "Sweden": "^OMXS30",           # OMX Stockholm 30
        "Norway": "^OBX",              # Oslo OBX
        "Finland": "^OMXH25",          # OMX Helsinki 25
        "Germany": "^GDAXI",           # DAX
        "France": "^FCHI",             # CAC 40
        "United Kingdom": "^FTSE",     # FTSE 100
        "Netherlands": "^AEX",         # AEX
        "Switzerland": "^SSMI",        # SMI
        "Italy": "FTSEMIB.MI",         # FTSE MIB
        "Spain": "^IBEX",              # IBEX 35
        "Japan": "^N225",              # Nikkei 225
        "China": "000001.SS",          # SSE Composite
        "Hong Kong": "^HSI",           # Hang Seng
        "Canada": "^GSPTSE",           # S&P/TSX
        "Australia": "^AXJO",          # ASX 200
        # Add more as needed
    }
    try:
        info = yf.Ticker(ticker).info
        country = info.get("country", None)
        if country is None:
            print(f"Could not determine country for {ticker}.")
            return None
        index = country_index_map.get(country, None)
        if index is None:
            print(f"No index mapping found for country: {country}")
        return index
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
        return None

def estimate_beta_against_index(ticker, period_years=5, plot=False, plot_path=None):
    """
    Estimate beta for a stock against its local major index.
    """
    index_ticker = get_country_index(ticker)
    if index_ticker is None:
        raise ValueError(f"Could not determine index for ticker {ticker}.")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_years * 365)
    
    # Download adjusted close prices
    data = yf.download([ticker, index_ticker], start=start_date, end=end_date, auto_adjust=True, progress=False)
    if 'Close' not in data.columns:
        raise ValueError("'Close' not found in downloaded data. Data may be empty or ticker symbols may be incorrect.")
    close = data['Close'].dropna()
    if ticker not in close.columns or index_ticker not in close.columns:
        raise ValueError(f"One or both tickers not found in 'Close': {ticker}, {index_ticker}")
    
    # Calculate daily returns
    returns = close.pct_change().dropna()
    stock_returns = returns[ticker]
    index_returns = returns[index_ticker]
    
    # Calculate covariance and variance
    cov = stock_returns.cov(index_returns)
    var = index_returns.var()
    
    # Beta calculation
    beta_est = cov / var

    # Plot if requested
    if plot or plot_path:
        plt.figure(figsize=(8, 5))
        plt.scatter(index_returns, stock_returns, alpha=0.5, label='Daily returns')
        # Regression line
        m, b = np.polyfit(index_returns, stock_returns, 1)
        plt.plot(index_returns, m * index_returns + b, color='red', label=f'Fit: y={m:.2f}x+{b:.2e}')
        plt.title(f"Beta estimation for {ticker} vs {index_ticker}\nBeta ≈ {beta_est:.3f}")
        plt.xlabel(f"{index_ticker} daily returns")
        plt.ylabel(f"{ticker} daily returns")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if plot_path:
            plt.savefig(plot_path)
        if plot:
            plt.show()
        plt.close()
    
    return beta_est


def compute_terminal_value(last_fcf: float, g: float, r: float) -> float:
    if r <= g:
        raise ValueError(f"WACC (r={r:.2%}) must be greater than g={g:.2%} for terminal value.")
    return last_fcf * (1 + g) / (r - g)

def discount_cash_flows(cash_flows: List[float], discount_rate: float) -> List[float]:
    discounted = []
    for t, cf in enumerate(cash_flows, start=1):
        discounted.append(cf / ((1 + discount_rate) ** t))
    return discounted

def get_cost_of_equity(
    ticker: str,
    risk_free_rate: float = 0.045,
    equity_premium: float = 0.0547
) -> float:
    """
    Calculate cost of equity using CAPM and estimated beta.
    Args:
        ticker: Stock ticker symbol (str)
        risk_free_rate: Risk-free rate (decimal)
        equity_premium: Expected market premium (decimal)
    Returns:
        Cost of equity (decimal)
    """
    # Use custom beta estimation instead of yfinance beta
    beta = estimate_beta_against_index(ticker, period_years=5, plot=False)
    if beta is None:
        raise ValueError(f"Beta not available for ticker {ticker}")
    return risk_free_rate + beta * equity_premium


def estimate_cost_of_debt(df_hist: pd.DataFrame, min_pre_tax: float = 0.04, plot: bool = True, plot_filename: str = "cost_of_debt.png") -> float:
    """
    Estimate cost of debt from historical data and optionally plot it.

    Args:
        df (pd.DataFrame): Must include 'Interest Expense', 'Total Debt', and 'effective_tax_rate'.
        min_pre_tax (float): Minimum pre-tax cost of debt.
        plot (bool): If True, save a dual-axis plot of cost of debt and total debt to output folder.
        plot_filename (str): Name of the plot file to save.

    Returns:
        float: Final conservative estimate of after-tax cost of debt.
    """
    df = df_hist.copy()

    # Calculate cost of debt
    df["Cost of Debt (pre-tax)"] = (
        df["Interest Expense"] / df["Total Debt"]
    )

    df["Cost of Debt (after-tax)"] = (
        df["Cost of Debt (pre-tax)"] * (1 - df["effective_tax_rate"])
    )

    # Drop NaNs
    df.dropna(subset=["Cost of Debt (after-tax)", "Total Debt"], inplace=True)

    if df.empty:
        raise ValueError("No valid rows to compute cost of debt.")

    # Compute average and current after-tax cost of debt
    avg_cod = df["Cost of Debt (after-tax)"].mean()
    current_cod = df.iloc[0]["Cost of Debt (after-tax)"]

    # Plot if requested
    if plot:
        output_dir = os.path.dirname(plot_filename)
        os.makedirs(output_dir, exist_ok=True)
        plot_path = plot_filename

        fig, ax1 = plt.subplots(figsize=(8, 5))

        x = df.index  # Use index for x-axis

        ax2 = ax1.twinx()
        ax1.plot(x, df["Cost of Debt (after-tax)"], label="After-tax Cost of Debt", linestyle=":", color="green")
        ax1.plot(x, df["Cost of Debt (pre-tax)"], label="Pre-tax Cost of Debt", linestyle="--", color="orange")
        ax2.plot(x, df["Total Debt"], label="Total Debt", color="blue")

        ax1.set_xlabel("Period")
        ax1.set_ylabel("Cost of Debt")
        ax2.set_ylabel("Total Debt")
        ax2.set_ylim(0, df["Total Debt"].max() * 1.1)  # Set y-limits for total debt
        ax1.set_title("Estimated Cost of Debt and Total Debt Over Time")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return round(max(avg_cod, current_cod, min_pre_tax*(1-df.iloc[0]["effective_tax_rate"])), 4)

def estimate_wacc(
    df_hist,
    cost_of_debt=0.0253,
    cost_of_equity=0.12,
    ticker: str = None,
    risk_free_rate: float = 0.045,
    equity_premium: float = 0.0547,
    beta_plot_path: str = None
):
    """
    Estimate WACC by:
      1. Using the last observation in df_hist for Total Debt and Equity
      2. Taking cost_of_debt and cost_of_equity as given, or fetching cost_of_equity using CAPM if ticker is provided
      3. Computing weights and WACC, plus a breakdown chart
    Args:
        df_hist: DataFrame with at least 'Total Debt' and 'Equity' columns (last row used)
        cost_of_debt: Cost of debt (as decimal)
        cost_of_equity: Cost of equity (as decimal)
        ticker: Stock ticker symbol (optional, if provided fetches cost_of_equity using CAPM)
        risk_free_rate: Risk-free rate (decimal)
        equity_premium: Expected market premium (decimal)
    Returns:
        dict with WACC breakdown
    """
    beta = None
    if ticker is not None:
        beta = estimate_beta_against_index(ticker, period_years=5, plot=False, plot_path=beta_plot_path)
        cost_of_equity = risk_free_rate + beta * equity_premium
    print(f"Using cost of equity: {cost_of_equity:.4f} (beta={beta:.4f})")
    # Get last observation for Total Debt and Equity
    total_debt = df_hist['Total Debt'].dropna().iloc[-1]
    equity = df_hist['Total Assets'].dropna().iloc[-1] - total_debt

    D = total_debt
    E = equity
    V = D + E
    w_d = D / V if V != 0 else 0
    w_e = E / V if V != 0 else 0

    # Calculate WACC
    debt_contrib   = w_d * cost_of_debt
    equity_contrib = w_e * cost_of_equity
    wacc = debt_contrib + equity_contrib

    return {
        'Debt (last)':       D,
        'Equity (last)':     E,
        'w_d':              w_d,
        'w_e':              w_e,
        'r_d (input)':      cost_of_debt,
        'r_e (input)':      cost_of_equity,
        'Debt contrib':     debt_contrib,
        'Equity contrib':   equity_contrib,
        'WACC':             wacc,
        'beta':             beta,
        'risk_free_rate':   risk_free_rate,
        'equity_premium':   equity_premium
    }

def run_dcf(
    df_hist: pd.DataFrame,
    df_proj: pd.DataFrame,
    terminal_growth: float,
    wacc: float,
    forecast_horizon: int = None,
    p_method: str = "cagr",
    number_shares: int = 1,
    share_price: float = 1.0
) -> tuple:
    
    """
    Computes DCF valuation using provided historical and projected values for a specified forecast horizon.

    Args:
        df_hist: Historical DataFrame (columns: Year, Revenue, EBITDA, FCF, ...)
        df_proj: Projected DataFrame (columns: Year, Revenue, EBITDA, FCF, ...)
        terminal_growth: Terminal growth rate (as a decimal, e.g., 0.025 for 2.5%)
        wacc: Weighted Average Cost of Capital (as a decimal, e.g., 0.08 for 8%)
        forecast_horizon: Number of projected years to include in DCF (first x years). If None, uses all available projections.
        p_method: Projection method to filter on (e.g., "cagr").
        number_shares: Number of shares outstanding.
        share_price: Current share price for upside calculation.

    Returns:
        full_df: Combined DataFrame with DCF calculations (historical + selected projections)
        valuation_summary: dict with valuation results
    """
 
    # Filter projections by method and variable
    proj_mask = (df_proj["method"] == p_method) & (df_proj["variable"] == "FCFF")
    df_proj_fcf = df_proj.loc[proj_mask, ["Year", 'value', "growth_rate"]].copy()
    df_proj_fcf.rename(columns={'value': "FCFF"}, inplace=True)
    # Limit to forecast_horizon if specified
    if forecast_horizon is not None:
        df_proj_fcf = df_proj_fcf.iloc[:forecast_horizon]

    # Extract cash flows series
    cash_flows = df_proj_fcf["FCFF"].copy().reset_index(drop=True)

    # Calculate growth series and last projected FCF
    growth_series = cash_flows.pct_change(fill_method=None).dropna()
    avg_growth = float(np.nanmean(growth_series)) if not growth_series.empty else np.nan
    last_proj_fcf = cash_flows.iloc[-1]

    # Calculate terminal value
    if wacc > terminal_growth:
        terminal_val = last_proj_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    else:
        terminal_val = np.nan

    # Discount cash flows (excluding terminal value for now)
    discount_factors = 1 / ((1 + wacc) ** np.arange(1, len(cash_flows) + 1))
    pv_vals = cash_flows * discount_factors
  
    # Discount terminal value using the last year's discount factor
    terminal_discount_factor = discount_factors[-1]
    pv_terminal = terminal_val * terminal_discount_factor

    # Build projected DataFrame with DCF metrics
    df_proj_sub = df_proj_fcf.copy().reset_index(drop=True)
    df_proj_sub["PV"] = pv_vals
    df_proj_sub["Discount Factor"] = discount_factors
    df_proj_sub["Type"] = "Projected calendar year"

    # Add terminal value as a separate row
    terminal_row = {
        "Year": None,
        "FCFF": terminal_val,
        "growth_rate": terminal_growth,
        "PV": pv_terminal,
        "Discount Factor": terminal_discount_factor,
        "Type": "Terminal value"
    }
    df_proj_sub = pd.concat([df_proj_sub, pd.DataFrame([terminal_row])], ignore_index=True)

    # Build historical DataFrame for output
    df_hist_sub = df_hist.reset_index()
    df_hist_sub = df_hist_sub.rename(columns={"index": "Year"})
    df_hist_sub = df_hist_sub[["Year", "FCFF", "ttm"]].copy()
    df_hist_sub['Year'] = pd.to_datetime(df_hist_sub['Year']).dt.year
    df_hist_sub["growth_rate"] = df_hist_sub["FCFF"].pct_change(fill_method=None).fillna(0)
    df_hist_sub["growth_rate"] = df_hist_sub["growth_rate"].infer_objects(copy=False)
    df_hist_sub["PV"] = np.nan
    df_hist_sub["Discount Factor"] = np.nan
    df_hist_sub["Type"] = np.where(df_hist_sub["ttm"] == 1, "Historical ttm", "Historical calendar year")
    df_hist_sub = df_hist_sub.drop(columns=["ttm"])

    # Combine historical and projected
    full_df = pd.concat([df_hist_sub, df_proj_sub], ignore_index=True)

    # Valuation summary calculations
    enterprise_value = float(np.nansum(pv_vals)) + float(pv_terminal)
    equity_value = enterprise_value - df_hist['Net Debt'].dropna().iloc[-1]
    per_share_value = equity_value / number_shares
    upside = 100 * (per_share_value - share_price) / share_price

    valuation_summary = {
        "enterprise_value": enterprise_value,
        "terminal_value": float(terminal_val),
        "npv": enterprise_value,
        "net_debt": df_hist['Net Debt'].dropna().iloc[-1],
        "equity_value": equity_value,
        "per_share_value": per_share_value,
        "upside": upside,
        "average_growth_rate": avg_growth,
    }

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



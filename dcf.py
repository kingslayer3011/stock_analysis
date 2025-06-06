import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def read_projected_fcf(csv_path: str, method: str = "cagr") -> pd.DataFrame:
    """
    Reads the projected FCF values from a CSV, filtered by the chosen projection method.

    Args:
        csv_path: Path to the projected_values.csv file.
        method: Projection method to use ("cagr" or "slope").

    Returns:
        DataFrame with columns: ['Year', 'value'] (sorted by Year).
    """
    df_proj = pd.read_csv(csv_path)
    # Filter for FCF and desired method
    fcf_proj = df_proj[(df_proj['variable'] == 'FCF') & (df_proj['method'] == method)]
    fcf_proj = fcf_proj.sort_values(by='Year')
    return fcf_proj[['Year', 'value']]

def compute_terminal_value(last_fcf: float, g: float, r: float) -> float:
    if r <= g:
        raise ValueError(f"WACC (r={r:.2%}) must be greater than g={g:.2%} for terminal value.")
    return last_fcf * (1 + g) / (r - g)

def discount_cash_flows(cash_flows: List[float], discount_rate: float) -> List[float]:
    discounted = []
    for t, cf in enumerate(cash_flows, start=1):
        discounted.append(cf / ((1 + discount_rate) ** t))
    return discounted
def run_dcf(
    projected_fcf: pd.DataFrame,
    terminal_growth: float,
    wacc: float,
    last_actual_year: int,
    shares_outstanding: float,
    method: str = "cagr"
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Runs a DCF using provided projected FCFs.

    Args:
        projected_fcf: DataFrame with columns ['Year', 'value'].
        terminal_growth: perpetual growth rate for the terminal value.
        wacc: discount rate.
        last_actual_year: the last year of historical data (for reporting/validation).
        shares_outstanding: number of shares outstanding for per-share value calculation.
        method: the method used for projection ("cagr" or "slope").

    Returns:
        dcf_df: DataFrame with columns ["Year", "ProjectedFCF", "PV(FCF+TV)"]
        valuation_summary: dict with keys ["sum_pv", "terminal_value", "npv", "per_share_value", "method"]
    """
    years = projected_fcf['Year'].values
    projected_fcf_values = projected_fcf['value'].values

    # Calculate terminal value on the final projected FCF
    last_proj_fcf = projected_fcf_values[-1]
    terminal_val = compute_terminal_value(last_proj_fcf, terminal_growth, wacc)

    # Append terminal value to the last year's cash flow
    cash_flows_with_term = projected_fcf_values.copy()
    cash_flows_with_term[-1] += terminal_val

    # Discount each cash flow (including terminal) back to present value
    pv_vals = discount_cash_flows(cash_flows_with_term, wacc)

    # Build the output DataFrame
    dcf_df = pd.DataFrame({
        "Year": years,
        "ProjectedFCF": projected_fcf_values,
        "PV(FCF+TV)": pv_vals,
    })

    sum_pv = float(np.nansum(pv_vals))
    npv = sum_pv
    per_share_value = sum_pv / shares_outstanding if shares_outstanding and shares_outstanding > 0 else np.nan
    valuation_summary = {
        "sum_pv": sum_pv,
        "terminal_value": terminal_val,
        "npv": npv,
        "per_share_value": per_share_value,
        "method": method
    }
    return dcf_df, valuation_summary

# Sensitivity functions remain unchanged
def perform_sensitivity_analysis(
    last_fcf: float,
    wacc: float,
    terminal_growth: float,
    projection_years: int,
    fcf_to_ebitda_ratio: float,
    market_cap: float,
) -> pd.DataFrame:
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
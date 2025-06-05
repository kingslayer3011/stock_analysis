# dcf.py

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def compute_terminal_value(last_fcf: float, g: float, r: float) -> float:
    """
    Calculates the terminal (continuing) value using the Gordon Growth model:
      TV = FCF_last * (1 + g) / (r – g),  with r > g.
    Raises ValueError if r <= g.
    """
    if r <= g:
        raise ValueError(f"WACC (r={r:.2%}) must be greater than g={g:.2%} for terminal value.")
    return last_fcf * (1 + g) / (r - g)


def discount_cash_flows(cash_flows: List[float], discount_rate: float) -> List[float]:
    """
    Given a list of future cash flows [CF1, CF2, ..., CFn],
    returns [PV1, PV2, ..., PVn], where PV_t = CF_t / (1+r)^t.
    """
    discounted = []
    for t, cf in enumerate(cash_flows, start=1):
        discounted.append(cf / ((1 + discount_rate) ** t))
    return discounted


def run_dcf(
    df_hist: pd.DataFrame,
    projection_years: int,
    terminal_growth: float,
    wacc: float,
    use_ebitda_base: bool = False,
    fcf_to_ebitda_ratio: float = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Runs a simple DCF based on historical FCF (or EBITDA + ratio) growth.

    Args:
      df_hist: historical DataFrame (index = Year [Timestamp], oldest→newest) with at least
               columns ["EBITDA", "FCF"].
      projection_years: number of years to project into the future.
      terminal_growth: perpetual growth rate for the terminal value.
      wacc: discount rate.
      use_ebitda_base: if True, base projection on EBITDA and convert to FCF via ratio.
      fcf_to_ebitda_ratio: ratio to convert projected EBITDA → FCF (required if use_ebitda_base=True).

    Returns:
      dcf_df: DataFrame with columns ["Year", "ProjectedEBITDA", "ProjectedFCF", "PV(FCF+TV)"]
      valuation_summary: dict with keys ["sum_pv", "terminal_value", "npv", "per_share_value"]
    """
    # Determine historical base and growth rate
    if use_ebitda_base:
        last_ebitda = df_hist["EBITDA"].iloc[-1]
        # CAGR of EBITDA over history
        try:
            e_first = df_hist["EBITDA"].iloc[0]
            e_last = df_hist["EBITDA"].iloc[-1]
            periods_eb = len(df_hist["EBITDA"]) - 1
            hist_eb_growth = (e_last / e_first) ** (1 / periods_eb) - 1
        except Exception:
            hist_eb_growth = np.nan

        if fcf_to_ebitda_ratio is None:
            raise ValueError("fcf_to_ebitda_ratio must be provided if use_ebitda_base=True")
        last_fcf = last_ebitda * fcf_to_ebitda_ratio
        growth_rate = hist_eb_growth
    else:
        last_fcf = df_hist["FCF"].iloc[-1]
        # CAGR of FCF over history
        try:
            f_first = df_hist["FCF"].iloc[0]
            f_last = df_hist["FCF"].iloc[-1]
            periods_fc = len(df_hist["FCF"]) - 1
            growth_rate = (f_last / f_first) ** (1 / periods_fc) - 1
        except Exception:
            growth_rate = np.nan

    # Build projections
    years: List[int] = []
    projected_ebitda: List[float] = []
    projected_fcf: List[float] = []

    # df_hist.index[-1] is a pandas.Timestamp (e.g., 2023-09-30)
    last_year = df_hist.index[-1].year

    for i in range(1, projection_years + 1):
        years.append(last_year + i)

        if use_ebitda_base:
            # Project EBITDA via historical CAGR
            ebitda_i = last_ebitda * ((1 + growth_rate) ** i)
            fcf_i = ebitda_i * fcf_to_ebitda_ratio
        else:
            # Project FCF directly via historical CAGR
            fcf_i = last_fcf * ((1 + growth_rate) ** i)
            # For completeness, also project EBITDA if needed downstream
            try:
                # Recompute EBITDA CAGR
                e_first = df_hist["EBITDA"].iloc[0]
                e_last = df_hist["EBITDA"].iloc[-1]
                periods_eb = len(df_hist["EBITDA"]) - 1
                eb_growth = (e_last / e_first) ** (1 / periods_eb) - 1
                ebitda_i = e_last * ((1 + eb_growth) ** i)
            except Exception:
                ebitda_i = np.nan

        projected_ebitda.append(ebitda_i)
        projected_fcf.append(fcf_i)

    # Calculate terminal value on the final projected FCF
    last_proj_fcf = projected_fcf[-1]
    terminal_val = compute_terminal_value(last_proj_fcf, terminal_growth, wacc)

    # Append terminal value to the last year's cash flow
    cash_flows_with_term = projected_fcf.copy()
    cash_flows_with_term[-1] += terminal_val

    # Discount each cash flow (including terminal) back to present value
    pv_vals = discount_cash_flows(cash_flows_with_term, wacc)

    # Build the output DataFrame
    dcf_df = pd.DataFrame({
        "Year": years,
        "ProjectedEBITDA": projected_ebitda,
        "ProjectedFCF": projected_fcf,
        "PV(FCF+TV)": pv_vals,
    })

    # Summarize valuation
    sum_pv = float(np.nansum(pv_vals))
    npv = sum_pv
    valuation_summary = {
        "sum_pv": sum_pv,
        "terminal_value": terminal_val,
        "npv": npv,
        "per_share_value": np.nan,  # to be filled in reporter.py if needed
    }

    return dcf_df, valuation_summary

#! Create new separate module for sensitivity analysis
def perform_sensitivity_analysis(
    last_fcf: float,
    wacc: float,
    terminal_growth: float,
    projection_years: int,
    fcf_to_ebitda_ratio: float,
    market_cap: float,
) -> pd.DataFrame:
    """
    Sensitivity #1: vary FCF→EBITDA ratio (0.4 to 0.8 in 9 steps) vs. EBITDA growth (±2% around historical).
    Computes (PV CF+TV)/market_cap. Returns a DataFrame indexed by ratio, columns by growth_rate.
    """
    # Range of ratios (e.g., 0.40, 0.45, ..., 0.80)
    ratios = np.linspace(0.4, 0.8, 9)

    # Range of growth rates ±2% around terminal_growth
    growth_rates = np.linspace(max(0.0, terminal_growth - 0.02), terminal_growth + 0.02, 9)

    heatmap_matrix = []
    for ratio in ratios:
        row = []
        for g in growth_rates:
            try:
                # If projecting from last_fcf via ratio: derive last_ebitda = last_fcf / ratio
                last_ebitda = last_fcf / ratio

                # Project EBITDA series
                projected_ebitda = [
                    last_ebitda * ((1 + g) ** i)
                    for i in range(1, projection_years + 1)
                ]
                # Convert EBITDA→FCF via this ratio
                projected_fcf = [eb * ratio for eb in projected_ebitda]

                # Terminal value at final projected FCF
                terminal_val = compute_terminal_value(projected_fcf[-1], terminal_growth, wacc)

                # Append TV to last year's cash flow
                cf_with_term = projected_fcf.copy()
                cf_with_term[-1] += terminal_val

                # Discount back
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

    # Historical base for EBITDA-to-FCF conversion
    last_ebitda = df_hist["EBITDA"].iloc[-1]

    rows = []
    for w in wacc_range:
        row = []
        for g in g_range:
            try:
                # Project EBITDA with historical CAGR from df_hist
                # Compute historical CAGR of EBITDA
                e_first = df_hist["EBITDA"].iloc[0]
                e_last = df_hist["EBITDA"].iloc[-1]
                periods = len(df_hist["EBITDA"]) - 1
                hist_eb_growth = (e_last / e_first) ** (1 / periods) - 1

                # Project EBITDA series
                projected_ebitda = [
                    last_ebitda * ((1 + hist_eb_growth) ** i)
                    for i in range(1, projection_years + 1)
                ]
                # Convert to FCF via ratio
                projected_fcf = [eb * fcf_to_ebitda_ratio for eb in projected_ebitda]

                # Terminal value on last projected FCF
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


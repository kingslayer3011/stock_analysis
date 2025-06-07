import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def compute_terminal_value(last_fcf: float, g: float, r: float) -> float:
    if r <= g:
        raise ValueError(f"WACC (r={r:.2%}) must be greater than g={g:.2%} for terminal value.")
    return last_fcf * (1 + g) / (r - g)

def discount_cash_flows(cash_flows: List[float], discount_rate: float) -> List[float]:
    discounted = []
    for t, cf in enumerate(cash_flows, start=1):
        discounted.append(cf / ((1 + discount_rate) ** t))
    return discounted

def calculate_regression_slope(series):
    y = pd.to_numeric(series, errors="coerce").values.astype(float)
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan, np.nan
    x = x[mask]
    y = y[mask]
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)

def _get_growth_vector(initial_growth, terminal_growth, projection_years, converge_growth):
    if not converge_growth or initial_growth is None or np.isnan(initial_growth):
        return [initial_growth] * projection_years
    else:
        N = projection_years
        growths = []
        for i in range(1, N+1):
            g = initial_growth * (1 - i/N) + terminal_growth * (i/N)
            growths.append(g)
        return growths

def run_dcf(
    df_hist: pd.DataFrame,
    projection_years: int,
    terminal_growth: float,
    wacc: float,
    use_ebitda_base: bool = False,
    fcf_to_ebitda_ratio: float = None,
    expected_growth: float = None,  # can be float or the string "slope"
    converge_growth: bool = False,  # NEW
) -> tuple:
    """
    Runs a DCF based on historical FCF (or EBITDA + ratio) growth,
    or using a user-supplied expected growth, or regression slope if expected_growth=="slope".

    Returns:
      full_df: DataFrame with historical and projected values with a 'Type' column and "Discount Factor".
      valuation_summary: dict with keys ["sum_pv", "terminal_value", "npv", "per_share_value", "average_growth_rate"]
    """

    # ========== Project Revenue ==========
    last_revenue = df_hist["Revenue"].iloc[-1]
    revenue_series = df_hist["Revenue"].dropna()
    projected_revenue = []
    # --- Compute revenue growth vector ---
    if expected_growth == "slope":
        s_rev = revenue_series
        slope_rev, intercept_rev = calculate_regression_slope(s_rev)
        projected_revenue = [
            float(slope_rev * (len(s_rev)-1+i) + intercept_rev)
            if slope_rev is not None and intercept_rev is not None else np.nan
            for i in range(1, projection_years+1)
        ]
        growth_rate_rev = None  # Not used for revenue
    else:
        if expected_growth is not None:
            growth_rate_rev = expected_growth
        else:
            try:
                r_first = revenue_series.iloc[0]
                r_last = revenue_series.iloc[-1]
                periods_rv = len(revenue_series) - 1
                growth_rate_rev = (r_last / r_first) ** (1 / periods_rv) - 1
            except Exception:
                growth_rate_rev = np.nan
        # Optionally converge growth
        rev_growth_vec = _get_growth_vector(growth_rate_rev, terminal_growth, projection_years, converge_growth)
        projected_revenue = []
        v = last_revenue
        for g in rev_growth_vec:
            v = v * (1 + g) if v is not None and not np.isnan(v) and g is not None and not np.isnan(g) else np.nan
            projected_revenue.append(v)

    # ========== Project EBITDA & FCF ==========
    if use_ebitda_base:
        last_ebitda = df_hist["EBITDA"].iloc[-1]
        eb_growth_vec = []
        if expected_growth == "slope":
            s = df_hist["EBITDA"].dropna()
            slope, intercept = calculate_regression_slope(s)
            projected_ebitda = [
                float(slope * (len(s)-1+i) + intercept) if slope is not None and intercept is not None else np.nan
                for i in range(1, projection_years+1)
            ]
            # FCF derived from EBITDA
            projected_fcf = [
                float(eb * fcf_to_ebitda_ratio) if eb is not None and not pd.isna(eb) else np.nan
                for eb in projected_ebitda
            ]
            # Estimate implied ebitda growth for average_growth_rate
            eb_growth_vec = []
            eb_v = last_ebitda
            for i in range(1, projection_years+1):
                next_eb = float(slope * (len(s)-1+i) + intercept) if slope is not None and intercept is not None else np.nan
                g = (next_eb/eb_v - 1) if eb_v and next_eb and not np.isnan(eb_v) and not np.isnan(next_eb) and eb_v != 0 else np.nan
                eb_growth_vec.append(g)
                eb_v = next_eb
        else:
            if expected_growth is not None:
                growth_rate = expected_growth
            else:
                try:
                    e_first = df_hist["EBITDA"].iloc[0]
                    e_last = df_hist["EBITDA"].iloc[-1]
                    periods_eb = len(df_hist["EBITDA"]) - 1
                    growth_rate = (e_last / e_first) ** (1 / periods_eb) - 1
                except Exception:
                    growth_rate = np.nan
            eb_growth_vec = _get_growth_vector(growth_rate, terminal_growth, projection_years, converge_growth)
            projected_ebitda = []
            v = last_ebitda
            for g in eb_growth_vec:
                v = v * (1 + g) if v is not None and not np.isnan(v) and g is not None and not np.isnan(g) else np.nan
                projected_ebitda.append(v)
            projected_fcf = [eb * fcf_to_ebitda_ratio if eb is not None and not np.isnan(eb) else np.nan for eb in projected_ebitda]
    else:
        last_fcf = df_hist["FCF"].iloc[-1]
        fcf_growth_vec = []
        if expected_growth == "slope":
            s = df_hist["FCF"].dropna()
            slope, intercept = calculate_regression_slope(s)
            projected_fcf = [
                float(slope * (len(s)-1+i) + intercept) if slope is not None and intercept is not None else np.nan
                for i in range(1, projection_years+1)
            ]
            # EBITDA for reporting
            s_eb = df_hist["EBITDA"].dropna()
            slope_eb, intercept_eb = calculate_regression_slope(s_eb)
            projected_ebitda = [
                float(slope_eb * (len(s_eb)-1+i) + intercept_eb) if slope_eb is not None and intercept_eb is not None else np.nan
                for i in range(1, projection_years+1)
            ]
            # Estimate implied fcf growth
            fcf_growth_vec = []
            fcf_v = last_fcf
            for i in range(1, projection_years+1):
                next_fcf = float(slope * (len(s)-1+i) + intercept) if slope is not None and intercept is not None else np.nan
                g = (next_fcf/fcf_v - 1) if fcf_v and next_fcf and not np.isnan(fcf_v) and not np.isnan(next_fcf) and fcf_v != 0 else np.nan
                fcf_growth_vec.append(g)
                fcf_v = next_fcf
        else:
            if expected_growth is not None:
                growth_rate = expected_growth
            else:
                try:
                    f_first = df_hist["FCF"].iloc[0]
                    f_last = df_hist["FCF"].iloc[-1]
                    periods_fc = len(df_hist["FCF"]) - 1
                    growth_rate = (f_last / f_first) ** (1 / periods_fc) - 1
                except Exception:
                    growth_rate = np.nan
            fcf_growth_vec = _get_growth_vector(growth_rate, terminal_growth, projection_years, converge_growth)
            projected_fcf = []
            v = last_fcf
            for g in fcf_growth_vec:
                v = v * (1 + g) if v is not None and not np.isnan(v) and g is not None and not np.isnan(g) else np.nan
                projected_fcf.append(v)
            try:
                e_first = df_hist["EBITDA"].iloc[0]
                e_last = df_hist["EBITDA"].iloc[-1]
                periods_eb = len(df_hist["EBITDA"]) - 1
                eb_growth = (e_last / e_first) ** (1 / periods_eb) - 1
                eb_growth_vec = _get_growth_vector(eb_growth, terminal_growth, projection_years, converge_growth)
                projected_ebitda = []
                v = e_last
                for g in eb_growth_vec:
                    v = v * (1 + g) if v is not None and not np.isnan(v) and g is not None and not np.isnan(g) else np.nan
                    projected_ebitda.append(v)
            except Exception:
                projected_ebitda = [np.nan for _ in range(projection_years)]

    # ========== DCF Calculation ==========
    last_proj_fcf = projected_fcf[-1]
    terminal_val = None
    try:
        terminal_val = compute_terminal_value(last_proj_fcf, terminal_growth, wacc)
    except Exception:
        terminal_val = np.nan
    cash_flows_with_term = projected_fcf.copy()
    if len(cash_flows_with_term) > 0:
        cash_flows_with_term[-1] += terminal_val if terminal_val is not None else 0
    else:
        cash_flows_with_term = []

    pv_vals = discount_cash_flows(cash_flows_with_term, wacc) if cash_flows_with_term else []

    # ========== Discount Factor Calculation (for projected part only) ==========
    discount_factors = []
    for i in range(1, projection_years + 1):
        discount_factors.append(1 / ((1 + wacc) ** i))

    # ========== Historical Part ==========
    hist_years = [idx.year for idx in df_hist.index]
    hist_df = pd.DataFrame({
        "Year": hist_years,
        "Type": ["historical"] * len(df_hist),
        "Revenue": df_hist["Revenue"].values,
        "EBITDA": df_hist["EBITDA"].values,
        "FCF": df_hist["FCF"].values,
        "PV(FCF+TV)": [np.nan]*len(df_hist),
        "Discount Factor": [np.nan]*len(df_hist)
    })

    # ========== Projected Part ==========
    last_year = df_hist.index[-1].year
    years = [last_year + i for i in range(1, projection_years+1)]
    dcf_df = pd.DataFrame({
        "Year": years,
        "Type": ["projected"] * projection_years,
        "Revenue": projected_revenue,
        "EBITDA": projected_ebitda,
        "FCF": projected_fcf,
        "PV(FCF+TV)": pv_vals,
        "Discount Factor": discount_factors
    })

    # ========== Combine ==========
    full_df = pd.concat([hist_df, dcf_df], axis=0, ignore_index=True)

    # ========== Valuation Summary ==========
    sum_pv = float(np.nansum(pv_vals))
    npv = sum_pv
    # --- Compute average projected growth rate (for FCF, main DCF input) ---
    if use_ebitda_base:
        avg_growth = np.nanmean(eb_growth_vec) if eb_growth_vec else np.nan
    else:
        avg_growth = np.nanmean(fcf_growth_vec) if fcf_growth_vec else np.nan

    valuation_summary = {
        "sum_pv": sum_pv,
        "terminal_value": terminal_val,
        "npv": npv,
        "per_share_value": np.nan,  # to be filled in main
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
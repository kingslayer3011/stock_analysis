import json
import pandas as pd
import numpy as np
import argparse
import os

from data_fetcher import get_historical_data

def project_values(ticker: str, config: dict) -> pd.DataFrame:
    """
    Fetches historical data for the given ticker and projects future cash flows (and EBITDA if configured).

    Args:
        ticker: Stock ticker symbol.
        config: Dictionary containing DCF configuration.

    Returns:
        DataFrame with columns: Year, Projected_FCF, and optionally Projected_EBITDA.
    """
    projection_years = config.get("projection_years")
    if projection_years is None:
        raise ValueError("'projection_years' must be specified in the configuration.")
    use_ebitda_base = config.get("use_ebitda_base", False)
    fcf_to_ebitda_ratio = config.get("fcf_ratio") if use_ebitda_base else None
    
    #! import historical_data.csv instead of creating it
    df_hist, _ = get_historical_data(ticker)
    #! make sure it is sort from first to last year
    df_hist = df_hist.sort_index()
    last_year = df_hist.index[-1].year

    #! Create separate fcf projection function for each method use_ebitda_base
    if use_ebitda_base:
        if fcf_to_ebitda_ratio is None:
            raise ValueError("'fcf_ratio' must be provided when 'use_ebitda_base' is True.")
        ebitda_series = df_hist["EBITDA"].dropna()
        if len(ebitda_series) < 2:
            raise ValueError("Not enough historical EBITDA data to compute CAGR.")
        e_first, e_last = ebitda_series.iloc[0], ebitda_series.iloc[-1]
        periods = len(ebitda_series) - 1
        growth_rate = (e_last / e_first) ** (1 / periods) - 1 if e_first > 0 else 0.0
        last_ebitda = e_last
    else:
        fcf_series = df_hist["FCF"].dropna()
        if len(fcf_series) < 2:
            raise ValueError("Not enough historical FCF data to compute CAGR.")
        f_first, f_last = fcf_series.iloc[0], fcf_series.iloc[-1]
        periods = len(fcf_series) - 1
        growth_rate = (f_last / f_first) ** (1 / periods) - 1 if f_first > 0 else 0.0
        last_fcf = f_last
        last_ebitda = df_hist["EBITDA"].iloc[-1] if "EBITDA" in df_hist.columns else None

    years = []
    projected_ebitda = [] if use_ebitda_base else None
    projected_fcf = []

    for i in range(1, projection_years + 1):
        year = last_year + i
        years.append(year)
        if use_ebitda_base:
            ebitda_i = last_ebitda * ((1 + growth_rate) ** i)
            fcf_i = ebitda_i * fcf_to_ebitda_ratio
            projected_ebitda.append(ebitda_i)
        else:
            fcf_i = last_fcf * ((1 + growth_rate) ** i)
        projected_fcf.append(fcf_i)

    data = {"Year": years, "Projected_FCF": projected_fcf}
    if use_ebitda_base:
        data["Projected_EBITDA"] = projected_ebitda

    df_proj = pd.DataFrame(data)
    df_proj = df_proj.sort_values(by="Year")
    return df_proj

def main():
    parser = argparse.ArgumentParser(description="Project future cash flows and output to CSV.")
    parser.add_argument("ticker", help="Ticker symbol for the company to project.")
    parser.add_argument("--config", default="dcf_config.json", help="Path to JSON config file.")
    parser.add_argument("--output", default="projected_values.csv", help="Output CSV filename.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file '{args.config}' not found.")
    with open(args.config, "r") as f:
        config = json.load(f)
    
    df_proj = project_values(args.ticker, config)
    df_proj.to_csv(args.output, index=False)
    print(f"Projected values saved to '{args.output}'")

if __name__ == "__main__":
    main()

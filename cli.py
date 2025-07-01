import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from data_fetcher import *
from projection import *
from dcf import *
from reporter import *
from sensitivity import *

import yfinance as yf
import matplotlib.pyplot as plt

def change_units(df, columns, to="ones"):
    scale = 1
    suffix = ""
    if to == "millions":
        scale = 1e-6
        suffix = " (M)"
    elif to == "billions":
        scale = 1e-9
        suffix = " (B)"
    elif to == "thousands":
        scale = 1e-3
        suffix = " (K)"
    elif to == "ones" or to is None:
        scale = 1
        suffix = ""
    else:
        raise ValueError("Unknown unit: %s" % to)

    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col] * scale
            if suffix and not df.columns[df.columns.get_loc(col)].endswith(suffix):
                df.rename(columns={col: col + suffix}, inplace=True)
    return df

def custom_round(val):
    if isinstance(val, pd.Series):
        return val.apply(custom_round)
    elif isinstance(val, pd.DataFrame):
        return val.applymap(custom_round)
    try:
        if pd.isnull(val):
            return val
        if abs(val) > 100:
            return round(val)
        else:
            return round(val, 2)
    except Exception:
        return val

def round_numeric_cols(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = custom_round(df[col])
    return df

def save_dcf_results(dcf_df, dcf_valuation_info, output_folder, suffix=""):
    dcf_csv_path = os.path.join(output_folder, f"dcf_results{suffix}.csv")
    dcf_df.to_csv(dcf_csv_path, index=False)
    summary_path = os.path.join(output_folder, f"dcf_summary{suffix}.json")
    with open(summary_path, 'w') as f:
        json.dump(dcf_valuation_info, f, indent=2)

def save_wacc_estimate(wacc_estimate_results, output_folder, suffix=""):
    if wacc_estimate_results is None:
        return
    wacc_csv_path = os.path.join(output_folder, f"wacc_estimate{suffix}.csv")
    wacc_json_path = os.path.join(output_folder, f"wacc_estimate{suffix}.json")
    wacc_df = pd.DataFrame([wacc_estimate_results])
    wacc_df.to_csv(wacc_csv_path, index=False)
    with open(wacc_json_path, 'w') as f:
        json.dump(wacc_estimate_results, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the DCF and related analyses.")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--ticker", help="Ticker symbol (overrides config).")
    parser.add_argument("--wacc", type=float, help="WACC override.")
    parser.add_argument("--terminal_growth", type=float, help="Terminal growth override.")
    parser.add_argument("--projection_years", type=int, help="Projection years override.")
    parser.add_argument("--use_ebitda_base", action="store_true", help="Use EBITDA base for projection.")
    parser.add_argument("--fcf_ratio", type=float, help="FCF-to-EBITDA ratio override.")
    parser.add_argument("--expected_growth", type=float, help="Expected FCF or EBITDA growth rate override.")
    parser.add_argument("--units", choices=["ones", "thousands", "millions", "billions"],
                        help="Units for output values (ones, thousands, millions, billions).")
    parser.add_argument("--converge_growth", action="store_true",
                        help="Enable converging projected growth rates toward the terminal growth rate.")
    return parser.parse_args()

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"ERROR: Config file '{path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    with open(path, 'r') as f:
        return json.load(f)


def main():
    args = parse_args()

    # Load base configuration
    with open("dcf_config.json", "r") as f:
        config = json.load(f)

    cli_units = args.units
    config_units = config.get("units", None)
    units = cli_units if cli_units is not None else (config_units if config_units is not None else "ones")

    final_params = config.copy()
    for field in [
        "ticker", "wacc", "terminal_growth", "projection_years",
        "use_ebitda_base", "fcf_ratio", "expected_growth"
    ]:
        val = getattr(args, field)
        if val is not None:
            final_params[field] = val

    ticker = final_params.get("ticker")
    if not ticker:
        print("ERROR: 'ticker' must be specified either in the config or via --ticker", file=sys.stderr)
        sys.exit(1)

    # Fetch company name via yfinance
    try:
        ticker_obj = yf.Ticker(ticker)
        company_name = ticker_obj.info.get('longName') or ticker_obj.info.get('shortName') or ticker
        print(f"Analyzing {company_name} ({ticker})...")
    except Exception as e:
        company_name = ticker
        print(f"Warning: Could not fetch company name for {ticker}: {e}")

    wacc_config = final_params["wacc_min"]
    terminal_growth = final_params["terminal_growth"]
    projection_years = final_params["projection_years"]
    plot_projections = final_params.get("plot_projections", None)
    cost_of_debt = final_params.get("cost_of_debt", 0.0253)
    cost_of_equity = final_params.get("cost_of_equity", 0.08)

    output_folder = f"dcf_results"
    os.makedirs(output_folder, exist_ok=True)

    output_folder = os.path.join(output_folder, f"{ticker}_results")
    os.makedirs(output_folder, exist_ok=True)

    # Estimate WACC once before running DCF
    wacc_estimate_results = estimate_wacc(
        ticker,
        cost_of_debt=cost_of_debt,
        cost_of_equity=cost_of_equity
    )
    # Use the higher of user/config WACC and estimated WACC for ALL DCFs
    wacc_used = max(wacc_config, wacc_estimate_results["WACC"])
    print(f"Using WACC={wacc_used:.4f} (max of input={wacc_config:.4f} and estimated={wacc_estimate_results['WACC']:.4f})")
    save_wacc_estimate(wacc_estimate_results, output_folder)

    df_hist, val_info = get_historical_data(ticker)
    df_hist = df_hist[df_hist['Revenue'].notna()]

    # Save historical
    df_hist.to_csv(os.path.join(output_folder, "historical_data.csv"), index=True)

    # Run projections
    df_proj = project_all(
        df_hist,
        projection_years,
        terminal_growth=terminal_growth,
        converge_growth=False,
        man_inp_growth=config.get("man_inp_growth", None)
    )

    df_proj_converge = project_all(
        df_hist,
        projection_years,
        terminal_growth=terminal_growth,
        converge_growth=True,
        man_inp_growth=config.get("man_inp_growth", None)
    )

    if plot_projections:
        for df_p, suffix in [(df_proj, ""), (df_proj_converge, "_converge")]:
            plot_folder = os.path.join(output_folder, f"proj_plots{suffix}")
            os.makedirs(plot_folder, exist_ok=True)
            plot_all_variables(ticker, df_hist, df_p, plot_folder)
            plot_projected_fcf_growth_rates(ticker, df_p, plot_folder)

    proj_csv_path = os.path.join(output_folder, "projected_values.csv")
    df_proj_out = change_units(df_proj, [c for c in ["Revenue", "EBITDA", "FCF"] if c in df_proj.columns], to=units)
    round_numeric_cols(df_proj_out).to_csv(proj_csv_path, index=False)

    # === Run DCF for multiple forecast horizons ===
    horizons = [5, 10]
    dcf_estimates_list = []
    unit_cols = ["Revenue", "EBITDA", "FCF", "PV(FCF+TV)"]
    number_shares = val_info.get("number_shares")
    share_price = val_info.get("share_price")

    proj_sets = [
        ("standard", df_proj, ""),
        ("converged", df_proj_converge, "_converge"),
    ]

    fcf_cols = ["FCFE"]

    projection_methods = [
        ("cagr", "_cagr", "historical_cagr"),
        ("slope", "_slope", "regression_slope"),
        ("man_inp_growth", "_manual_input", "manual_input_growth"),
    ]

    for horizon in horizons:
        for proj_name, proj_df, proj_suffix in proj_sets:
            for fcf_col in fcf_cols:
                for p_method, method_suffix, method_name in projection_methods:
                    suffix = f"{method_suffix}_{fcf_col}{proj_suffix}_h{horizon}"
                    label = method_name.replace("_", " ").title()
                 
                    dcf_df, dcf_info = run_dcf(
                        df_hist,
                        proj_df,
                        terminal_growth,
                        wacc_used,
                        forecast_horizon=horizon,
                        FCF_col=fcf_col,
                        p_method=p_method,
                        number_shares=number_shares,
                        share_price=share_price
                    )

                    dcf_out = change_units(dcf_df, [c for c in unit_cols if c in dcf_df.columns], to=units)
                    save_dcf_results(dcf_out, dcf_info, output_folder, suffix=suffix)
            
                    dcf_estimates_list.append({
                        "ticker": ticker,
                        "name": company_name,
                        "horizon": horizon,
                        "proj_type": proj_name,
                        "fcf_source": fcf_col,
                        "p_method": p_method,
                        "method_label": label,
                        "used_wacc": wacc_used,
                        **dcf_info
                    })

    dcf_estimates = pd.DataFrame(dcf_estimates_list)
    to_conv = [c for c in ["sum_pv", "terminal_value", "npv"] if c in dcf_estimates]
    if to_conv:
        dcf_estimates = change_units(dcf_estimates, to_conv, to=units)
    dcf_estimates = round_numeric_cols(dcf_estimates)
    dcf_estimates.to_csv(os.path.join(output_folder, "dcf_estimates.csv"), index=False)

if __name__ == "__main__":
    main()
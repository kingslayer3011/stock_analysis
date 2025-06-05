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

def save_dcf_results(dcf_df, dcf_valuation_info, output_folder):
    dcf_csv_path = os.path.join(output_folder, "dcf_results.csv")
    dcf_df.to_csv(dcf_csv_path, index=False)
    summary_path = os.path.join(output_folder, "dcf_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(dcf_valuation_info, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the DCF and related analyses.")
    parser.add_argument("--config", help="Path to JSON config file.")
    # CLI overrides
    parser.add_argument("--ticker", help="Ticker symbol (overrides config).")
    parser.add_argument("--wacc", type=float, help="WACC override.")
    parser.add_argument("--terminal_growth", type=float, help="Terminal growth override.")
    parser.add_argument("--projection_years", type=int, help="Projection years override.")
    parser.add_argument("--use_ebitda_base", action="store_true", help="Use EBITDA base for projection.")
    parser.add_argument("--fcf_ratio", type=float, help="FCF-to-EBITDA ratio override.")
    return parser.parse_args()

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"ERROR: Config file '{path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    with open(path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()

    # 1) Ensure config was passed
    if args.config is None:
        print("ERROR: Must specify --config <path to JSON config>", file=sys.stderr)
        sys.exit(1)

    # 2) Load config JSON
    config = load_config(args.config)

    # 3) Overlay CLI arguments onto config (if provided)
    final_params = config.copy()
    for field in ["ticker", "wacc", "terminal_growth", "projection_years", "use_ebitda_base", "fcf_ratio"]:
        val = getattr(args, field)
        if val is not None:
            final_params[field] = val

    ticker = final_params.get("ticker")
    if ticker is None:
        print("ERROR: 'ticker' must be specified either in the config or via --ticker", file=sys.stderr)
        sys.exit(1)

    wacc = final_params["wacc"]
    terminal_growth = final_params["terminal_growth"]
    projection_years = final_params["projection_years"]
    use_ebitda_base = final_params.get("use_ebitda_base", False)
    fcf_ratio = final_params.get("fcf_ratio", None)

    # 4) Create output folder "<TICKER>_results"
    output_folder = f"{ticker}_results"
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create output folder '{output_folder}': {e}", file=sys.stderr)
        sys.exit(1)

    # 5) Fetch historical data
    print(f"Fetching historical data for '{ticker}'…")
    try:
        df_hist, val_info = get_historical_data(ticker)
        df_hist = df_hist[df_hist['Revenue'].notna()]
    except Exception as e:
        print(f"ERROR: Could not retrieve historical data for '{ticker}': {e}", file=sys.stderr)
        sys.exit(1)

    # 6) Save historical data to CSV
    hist_csv_path = os.path.join(output_folder, "historical_data.csv")
    try:
        #df_hist[df_hist['Revenue'].notna()].to_csv(hist_csv_path)
        df_hist.to_csv(hist_csv_path, index=True)
        print(f"Saved historical data to '{hist_csv_path}'")
    except Exception as e:
        print(f"WARNING: Could not save historical_data CSV: {e}", file=sys.stderr)


    # 8) Project future values and save CSV
    print("Projecting future values…")
    df_proj = project_all(df_hist, projection_years)
    print("DEBUG: project_all output shape", df_proj.shape)
    print("DEBUG: project_all output head:\n", df_proj.head())
    proj_csv_path = os.path.join(output_folder, "projected_values.csv")
    df_proj.to_csv(proj_csv_path, index=False)
    print(f"Saved projected values to '{proj_csv_path}'")


    # 9) Run the main DCF calculation
    print("Running DCF calculation…")
    try:
        dcf_df, dcf_valuation_info = run_dcf(
            df_hist,
            projection_years,
            terminal_growth,
            wacc,
            use_ebitda_base,
            fcf_ratio
        )
    except Exception as e:
        print(f"ERROR in DCF calculation: {e}", file=sys.stderr)
        sys.exit(1)

    # 10) Save DCF results
    try:
        save_dcf_results(dcf_df, dcf_valuation_info, output_folder)
    except Exception as e:
        print(f"WARNING: Could not save DCF results: {e}", file=sys.stderr)

    # 11) Run sensitivity analyses
    sens1_df = None
    if use_ebitda_base:
        print("Running sensitivity #1 (Growth vs. FCF/EBITDA)…")
        try:
            sens1_df = perform_sensitivity_analysis(
                last_fcf=val_info.get("last_fcf"),
                wacc=wacc,
                terminal_growth=terminal_growth,
                projection_years=projection_years,
                fcf_to_ebitda_ratio=fcf_ratio,
                market_cap=val_info.get("market_cap", 0),
            )
            sens1_csv = os.path.join(output_folder, "sensitivity_growth_vs_ratio.csv")
            sens1_df.to_csv(sens1_csv)
            print(f"Saved sensitivity #1 to '{sens1_csv}'")
        except Exception as e:
            print(f"WARNING: Could not save sensitivity #1: {e}", file=sys.stderr)

    print("Running sensitivity #2 (WACC vs. Terminal Growth)…")
    sens2_df = None
    try:
        sens2_df = perform_additional_sensitivity(
            df_hist=df_hist,
            market_cap=val_info.get("market_cap", 0),
            wacc=wacc,
            terminal_growth=terminal_growth,
            projection_years=projection_years,
            fcf_to_ebitda_ratio=fcf_ratio,
        )
        sens2_csv = os.path.join(output_folder, "sensitivity_wacc_vs_growth.csv")
        sens2_df.to_csv(sens2_csv)
        print(f"Saved sensitivity #2 to '{sens2_csv}'")
    except Exception as e:
        print(f"WARNING: Could not save sensitivity #2: {e}", file=sys.stderr)

    # 12) Generate PDF report
    print("Generating PDF report…")
    try:
        generate_pdf_report(
            ticker,
            df_hist,
            dcf_df,
            dcf_valuation_info,
            sens1_df,
            sens2_df,
            output_folder
        )
    except Exception as e:
        print(f"ERROR generating PDF report: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"All outputs have been saved into the folder: '{output_folder}'")

if __name__ == "__main__":
    main()


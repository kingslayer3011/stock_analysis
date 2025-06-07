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

def change_units(df, columns, to="ones"):
    """
    Change the units of specified columns in a dataframe.
    Args:
        df: DataFrame containing columns to convert
        columns: list of column names to convert
        to: target units, "millions" or "billions" or "thousands" or "ones"
    Returns:
        DataFrame with converted columns (values and column names updated)
    """
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
            # Update column name with suffix if not already present
            if suffix and not df.columns[df.columns.get_loc(col)].endswith(suffix):
                df.rename(columns={col: col + suffix}, inplace=True)
    return df

def custom_round(val):
    """
    Round values > 100 to 0 decimals, <= 100 to 2 decimals.
    Handles pd.Series and pd.DataFrame as well as scalars.
    """
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

    if args.config is None:
        print("ERROR: Must specify --config <path to JSON config>", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    # Determine units: CLI overrides config, config overrides default
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

    # NEW: growth convergence option
    converge_growth = args.converge_growth or final_params.get("converge_growth", False)

    ticker = final_params.get("ticker")
    if ticker is None:
        print("ERROR: 'ticker' must be specified either in the config or via --ticker", file=sys.stderr)
        sys.exit(1)

    wacc = final_params["wacc"]
    terminal_growth = final_params["terminal_growth"]
    projection_years = final_params["projection_years"]
    use_ebitda_base = final_params.get("use_ebitda_base", False)
    fcf_ratio = final_params.get("fcf_ratio", None)
    expected_growth = final_params.get("expected_growth", None)

    output_folder = f"{ticker}_results"
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Could not create output folder '{output_folder}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching historical data for '{ticker}'…")
    try:
        df_hist, val_info = get_historical_data(ticker)
        df_hist = df_hist[df_hist['Revenue'].notna()]
    except Exception as e:
        print(f"ERROR: Could not retrieve historical data for '{ticker}': {e}", file=sys.stderr)
        sys.exit(1)

    hist_csv_path = os.path.join(output_folder, "historical_data.csv")
    unit_cols_hist = [col for col in ["Revenue", "EBITDA", "FCF"] if col in df_hist.columns]
    df_hist_out = change_units(df_hist, unit_cols_hist, to=units)
    df_hist_out = round_numeric_cols(df_hist_out)
    try:
        df_hist_out.to_csv(hist_csv_path, index=True)
        print(f"Saved historical data to '{hist_csv_path}'")
    except Exception as e:
        print(f"WARNING: Could not save historical_data CSV: {e}", file=sys.stderr)

    print("Projecting future values…")
    df_proj = project_all(
        df_hist,
        projection_years,
        terminal_growth=terminal_growth,
        converge_growth=converge_growth
    )
    proj_csv_path = os.path.join(output_folder, "projected_values.csv")
    df_proj_out = change_units(df_proj, [col for col in ["Revenue", "EBITDA", "FCF"] if col in df_proj.columns], to=units)
    df_proj_out = round_numeric_cols(df_proj_out)
    df_proj_out.to_csv(proj_csv_path, index=False)
    print(f"Saved projected values to '{proj_csv_path}'")

    # --- NEW: Plot projected FCF growth rates ---
    from projection import plot_projected_fcf_growth_rates
    plot_projected_fcf_growth_rates(ticker, df_proj, output_folder)

    # === Run all DCF methods and collect results ===
    dcf_estimates_list = []
    number_shares = val_info.get("number_shares")
    if number_shares is None or number_shares == 0 or pd.isna(number_shares):
        print("WARNING: Number of shares is missing or invalid. Per-share value will be empty.")
        number_shares = None

    # Columns to convert for DCF outputs (historical and projected)
    unit_cols = ["Revenue", "EBITDA", "FCF", "PV(FCF+TV)"]

    # 1. DCF (historical CAGR)
    print("Running DCF calculation (historical CAGR)…")
    try:
        dcf_df, dcf_valuation_info = run_dcf(
            df_hist,
            projection_years,
            terminal_growth,
            wacc,
            use_ebitda_base,
            fcf_ratio,
            expected_growth=None,
            converge_growth=converge_growth
        )
        dcf_df_out = change_units(dcf_df, [col for col in unit_cols if col in dcf_df.columns], to=units)
        dcf_df_out = round_numeric_cols(dcf_df_out)
        if number_shares:
            dcf_valuation_info["per_share_value"] = dcf_valuation_info["npv"] / number_shares
        else:
            dcf_valuation_info["per_share_value"] = np.nan
        dcf_estimates_list.append({
            "method": "historical_cagr",
            **dcf_valuation_info
        })
        save_dcf_results(dcf_df_out, dcf_valuation_info, output_folder, suffix="")
    except Exception as e:
        print(f"ERROR in DCF calculation: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. DCF (regression slope)
    print("Running DCF calculation (regression slope)…")
    try:
        dcf_df_slope, dcf_valuation_info_slope = run_dcf(
            df_hist,
            projection_years,
            terminal_growth,
            wacc,
            use_ebitda_base,
            fcf_ratio,
            expected_growth="slope",
            converge_growth=converge_growth
        )
        dcf_df_slope_out = change_units(dcf_df_slope, [col for col in unit_cols if col in dcf_df_slope.columns], to=units)
        dcf_df_slope_out = round_numeric_cols(dcf_df_slope_out)
        if number_shares:
            dcf_valuation_info_slope["per_share_value"] = dcf_valuation_info_slope["npv"] / number_shares
        else:
            dcf_valuation_info_slope["per_share_value"] = np.nan
        dcf_estimates_list.append({
            "method": "regression_slope",
            **dcf_valuation_info_slope
        })
        save_dcf_results(dcf_df_slope_out, dcf_valuation_info_slope, output_folder, suffix="_slope")
        print(f"Saved DCF (regression slope) to '{os.path.join(output_folder, 'dcf_results_slope.csv')}'")
    except Exception as e:
        print(f"WARNING: Could not run/save DCF (regression slope): {e}", file=sys.stderr)

    # 3. DCF (expected growth if specified and not "slope")
    if expected_growth is not None and expected_growth != "slope":
        print(f"Running DCF calculation (expected growth rate = {expected_growth:.2%})…")
        try:
            dcf_df_exp, dcf_valuation_info_exp = run_dcf(
                df_hist,
                projection_years,
                terminal_growth,
                wacc,
                use_ebitda_base,
                fcf_ratio,
                expected_growth=expected_growth,
                converge_growth=converge_growth
            )
            dcf_df_exp_out = change_units(dcf_df_exp, [col for col in unit_cols if col in dcf_df_exp.columns], to=units)
            dcf_df_exp_out = round_numeric_cols(dcf_df_exp_out)
            if number_shares:
                dcf_valuation_info_exp["per_share_value"] = dcf_valuation_info_exp["npv"] / number_shares
            else:
                dcf_valuation_info_exp["per_share_value"] = np.nan
            dcf_estimates_list.append({
                "method": f"expected_growth_{expected_growth:.4f}",
                **dcf_valuation_info_exp
            })
            save_dcf_results(dcf_df_exp_out, dcf_valuation_info_exp, output_folder, suffix="_expected")
            print(f"Saved DCF (expected growth) to '{os.path.join(output_folder, 'dcf_results_expected.csv')}'")
        except Exception as e:
            print(f"WARNING: Could not run/save DCF (expected growth): {e}", file=sys.stderr)

    # === Save all DCF estimates as a DataFrame ===
    dcf_estimates = pd.DataFrame(dcf_estimates_list)
    # Convert units for summary columns except per_share_value
    cols_to_convert = [col for col in ["sum_pv", "terminal_value", "npv"] if col in dcf_estimates.columns]
    dcf_estimates_out = dcf_estimates.copy()
    if cols_to_convert:
        dcf_estimates_out = change_units(dcf_estimates_out, cols_to_convert, to=units)
    dcf_estimates_out = round_numeric_cols(dcf_estimates_out)
    dcf_estimates_path = os.path.join(output_folder, "dcf_estimates.csv")
    dcf_estimates_out.to_csv(dcf_estimates_path, index=False)
    print(f"Saved summary of all DCF estimates to '{dcf_estimates_path}'")

    # === Continue with sensitivity and reporting as before ===
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
            sens1_df_out = change_units(sens1_df, [col for col in unit_cols if col in sens1_df.columns], to=units)
            sens1_df_out = round_numeric_cols(sens1_df_out)
            sens1_df_out.to_csv(sens1_csv)
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
        sens2_df_out = change_units(sens2_df, [col for col in unit_cols if col in sens2_df.columns], to=units)
        sens2_df_out = round_numeric_cols(sens2_df_out)
        sens2_df_out.to_csv(sens2_csv)
        print(f"Saved sensitivity #2 to '{sens2_csv}'")
    except Exception as e:
        print(f"WARNING: Could not save sensitivity #2: {e}", file=sys.stderr)

    print("Generating PDF report…")
    try:
        # For reporting, convert historical to units as well (for consistency in PDF)
        df_hist_report = change_units(df_hist, unit_cols_hist, to=units)
        df_hist_report = round_numeric_cols(df_hist_report)
        generate_pdf_report(
            ticker,
            df_hist_report,
            dcf_df_out,
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
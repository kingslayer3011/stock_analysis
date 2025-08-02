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
from data_fetcher import plot_fcff_components

import yfinance as yf
import matplotlib.pyplot as plt
import dcf

"""import debugpy
debugpy.listen(("localhost", 5682))  # You can use any open port, e.g., 5678
print("Waiting for debugger attach...")
debugpy.wait_for_client()"""

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
            df[col] = (df[col] * scale).round().astype("Int64")  # Nullable int in case of NaNs
            if suffix and not df.columns[df.columns.get_loc(col)].endswith(suffix):
                df.rename(columns={col: col + suffix}, inplace=True)
    return df

def reduce_decimals(df, columns, decimals=4):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].round(decimals)
    return df



def save_dcf_results(dcf_df, output_folder, suffix=""):
    # Ensure results are saved in a subfolder called 'dcf_results'
    dcf_results_folder = os.path.join(output_folder, "dcf_results")
    os.makedirs(dcf_results_folder, exist_ok=True)
    dcf_csv_path = os.path.join(dcf_results_folder, f"dcf_results{suffix}.csv")
    dcf_df.to_csv(dcf_csv_path, index=False)


def save_wacc_estimate(wacc_estimate_results, output_folder, suffix=""):
    if wacc_estimate_results is None:
        return
    wacc_csv_path = os.path.join(output_folder, f"wacc_estimate{suffix}.csv")
    wacc_json_path = os.path.join(output_folder, f"wacc_estimate{suffix}.json")
    wacc_df = pd.DataFrame([wacc_estimate_results])
    #wacc_df.to_csv(wacc_csv_path, index=False)
    with open(wacc_json_path, 'w') as f:
        json.dump(wacc_estimate_results, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the DCF and related analyses.")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--ticker", help="Ticker symbol (overrides config).")
    parser.add_argument("--wacc", type=float, help="WACC override.")
    parser.add_argument("--terminal_growth", type=float, help="Terminal growth override.")
    parser.add_argument("--expected_growth", type=float, help="Expected FCF or EBITDA growth rate override.")
    parser.add_argument("--target_operating_margin", type=float, default=None, help="Target operating margin override.")
    parser.add_argument("--target_effective_tax_rate", type=float, default=None, help="Target effective tax rate override.")
    parser.add_argument("--target_sales_to_capital", type=float, default=None, help="Target sales to capital override.")
    parser.add_argument("--initial_operating_margin", type=float, default=None, help="Initial operating margin override.")
    parser.add_argument("--initial_effective_tax_rate", type=float, default=None, help="Initial effective tax rate override.")
    parser.add_argument("--initial_sales_to_capital", type=float, default=None, help="Initial sales to capital override.")
    parser.add_argument("--units", choices=["ones", "thousands", "millions", "billions"],
                        help="Units for output values (ones, thousands, millions, billions).")
    parser.add_argument("--converge_growth", action="store_true",
                        help="Enable converging projected growth rates toward the terminal growth rate.")
    parser.add_argument('--risk_free_rate', type=float, default=0.045, help='Risk-free rate for CAPM')
    parser.add_argument('--equity_premium', type=float, default=0.0547, help='Market premium for CAPM')
    parser.add_argument('--plot_projections', action="store_true", help="Enable plotting of projections")  # <-- Add this line
    parser.add_argument('--plot_regression_plots', action="store_true", help="Enable saving regression plots in regression_plots folder")
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

    # --- Load country equity premium mapping ---
    with open("country_eq_prem.json", "r") as f:
        country_eq_prem = json.load(f)

    cli_units = args.units
    config_units = config.get("units", None)
    units = cli_units if cli_units is not None else (config_units if config_units is not None else "ones")

    final_params = config.copy()
    for field in [
        "ticker", "wacc", "terminal_growth",
        "expected_growth"
    ]:
        val = getattr(args, field)
        if val is not None:
            final_params[field] = val

    ticker = final_params.get("ticker")
    if not ticker:
        print("ERROR: 'ticker' must be specified either in the config or via --ticker", file=sys.stderr)
        sys.exit(1)

    # --- Load config targets ---
    target_operating_margin = final_params.get("target_operating_margin", None)
    target_effective_tax_rate = final_params.get("target_effective_tax_rate", None)
    target_sales_to_capital = final_params.get("target_sales_to_capital", None)
    initial_operating_margin = final_params.get("initial_operating_margin", None)
    initial_effective_tax_rate = final_params.get("initial_effective_tax_rate", None)
    initial_sales_to_capital = final_params.get("initial_sales_to_capital", None)

    # Fetch company name and country via yfinance
    try:
        ticker_obj = yf.Ticker(ticker)
        company_name = ticker_obj.info.get('longName') or ticker_obj.info.get('shortName') or ticker
        country = ticker_obj.info.get('country', None)
        print(f"Analyzing {company_name} ({ticker})...")
    except Exception as e:
        company_name = ticker
        country = None
        print(f"Warning: Could not fetch company name or country for {ticker}: {e}")

    # --- Set equity premium from country_eq_prem.json if possible ---
    equity_premium = None
    if country and country in country_eq_prem:
        equity_premium = country_eq_prem[country]
        print(f"Using equity premium for {country}: {equity_premium:.4f}")
    else:
        # fallback to config or CLI/default
        equity_premium = final_params.get("equity_premium", args.equity_premium)
        if country:
            print(f"Warning: No equity premium found for country '{country}', using config/CLI/default: {equity_premium:.4f}")
        else:
            print(f"Warning: Country not found for ticker '{ticker}', using config/CLI/default equity premium: {equity_premium:.4f}")

    wacc_config = final_params["wacc_min"]
    terminal_growth = final_params["terminal_growth"]
    plot_projections = final_params.get("plot_projections", None)


    output_folder = f"dcf_results"
    os.makedirs(output_folder, exist_ok=True)

    output_folder = os.path.join(output_folder, f"{ticker}_results")
    os.makedirs(output_folder, exist_ok=True)

    # --- Save config/arguments to output folder ---
    config_to_save = {
        "config": config
    }
    config_path = os.path.join(output_folder, f"{ticker}_config.json")
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)

    df_hist, val_info = get_historical_data(ticker)
    df_hist = df_hist[df_hist['Revenue'].notna()]
    
    cost_of_debt_plot_path = os.path.join(output_folder, "cost_of_debt.png")
    cost_of_debt = estimate_cost_of_debt(df_hist, min_pre_tax=0.04, plot=True, plot_filename=cost_of_debt_plot_path)
    print(f"Estimated cost of debt (after-tax): {cost_of_debt:.4f}")
    # Estimate WACC once before running DCF, using df_hist
    beta_plot_path = os.path.join(output_folder, f"{ticker}_beta_regression.png")
    wacc_estimate_results = estimate_wacc(
        df_hist,
        cost_of_debt=cost_of_debt,
        ticker=ticker,
        risk_free_rate=args.risk_free_rate,
        equity_premium=equity_premium,
        beta_plot_path=beta_plot_path
    )
    # Use the higher of user/config WACC and estimated WACC for ALL DCFs
    wacc_used = max(wacc_config, wacc_estimate_results["WACC"])
    print(f"Using WACC={wacc_used:.4f} (max of input={wacc_config:.4f} and estimated={wacc_estimate_results['WACC']:.4f})")
    save_wacc_estimate(wacc_estimate_results, output_folder)

    # Save historical
    df_hist.to_csv(os.path.join(output_folder, "historical_data.csv"), index=True)

    regression_plots_dir = os.path.join(output_folder, "regression_plots")
    # Run projections
    horizons = [5, 10]
    df_proj = project_all(
        df_hist,
        max(horizons),
        terminal_growth=terminal_growth,
        converge_growth=False,
        man_inp_growth=config.get("man_inp_growth", None),
        plot_projections=plot_projections,
        regression_plots_dir=regression_plots_dir,
        plot_regression_plots=args.plot_regression_plots or final_params.get("plot_regression_plots", True),
        variables=["Revenue", "operating_margin", "effective_tax_rate", "sales_to_capital"],
        target_operating_margin=target_operating_margin,
        target_effective_tax_rate=target_effective_tax_rate,
        target_sales_to_capital=target_sales_to_capital,
        initial_operating_margin=initial_operating_margin,
        initial_effective_tax_rate=initial_effective_tax_rate,
        initial_sales_to_capital=initial_sales_to_capital
    )

    df_proj_converge = project_all(
        df_hist,
        max(horizons),
        terminal_growth=terminal_growth,
        converge_growth=True,
        man_inp_growth=config.get("man_inp_growth", None),
        plot_projections=plot_projections,
        regression_plots_dir=regression_plots_dir,
        plot_regression_plots=args.plot_regression_plots or final_params.get("plot_regression_plots", True),
        variables=["Revenue", "operating_margin", "effective_tax_rate", "sales_to_capital"],
        target_operating_margin=target_operating_margin,
        target_effective_tax_rate=target_effective_tax_rate,
        target_sales_to_capital=target_sales_to_capital,
        initial_operating_margin=initial_operating_margin,
        initial_effective_tax_rate=initial_effective_tax_rate,
        initial_sales_to_capital=initial_sales_to_capital
    )

    if plot_projections:
        for df_p, suffix in [(df_proj, ""), (df_proj_converge, "_converge")]:
            plot_folder = os.path.join(output_folder, f"proj_plots{suffix}")
            os.makedirs(plot_folder, exist_ok=True)
            plot_all_variables(ticker, df_hist, df_p, plot_folder)
            plot_projected_fcf_growth_rates(ticker, df_p, plot_folder)

    proj_csv_path = os.path.join(output_folder, "projected_values.csv")
    df_proj_out = change_units(df_proj, [c for c in ["Revenue", "EBITDA", "FCF"] if c in df_proj.columns], to=units)
    df_proj_out.to_csv(proj_csv_path, index=False)


    # === Run DCF for multiple forecast horizons ===
    dcf_estimates_list = []
    unit_cols = ["Revenue", "EBIT", "nopat", "reinvest", "capital_invested", "FCFF", "pv", "Enterprise_value", "Terminal_value", "Npv", "Net_debt", "Equity_value"]
    dec_cols = ["growth_rate", "operating_margin", "effective_tax_rate", "ROCE", "discount_factor", "sales_to_capital", "per_share_value", "upside", "average_growth_rate"]
    number_shares = val_info.get("number_shares")
    share_price = val_info.get("share_price")

    proj_sets = [
        ("standard", df_proj, ""),
        ("converged", df_proj_converge, "_converge"),
    ]

    projection_methods = [
        ("cagr", "_cagr", "historical_cagr"),
        ("slope", "_slope", "regression_slope"),
        ("man_inp_growth", "_manual_input", "manual_input_growth"),
    ]

    for horizon in horizons:
        for proj_name, proj_df, proj_suffix in proj_sets:
            for p_method, method_suffix, method_name in projection_methods:
                suffix = f"{method_suffix}_{proj_suffix}_h{horizon}"
                label = method_name.replace("_", " ").title()
                
                dcf_df, dcf_info = run_dcf(
                    df_hist,
                    proj_df,
                    terminal_growth,
                    wacc_used,
                    forecast_horizon=horizon,
                    p_method=p_method,
                    number_shares=number_shares,
                    share_price=share_price,
                    output_folder= output_folder,
                    proj_name=proj_name,
                )

                dcf_out = change_units(dcf_df, [c for c in unit_cols if c in dcf_df.columns], to=units)
                dcf_out = reduce_decimals(dcf_out, [c for c in dec_cols if c in dcf_out.columns])
                save_dcf_results(dcf_out, output_folder, suffix=suffix)
        
                # Insert current share price after per_share_value
                dcf_row = {
                    "ticker": ticker,
                    "name": company_name,
                    "horizon": horizon,
                    "proj_type": proj_name,
                    "fcf_source": "FCFF",
                    "p_method": p_method,
                    "method_label": label,
                    "used_wacc": wacc_used,
                }
                # dcf_info contains: enterprise_value, terminal_value, npv, per_share_value, upside, average_growth_rate
                # We want: ... per_share_value, current_share_price, upside ...
                for k in ["enterprise_value", "terminal_value", "npv", "net_debt"]:
                    dcf_row[k] = dcf_info[k]
                dcf_row["per_share_value"] = dcf_info["per_share_value"]
                dcf_row["current_share_price"] = share_price
                dcf_row["n_shares"] = number_shares
                dcf_row["upside"] = dcf_info["upside"]
                dcf_row["average_growth_rate"] = dcf_info["average_growth_rate"]
                dcf_estimates_list.append(dcf_row)


    dcf_estimates = pd.DataFrame(dcf_estimates_list)
    dcf_estimates = change_units(dcf_estimates, [c for c in unit_cols if c in dcf_estimates.columns], to=units)
    dcf_estimates = reduce_decimals(dcf_estimates, [c for c in dec_cols if c in dcf_estimates.columns])
    # Save DCF estimates to CSV
    col_order = [
        "ticker", "name", "upside", "horizon", "proj_type", "fcf_source", "p_method", "method_label", "used_wacc",
        "enterprise_value", "terminal_value", "npv", "net_debt", "per_share_value", "current_share_price", "n_shares", "average_growth_rate"
    ]
    dcf_estimates = dcf_estimates[[c for c in col_order if c in dcf_estimates.columns]]
    dcf_estimates.to_csv(os.path.join(output_folder, "dcf_estimates.csv"), index=False)

    # --- Plotting Upside by Method, Horizon, and Projection Type ---
    try:
        import seaborn as sns
        # Only plot if there is enough data
        if not dcf_estimates.empty and 'upside' in dcf_estimates.columns:
            df_upside = dcf_estimates.copy()
            horizons = sorted(df_upside['horizon'].unique())
            proj_types = sorted(df_upside['proj_type'].unique())
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
            for i, horizon in enumerate(horizons):
                for j, proj_type in enumerate(proj_types):
                    if i < axes.shape[0] and j < axes.shape[1]:
                        ax = axes[i, j]
                        subset = df_upside[(df_upside['horizon'] == horizon) & (df_upside['proj_type'] == proj_type)]
                        if not subset.empty:
                            sns.barplot(
                                data=subset,
                                x='method_label',
                                y='upside',
                                hue='method_label',
                                ax=ax,
                                palette='viridis',
                                legend=False
                            )
                        ax.set_title(f'Horizon: {horizon} yrs | Proj: {proj_type}')
                        ax.set_xlabel('Method')
                        ax.set_ylabel('Upside (%)' if j == 0 else '')
                        ax.set_ylim(0, 100)  
                        ax.tick_params(axis='x', rotation=20)
            plt.tight_layout()
            plot_path = os.path.join(output_folder, "dcf_upside_methods.png")
            plt.savefig(plot_path)
            # Optionally show plot interactively if running in a notebook or interactive session
            # plt.show()
    except Exception as e:
        print(f"Warning: Could not generate upside plot: {e}")

    plot_fcff_components(df_hist, output_folder)


if __name__ == "__main__":
    main()
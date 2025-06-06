import json
import pandas as pd
import numpy as np
import argparse
import os
from scipy.stats import linregress
import matplotlib.pyplot as plt


def calculate_growth_stats(series: pd.Series):
    """
    Given a pd.Series of annual numbers indexed by year (e.g. Revenue),
    returns:
      - CAGR over the full span ([n_last / n_first]^(1/(N–1)) – 1)
      - Latest year‐over‐year change (last / prior – 1)

    If series has fewer than 2 points, returns (np.nan, np.nan).
    """
    try:
        n_first = series.iloc[0]
        n_last = series.iloc[-1]
        periods = len(series) - 1
        cagr = (n_last / n_first) ** (1 / periods) - 1
    except Exception:
        cagr = np.nan

    try:
        recent = series.iloc[-1] / series.iloc[-2] - 1
    except Exception:
        recent = np.nan

    return cagr, recent

def calc_linreg_slope(series):
    s = pd.Series(series).dropna()
    if not isinstance(s, pd.Series):
        print(f"DEBUG: Not a Series, got {type(s)}")
        return np.nan, np.nan
    if len(s) < 2:
        print(f"DEBUG: Not enough data for linregress: {s.values}")
        return np.nan, np.nan
    x = np.arange(len(s))
    if x.shape != s.values.shape or len(s) < 2:
        print(f"DEBUG: Shape mismatch or insufficient data: x={x.shape}, s={s.values.shape}")
        return np.nan, np.nan
    try:
        s = np.array(s.values, dtype=float)
        result = linregress(x, s)
        slope, intercept = result.slope, result.intercept
        return slope, intercept
    except Exception as e:
        print(f"DEBUG: linregress failed: {e}")
        return np.nan, np.nan


def project_cagr(series, last_year, projection_years):
    cagr, _ = calculate_growth_stats(series)
    last_value = series.dropna().iloc[-1] if len(series.dropna()) > 0 else np.nan
    projections = []
    for i in range(1, projection_years + 1):
        year = last_year + i
        value = last_value * ((1 + cagr) ** i) if not np.isnan(last_value) and not np.isnan(cagr) else np.nan
        projections.append({'Year': year, 'method': 'cagr', 'value': value, 'parameter_value': cagr})
    return projections

def project_slope(series, last_year, projection_years):
    s = series.dropna()
    if len(s) < 2:
        return [{'Year': last_year + i, 'method': 'slope', 'value': np.nan, 'parameter_value': np.nan} for i in range(1, projection_years + 1)]
    slope, intercept = calc_linreg_slope(s)

    projections = []
    for i in range(1, projection_years + 1):
        year = last_year + i
        x_proj = len(s) - 1 + i
        value = slope * x_proj + intercept
        projections.append({'Year': year, 'method': 'slope', 'value': value, 'parameter_value': slope})
    return projections

def project_all(df_hist, projection_years):
    last_year = pd.to_datetime(df_hist.index).year.max()
    results = []
    for col in df_hist.columns:
        s = df_hist[col]
        cagr_proj = project_cagr(s, last_year, projection_years)
        for p in cagr_proj:
            results.append({'variable': col, **p})
        slope_proj = project_slope(s, last_year, projection_years)
        for p in slope_proj:
            results.append({'variable': col, **p})
    df_proj = pd.DataFrame(results)

    return df_proj[['Year', 'variable', 'method', 'value', 'parameter_value']]

def plot_variable_projection(ticker, variable, df_hist, df_proj, output_folder):
    # Prepare data
    years_hist = pd.to_datetime(df_hist.index).year
    values_hist = df_hist[variable]
    
    # Pivot projected values for this variable
    df_proj_var = df_proj[df_proj['variable'] == variable]
    proj_cagr = df_proj_var[df_proj_var['method'] == 'cagr'].set_index('Year')['value']
    proj_slope = df_proj_var[df_proj_var['method'] == 'slope'].set_index('Year')['value']
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(years_hist.to_numpy(), values_hist.to_numpy(), 'o-', color='black', label='Historical')
    if not proj_cagr.empty:
        plt.plot(proj_cagr.index.to_numpy(), proj_cagr.values, 'r--', label='Projected (CAGR)')
    if not proj_slope.empty:
        plt.plot(proj_slope.index.to_numpy(), proj_slope.values, 'b:', label='Projected (Regression Slope)')
    plt.title(f"{ticker} - {variable}: Historical and Projected Values")
    plt.xlabel("Year")
    plt.ylabel(variable)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save
    safe_var = variable.replace("/", "_").replace("\\", "_")
    filename = os.path.join(output_folder, f"{ticker}_{safe_var}_projection.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

def plot_all_variables(ticker, df_hist, df_proj, output_folder):
    variables = [col for col in df_hist.columns if col in df_proj['variable'].unique()]
    for variable in variables:
        plot_variable_projection(ticker, variable, df_hist, df_proj, output_folder)

def main():
    parser = argparse.ArgumentParser(description="Project all historical variables using both CAGR and regression slope and plot results.")
    parser.add_argument("ticker", help="Ticker symbol for the company to project.")
    parser.add_argument("--config", default="dcf_config.json", help="Path to JSON config file.")
    parser.add_argument("--output", default="projected_values.csv", help="Output CSV filename.")
    args = parser.parse_args()

    ticker = args.ticker
    output_folder = f"{ticker}_results"
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file '{args.config}' not found.")
    with open(args.config, "r") as f:
        config = json.load(f)

    projection_years = config.get("projection_years")
    if projection_years is None:
        raise ValueError("'projection_years' must be specified in the configuration.")

    hist_csv_path = os.path.join(output_folder, "historical_data.csv")
    if not os.path.exists(hist_csv_path):
        raise FileNotFoundError(f"Historical data CSV '{hist_csv_path}' not found. Please run data fetch first.")
    df_hist = pd.read_csv(hist_csv_path, index_col=0)
    df_hist.index = pd.to_datetime(df_hist.index)
    df_hist = df_hist.sort_index()

    df_proj = project_all(df_hist, projection_years)
    df_proj.to_csv(args.output, index=False)
    print(f"Projected values saved to '{args.output}'")

    plot_all_variables(ticker, df_hist, df_proj, output_folder)

if __name__ == "__main__":
    main()
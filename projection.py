import json
import pandas as pd
import numpy as np
import argparse
import os
from scipy.stats import linregress
import matplotlib.pyplot as plt

def calculate_growth_stats(series: pd.Series):
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

def project_cagr(series, last_year, projection_years, terminal_growth=None, converge_growth=False):
    cagr, _ = calculate_growth_stats(series)
    last_value = series.dropna().iloc[-1] if len(series.dropna()) > 0 else np.nan
    projections = []
    growth_vec = _get_growth_vector(cagr, terminal_growth, projection_years, converge_growth)
    v = last_value
    for i, g in enumerate(growth_vec):
        year = last_year + i + 1
        v = v * (1 + g) if v is not None and not np.isnan(v) and g is not None and not np.isnan(g) else np.nan
        projections.append({'Year': year, 'method': 'cagr', 'value': v, 'parameter_value': cagr, 'growth_rate': g})
    return projections

def project_slope(series, last_year, projection_years, terminal_growth=None, converge_growth=False):
    s = series.dropna()
    if len(s) < 2:
        return [{'Year': last_year + i, 'method': 'slope', 'value': np.nan, 'parameter_value': np.nan, 'growth_rate': np.nan} for i in range(1, projection_years + 1)]
    slope, intercept = calc_linreg_slope(s)
    projections = []
    v = s.iloc[-1]
    for i in range(1, projection_years + 1):
        year = last_year + i
        x_proj = len(s) - 1 + i
        value = slope * x_proj + intercept
        g = (value/v - 1) if v and value and not np.isnan(v) and not np.isnan(value) and v != 0 else np.nan
        projections.append({'Year': year, 'method': 'slope', 'value': value, 'parameter_value': slope, 'growth_rate': g})
        v = value
    if converge_growth and terminal_growth is not None:
        growths = [p['growth_rate'] for p in projections]
        growth_vec = _get_growth_vector(growths[0], terminal_growth, projection_years, True)
        v = s.iloc[-1]
        for i, g in enumerate(growth_vec):
            value = v * (1 + g) if v is not None and not np.isnan(v) and g is not None and not np.isnan(g) else np.nan
            projections[i]['value'] = value
            projections[i]['growth_rate'] = g
            v = value
    return projections

def project_expected(series, last_year, projection_years, expected_growth, terminal_growth=None, converge_growth=False):
    """
    Projects values using a fixed expected growth (float) or converged growth vector.
    """
    last_value = series.dropna().iloc[-1] if len(series.dropna()) > 0 else np.nan
    projections = []
    growth_vec = _get_growth_vector(expected_growth, terminal_growth, projection_years, converge_growth)
    v = last_value
    for i, g in enumerate(growth_vec):
        year = last_year + i + 1
        v = v * (1 + g) if v is not None and not np.isnan(v) and g is not None and not np.isnan(g) else np.nan
        projections.append({'Year': year, 'method': 'expected_growth', 'value': v, 'parameter_value': expected_growth, 'growth_rate': g})
    return projections

def project_all(df_hist, projection_years, terminal_growth=None, converge_growth=False, expected_growth=None):
    last_year = pd.to_datetime(df_hist.index).year.max()
    results = []
    for col in df_hist.columns:
        s = df_hist[col]
        cagr_proj = project_cagr(s, last_year, projection_years, terminal_growth, converge_growth)
        for p in cagr_proj:
            results.append({'variable': col, **p})
        slope_proj = project_slope(s, last_year, projection_years, terminal_growth, converge_growth)
        for p in slope_proj:
            results.append({'variable': col, **p})
        # Add expected_growth projections only if expected_growth is not None and not 'slope'
        if expected_growth is not None and expected_growth != "slope":
            expected_proj = project_expected(s, last_year, projection_years, expected_growth, terminal_growth, converge_growth)
            for p in expected_proj:
                results.append({'variable': col, **p})
    df_proj = pd.DataFrame(results)
    return df_proj[['Year', 'variable', 'method', 'value', 'parameter_value', 'growth_rate']]

def plot_variable_projection(ticker, variable, df_hist, df_proj, output_folder):
    years_hist = pd.to_datetime(df_hist.index).year
    values_hist = df_hist[variable]
    df_proj_var = df_proj[df_proj['variable'] == variable]
    proj_cagr = df_proj_var[df_proj_var['method'] == 'cagr'].set_index('Year')['value']
    proj_slope = df_proj_var[df_proj_var['method'] == 'slope'].set_index('Year')['value']
    proj_expected = None
    if 'expected_growth' in df_proj_var['method'].unique():
        proj_expected = df_proj_var[df_proj_var['method'] == 'expected_growth'].set_index('Year')['value']
    plt.figure(figsize=(8, 5))
    plt.plot(years_hist.to_numpy(), values_hist.to_numpy(), 'o-', color='black', label='Historical')
    if not proj_cagr.empty:
        plt.plot(proj_cagr.index.to_numpy(), proj_cagr.values, 'r--', label='Projected (CAGR)')
    if not proj_slope.empty:
        plt.plot(proj_slope.index.to_numpy(), proj_slope.values, 'b:', label='Projected (Regression Slope)')
    if proj_expected is not None and not proj_expected.empty:
        plt.plot(proj_expected.index.to_numpy(), proj_expected.values, 'g-.', label='Projected (Expected Growth)')
    plt.title(f"{ticker} - {variable}: Historical and Projected Values")
    plt.xlabel("Year")
    plt.ylabel(variable)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    safe_var = variable.replace("/", "_").replace("\\", "_")
    filename = os.path.join(output_folder, f"{ticker}_{safe_var}_projection.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

def plot_all_variables(ticker, df_hist, df_proj, output_folder):
    variables = [col for col in df_hist.columns if col in df_proj['variable'].unique()]
    for variable in variables:
        plot_variable_projection(ticker, variable, df_hist, df_proj, output_folder)

def plot_projected_fcf_growth_rates(ticker, df_proj, output_folder, terminal_growth=None):
    """
    Plots the yearly growth rates for projected FCF for each projection method (CAGR, slope, expected_growth, etc).
    Also plots the terminal growth rate as a horizontal line.
    """
    methods = df_proj['method'].unique()
    years = sorted(df_proj['Year'].unique())
    fcf_proj = df_proj[df_proj['variable'] == 'FCF']

    plt.figure(figsize=(8, 5))
    for method in methods:
        fcf_method = fcf_proj[fcf_proj['method'] == method]
        if fcf_method.empty:
            continue
        years_plot = fcf_method['Year'].values
        growths_plot = fcf_method['growth_rate'].values
        mask = ~pd.isnull(growths_plot)
        if not np.any(mask):
            continue
        label = f'Projected FCF growth ({method})'
        plt.plot(years_plot[mask], np.array(growths_plot)[mask], marker='o', label=label)
    # Terminal growth rate line
    if terminal_growth is not None:
        plt.axhline(y=terminal_growth, color='purple', linestyle='--', linewidth=1.3, label='Terminal Growth Rate')
    plt.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    plt.title(f"{ticker} - Yearly Projected FCF Growth Rates by Method")
    plt.xlabel("Year")
    plt.ylabel("Projected FCF Growth Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_folder, f"{ticker}_fcf_growth_rates.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Project all historical variables using both CAGR and regression slope and plot results.")
    parser.add_argument("ticker", help="Ticker symbol for the company to project.")
    parser.add_argument("--config", default="dcf_config.json", help="Path to JSON config file.")
    parser.add_argument("--output", default="projected_values.csv", help="Output CSV filename.")
    parser.add_argument("--converge_growth", action="store_true", help="Enable convergence of projected growth rates to terminal growth.")
    parser.add_argument("--expected_growth", type=float, help="Expected FCF or EBITDA growth rate override.")
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

    # Support both terminal_growth and terminal_growth_rate in config
    terminal_growth = config.get("terminal_growth")
    if terminal_growth is None:
        terminal_growth = config.get("terminal_growth_rate")

    converge_growth = args.converge_growth or config.get("converge_growth", False)
    # Accept expected_growth as CLI or config, but only pass if not 'slope'
    expected_growth = args.expected_growth if args.expected_growth is not None else config.get("expected_growth", None)
    if expected_growth == "slope":
        expected_growth = None  # We don't want to run expected as "slope" here

    hist_csv_path = os.path.join(output_folder, "historical_data.csv")
    if not os.path.exists(hist_csv_path):
        raise FileNotFoundError(f"Historical data CSV '{hist_csv_path}' not found. Please run data fetch first.")
    df_hist = pd.read_csv(hist_csv_path, index_col=0)
    df_hist.index = pd.to_datetime(df_hist.index)
    df_hist = df_hist.sort_index()

    df_proj = project_all(
        df_hist,
        projection_years,
        terminal_growth=terminal_growth,
        converge_growth=converge_growth,
        expected_growth=expected_growth
    )
    df_proj.to_csv(args.output, index=False)
    print(f"Projected values saved to '{args.output}'")

    plot_all_variables(ticker, df_hist, df_proj, output_folder)
    plot_projected_fcf_growth_rates(ticker, df_proj, output_folder, terminal_growth=terminal_growth)

if __name__ == "__main__":
    main()
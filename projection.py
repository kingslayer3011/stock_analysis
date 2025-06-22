import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
import numpy as np
import pandas as pd
import debugpy

def calculate_growth_stats(series: pd.Series):
    try:
        series = series.dropna()
        if len(series) < 2:
            return np.nan, np.nan

        n_first = series.iloc[0]
        n_last = series.iloc[-1]
        periods = len(series) - 1

        # Robust CAGR calculation
        if n_first <= 0 or n_last <= 0 or periods <= 0:
            cagr = np.nan
        else:
            cagr = (n_last / n_first) ** (1 / periods) - 1
            if isinstance(cagr, complex):
                cagr = np.nan
    except Exception:
        cagr = np.nan

    # Recent growth (last value vs. second-to-last)
    try:
        if series.iloc[-2] == 0:
            recent = np.nan
        else:
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

def project_cagr(series, last_year, projection_years, terminal_growth=None, converge_growth=False, last_value=None):
    """
    Projects values using CAGR, optionally converging to a terminal growth rate.
    last_value: Optionally specify the starting value for projections (default: last non-NA in series).
    """
    cagr, _ = calculate_growth_stats(series)
    if last_value is None:
        last_value = series.dropna().iloc[-1] if len(series.dropna()) > 0 else np.nan
    projections = []
    growth_vector = _get_growth_vector(cagr, terminal_growth, projection_years, converge_growth)
    for projection_index, growth_rate in enumerate(growth_vector):
        projected_year = last_year + projection_index + 1
        last_value = (
            last_value * (1 + growth_rate)
            if last_value is not None and not np.isnan(last_value) and growth_rate is not None and not np.isnan(growth_rate)
            else np.nan
        )
        projections.append({
            'Year': projected_year,
            'method': 'cagr',
            'value': last_value,
            'parameter_value': cagr,
            'growth_rate': growth_rate
        })
    return projections

def project_slope(series, last_year, projection_years, terminal_growth=None, converge_growth=False, last_value=None):
    """
    Projects values using regression slope, optionally converging to a terminal growth rate.
    last_value: Optionally specify the starting value for projections (default: last non-NA in series).
    """
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return [
            {
                'Year': last_year + i,
                'method': 'slope',
                'value': np.nan,
                'parameter_value': np.nan,
                'growth_rate': np.nan
            }
            for i in range(1, projection_years + 1)
        ]
    slope, intercept = calc_linreg_slope(clean_series)
    if last_value is None:
        last_value = clean_series.iloc[-1]
    projections = []
    for projection_index in range(1, projection_years + 1):
        projected_year = last_year + projection_index
        regression_x = len(clean_series) - 1 + projection_index
        projected_value = slope * regression_x + intercept
        projected_growth = (
            (projected_value / last_value - 1)
            if last_value and projected_value and not np.isnan(last_value) and not np.isnan(projected_value) and last_value != 0
            else np.nan
        )
        projections.append({
            'Year': projected_year,
            'method': 'slope',
            'value': projected_value,
            'parameter_value': slope,
            'growth_rate': projected_growth
        })
        last_value = projected_value
    if converge_growth and terminal_growth is not None:
        initial_growth = projections[0]['growth_rate']
        growth_vector = _get_growth_vector(initial_growth, terminal_growth, projection_years, True)
        last_value = clean_series.iloc[-1]
        for i, growth_rate in enumerate(growth_vector):
            projected_value = (
                last_value * (1 + growth_rate)
                if last_value is not None and not np.isnan(last_value) and growth_rate is not None and not np.isnan(growth_rate)
                else np.nan
            )
            projections[i]['value'] = projected_value
            projections[i]['growth_rate'] = growth_rate
            last_value = projected_value
    return projections

def project_man_inp_growth(series, last_year, projection_years, man_inp_growth, terminal_growth=None, converge_growth=False, last_value=None):
    """
    Projects values using a fixed manually input growth (float) or converged growth vector.
    last_value: Optionally specify the starting value for projections (default: last non-NA in series).
    """
    if last_value is None:
        last_value = series.dropna().iloc[-1] if len(series.dropna()) > 0 else np.nan
    projections = []
    growth_vec = _get_growth_vector(man_inp_growth, terminal_growth, projection_years, converge_growth)
    for i, growth_rate in enumerate(growth_vec):
        year = last_year + i + 1
        last_value = (
            last_value * (1 + growth_rate)
            if last_value is not None and not np.isnan(last_value) and growth_rate is not None and not np.isnan(growth_rate)
            else np.nan
        )
        projections.append({
            'Year': year,
            'method': 'man_inp_growth',
            'value': last_value,
            'parameter_value': man_inp_growth,
            'growth_rate': growth_rate
        })
    return projections

def project_all(df_hist, projection_years, terminal_growth=None, converge_growth=False, man_inp_growth=None):
    """
    Projects every column in df_hist using CAGR, regression‐slope, and manual‐input methods,
    but excludes any 'TTM' row so that only full‐year data is used.
    """
    # 1) Filter out any non‐year index (e.g. 'TTM') – assumes full years are datetime‐like or ints
    #    We coerce to datetime, drop failures (like 'TTM'), then extract unique years.
    valid_idx = pd.to_datetime(df_hist.index, errors='coerce').dropna()
    df_full_years = df_hist.loc[valid_idx]

    # 2) Compute last_year from full-year data (where ttm == 0)
    df_full_years_no_ttm = df_full_years[df_full_years.get('ttm', 0) == 0]
    last_year = pd.to_datetime(df_full_years_no_ttm.index).year.max()

    # 3) Get TTM row (where ttm == 1) for last_value
    if 'ttm' in df_full_years.columns:
        ttm_rows = df_full_years[df_full_years['ttm'] == 1]
    else:
        ttm_rows = pd.DataFrame()
    results = []
    for col in df_full_years.columns:
        if col == 'ttm':
            continue  # skip the ttm indicator column itself

        s = df_full_years[df_full_years['ttm'] == 0][col]

        # Use TTM value as last_value if available, else fallback to last non-NA
        if not ttm_rows.empty and col in ttm_rows.columns:
            ttm_val = ttm_rows[col].dropna()
            last_value = ttm_val.iloc[0] if len(ttm_val) > 0 else s.dropna().iloc[-1] if len(s.dropna()) > 0 else np.nan
        else:
            last_value = s.dropna().iloc[-1] if len(s.dropna()) > 0 else np.nan

        # CAGR‐based projections
        cagr_proj = project_cagr(s, last_year, projection_years, terminal_growth, converge_growth, last_value=last_value)
        for p in cagr_proj:
            results.append({'variable': col, **p})

        # Regression‐slope projections
        slope_proj = project_slope(s, last_year, projection_years, terminal_growth, converge_growth, last_value=last_value)
        for p in slope_proj:
            results.append({'variable': col, **p})

        # Manual‐input projections
        man_inp_proj = project_man_inp_growth(s, last_year, projection_years, man_inp_growth, terminal_growth, converge_growth, last_value=last_value)
        for p in man_inp_proj:
            results.append({'variable': col, **p})
    debugpy.breakpoint()
    df_proj = pd.DataFrame(results)
    return df_proj[['Year', 'variable', 'method', 'value', 'parameter_value', 'growth_rate']]

def plot_variable_projection(ticker, variable, df_hist, df_proj, output_folder):
    years_hist = pd.to_datetime(df_hist.index).year
    values_hist = df_hist[variable]
    df_proj_var = df_proj[df_proj['variable'] == variable]
    proj_cagr = df_proj_var[df_proj_var['method'] == 'cagr'].set_index('Year')['value']
    proj_slope = df_proj_var[df_proj_var['method'] == 'slope'].set_index('Year')['value']
    proj_man_inp = None
    if 'man_inp_growth' in df_proj_var['method'].unique():
        proj_man_inp = df_proj_var[df_proj_var['method'] == 'man_inp_growth'].set_index('Year')['value']
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(years_hist.to_numpy(), values_hist.to_numpy(), 'o-', color='black', label='Historical')
    if not proj_cagr.empty:
        plt.plot(proj_cagr.index.to_numpy(), proj_cagr.values, 'r--', label='Projected (CAGR)')
    if not proj_slope.empty:
        plt.plot(proj_slope.index.to_numpy(), proj_slope.values, 'b:', label='Projected (Regression Slope)')
    if proj_man_inp is not None and not proj_man_inp.empty:
        plt.plot(proj_man_inp.index.to_numpy(), proj_man_inp.values, 'g-.', label='Projected (Manual Input Growth)')
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
    variables = [
    "Revenue", "EBITDA", "Net Income", "FCF", "FCFE", "CapEx",
    "TaxRate", "NOPAT", "ReinvestmentRate", "RetentionRatio", "ROE", "ROC"]

    for variable in variables:
        plot_variable_projection(ticker, variable, df_hist, df_proj, output_folder)

def plot_projected_fcf_growth_rates(ticker, df_proj, output_folder, terminal_growth=None):
    import matplotlib.pyplot as plt
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
        if method == 'man_inp_growth':
            plt.plot(years_plot[mask], np.array(growths_plot)[mask], marker='o', color='green', linestyle='-.', linewidth=2, label=label)
        else:
            plt.plot(years_plot[mask], np.array(growths_plot)[mask], marker='o', label=label)
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



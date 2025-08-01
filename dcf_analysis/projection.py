import pandas as pd
import numpy as np
from scipy.stats import linregress
import os
import numpy as np
import pandas as pd
import debugpy
from data_fetcher import *


"""import debugpy
debugpy.listen(("localhost", 5682))  # You can use any open port, e.g., 5678
print("Waiting for debugger attach...")
debugpy.wait_for_client()"""  # This will pause execution until you attach the debugger


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
    # Cap CAGR at 0.15 if above
    if cagr is not None and not np.isnan(cagr) and cagr > 0.15:
        cagr = 0.15
    if last_value is None:
        last_value = series.dropna().iloc[-1] if len(series.dropna()) > 0 else np.nan
    projections = []
    growth_vector = _get_growth_vector(cagr, terminal_growth, projection_years, converge_growth)
    for projection_index, growth_rate in enumerate(growth_vector):
        # Cap each projected growth rate at 0.15 as well
        if growth_rate is not None and not np.isnan(growth_rate) and growth_rate > 0.15:
            growth_rate = 0.15
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

def project_slope(
    series, last_year, projection_years, terminal_growth=None, converge_growth=False,
    last_value=None, plot_projections=False, variable=None, regression_plots_dir=None,
    plot_regression_plots=True,  # <-- Add this parameter
    **kwargs
):
    """
    Projects values using regression slope, optionally converging to a terminal growth rate.
    last_value: Optionally specify the starting value for projections (default: last non-NA in series).
    If plot_projections is True, saves a regression plot in regression_plots/ (or regression_plots_dir if provided).
    variable: Name of the variable (for plot filename).
    regression_plots_dir: Optional path to save regression plots.
    plot_regression_plots: If False, disables saving regression plots even if plot_projections is True.
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

    # Plot regression if requested
    if plot_projections and plot_regression_plots:
        import matplotlib.pyplot as plt
        import os
        folder = regression_plots_dir if regression_plots_dir is not None else 'regression_plots'
        os.makedirs(folder, exist_ok=True)
        x = np.arange(len(clean_series))
        y = clean_series.values
        y_pred = slope * x + intercept
        plt.figure(figsize=(7, 4))
        plt.plot(x, y, 'o', label='Historical')
        plt.plot(x, y_pred, '-', label='Regression line')
        plt.title(f"Regression Slope Projection\n{variable if variable else ''}")
        plt.xlabel('Time Index')
        plt.ylabel(variable if variable else 'Value')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0)
        # Add slope annotation box
        slope_text = f"Slope: {slope:.4g}"
        plt.gca().text(0.98, 0.02, slope_text, transform=plt.gca().transAxes,
                      fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        fname = f"{variable if variable else 'variable'}_regression.png"
        fname = fname.replace('/', '_').replace('\\', '_')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, fname))
        plt.close()
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

def project_operating_margin(
    df_hist, projection_years, initial_value=None, terminal_value=None):
    """
    Projects operating margin using convex combination of last value and terminal value.
    terminal_value: Optional terminal value to converge to.
    """
    if 'operating_margin' not in df_hist.columns:
        raise ValueError("DataFrame must contain 'operating_margin' column")
    s = df_hist['operating_margin'].dropna()
    if len(s) < 1:
        return pd.DataFrame(columns=['Year', 'method', 'value', 'parameter_value', 'growth_rate'])
    last_year = df_hist[df_hist.get("ttm", 0) == 0].index.max().year

    if initial_value is None:
        initial_value = s.iloc[-1] if not s.empty else np.nan

    projections = []
    for i in range(1, projection_years + 1):
        projected_year = last_year + i
        if terminal_value is not None:
            alpha = i / projection_years
            projected_value = (1 - alpha) * initial_value + alpha * terminal_value
            growth_rate = (projected_value / initial_value - 1) if initial_value != 0 and not np.isnan(initial_value) else np.nan
        else:
            projected_value = initial_value
            growth_rate = 0
        projections.append({
            'variable': 'operating_margin',
            'Year': projected_year,
            'method': str(projection_years),
            'value': projected_value,
            'parameter_value': np.nan,
            'growth_rate': growth_rate
        })
    df_proj = pd.DataFrame(projections)
    return df_proj

def project_effective_tax_rate(df_hist, projection_years, initial_value=None, terminal_value=None):
    """
    Projects effective tax rate using convex combination of last value and terminal value.
    terminal_value: Optional terminal value to converge to.
    """
    if 'effective_tax_rate' not in df_hist.columns:
        raise ValueError("DataFrame must contain 'effective_tax_rate' column")
    s = df_hist['effective_tax_rate'].dropna()
    if len(s) < 1:
        return pd.DataFrame(columns=['Year', 'method', 'value', 'parameter_value', 'growth_rate'])
    last_year = df_hist[df_hist.get("ttm", 0) == 0].index.max().year

    if initial_value is None:
        initial_value = s.iloc[-1] if not s.empty else np.nan

    projections = []
    for i in range(1, projection_years + 1):
        projected_year = last_year + i
        if terminal_value is not None:
            alpha = i / projection_years
            projected_value = (1 - alpha) * initial_value + alpha * terminal_value
            growth_rate = (projected_value / initial_value - 1) if initial_value != 0 and not np.isnan(initial_value) else np.nan
        else:
            projected_value = initial_value
            growth_rate = 0
        projections.append({
            'variable': 'effective_tax_rate',
            'Year': projected_year,
            'method': str(projection_years),
            'value': projected_value,
            'parameter_value': np.nan,
            'growth_rate': growth_rate
        })
    df_proj = pd.DataFrame(projections)
    return df_proj

def project_sales_to_capital(
    df_hist, projection_years, initial_value=None, terminal_value=None):
    """
    Projects sales to capital using convex combination of last value and terminal value.
    terminal_value: Optional terminal value to converge to.
    """
    if 'sales_to_capital' not in df_hist.columns:
        raise ValueError("DataFrame must contain 'sales_to_capital' column")
    s = df_hist['sales_to_capital'].dropna()
    if len(s) < 1:
        return pd.DataFrame(columns=['Year', 'method', 'value', 'parameter_value', 'growth_rate'])
    last_year = df_hist[df_hist.get("ttm", 0) == 0].index.max().year

    if initial_value is None:
        initial_value = s.iloc[-1] if not s.empty else np.nan

    projections = []
    for i in range(1, projection_years + 1):
        projected_year = last_year + i
        if terminal_value is not None:
            alpha = i / projection_years
            projected_value = (1 - alpha) * initial_value + alpha * terminal_value
            growth_rate = (projected_value / initial_value - 1) if initial_value != 0 and not np.isnan(initial_value) else np.nan
        else:
            projected_value = initial_value
            growth_rate = 0
        projections.append({
            'variable': 'sales_to_capital',
            'Year': projected_year,
            'method': str(projection_years),
            'value': projected_value,
            'parameter_value': np.nan,
            'growth_rate': growth_rate
        })
    df_proj = pd.DataFrame(projections)
    return df_proj



def project_all(
    df_hist, projection_years, target_horizons=[5,10], terminal_growth=None, converge_growth=False, man_inp_growth=None,
    plot_projections=False, regression_plots_dir=None, plot_regression_plots=True,
    variables=None, target_operating_margin=None, target_effective_tax_rate=None, target_sales_to_capital=None, 
    initial_operating_margin=None, initial_effective_tax_rate=None, initial_sales_to_capital=None
):
    """
    Projects every column in df_hist using CAGR, regression‐slope, and manual‐input methods,
    but excludes any 'TTM' row so that only full‐year data is used.
    Only projects columns listed in `variables` if provided.
    plot_regression_plots: If False, disables saving regression plots even if plot_projections is True.
    """
    # 1) Filter out any non‐year index (e.g. 'TTM') – assumes full years are datetime‐like or ints
    #    We coerce to datetime, drop failures (like 'TTM'), then extract unique years.
    valid_idx = pd.to_datetime(df_hist.index, errors='coerce').dropna()
    df_full_years = df_hist.loc[valid_idx]

    # 2) Compute last_year from full-year data (where ttm == 0)
    df_full_years_no_ttm = df_full_years[df_full_years.get('ttm', 0) == 0]
    if not df_full_years_no_ttm.empty:
        last_year = pd.to_datetime(df_full_years_no_ttm.index).year.max()
    else:
        last_year = None

    # 3) Get TTM row (where ttm == 1) for last_value
    if 'ttm' in df_full_years.columns:
        ttm_rows = df_full_years[df_full_years['ttm'] == 1]
        if not ttm_rows.empty:
            quarter = "Q" + str(ttm_rows.index.to_series().dt.quarter.values[0])
        else:
            quarter = None
    else:
        ttm_rows = pd.DataFrame()
        quarter = None
    results = []
    for col in df_full_years.columns:
        if col == 'ttm':
            continue  # skip the ttm indicator column itself
        if variables is not None and col not in variables:
            continue  # skip columns not in the selected list

        s = df_full_years[df_full_years['ttm'] == 0][col]

        # Use TTM value as last_value if available, else fallback to last non-NA
        if not ttm_rows.empty and col in ttm_rows.columns:
            ttm_val = ttm_rows[col].dropna()
            last_value = ttm_val.iloc[0] if len(ttm_val) > 0 else s.dropna().iloc[-1] if len(s.dropna()) > 0 else np.nan
        else:
            last_value = s.dropna().iloc[-1] if len(s.dropna()) > 0 else np.nan

        # Use specialized projection functions for Operating Margin and effective_tax_rate
        if col == "operating_margin":
            for h in target_horizons:
                op_margin_proj = project_operating_margin(df_full_years, h, initial_value=initial_operating_margin, terminal_value=target_operating_margin)
                for p in op_margin_proj.to_dict('records'):
                    p['variable'] = col
                    results.append(p)

            continue  # skip generic projections for this column

        if col == "effective_tax_rate":
            for h in target_horizons:
                tax_proj = project_effective_tax_rate(df_full_years, h, initial_value=initial_effective_tax_rate, terminal_value=target_effective_tax_rate)
                for p in tax_proj.to_dict('records'):
                    p['variable'] = col
                    results.append(p)
                continue  # skip generic projections for this column

        if col == "sales_to_capital":
            for h in target_horizons:
                sales_to_capital_proj = project_sales_to_capital(df_full_years, h, initial_value=initial_sales_to_capital, terminal_value=target_sales_to_capital)
                for p in sales_to_capital_proj.to_dict('records'):
                    p['variable'] = col
                    results.append(p)
                continue

        # CAGR‐based projections
        cagr_proj = project_cagr(s, last_year, projection_years, terminal_growth, converge_growth, last_value=last_value)
        for p in cagr_proj:
            results.append({'variable': col, **p})

        # Regression‐slope projections
        slope_proj = project_slope(
            s, last_year, projection_years, terminal_growth, converge_growth,
            last_value=last_value, plot_projections=plot_projections, variable=col,
            regression_plots_dir=regression_plots_dir, plot_regression_plots=plot_regression_plots
        )
        for p in slope_proj:
            results.append({'variable': col, **p})

        # Manual‐input projections
        man_inp_proj = project_man_inp_growth(s, last_year, projection_years, man_inp_growth, terminal_growth, converge_growth, last_value=last_value)
        for p in man_inp_proj:
            results.append({'variable': col, **p})

    # After the main projection loop, add TTM row(s) if present
    if not ttm_rows.empty:
        for col in df_full_years.columns:
            if col == 'ttm':
                continue
            ttm_val = ttm_rows[col].dropna()
            if len(ttm_val) > 0:
                ttm_date = ttm_val.index[0]
                ttm_quarter = "Q" + str(ttm_date.quarter)
                results.append({
                    'Year': last_year,
                    'variable': col,
                    'method': 'ttm',
                    'value': ttm_val.iloc[0],
                    'parameter_value': np.nan,
                    'growth_rate': np.nan
                })


    df_proj = pd.DataFrame(results)
    if 'quarter' in locals() and isinstance(quarter, str):
        df_proj["period"] = df_proj["Year"].astype(str) + " + " + quarter
    else:
        df_proj["period"] = df_proj["Year"].astype(str)
    
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

def plot_all_variables(ticker, df_hist, df_proj, output_folder):
    variables = [
    "Revenue", "EBITDA", "Net Income", "FCF", "FCFF", "CapEx",
    "effective_tax_rate", "Delta WC", "Depreciation And Amortization"]

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



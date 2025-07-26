import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



def safe_divide(numerator, denominator):
    # if either is a Series with one element, convert to scalar
    if isinstance(numerator, pd.Series) and len(numerator) == 1:
        numerator = numerator.iloc[0]
    if isinstance(denominator, pd.Series) and len(denominator) == 1:
        denominator = denominator.iloc[0]
    if pd.notna(numerator) and pd.notna(denominator) and denominator != 0:
        return numerator / denominator
    else:
        return np.nan

def calculate_fcff(
    df: pd.Series
) -> float:
    """
    Calculate Free Cash Flow to the Firm (FCFF) using the formula:
    FCFF = EBIT - Tax Provision + D&A – CapEx – Change in NWC
    Expects a row (Series) with keys: 'EBIT', 'Depreciation And Amortization', 'CapEx', 'Delta WC', 'Tax Provision'.
    """
    revenue = df.get("Revenue", np.nan)
    operating_margin = df.get("operating_margin", np.nan)
    depreciation_amortization = df.get("Depreciation And Amortization", np.nan)
    capex = df.get("CapEx", np.nan)
    change_in_nwc = df.get("Delta WC", np.nan)
    effective_tax_rate = df.get("effective_tax_rate", np.nan)

    # Check if any value is NaN
    if any(np.isnan(x) for x in [revenue, operating_margin, depreciation_amortization, capex, change_in_nwc, effective_tax_rate]):
        return np.nan

    ebit = revenue * operating_margin
    tax_provision = ebit * effective_tax_rate

    return ebit - tax_provision + depreciation_amortization - capex - change_in_nwc
  # Set a breakpoint here to inspect the values

def calculate_fcfe(
    df: pd.Series
) -> float:
    """
    Calculate Free Cash Flow to Equity (FCFE) using the formula:
    FCFE = FCFF + Change in Debt
    Expects a row (Series) with keys: 'EBIT', 'Depreciation And Amortization', 'CapEx', 'Delta WC', 'Tax Provision', 'Change in Debt'.
    """
    fcff = calculate_fcff(df)
    change_in_debt = df.get("Change in Debt", np.nan)
    return fcff + change_in_debt if pd.notna(change_in_debt) else fcff

def calculate_reinvestment_rate(
    capex: float,
    depreciation: float,
    delta_wc: float,
    ebit: float,
    tax_provision: float,
) -> float:
    """Calculate the reinvestment rate using the formula:
    Reinvestment Rate = (CapEx + Depreciation + ΔWC) / NOPAT
    where NOPAT = EBIT - Tax Provision
    """
    nopat = ebit - tax_provision
    return safe_divide(capex + depreciation + delta_wc, nopat)


def safe_value(df, row):
    if df is not None and row in df.index:
        val = df.loc[row]
        if isinstance(val, pd.Series):
            return val.iloc[0]
        return val
    return np.nan

def calculate_ttm_change_in_working_capital(raw_balance_ttm):
    """
    Calculate working capital and its change from a balance sheet DataFrame.
    Returns (working_capital: pd.Series, change_in_wc: pd.Series)
    """
    working_capital = raw_balance_ttm.loc['Current Assets'] - raw_balance_ttm.loc['Current Liabilities']

    # Convert columns to DatetimeIndex for sorting
    working_capital.index = pd.to_datetime(working_capital.index)
    working_capital = working_capital.sort_index()

    # Calculate change in working capital compared to the same date one year prior
    change_in_wc = working_capital - working_capital.shift(4)
    change_in_wc = change_in_wc.dropna()

    return change_in_wc.iloc[-1]

def plot_fcff_components(df_hist: pd.DataFrame, output_folder: str = None):
    """
    Plot only the indexed (first year = 100%) FCFF components.
    Saves the figure to output_folder if provided, else to current directory.
    """
    components = [
        "EBITDA",
        "Depreciation And Amortization",
        "CapEx",
        "Delta WC",
        "effective_tax_rate"
    ]
    # Filter only historical (not TTM) rows
    df = df_hist[df_hist['ttm'] == 0].copy()
    df = df.sort_index()
    if df.empty:
        print("No historical data to plot.")
        return
    years = df.index.year

    fig, ax = plt.subplots(figsize=(7, 6))

    # Indexed to 100% at first year
    for comp in components:
        if comp in df.columns:
            base = df[comp].iloc[0]
            if pd.notna(base) and base != 0:
                indexed = df[comp] / base * 100
                ax.plot(years, indexed, marker='o', label=comp)
    ax.set_title("FCFF Components (Indexed, First Year = 100%)")
    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        plot_path = os.path.join(output_folder, "fcff_components_indexed.png")
        plt.savefig(plot_path)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


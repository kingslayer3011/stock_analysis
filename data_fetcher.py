import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from functools import reduce

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



def calculate_historical_reinvestment_rate(cashflow: pd.DataFrame,
                                           balance_sheet: pd.DataFrame,
                                           income_statement: pd.DataFrame,
                                           periods: int = None) -> pd.DataFrame:
    """
    Calculate historical reinvestment rates for given financial statements.

    Returns DataFrame indexed by period (column labels) with columns:
    ['CapEx', 'Depreciation', 'DeltaWC', 'EBIT', 'TaxRate', 'NOPAT',
     'ReinvestmentRate', 'ROE', 'RetentionRatio', 'ROC']
    """
    common_cols = cashflow.columns.intersection(balance_sheet.columns).intersection(income_statement.columns)
    common_cols = common_cols.sort_values()
    if periods is not None:
        common_cols = common_cols[:periods]

    records = []
    for col in common_cols:
        cf = cashflow[col]
        bs = balance_sheet[col]
        is_ = income_statement[col]

        capex = -cf.get('Capital Expenditures', cf.get('Capital Expenditure', 0))

        depreciation = (
            cf.get('Depreciation', cf.get('Depreciation And Amortization')) or
            is_.get('Depreciation', is_.get('Depreciation And Amortization', 0))
        )

        delta_wc = cf.get("Change In Working Capital", 0)
        ebit = is_.get('EBIT', is_.get('Ebit', is_.get('Operating Income', 0)))

        tax_rate = is_.get('Tax Rate For Calcs')
        if tax_rate is None:
            tax_expense = is_.get('Tax Provision', 0)
            pretax = is_.get('Pretax Income', ebit)
            tax_rate = safe_divide(tax_expense, pretax)
            if pd.isna(tax_rate):
                tax_rate = 0.25  # fallback default

        nopat = ebit * (1 - tax_rate) if pd.notna(ebit) else np.nan

        reinvest_rate = safe_divide((capex + depreciation + delta_wc), nopat)

        net_income = is_.get('Net Income')
        dividends = cf.get('Common Stock Dividend Paid', cf.get('Cash Dividends Paid', 0))
        stockholders_equity = bs.get('Stockholders Equity', bs.get('Common Stock Equity', np.nan))
        invested_capital = bs.get('Invested Capital', np.nan)

        retention_ratio = 1 - safe_divide(dividends, net_income) if pd.notna(net_income) else np.nan
        roe = safe_divide(net_income, stockholders_equity)
        roc = safe_divide(nopat, invested_capital)

        records.append({
            'Period': col,
            'CapEx': capex,
            'Depreciation': depreciation,
            'DeltaWC': delta_wc,
            'EBIT': ebit,
            'TaxRate': tax_rate,
            'NOPAT': nopat,
            'ReinvestmentRate': reinvest_rate,
            'RetentionRatio': retention_ratio,
            'ROE': roe,
            'ROC': roc
        })

    result = pd.DataFrame(records).set_index('Period')
    return result


def get_historical_data(ticker: str):
    """
    Fetch historical and TTM financial data and key metrics.
    Returns:
        - df: DataFrame with yearly and TTM rows
        - valuation_info: Dict of current market valuation metrics
    """
    t = yf.Ticker(ticker)

    raw_income        = t.financials
    raw_cashflow      = t.cashflow
    raw_balance       = t.balance_sheet
    raw_income_ttm    = t.ttm_financials
    raw_cashflow_ttm  = t.ttm_cashflow
    raw_balance_ttm   = t.quarterly_balancesheet

    def safe_row(df, row_name): 
        return df.loc[row_name] if df is not None and row_name in df.index else None

    # Historical lines
    hist_revenue    = safe_row(raw_income, "Total Revenue")
    hist_ebitda     = safe_row(raw_income, "EBITDA")
    hist_income     = safe_row(raw_income, "Net Income")
    hist_fcf        = safe_row(raw_cashflow, "Free Cash Flow")
    hist_ocf        = safe_row(raw_cashflow, "Operating Cash Flow")
    hist_net_debt   = safe_row(raw_cashflow, "Net Issuance Payments Of Debt")
    hist_capex      = safe_row(raw_cashflow, "Capital Expenditures") or safe_row(raw_cashflow, "Capital Expenditure")
    hist_debt       = safe_row(raw_balance, "Total Debt")
    hist_cash       = safe_row(raw_balance, "Cash")

    # Create unified index
    indices = [s.index for s in [hist_revenue, hist_ebitda, hist_income, hist_fcf, hist_ocf, hist_net_debt, hist_capex, hist_debt, hist_cash] if s is not None]
    if indices:
        common_index = reduce(lambda x, y: x.union(y), indices).sort_values()
    else:
        common_index = pd.Index([])

    def reindexed_or_empty(series):
        return series.reindex(common_index) if series is not None else pd.Series(np.nan, index=common_index)

    # Reindex
    df = pd.DataFrame({
        "Revenue": reindexed_or_empty(hist_revenue),
        "EBITDA": reindexed_or_empty(hist_ebitda),
        "Net Income": reindexed_or_empty(hist_income),
        "FCF": reindexed_or_empty(hist_fcf),
        "Total Debt": reindexed_or_empty(hist_debt),
        "Cash": reindexed_or_empty(hist_cash)
    })
    df.index.name = "Year"
    df = df.sort_index()

    s_ocf      = reindexed_or_empty(hist_ocf)
    s_net_debt = reindexed_or_empty(hist_net_debt)
    s_capex    = reindexed_or_empty(hist_capex)

    # Metrics with safe division
    df["FCF/EBITDA"] = np.where(
        (df["EBITDA"] != 0) & pd.notna(df["EBITDA"]) & pd.notna(df["FCF"]),
        df["FCF"] / df["EBITDA"],
        np.nan
    )
    mean_ratio = df["FCF/EBITDA"].mean()

    df["FCF_from_EBITDA"] = df["EBITDA"] * mean_ratio

    df["EBITDA/Revenue"] = np.where(
        (df["Revenue"] != 0) & pd.notna(df["Revenue"]) & pd.notna(df["EBITDA"]),
        df["EBITDA"] / df["Revenue"],
        np.nan
    )
    df["FCF/Revenue"] = np.where(
        (df["Revenue"] != 0) & pd.notna(df["Revenue"]) & pd.notna(df["FCF"]),
        df["FCF"] / df["Revenue"],
        np.nan
    )

    sp    = t.info.get("currentPrice", np.nan)
    sh    = t.info.get("sharesOutstanding", np.nan)
    mcap  = sp * sh if not np.isnan(sp) and not np.isnan(sh) else np.nan
    ev    = t.info.get("enterpriseValue", np.nan)

    df["P/FCF"] = np.where(
        (df["FCF"] != 0) & pd.notna(df["FCF"]) & pd.notna(mcap),
        mcap / df["FCF"],
        np.nan
    )
    df["EV/EBITDA"] = np.where(
        (df["EBITDA"] != 0) & pd.notna(df["EBITDA"]) & pd.notna(ev),
        ev / df["EBITDA"],
        np.nan
    )

    df["FCFE"] = s_ocf - (-s_capex) + (-s_net_debt)

    latest_debt = df["Total Debt"].dropna().iloc[-1] if not df["Total Debt"].dropna().empty else np.nan
    latest_cash = df["Cash"].dropna().iloc[-1] if not df["Cash"].dropna().empty else np.nan
    valuation_info = {
        "market_cap":       mcap,
        "share_price":      sp,
        "number_shares":    sh,
        "enterprise_value": ev,
        "total_debt":       latest_debt,
        "cash":             latest_cash,
        "net_debt":         (latest_debt - latest_cash) if pd.notna(latest_debt) and pd.notna(latest_cash) else np.nan
    }

    # Reinvestment metrics
    reinvest_df = calculate_historical_reinvestment_rate(raw_cashflow, raw_balance, raw_income)
    df.index = pd.to_datetime(df.index).normalize()
    reinvest_df.index = pd.to_datetime(reinvest_df.index).normalize()
    df = df.merge(reinvest_df, how="left", left_index=True, right_index=True)

    # ===== TTM Section =====
    def safe_value(df, row):
        if df is not None and row in df.index:
            val = df.loc[row]
            if isinstance(val, pd.Series):
                return val.iloc[0]
            return val
        return np.nan


    ttm_row = {
        "Revenue":     safe_value(raw_income_ttm, "Total Revenue"),
        "EBITDA":      safe_value(raw_income_ttm, "EBITDA"),
        "Net Income":  safe_value(raw_income_ttm, "Net Income"),
        "FCF":         safe_value(raw_cashflow_ttm, "Free Cash Flow"),
        "Total Debt":  safe_value(raw_balance_ttm, "Total Debt"),
        "Cash":        safe_value(raw_balance_ttm, "Cash"),
    }

    ttm_row["FCF/EBITDA"] = safe_divide(ttm_row["FCF"], ttm_row["EBITDA"])
    ttm_row["FCF_from_EBITDA"] = (
        ttm_row["EBITDA"] * mean_ratio if pd.notna(ttm_row["EBITDA"]) else np.nan
    )
    ttm_row["EBITDA/Revenue"] = safe_divide(ttm_row["EBITDA"], ttm_row["Revenue"])
    ttm_row["FCF/Revenue"] = safe_divide(ttm_row["FCF"], ttm_row["Revenue"])
    ttm_row["P/FCF"] = safe_divide(mcap, ttm_row["FCF"])
    ttm_row["EV/EBITDA"] = safe_divide(ev, ttm_row["EBITDA"])

    # Add TTM row at the end
    ttm_index = raw_income_ttm.columns[0]
    ttm_df = pd.DataFrame(ttm_row, index=[ttm_index])
    df['ttm'] = 0
    ttm_df['ttm'] = 1 
    full_df = pd.concat([df, ttm_df], axis=0)

    return full_df, valuation_info


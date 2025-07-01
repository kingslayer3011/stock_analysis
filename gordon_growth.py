import numpy as np
import pandas as pd
from scipy.stats import linregress
import yfinance as yf
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, Tuple, Dict, Any

# Simple cache for yfinance Ticker objects to improve performance
_yf_cache = {}
def get_yf_ticker(ticker: str) -> yf.Ticker:
    if ticker not in _yf_cache:
        _yf_cache[ticker] = yf.Ticker(ticker)
    return _yf_cache[ticker]

def estimate_growth_rate(series: pd.Series, method: str = "cagr", periods: int = 5) -> float:
    """Estimate growth rate from a pandas Series using CAGR, average, or regression."""
    series = pd.Series(series).dropna()
    if len(series) < 2:
        raise ValueError("Not enough data to estimate growth rate.")
    if periods is not None and len(series) > periods:
        series = series.iloc[-periods:]
    if method == "cagr":
        start, end = series.iloc[0], series.iloc[-1]
        n = len(series) - 1
        if start <= 0 or end <= 0:
            raise ValueError("CAGR requires all positive values.")
        return (end / start) ** (1 / n) - 1
    elif method == "average":
        return series.pct_change().dropna().mean()
    elif method == "regression":
        # Regression on log values to estimate average growth rate
        y = np.log(series.values)
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return np.exp(slope) - 1
    else:
        raise ValueError("Unknown method for growth rate.")

def estimate_required_return(
    ticker: str,
    risk_free_rate: float = 0.04,
    market_return: float = 0.08,
    beta: Optional[float] = None,
    use_historical_beta: bool = False,
    market_ticker: str = "^GSPC",
    period: str = "5y"
) -> float:
    """Estimate required return using CAPM: r = rf + beta * (rm - rf)"""
    if beta is None:
        if use_historical_beta:
            beta = calculate_historical_beta(ticker, market_ticker, period)
        if beta is None:
            try:
                stock = get_yf_ticker(ticker)
                beta = stock.info.get("beta", None)
            except Exception:
                beta = None
    if beta is None or not np.isfinite(beta):
        beta = 1.0  # fallback to market average
    return risk_free_rate + beta * (market_return - risk_free_rate)

def gordon_growth_model(
    series: pd.Series,
    required_return: Optional[float] = None,
    growth_rate: Optional[float] = None,
    long_term_growth_rate: float = 0.02,
    method: str = "cagr",
    periods: int = 5,
    ticker: Optional[str] = None,
    risk_free_rate: float = 0.025,
    market_return: float = 0.07,
    use_historical_beta: bool = False,
    market_ticker: str = "^GSPC",
    beta_period: str = "5y",
    use_analyst_D1: bool = False
) -> Dict[str, Any]:
    """
    Flexible Gordon Growth Model.
    - series: pandas Series of dividends (index: years or dates)
    - required_return: if None, estimate using CAPM
    - growth_rate: if None, estimate from series
    - method: 'cagr', 'average', or 'regression' for growth rate
    - periods: how many periods to use for estimation
    - ticker: for CAPM beta lookup
    - use_analyst_D1: if True, use analyst forward dividend as D1 (next dividend)
    """
    series = series.dropna()
    if len(series) == 0:
        raise ValueError("No dividend data available for valuation.")

    # Estimate growth rate if not provided
    if growth_rate is None:
        growth_rate = estimate_growth_rate(series, method=method, periods=periods)

    # Estimate required return and beta if not provided
    beta_used = None
    if required_return is None:
        if ticker is None:
            raise ValueError("Ticker required to estimate required return.")
        if use_historical_beta:
            beta_used = calculate_historical_beta(ticker, market_ticker, beta_period)
        else:
            try:
                stock = get_yf_ticker(ticker)
                beta_used = stock.info.get("beta", None)
            except Exception:
                beta_used = None
        if beta_used is None or not np.isfinite(beta_used):
            beta_used = 1.0
        required_return = risk_free_rate + beta_used * (market_return - risk_free_rate)
    else:
        if ticker is not None:
            try:
                stock = get_yf_ticker(ticker)
                beta_used = stock.info.get("beta", None)
            except Exception:
                beta_used = None
        if beta_used is None or not np.isfinite(beta_used):
            beta_used = 1.0

    # Use analyst estimate for D1 if requested, else project from history
    if use_analyst_D1:
        stock = get_yf_ticker(ticker)
        analyst_forward_div = stock.info.get("dividendRate", None)
        if analyst_forward_div is not None:
            D1 = analyst_forward_div
        else:
            raise ValueError("Analyst forward dividend estimate not available.")
    else:
        D1 = series.iloc[-1] * (1 + growth_rate)

    if required_return <= long_term_growth_rate:
        raise ValueError(f"Required return ({required_return:.2%}) must be greater than long-term growth rate ({long_term_growth_rate:.2%}).")

    value = D1 / (required_return - long_term_growth_rate)
    return {
        "value": value,
        "D1": D1,
        "required_return": required_return,
        "growth_rate": growth_rate,
        "last_dividend": series.iloc[-1],
        "method": method,
        "periods": periods,
        "beta": beta_used
    }

def calculate_historical_beta(ticker: str, market_ticker: str = "^GSPC", period: str = "5y") -> Optional[float]:
    """
    Calculate beta using historical price data.
    ticker: stock ticker (e.g., 'AAPL')
    market_ticker: market index ticker (default: S&P 500)
    period: period for historical data (e.g., '5y')
    """
    try:
        stock = get_yf_ticker(ticker).history(period=period)["Close"].pct_change().dropna()
        market = get_yf_ticker(market_ticker).history(period=period)["Close"].pct_change().dropna()
        df = pd.concat([stock, market], axis=1, keys=["stock", "market"]).dropna()
        if len(df) < 2:
            return None
        slope, *_ = linregress(df["market"], df["stock"])
        return slope
    except Exception:
        return None

def fetch_dividends(ticker: str, periods: Optional[int] = None) -> pd.Series:
    """Fetch dividend history for a ticker using yfinance and annualize the current year if incomplete."""
    stock = get_yf_ticker(ticker)
    div = stock.dividends
    if div.empty:
        raise ValueError(f"No dividend data found for {ticker}")
    div = div.resample("YE").sum()  # annualize

    # Annualize the current year if incomplete
    if not div.empty:
        last_idx = div.index[-1]
        year = last_idx.year
        today = pd.Timestamp.today()
        if year == today.year and today.month < 12:
            # Get all dividends for the current year
            divs_this_year = stock.dividends[stock.dividends.index.year == year]
            count_this_year = len(divs_this_year)
            # Get previous year's dividend count
            prev_year = year - 1
            divs_prev_year = stock.dividends[stock.dividends.index.year == prev_year]
            count_prev_year = len(divs_prev_year)
            if count_prev_year > 0 and count_this_year < count_prev_year:
                # Assume last paid dividend is repeated for missing periods
                last_div = divs_this_year.iloc[-1] if not divs_this_year.empty else 0
                estimated_total = divs_this_year.sum() + last_div * (count_prev_year - count_this_year)
                div.at[last_idx] = estimated_total

    if periods is not None and len(div) > periods:
        div = div.iloc[-periods:]
    return div

def plot_dividend_history(series: pd.Series, ticker: str, output_folder: str, show: bool = False) -> None:
    """Plot and save the dividend history for a ticker. Optionally show the plot."""
    plt.figure(figsize=(8,4))
    plt.plot(series.index.year, series.values, marker='o')
    plt.title(f"Dividend History for {ticker}")
    plt.xlabel("Year")
    plt.ylabel("Dividend")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "dividend_history.png"))
    if show:
        plt.show()
    plt.close()

def plot_dividend_growth(series: pd.Series, ticker: str, output_folder: str, show: bool = False) -> None:
    """Plot and save the dividend growth rate for a ticker. Optionally show the plot."""
    growth = series.pct_change().dropna()
    plt.figure(figsize=(8,4))
    plt.bar(growth.index.year, growth.values*100)
    plt.title(f"Dividend Growth Rate for {ticker}")
    plt.xlabel("Year")
    plt.ylabel("Growth Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "dividend_growth.png"))
    if show:
        plt.show()
    plt.close()

def plot_ggm_sensitivity(D1: float, required_return: float, growth_rate: float, output_folder: str, show: bool = False) -> None:
    """Plot and save a heatmap of GGM value sensitivity to required return and growth rate. Optionally show the plot."""
    r_vals = np.linspace(required_return-0.03, required_return+0.03, 50)
    g_vals = np.linspace(growth_rate-0.02, growth_rate+0.02, 50)
    values = np.zeros((len(g_vals), len(r_vals)))
    for i, g in enumerate(g_vals):
        for j, r in enumerate(r_vals):
            values[i, j] = D1 / (r - g) if r > g else np.nan
    plt.figure(figsize=(8,6))
    sns.heatmap(values, xticklabels=np.round(r_vals,3), yticklabels=np.round(g_vals,3), cmap="viridis")
    plt.title("GGM Value Sensitivity")
    plt.xlabel("Required Return (r)")
    plt.ylabel("Growth Rate (g)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "ggm_sensitivity.png"))
    if show:
        plt.show()
    plt.close()

def plot_regression_fit(series: pd.Series, ticker: str, output_folder: str, show: bool = False) -> None:
    """Plot and save the regression fit of log dividends. Optionally show the plot."""
    y = np.log(series.values)
    x = np.arange(len(series))
    slope, intercept, *_ = linregress(x, y)
    plt.figure(figsize=(8,4))
    plt.plot(series.index.year, y, 'o', label='Log(Dividend)')
    plt.plot(series.index.year, intercept + slope*x, '-', label='Regression Line')
    plt.title(f"Log Dividend Regression for {ticker}")
    plt.xlabel("Year")
    plt.ylabel("Log(Dividend)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "regression_fit.png"))
    if show:
        plt.show()
    plt.close()

def multi_stage_gordon_growth_model(
    series: pd.Series,
    required_return: Optional[float] = None,
    high_growth_rate: Optional[float] = None,
    high_growth_years: int = 5,
    long_term_growth_rate: float = 0.02,
    ticker: Optional[str] = None,
    risk_free_rate: float = 0.025,
    market_return: float = 0.07,
    use_historical_beta: bool = False,
    market_ticker: str = "^GSPC",
    beta_period: str = "5y",
    use_analyst_D1: bool = False,
    method: str = "cagr",
    periods: int = 5
) -> Dict[str, Any]:
    """
    Two-stage Gordon Growth Model.
    - high_growth_rate: if None, estimate from dividend history using estimate_growth_rate
    - method: method for growth rate estimation ('cagr', 'average', 'regression')
    - periods: number of periods to use for growth estimation
    """
    series = series.dropna()
    if len(series) == 0:
        raise ValueError("No dividend data available for valuation.")

    # Estimate high growth rate if not provided
    if high_growth_rate is None:
        high_growth_rate = estimate_growth_rate(series, method=method, periods=periods)

    # Estimate required return and beta if not provided
    beta_used = None
    if required_return is None:
        if ticker is None:
            raise ValueError("Ticker required to estimate required return.")
        if use_historical_beta:
            beta_used = calculate_historical_beta(ticker, market_ticker, beta_period)
        else:
            try:
                stock = get_yf_ticker(ticker)
                beta_used = stock.info.get("beta", None)
            except Exception:
                beta_used = None
        if beta_used is None or not np.isfinite(beta_used):
            beta_used = 1.0
        required_return = risk_free_rate + beta_used * (market_return - risk_free_rate)
    else:
        if ticker is not None:
            try:
                stock = get_yf_ticker(ticker)
                beta_used = stock.info.get("beta", None)
            except Exception:
                beta_used = None
        if beta_used is None or not np.isfinite(beta_used):
            beta_used = 1.0

    # Always use the last historical dividend as the base
    dividends = []
    prev_div = series.iloc[-1]
    for n in range(1, high_growth_years + 1):
        g = high_growth_rate + (long_term_growth_rate - high_growth_rate) * (n - 1) / (high_growth_years - 1) if high_growth_years > 1 else high_growth_rate
        next_div = prev_div * (1 + g)
        dividends.append(next_div)
        prev_div = next_div

    D1 = dividends[0]  # First projected dividend

    # Calculate the terminal value at the end of the high growth period
    terminal_dividend = dividends[-1] * (1 + long_term_growth_rate)
    if required_return <= long_term_growth_rate:
        raise ValueError(f"Required return ({required_return:.2%}) must be greater than long-term growth rate ({long_term_growth_rate:.2%}).")
    terminal_value = terminal_dividend / (required_return - long_term_growth_rate)

    # Discount all dividends and the terminal value to present value
    value = 0
    for n, div in enumerate(dividends, 1):
        value += div / (1 + required_return) ** n
    value += terminal_value / (1 + required_return) ** high_growth_years

    return {
        "value": value,
        "D1": D1,
        "required_return": required_return,
        "growth_rate": high_growth_rate,
        "high_growth_rate": high_growth_rate,
        "high_growth_years": high_growth_years,
        "long_term_growth_rate": long_term_growth_rate,
        "last_dividend": series.iloc[-1],
        "method": "multi-stage",
        "periods": periods,
        "beta": beta_used,
        "projected_dividends": dividends
    }

def plot_dividend_forecast(series: pd.Series, forecasted_dividends: list, high_growth_years: int, ticker: str, output_folder: str, show: bool = False) -> None:
    """Plot historical and forecasted dividends for visual comparison. Optionally show the plot."""
    plt.figure(figsize=(10,5))
    # Plot historical dividends
    plt.plot(series.index.year, series.values, marker='o', label='Historical Dividends')
    # Forecast years
    last_year = series.index.year[-1]
    forecast_years = [last_year + i for i in range(1, high_growth_years + 1)]
    plt.plot(forecast_years, forecasted_dividends, marker='o', linestyle='--', color='orange', label='Forecasted Dividends')
    plt.title(f"Historical and Forecasted Dividends for {ticker}")
    plt.xlabel("Year")
    plt.ylabel("Dividend")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "dividend_forecast.png"))
    if show:
        plt.show()
    plt.close()

def save_projected_dividends_csv(series: pd.Series, forecasted_dividends: list, high_growth_years: int, output_folder: str) -> None:
    """Save historical and projected dividends to a CSV file."""
    import pandas as pd
    last_year = series.index.year[-1]
    forecast_years = [last_year + i for i in range(1, high_growth_years + 1)]
    # Historical
    hist_df = pd.DataFrame({
        "Year": series.index.year,
        "Dividend": series.values,
        "Type": "Historical"
    })
    # Forecasted
    proj_df = pd.DataFrame({
        "Year": forecast_years,
        "Dividend": forecasted_dividends,
        "Type": "Projected"
    })
    out_df = pd.concat([hist_df, proj_df], ignore_index=True)
    out_df.to_csv(os.path.join(output_folder, "projected_dividends.csv"), index=False)

def plot_required_return_breakdown(risk_free_rate: float, market_return: float, beta: float, required_return: float, ticker: str, output_folder: str, show: bool = False) -> None:
    """
    Visualize the required return calculation using CAPM.
    Args:
        risk_free_rate: The risk-free rate used in CAPM.
        market_return: The expected market return.
        beta: The beta value used.
        required_return: The total required return (cost of equity).
        ticker: The stock ticker.
        output_folder: Where to save the plot.
        show: If True, display the plot interactively.
    """
    components = [
        ("Risk-Free Rate", risk_free_rate),
        ("Beta × (Market Return - Risk-Free Rate)", beta * (market_return - risk_free_rate))
    ]
    values = [c[1] for c in components]
    labels = [c[0] for c in components]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=["#4F81BD", "#C0504D"])
    plt.title(f"Required Return Breakdown for {ticker}")
    plt.ylabel("Return")
    plt.ylim(0, max(required_return * 1.2, 0.1))
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2%}", ha='center', va='bottom')
    plt.axhline(required_return, color="green", linestyle="--", label=f"Total Required Return: {required_return:.2%}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "required_return_breakdown.png"))
    if show:
        plt.show()
    plt.close()

def plot_beta_regression(ticker: str, market_ticker: str, period: str, output_folder: str, show: bool = False) -> None:
    """
    Plot the regression of stock returns vs. market returns to visualize beta.
    Args:
        ticker: Stock ticker.
        market_ticker: Market index ticker.
        period: Period for historical data.
        output_folder: Where to save the plot.
        show: If True, display the plot interactively.
    """
    try:
        stock = get_yf_ticker(ticker).history(period=period)["Close"].pct_change().dropna()
        market = get_yf_ticker(market_ticker).history(period=period)["Close"].pct_change().dropna()
        df = pd.concat([stock, market], axis=1, keys=["stock", "market"]).dropna()
        if len(df) < 2:
            print("Not enough data to plot beta regression.")
            return
        slope, intercept, r_value, p_value, std_err = linregress(df["market"], df["stock"])
        plt.figure(figsize=(7, 5))
        plt.scatter(df["market"], df["stock"], alpha=0.5, label="Daily Returns")
        plt.plot(df["market"], intercept + slope * df["market"], color="red", label=f"Regression Line (Beta={slope:.2f})")
        plt.title(f"Beta Regression: {ticker} vs {market_ticker}")
        plt.xlabel(f"{market_ticker} Returns")
        plt.ylabel(f"{ticker} Returns")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "beta_regression.png"))
        if show:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"Could not plot beta regression: {e}")

def fetch_roe(ticker: str) -> Optional[float]:
    """Fetch Return on Equity (ROE) for the given ticker using yfinance."""
    try:
        stock = get_yf_ticker(ticker)
        roe = stock.info.get("returnOnEquity", None)
        if roe is not None and np.isfinite(roe):
            return roe
    except Exception:
        pass
    return None

def fetch_debt_ratios(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """Fetch Debt/Equity and Interest Coverage ratios using yfinance."""
    try:
        stock = get_yf_ticker(ticker)
        info = stock.info
        # Debt/Equity
        debt_to_equity = info.get("debtToEquity", None)
        if debt_to_equity is not None and np.isfinite(debt_to_equity):
            debt_to_equity = float(debt_to_equity)
        else:
            debt_to_equity = None
        # Interest Coverage: EBIT / Interest Expense
        ebit = info.get("ebit", None)
        interest_expense = info.get("interestExpense", None)
        if ebit is not None and interest_expense is not None and interest_expense != 0:
            interest_coverage = ebit / abs(interest_expense)
        else:
            interest_coverage = None
        return debt_to_equity, interest_coverage
    except Exception:
        return None, None

def format_financial_ratios(roe: Optional[float], required_return: float, debt_to_equity: Optional[float], interest_coverage: Optional[float]) -> Tuple[str, str]:
    """
    Helper to format financial ratios and warnings for output.
    Returns (output_str, warning_str)
    """
    roe_str = f"{roe:.2%}" if roe is not None else "N/A"
    roe_warning = ""
    if roe is not None and roe < required_return:
        roe_warning = "WARNING: ROE is less than the required return (cost of equity). Long-term growth may be questionable.\n"
    elif roe is None:
        roe_warning = "ROE data not available.\n"
    debt_str = f"{debt_to_equity:.2f}" if debt_to_equity is not None else "N/A"
    ic_str = f"{interest_coverage:.2f}" if interest_coverage is not None else "N/A"
    debt_warning = ""
    if debt_to_equity is not None and debt_to_equity > 2:
        debt_warning += "WARNING: Debt/Equity ratio is high. Financial risk may be elevated.\n"
    if interest_coverage is not None and interest_coverage < 2:
        debt_warning += "WARNING: Interest coverage is low. Company may have trouble servicing debt.\n"
    if debt_to_equity is None and interest_coverage is None:
        debt_warning = "Debt/Equity and Interest Coverage data not available.\n"
    output = (
        f"Return on Equity (ROE): {roe_str}\n"
        f"Cost of Equity (Required Return): {required_return:.2%}\n"
        f"Debt/Equity Ratio: {debt_str}\n"
        f"Interest Coverage Ratio: {ic_str}\n"
    )
    warnings = roe_warning + debt_warning
    return output, warnings

def main():
    parser = argparse.ArgumentParser(description="Gordon Growth Model Valuation (Dividend Discount Model)")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--method", choices=["cagr", "average", "regression"], default="cagr", help="Growth rate estimation method")
    parser.add_argument("--periods", type=int, default=5, help="Number of periods to use for growth estimation")
    parser.add_argument("--risk_free_rate", type=float, default=0.04, help="Risk-free rate (e.g., 0.04 for 4%)")
    parser.add_argument("--market_return", type=float, default=0.08, help="Market return (e.g., 0.08 for 8%)")
    parser.add_argument("--use_historical_beta", action="store_true", help="Estimate beta from historical price data")
    parser.add_argument("--market_ticker", default="^GSPC", help="Market index ticker for beta calculation")
    parser.add_argument("--beta_period", default="5y", help="Period for historical beta calculation (e.g., '5y')")
    parser.add_argument("--use_analyst_d1", action="store_true", help="Use analyst forward dividend as D1 in GGM")
    parser.add_argument("--multi_stage", action="store_true", help="Use multi-stage GGM")
    parser.add_argument("--high_growth_rate", type=float, default=None, help="High growth rate for first stage (e.g., 0.05 for 5%)")
    parser.add_argument("--high_growth_years", type=int, default=5, help="Number of years for high growth stage")
    parser.add_argument("--long_term_growth_rate", type=float, default=0.02, help="Long-term perpetual growth rate (e.g., 0.02 for 2%)")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file for CLI defaults")
    args = parser.parse_args()

    # Load CLI defaults from config file if provided
    if args.config:
        import json
        with open(args.config, "r") as f:
            config = json.load(f)
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    # Create output folder
    output_folder = f"div_model_{args.ticker}"
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Early check for dividend history
        try:
            div_series = fetch_dividends(args.ticker, periods=args.periods)
        except ValueError as ve:
            print(f"Error: {ve}")
            return

        # Fetch analyst forward dividend for comparison
        stock = get_yf_ticker(args.ticker)
        analyst_forward_div = stock.info.get("dividendRate", None)

        if args.multi_stage:
            result = multi_stage_gordon_growth_model(
                div_series,
                ticker=args.ticker,
                required_return=None,
                high_growth_rate=args.high_growth_rate,  # Always estimate unless overridden
                high_growth_years=args.high_growth_years,
                long_term_growth_rate=args.long_term_growth_rate,
                risk_free_rate=args.risk_free_rate,
                market_return=args.market_return,
                use_historical_beta=args.use_historical_beta,
                market_ticker=args.market_ticker,
                beta_period=args.beta_period,
                method=args.method,
                periods=args.periods
            )
        else:
            result = gordon_growth_model(
                div_series,
                method=args.method,
                periods=args.periods,
                ticker=args.ticker,
                risk_free_rate=args.risk_free_rate,
                market_return=args.market_return,
                use_historical_beta=args.use_historical_beta,
                market_ticker=args.market_ticker,
                beta_period=args.beta_period,
                use_analyst_D1=args.use_analyst_d1
            )

        # Fetch current price
        current_price = None
        try:
            current_price = stock.history(period="1d")["Close"].iloc[-1]
        except Exception:
            pass

        # Calculate upside
        upside = None
        if current_price is not None and current_price > 0:
            upside = 100 * (result['value'] - current_price) / current_price

        # Save results summary
        summary_path = os.path.join(output_folder, "ggm_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Gordon Growth Model Value for {args.ticker}: {result['value']:.2f}\n")
            f.write(f"D1 (next dividend): {result['D1']:.2f}\n")
            f.write(f"Required Return: {result['required_return']:.4f}\n")
            f.write(f"Growth Rate: {result['growth_rate']:.4f}\n")
            f.write(f"Last Dividend: {result['last_dividend']:.2f}\n")
            f.write(f"Method: {result['method']}, Periods: {result['periods']}\n")
            if current_price is not None:
                f.write(f"Current Price: {current_price:.2f}\n")
            if upside is not None:
                f.write(f"Potential Upside: {upside:.2f}%\n")
            if args.use_historical_beta:
                f.write("Beta estimated from historical price data.\n")

        comparison_str = ""
        if analyst_forward_div is not None:
            comparison_str = (
                f"Analyst Forward Dividend: {analyst_forward_div:.2f}\n"
                f"Your Projected D1: {result['D1']:.2f}\n"
                f"Difference: {result['D1'] - analyst_forward_div:.2f} "
                f"({100 * (result['D1'] - analyst_forward_div) / analyst_forward_div:.2f}%)\n"
            )
        else:
            comparison_str = "Analyst forward dividend estimate not available.\n"

        with open(summary_path, "a") as f:
            f.write(comparison_str)

        # Fetch and format financial ratios
        roe = fetch_roe(args.ticker)
        debt_to_equity, interest_coverage = fetch_debt_ratios(args.ticker)
        ratios_str, warnings_str = format_financial_ratios(roe, result["required_return"], debt_to_equity, interest_coverage)
        with open(summary_path, "a") as f:
            f.write(ratios_str)
            if warnings_str:
                f.write(warnings_str)

        # Print summary to console
        print(f"\nGordon Growth Model Value for {args.ticker}: {result['value']:.2f}")
        print(f"  D1 (next dividend): {result['D1']:.2f}")
        print(f"  Required Return: {result['required_return']:.2%}")
        print(f"  Growth Rate: {result['growth_rate']:.2%}")
        print(f"  Last Dividend: {result['last_dividend']:.2f}")
        print(f"  Method: {result['method']}, Periods: {result['periods']}")
        if current_price is not None:
            print(f"  Current Price: {current_price:.2f}")
        if upside is not None:
            print(f"  Potential Upside: {upside:.2f}%")
        if args.use_historical_beta:
            print("  Beta estimated from historical price data.")
        print(ratios_str, end="")
        if warnings_str:
            print(warnings_str, end="")

        # Save dividend history
        div_series.to_csv(os.path.join(output_folder, "dividend_history.csv"))

        # Plot and save figures (do not show by default)
        plot_dividend_history(div_series, args.ticker, output_folder)
        plot_dividend_growth(div_series, args.ticker, output_folder)
        plot_ggm_sensitivity(result['D1'], result['required_return'], result['growth_rate'], output_folder)
        plot_regression_fit(div_series, args.ticker, output_folder)

        # Plot historical + forecasted dividends for multi-stage model
        if args.multi_stage:
            forecasted_dividends = result["projected_dividends"]
            plot_dividend_forecast(div_series, forecasted_dividends, args.high_growth_years, args.ticker, output_folder)
            save_projected_dividends_csv(div_series, forecasted_dividends, args.high_growth_years, output_folder)

        # Visualize required return calculation
        beta_val = result["beta"] if result["beta"] is not None else 1.0
        plot_required_return_breakdown(
            args.risk_free_rate,
            args.market_return,
            beta_val,
            result["required_return"],
            args.ticker,
            output_folder
        )

        # Visualize beta regression if using historical beta
        if args.use_historical_beta:
            plot_beta_regression(
                args.ticker,
                args.market_ticker,
                args.beta_period,
                output_folder
            )

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
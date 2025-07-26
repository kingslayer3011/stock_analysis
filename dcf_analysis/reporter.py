# reporter.py

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os

def plot_valuation_summary(valuation_info):
    plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    summary_txt = (
        f"Market Cap: {valuation_info.get('market_cap', float('nan')):,.0f}\n"
        f"Enterprise Value: {valuation_info.get('enterprise_value', float('nan')):,.0f}\n"
        f"Total Debt: {valuation_info.get('total_debt', float('nan')):,.0f}\n"
        f"Cash: {valuation_info.get('cash', float('nan')):,.0f}\n"
        f"Net Debt: {valuation_info.get('net_debt', float('nan')):,.0f}\n\n"
        f"Sum of PV(FCF+TV): {valuation_info.get('sum_pv', float('nan')):,.0f}\n"
        f"Terminal Value: {valuation_info.get('terminal_value', float('nan')):,.0f}\n"
        f"NPV: {valuation_info.get('npv', float('nan')):,.0f}\n"
        f"Implied Per Share Value: {valuation_info.get('per_share_value', float('nan')):,.2f}\n"
    )
    plt.text(0.1, 0.5, summary_txt, ha="left", va="center", fontsize=12)
    return plt

def plot_historical_table(df_hist):
    plt.figure(figsize=(11, 8.5))
    plt.axis("off")
    tbl = plt.table(
        cellText=df_hist.round(2).fillna("").values,
        colLabels=df_hist.columns,
        rowLabels=[str(idx.year) for idx in df_hist.index],
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    return plt

def plot_dcf_table(dcf_results):
    plt.figure(figsize=(11, 8.5))
    plt.axis("off")
    tbl2 = plt.table(
        cellText=dcf_results.round(2).fillna("").values,
        colLabels=dcf_results.columns,
        rowLabels=[str(int(x)) for x in dcf_results["Year"].to_list()] if "Year" in dcf_results.columns else [],
        cellLoc="center",
        loc="center"
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(8)
    return plt

def plot_sensitivity_table(df_sens):
    plt.figure(figsize=(11, 8.5))
    plt.axis("off")
    tbl = plt.table(
        cellText=df_sens.values,
        colLabels=df_sens.columns.astype(str),
        rowLabels=df_sens.index.astype(str),
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    return plt

def generate_pdf_report(
    ticker: str,
    df_hist: pd.DataFrame,
    dcf_results: pd.DataFrame,
    valuation_info: dict,
    df_sens: pd.DataFrame,
    df_sens_5y: pd.DataFrame,
    output_folder: str
):
    pdf_name = f"{ticker}_DCF_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(output_folder, pdf_name)
    with PdfPages(pdf_path) as pdf:
        # Title Page
        plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        title_text = f"{ticker} Discounted Cash Flow Analysis\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        plt.text(0.5, 0.5, title_text, ha="center", va="center", fontsize=20)
        pdf.savefig()
        plt.close()
        # Valuation Summary
        plot_valuation_summary(valuation_info)
        pdf.savefig()
        plt.close()
        # Historical Data Table
        plot_historical_table(df_hist)
        pdf.savefig()
        plt.close()
        # DCF Table
        plot_dcf_table(dcf_results)
        pdf.savefig()
        plt.close()
        # Sensitivity #1 Table (if present)
        if df_sens is not None:
            plot_sensitivity_table(df_sens)
            pdf.savefig()
            plt.close()
        # Sensitivity #2 Table (if present)
        if df_sens_5y is not None:
            plot_sensitivity_table(df_sens_5y)
            pdf.savefig()
            plt.close()
    print(f"PDF report saved as: '{pdf_path}'")
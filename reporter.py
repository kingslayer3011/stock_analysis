# reporter.py

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os


def generate_pdf_report(
    ticker: str,
    df_hist: pd.DataFrame,
    dcf_results: pd.DataFrame,
    valuation_info: dict,
    df_sens: pd.DataFrame,
    df_sens_5y: pd.DataFrame,
    output_folder: str
):
    """
    Generates a multipage PDF report and saves it into `output_folder`.

    Pages include:
      1) Title page
      2) Final summary (with multiples and DCF valuation)
      3) Historical tables (Income, Cash Flow, etc.)
      4) DCF projection table
      5) Sensitivity tables
    """

    pdf_name = f"{ticker}_DCF_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(output_folder, pdf_name)

    with PdfPages(pdf_path) as pdf:
        # --- Page 1: Title Page ---
        plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        title_text = f"{ticker} Discounted Cash Flow Analysis\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        plt.text(0.5, 0.5, title_text, ha="center", va="center", fontsize=20)
        pdf.savefig()
        plt.close()

        #! move to data_fetcher.py
        # --- Page 2: Valuation Summary ---
        plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        summary_txt = (
            f"Market Cap: {valuation_info.get('market_cap', np.nan):,.0f}\n"
            f"Enterprise Value: {valuation_info.get('enterprise_value', np.nan):,.0f}\n"
            f"Total Debt: {valuation_info.get('total_debt', np.nan):,.0f}\n"
            f"Cash: {valuation_info.get('cash', np.nan):,.0f}\n"
            f"Net Debt: {valuation_info.get('net_debt', np.nan):,.0f}\n\n"
            f"Sum of PV(FCF+TV): {valuation_info.get('sum_pv', np.nan):,.0f}\n"
            f"Terminal Value: {valuation_info.get('terminal_value', np.nan):,.0f}\n"
            f"NPV: {valuation_info.get('npv', np.nan):,.0f}\n"
            f"Implied Per Share Value: {valuation_info.get('per_share_value', np.nan):,.2f}\n"
        )
        plt.text(0.1, 0.5, summary_txt, ha="left", va="center", fontsize=12)
        pdf.savefig()
        plt.close()

        #! move to data_fetcher.py
        # --- Page 3: Historical Data Table ---
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
        pdf.savefig()
        plt.close()

        #! move to dcf.py
        # --- Page 4: DCF Projections Table ---
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
        pdf.savefig()
        plt.close()

        #! move to new separate module for sensitivity analysis
        # --- Page 5: Sensitivity #1 Table (if present) ---
        if df_sens is not None:
            plt.figure(figsize=(11, 8.5))
            plt.axis("off")
            tbl3 = plt.table(
                cellText=df_sens.values,
                colLabels=df_sens.columns.astype(str),
                rowLabels=df_sens.index.astype(str),
                cellLoc="center",
                loc="center"
            )
            tbl3.auto_set_font_size(False)
            tbl3.set_fontsize(8)
            tbl3.scale(1, 1.5)
            pdf.savefig()
            plt.close()

        #! move to new separate module for sensitivity analysis
        # --- Page 6: Sensitivity #2 Table (if present) ---
        if df_sens_5y is not None:
            plt.figure(figsize=(11, 8.5))
            plt.axis("off")
            tbl4 = plt.table(
                cellText=df_sens_5y.values,
                colLabels=df_sens_5y.columns.astype(str),
                rowLabels=df_sens_5y.index.astype(str),
                cellLoc="center",
                loc="center"
            )
            tbl4.auto_set_font_size(False)
            tbl4.set_fontsize(8)
            tbl4.scale(1, 1.5)
            pdf.savefig()
            plt.close()

    print(f"PDF report saved as: '{pdf_path}'")

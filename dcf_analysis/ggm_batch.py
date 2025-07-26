import subprocess
import argparse
import os
import sys
import csv

def run_ggm_for_ticker(ticker, ggm_args=None, results_folder="ggm_results", country=None):
    """
    Run gordon_growth.py for a single ticker and return the summary file path and value.
    Results are copied to the results_folder/country if country is provided.
    """
    import shutil
    script = os.path.join(os.path.dirname(__file__), 'gordon_growth.py')
    cmd = [sys.executable, script, '--ticker', ticker]
    if ggm_args:
        cmd += ggm_args
    try:
        subprocess.run(cmd, check=True)
        output_folder = f"div_model_{ticker}"
        summary_path = os.path.join(output_folder, "ggm_summary.txt")
        value = None
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                for line in f:
                    if line.lower().startswith('gordon growth model value'):
                        value = line.strip().split(':')[-1].strip()
                        break
        # Determine country subfolder
        if country:
            results_folder = os.path.join(results_folder, country)
        os.makedirs(results_folder, exist_ok=True)
        result_summary_path = os.path.join(results_folder, f"{ticker}_ggm_summary.txt")
        if os.path.exists(summary_path):
            shutil.copy2(summary_path, result_summary_path)
        else:
            result_summary_path = ''
        # Copy the entire div_model_{ticker} folder into ggm_results/country
        dest_model_folder = os.path.join(results_folder, output_folder)
        if os.path.exists(output_folder):
            if os.path.exists(dest_model_folder):
                shutil.rmtree(dest_model_folder)
            shutil.copytree(output_folder, dest_model_folder)
            # Remove the original folder after copying
            shutil.rmtree(output_folder)
        return result_summary_path, value
    except Exception as e:
        print(f"Error running GGM for {ticker}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Batch run Gordon Growth Model for multiple tickers.")
    parser.add_argument('--tickers', nargs='+', help='List of tickers to process')
    parser.add_argument('--ticker_file', help='Path to a file with tickers (one per line)')
    parser.add_argument('--ggm_args', nargs=argparse.REMAINDER, help='Additional arguments to pass to gordon_growth.py')
    parser.add_argument('--output_csv', default='ggm_batch_results.csv', help='CSV file to save summary results')
    parser.add_argument('--country', help='Country name for organizing results in subfolders')
    args = parser.parse_args()

    tickers = set()
    if args.tickers:
        tickers.update(args.tickers)
    if args.ticker_file:
        with open(args.ticker_file, 'r') as f:
            for line in f:
                t = line.strip()
                if t:
                    tickers.add(t)
    if not tickers:
        print("No tickers provided.")
        return

    results = []
    for ticker in sorted(tickers):
        print(f"\nRunning GGM for {ticker}...")
        summary_path, value = run_ggm_for_ticker(ticker, args.ggm_args, country=args.country)
        # Try to extract upside, company name, and high_growth_rate from the summary file
        upside = ''
        company_name = ''
        high_growth_rate = ''
        industry = ''
        if summary_path and os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                for line in f:
                    if 'Potential Upside:' in line:
                        try:
                            upside = line.split('Potential Upside:')[1].strip().replace('%','')
                        except Exception:
                            upside = ''
                    if line.lower().startswith('gordon growth model value for'):
                        pass
                    if 'Growth Rate:' in line or 'High Growth Rate:' in line:
                        try:
                            val = line.split(':')[1].strip()
                            if val.replace('.','',1).replace('-','',1).isdigit() or '%' in val:
                                high_growth_rate = val
                        except Exception:
                            high_growth_rate = ''
            # Try to fetch company name and industry from yfinance if not present in summary
            if not company_name or not industry:
                try:
                    import yfinance as yf
                    stock = yf.Ticker(ticker)
                    if not company_name:
                        company_name = stock.info.get('longName', '') or stock.info.get('shortName', '') or ''
                    industry = stock.info.get('industry', '') or ''
                except Exception:
                    if not company_name:
                        company_name = ''
                    industry = ''
            # Move div_model_{ticker} folder to upside/no upside subfolder
            try:
                if args.country:
                    country_folder = os.path.join('ggm_results', args.country)
                    upside_folder = os.path.join(country_folder, 'upside')
                    no_upside_folder = os.path.join(country_folder, 'no upside')
                    os.makedirs(upside_folder, exist_ok=True)
                    os.makedirs(no_upside_folder, exist_ok=True)
                    model_folder = os.path.join(country_folder, f'div_model_{ticker}')
                    # Determine if upside is positive
                    try:
                        upside_val = float(upside)
                    except Exception:
                        upside_val = 0
                    if os.path.exists(model_folder):
                        if upside_val > 0:
                            dest_folder = os.path.join(upside_folder, f'div_model_{ticker}')
                        else:
                            dest_folder = os.path.join(no_upside_folder, f'div_model_{ticker}')
                        import shutil
                        if os.path.exists(dest_folder):
                            shutil.rmtree(dest_folder)
                        shutil.move(model_folder, dest_folder)
            except Exception as e:
                print(f"Error moving model folder for {ticker}: {e}")
        results.append({'Ticker': ticker, 'Company Name': company_name, 'Industry': industry, 'SummaryFile': summary_path or '', 'Value': value or '', 'High Growth Rate': high_growth_rate, 'Upside (%)': upside})

    # Save results to CSV
    fieldnames = ['Ticker', 'Company Name', 'Industry', 'SummaryFile', 'Value', 'High Growth Rate', 'Upside (%)']
    if args.country:
        csv_path = os.path.join('ggm_results', args.country, args.output_csv)
    else:
        csv_path = args.output_csv
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nBatch results saved to {csv_path}")

if __name__ == "__main__":
    main()

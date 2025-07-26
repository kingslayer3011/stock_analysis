import json
import subprocess
import tempfile
import os
import pandas as pd

# Load list of tickers from JSON file
with open("screener_tickers.json", "r") as f:
    tickers_data = json.load(f)
    tickers = tickers_data["tickers"]

# Load base configuration
with open("dcf_config.json", "r") as f:
    base_config = json.load(f)

all_dcf_data = []

for ticker in tickers:
    print(f"Processing {ticker}...")

    # Update ticker in config
    config = base_config.copy()
    config["ticker"] = ticker
    config["plot_projections"] = False

    # Write temporary config file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json") as temp_config_file:
        json.dump(config, temp_config_file)
        temp_config_file_path = temp_config_file.name

    # Run cli.py with updated config
    subprocess.run(["python", "cli.py", "--config", temp_config_file_path])

    # Remove temp config file
    os.remove(temp_config_file_path)

    # Locate output file
    output_folder = os.path.join(f"dcf_results", f"{ticker}_results")
    result_path = os.path.join(output_folder, "dcf_estimates.csv")

    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        df["Ticker"] = ticker  # Add ticker info
        all_dcf_data.append(df)
    else:
        print(f"⚠️  Warning: {result_path} not found for {ticker}")

# Combine all results
if all_dcf_data:
    combined_df = pd.concat(all_dcf_data, ignore_index=True)
    # Save full combined results
    combined_df.to_csv("screener_results.csv", sep='\t', index=False)
    print("✅ Combined results saved to screener_results.csv (tab-separated)")

    # Define filters and save filtered datasets
    horizons = [5, 10]
    proj_types = ["standard", "converged"]

    for h in horizons:
        for p in proj_types:
            subset = combined_df[
                (combined_df['horizon'] == h) &
                (combined_df['proj_type'] == p)
            ].copy()

            if not subset.empty:
                # Sort by upside descending
                subset.sort_values(by='upside', ascending=False, inplace=True)

                # File name pattern: screener_results_h{h}_p{p}.csv
                filename = f"screener_results_h{h}_p{p}.csv"
                subset.to_csv(filename, sep='\t', index=False)
                print(f"✅ Filtered results saved to {filename} (horizon={h}, proj_type={p})")
            else:
                print(f"⚠️  No data for horizon={h}, proj_type={p}")
else:
    print("❌ No data combined. Check output folders.")

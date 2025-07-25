import pandas as pd
import numpy as np
import os

# === SETTING GLOBAL SEED FOR REPRODUCIBILITY ===
np.random.seed(42)

# Creating output folder if it doesn't exist
os.makedirs("processed_data", exist_ok=True)

# === LOADING DATA ===
sample_types_path = "raw_data/TO USE CT_sample_types.xlsx"
timeseries_path = "raw_data/TO USE CT_timeseries_long_format.csv"

sample_df = pd.read_excel(sample_types_path)
timeseries_df = pd.read_csv(timeseries_path)

# === CLEAN COLUMN NAMES ===
sample_df.columns = sample_df.columns.str.strip().str.lower().str.replace(" ", "_")
timeseries_df.columns = timeseries_df.columns.str.strip().str.lower().str.replace(" ", "_")

# === DROP ONLY WHERE sample_id IS MISSING ===
sample_df.dropna(subset=["sample_id"], inplace=True)
timeseries_df.dropna(subset=["sample_id"], inplace=True)

# === STANDARDIZE sample_id for merging ===
sample_df["sample_id"] = sample_df["sample_id"].astype(str).str.strip()
timeseries_df["sample_id"] = timeseries_df["sample_id"].astype(str).str.strip()

# === SAVE CLEANED FILES ===
sample_df.to_csv("processed_data/sample_metadata_clean.csv", index=False)
timeseries_df.to_csv("processed_data/timeseries_clean.csv", index=False)

print("✅ Preprocessing complete. Files saved to /processed_data/")

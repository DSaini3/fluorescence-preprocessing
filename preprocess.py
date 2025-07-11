import os
import pandas as pd
import numpy as np

# File paths
RAW_TIMESERIES_PATH = 'raw_data/TO USE CT_timeseries_long_format.csv'
RAW_METADATA_PATH = 'raw_data/TO USE CT_sample_types.xlsx'
PROCESSED_DATA_PATH = 'processed_data/normalized_data.csv'

# Ensure processed_data directory exists
os.makedirs('processed_data', exist_ok=True)

# Load data
print("🔄 Loading timeseries and metadata...")
df_time = pd.read_csv(RAW_TIMESERIES_PATH)
df_meta = pd.read_excel(RAW_METADATA_PATH)

# Remove '-dup' from sample IDs in df_time
df_time['sample_id'] = df_time['sample_id'].astype(str).str.replace('-dup', '', regex=False)

# Merge timeseries with metadata on sample_id
df = pd.merge(df_time, df_meta, on='sample_id', how='inner')

# Drop rows with missing key columns
df.dropna(subset=['sample_id', 'seconds', 'result', 'overall_result'], inplace=True)

# Convert 'result' to float (in case it's read as object)
df['result'] = pd.to_numeric(df['result'], errors='coerce')
df.dropna(subset=['result'], inplace=True)

# Normalize fluorescence per sample_id using Min-Max
print("⚙️ Applying min-max normalization...")
def normalize(group):
    vals = group['result']
    group['normalized_result'] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    return group

df = df.groupby('sample_id').apply(normalize).reset_index(drop=True)

# Save to processed_data folder
print("💾 Saving processed data to:", PROCESSED_DATA_PATH)
df.to_csv(PROCESSED_DATA_PATH, index=False)
print("✅ Preprocessing complete.")

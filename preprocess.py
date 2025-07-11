import os
import pandas as pd
import numpy as np

# Paths
RAW_DATA_PATH = 'raw_data/TO_USE_CT_timeseries_long_format.csv'
PROCESSED_DATA_PATH = 'processed_data/normalized_data.csv'

# Create processed_data directory if it doesn't exist
os.makedirs('processed_data', exist_ok=True)

# Load data
print("🔄 Loading raw data...")
df = pd.read_csv(RAW_DATA_PATH)

# Basic checks
if 'sample_id' not in df.columns or 'seconds' not in df.columns or 'result' not in df.columns:
    raise ValueError("Missing required columns: ['sample_id', 'seconds', 'result']")

# Drop rows with missing values
df = df.dropna(subset=['sample_id', 'seconds', 'result'])

# Normalize 'result' column per sample_id (Min-Max normalization)
print("⚙️ Normalizing fluorescence values per sample...")
def min_max_normalize(group):
    vals = group['result']
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    group['normalized_result'] = norm
    return group

df = df.groupby('sample_id').apply(min_max_normalize)

# Save normalized dataset
print("💾 Saving normalized data to processed_data/normalized_data.csv")
df.to_csv(PROCESSED_DATA_PATH, index=False)

print("✅ Done.")

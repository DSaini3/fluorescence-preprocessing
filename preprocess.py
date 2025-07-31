import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Set seed for reproducibility
np.random.seed(42)

# === PATH SETUP ===
RAW_PATH = "raw_data"
OUT_PATH = "processed_data"
os.makedirs(OUT_PATH, exist_ok=True)

# === FILES ===
sample_file = os.path.join(RAW_PATH, "TO USE CT_sample_types.xlsx")
timeseries_file = os.path.join(RAW_PATH, "TO USE CT_timeseries_long_format.csv")

# === LOAD FILES ===
sample_df = pd.read_excel(sample_file)
timeseries_df = pd.read_csv(timeseries_file)

# === CLEAN COLUMN NAMES ===
sample_df.columns = sample_df.columns.str.strip().str.lower().str.replace(" ", "_")
timeseries_df.columns = timeseries_df.columns.str.strip().str.lower().str.replace(" ", "_")

# === DROP ROWS WITH MISSING sample_id ===
sample_df.dropna(subset=["sample_id"], inplace=True)
timeseries_df.dropna(subset=["sample_id"], inplace=True)

# === CLEAN sample_id ===
sample_df["sample_id"] = sample_df["sample_id"].astype(str).str.strip()
timeseries_df["sample_id"] = timeseries_df["sample_id"].astype(str).str.strip()

# === SAVE CLEANED FILES ===
sample_df.to_csv(os.path.join(OUT_PATH, "sample_metadata_clean.csv"), index=False)
timeseries_df.to_csv(os.path.join(OUT_PATH, "timeseries_clean.csv"), index=False)

# === MERGE ===
df = pd.merge(timeseries_df, sample_df, on="sample_id", how="left")
df = df[df['replicate'].isin([0, 1, 2, 3])]
df['overall_result'] = df['overall_result'].astype(str).str.strip().str.lower()
df = df.dropna(subset=["overall_result"])

# === FEATURE EXTRACTION ===
def extract_features(data):
    df_sorted = data.sort_values(by=["sample_id", "mins"])
    
    grouped = df_sorted.groupby("sample_id").agg({
        'result': ['max', 'min', 'mean', 'std', 'median']
    })
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.reset_index(inplace=True)

    def compute_iqr(x):
        return np.percentile(x, 75) - np.percentile(x, 25) if len(x) >= 2 else np.nan
    iqr = df_sorted.groupby("sample_id")["result"].apply(compute_iqr).reset_index(name="result_iqr")

    peak_time = df_sorted.loc[df_sorted.groupby("sample_id")["result"].idxmax()][["sample_id", "mins"]]
    peak_time.columns = ["sample_id", "time_to_peak"]

    auc = df_sorted.groupby("sample_id")["result"].sum().reset_index(name="area_under_curve")

    df_sorted["phase"] = np.where(df_sorted["mins"] < 17, "early", "late")
    phase_avg = df_sorted.groupby(["sample_id", "phase"])["result"].mean().unstack(fill_value=0).reset_index()
    phase_avg.columns = ["sample_id", "early_mean", "late_mean"]

    labels = df_sorted[["sample_id", "overall_result"]].drop_duplicates()
    gender = df_sorted[["sample_id", "gender"]].drop_duplicates()

    features = grouped.merge(peak_time, on="sample_id")
    features = features.merge(auc, on="sample_id")
    features = features.merge(phase_avg, on="sample_id")
    features = features.merge(iqr, on="sample_id")
    features = features.merge(labels, on="sample_id")
    features = features.merge(gender, on="sample_id")

    # Print for verification
    print(f"⚠️ Missing IQR values: {features['result_iqr'].isna().sum()}")

    return features

# === TtA FEATURE ===
def compute_tta(df, baseline_points=10, sigma_multiplier=10):
    tta_records = []
    df_sorted = df.sort_values(by=["sample_id", "mins"])
    for sample_id, group in df_sorted.groupby("sample_id"):
        signal = group["result"].values
        time = group["mins"].values
        if len(signal) < baseline_points:
            tta_records.append((sample_id, np.nan))
            continue
        baseline = signal[:baseline_points]
        mean = baseline.mean()
        std = baseline.std()
        threshold = mean + sigma_multiplier * std
        above = np.where(signal > threshold)[0]
        tta_time = time[above[0]] if len(above) > 0 else np.nan
        tta_records.append((sample_id, tta_time))
    return pd.DataFrame(tta_records, columns=["sample_id", "time_to_amplification"])

# === SPLIT BY REPLICATE ===
rep0 = df[df["replicate"] == 0]
rep1 = df[df["replicate"] == 1]
rep2 = df[df["replicate"] == 2]
rep3 = df[df["replicate"] == 3]

df_train = extract_features(pd.concat([rep0, rep1]))
df_val = extract_features(rep2)
df_test = extract_features(rep3)

# === ENCODE LABEL ===
for part in [df_train, df_val, df_test]:
    part["label"] = part["overall_result"].map({"negative": 0, "positive": 1})

# === ADD TtA FEATURE (without filling missing) ===
tta_df = compute_tta(df)

# Print for verification
print(f"⚠️ Missing TtA values: {tta_df['time_to_amplification'].isna().sum()}")

df_train = df_train.merge(tta_df, on="sample_id", how="left")
df_val = df_val.merge(tta_df, on="sample_id", how="left")
df_test = df_test.merge(tta_df, on="sample_id", how="left")

# === GENDER ONE-HOT ENCODING ===
df_train = pd.get_dummies(df_train, columns=["gender"], drop_first=True)
df_val = pd.get_dummies(df_val, columns=["gender"], drop_first=True)
df_test = pd.get_dummies(df_test, columns=["gender"], drop_first=True)

# === FEATURE LIST ===
feature_cols = [
    'result_max', 'result_min', 'result_mean', 'result_median',
    'result_std', 'result_iqr', 'time_to_peak',
    'area_under_curve', 'early_mean', 'late_mean',
    'time_to_amplification'
]

gender_cols = [col for col in df_train.columns if col.startswith("gender_")]
feature_cols.extend(gender_cols)

# === FINAL TRAIN/VAL/TEST SPLIT ===
X_train = df_train[feature_cols]
X_val = df_val[feature_cols]
X_test = df_test[feature_cols]
y_train = df_train["label"]
y_val = df_val["label"]
y_test = df_test["label"]

# === SCALE FEATURES ===
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# === SAVE TO CSV ===
pd.DataFrame(X_train_scaled, columns=feature_cols).assign(label=y_train.values).to_csv(
    os.path.join(OUT_PATH, "preprocessed_train.csv"), index=False)
pd.DataFrame(X_val_scaled, columns=feature_cols).assign(label=y_val.values).to_csv(
    os.path.join(OUT_PATH, "preprocessed_val.csv"), index=False)
pd.DataFrame(X_test_scaled, columns=feature_cols).assign(label=y_test.values).to_csv(
    os.path.join(OUT_PATH, "preprocessed_test.csv"), index=False)

print("✅ All statistical ML preprocessing completed. Files saved in /processed_data/")

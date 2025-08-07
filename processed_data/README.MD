✅ Data Loading and Cleaning
Loaded:

timeseries_clean.csv (time series data)

sample_metadata_clean.csv (sample metadata)

Merged both datasets on sample_id

Added seconds column as:
seconds = mins * 60

Filtered to keep only replicate values in [0, 1, 2, 3]

Cleaned overall_result:

Converted to lowercase

Stripped whitespace

Dropped rows with missing overall_result

✅ Feature Engineering
Aggregated features per sample:

result_max, result_min, result_mean, result_std, result_median

mins_max, mins_min

result_iqr (Interquartile Range)

Computed dynamic signal features:

time_to_peak: time of max signal

area_under_curve: sum of signal values

early_mean and late_mean: average signal before and after 17 mins

Label column created:

overall_result → label (positive = 1, negative = 0)

✅ Time to Amplification (TtA)
For each sample:

Calculated threshold: baseline mean + 10 × std (baseline = first 10 time points)

Identified first time signal crosses threshold → time_to_amplification

Missing values filled with median TtA

✅ Metadata Integration
Added gender from metadata using sample_id

One-hot encoded gender with drop_first=True to avoid dummy variable trap

✅ Final Feature Set
Constructed feature list:

Signal stats + time features + gender dummies

Created:

X_train, X_val, X_test with selected feature columns

y_train, y_val, y_test with encoded labels

✅ Normalization
Applied MinMaxScaler():

Fit on X_train

Transformed X_val, X_test

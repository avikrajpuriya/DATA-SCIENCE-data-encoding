import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("urban_climate.csv")

print("ORIGINAL DATASET")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

print("\nSKEWNESS OF NUMERICAL FEATURES")
skewness = df[numeric_cols].skew()
print(skewness)

for col in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title("Histogram and KDE - " + col)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

scale_cols = [
    "temperature_celsius",
    "humidity_percent",
    "precipitation_mm",
    "wind_speed_ms",
    "urban_heat_island_intensity"
]
scale_cols = [col for col in scale_cols if col in df.columns]

manual_scaled = df[scale_cols].copy()
for col in scale_cols:
    minimum = df[col].min()
    maximum = df[col].max()
    if maximum != minimum:
        manual_scaled[col] = (df[col] - minimum) / (maximum - minimum)
    else:
        manual_scaled[col] = 0.0

print("\nMANUAL MIN-MAX SCALING")
print(manual_scaled.head())

minmax_scaler = MinMaxScaler()
sklearn_scaled = pd.DataFrame(
    minmax_scaler.fit_transform(df[scale_cols]),
    columns=scale_cols,
    index=df.index
)

print("\nSCIKIT-LEARN MIN-MAX SCALING")
print(sklearn_scaled.head())
print("\nMIN-MAX VERIFICATION")
print("Minimum values:\n", sklearn_scaled.min())
print("Maximum values:\n", sklearn_scaled.max())

# The source dataset has no original ordinal education-like feature.
# Create a derived ordinal precipitation category for demonstrating ordered labels.
df_encoded = df.copy()
df_encoded["precipitation_level"] = pd.qcut(
    df_encoded["precipitation_mm"].rank(method="first"),
    q=3,
    labels=["Low", "Moderate", "High"]
)
ordinal_mapping = {"Low": 0, "Moderate": 1, "High": 2}
df_encoded["precipitation_level_encoded"] = (
    df_encoded["precipitation_level"].map(ordinal_mapping).astype(int)
)

print("\nORDINAL ENCODING")
print(df_encoded[[
    "precipitation_level",
    "precipitation_level_encoded"
]].head())

nominal_cols = [
    col for col in [
        "city", "country", "continent", "subregion",
        "data_source", "capital_city", "iso3_code", "currency"
    ] if col in df_encoded.columns
]

encoded_df = pd.get_dummies(
    df_encoded,
    columns=nominal_cols,
    drop_first=True,
    dtype=int
)

print("\nONE-HOT ENCODED DATASET")
print(encoded_df.head())

# Standardize original numerical features; keep one-hot and ordinal columns unchanged.
standardized_df = encoded_df.copy()
standard_scaler = StandardScaler()
standardized_df[numeric_cols] = standard_scaler.fit_transform(
    standardized_df[numeric_cols]
)

print("\nSTANDARDIZATION VERIFICATION (population standard deviation)")
print("Means:\n", standardized_df[numeric_cols].mean().round(4))
print("Standard deviations:\n", standardized_df[numeric_cols].std(ddof=0).round(4))

print("\nORIGINAL DATASET (first five rows)")
print(df.head())
print("\nTRANSFORMED DATASET (first five rows)")
print(standardized_df.head())
print("\nOriginal shape:", df.shape)
print("Transformed shape:", standardized_df.shape)

# Save the normalized numeric columns and the fully encoded/standardized dataset.
normalized_df = df.copy()
normalized_df[scale_cols] = sklearn_scaled
normalized_df.to_csv("urban_climate_minmax_scaled.csv", index=False)
standardized_df.to_csv("urban_climate_preprocessed.csv", index=False)

print("\nSaved:")
print("- urban_climate_minmax_scaled.csv")
print("- urban_climate_preprocessed.csv")
print("\nEXPERIMENT 4 COMPLETED SUCCESSFULLY")

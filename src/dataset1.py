import pandas as pd

# Load the dataset
file_path = "data/Raw/Hydrographic_Dataset1.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nNumerical summary:")
print(df.describe())

print("\nUnique values in important categorical columns:")

categorical_columns = [
    "STATION",
    "TYPE OF MOLLUSC",
    "SPECIES",
    "BIOTOXIN GROUP",
    "VARIABLE (abbreviation)",
    "<=>",
    "UNITS (abbreviation)"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))

print("\n--- Step 3G: YTX Date + Station Matching ---")

# Select YTX records
ytx_df = df[df["VARIABLE (abbreviation)"] == "YTX"].copy()

# Keep only actually measured values
ytx_measured = ytx_df[ytx_df["<=>"] == "="].copy()

print("Total YTX records:", len(ytx_df))
print("Measured YTX records:", len(ytx_measured))

# Convert date to string for easier matching
ytx_measured["DATE"] = ytx_measured["UTC DATE (YYYYMMDD)"].astype(str)

# Check stations
print("\nYTX measured records by station:")
print(ytx_measured["STATION"].value_counts(dropna=False))

# Check date range
print("\nYTX measured date range:")
print("Start:", ytx_measured["UTC DATE (YYYYMMDD)"].min())
print("End:", ytx_measured["UTC DATE (YYYYMMDD)"].max())

# Check number of unique dates
print("\nUnique YTX measured dates:")
print(ytx_measured["DATE"].nunique())

# Check environmental rows available on the same dates
environment_df = df[
    df["TEMPERATURE (°C)"].notna()
].copy()

environment_df["DATE"] = environment_df["UTC DATE (YYYYMMDD)"].astype(str)

matched_dates = ytx_measured["DATE"].isin(
    environment_df["DATE"]
)

print("\nYTX measured records with matching environmental date:")
print(matched_dates.sum(), "out of", len(ytx_measured))

# Check exact DATE + STATION matching
ytx_keys = set(
    zip(
        ytx_measured["DATE"],
        ytx_measured["STATION"]
    )
)

environment_keys = set(
    zip(
        environment_df["DATE"],
        environment_df["STATION"]
    )
)

matched_keys = ytx_keys.intersection(environment_keys)

print("\nUnique YTX DATE + STATION combinations:", len(ytx_keys))
print("Matching DATE + STATION combinations:", len(matched_keys))

print("\nPercentage of YTX DATE + STATION combinations matched:")
print(round(len(matched_keys) / len(ytx_keys) * 100, 2), "%")


print("\n--- Step 3H: Inspect YTX + Environmental Matching ---")

# -----------------------------
# 1. Select measured YTX records
# -----------------------------

ytx_measured = df[
    (df["VARIABLE (abbreviation)"] == "YTX") &
    (df["<=>"] == "=")
].copy()

ytx_measured["DATE"] = ytx_measured["UTC DATE (YYYYMMDD)"].astype(str)

# Keep only the columns needed from YTX
ytx_target = ytx_measured[
    [
        "DATE",
        "STATION",
        "VALUE",
        "<=>",
        "TYPE OF MOLLUSC",
        "SPECIES"
    ]
].copy()


# -----------------------------
# 2. Prepare environmental data
# -----------------------------

environment_df = df[
    df["TEMPERATURE (°C)"].notna()
].copy()

environment_df["DATE"] = environment_df[
    "UTC DATE (YYYYMMDD)"
].astype(str)

environment_columns = [
    "DATE",
    "STATION",
    "TEMPERATURE (°C)",
    "SALINITY",
    "SIGMA THETA (kg m-3)",
    "LT (%)",
    "PAR (µmol m-2 s-1)",
    "CHLOROPHYLL-a (µg l-1)",
    "DISSOLVED OXYGEN (ml l-1)",
    "OXYGEN SATURATION (%)",
    "PRESSURE (dbar)"
]

environment_unique = environment_df[
    environment_columns
].drop_duplicates()


# -----------------------------
# 3. Check duplicates
# -----------------------------

print("\nEnvironmental rows before removing duplicates:",
      len(environment_df))

print("Environmental rows after removing exact duplicates:",
      len(environment_unique))

print("\nEnvironmental observations per DATE + STATION:")

duplicate_counts = (
    environment_unique
    .groupby(["DATE", "STATION"])
    .size()
    .value_counts()
    .sort_index()
)

print(duplicate_counts)


# -----------------------------
# 4. Create DATE + STATION keys
# -----------------------------

ytx_keys = (
    ytx_target[["DATE", "STATION"]]
    .drop_duplicates()
)

environment_keys = (
    environment_unique[["DATE", "STATION"]]
    .drop_duplicates()
)

matched_keys = ytx_keys.merge(
    environment_keys,
    on=["DATE", "STATION"],
    how="inner"
)

print("\nUnique YTX DATE + STATION combinations:",
      len(ytx_keys))

print("Matching DATE + STATION combinations:",
      len(matched_keys))


# -----------------------------
# 5. Inspect matching structure
# -----------------------------

ytx_match_counts = (
    ytx_target
    .groupby(["DATE", "STATION"])
    .size()
)

environment_match_counts = (
    environment_unique
    .groupby(["DATE", "STATION"])
    .size()
)

print("\nYTX records per DATE + STATION:")
print(ytx_match_counts.value_counts().sort_index())

print("\nEnvironmental records per DATE + STATION:")
print(environment_match_counts.value_counts().sort_index())


# -----------------------------
# 6. Show a small sample
# -----------------------------

print("\nFirst 10 YTX observations:")

print(
    ytx_target[
        [
            "DATE",
            "STATION",
            "VALUE",
            "TYPE OF MOLLUSC",
            "SPECIES"
        ]
    ].head(10)
)

print("\nFirst 10 environmental observations:")

print(
    environment_unique[
        [
            "DATE",
            "STATION",
            "TEMPERATURE (°C)",
            "SALINITY",
            "CHLOROPHYLL-a (µg l-1)",
            "DISSOLVED OXYGEN (ml l-1)"
        ]
    ].head(10)
)

print("\n--- Step 3I: Inspect YTX Sampling Structure ---")

# Measured YTX records
ytx_measured = df[
    (df["VARIABLE (abbreviation)"] == "YTX") &
    (df["<=>"] == "=")
].copy()

print("\nYTX sampling time distribution:")
print(
    ytx_measured["UTC TIME (hhmmss)"]
    .value_counts()
    .head(20)
)

print("\nYTX sampling depth distribution:")
print(
    ytx_measured["SAMPLING DEPTH (m)"]
    .value_counts(dropna=False)
    .head(20)
)

print("\nYTX station + sampling depth:")
print(
    ytx_measured[
        [
            "STATION",
            "SAMPLING DEPTH (m)",
            "UTC TIME (hhmmss)"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        ["STATION", "SAMPLING DEPTH (m)", "UTC TIME (hhmmss)"]
    )
    .head(50)
)

print("\nNumber of unique YTX sampling depths:")
print(
    ytx_measured["SAMPLING DEPTH (m)"].nunique(
        dropna=True
    )
)

print("\nNumber of unique YTX sampling times:")
print(
    ytx_measured["UTC TIME (hhmmss)"].nunique()
)

print("\nMissing YTX sampling depth:")
print(
    ytx_measured["SAMPLING DEPTH (m)"].isna().sum(),
    "out of",
    len(ytx_measured)
)

print("\nMissing YTX sampling time:")
print(
    ytx_measured["UTC TIME (hhmmss)"].isna().sum(),
    "out of",
    len(ytx_measured)
)

print("\n--- Step 3J: Exact YTX Time + Depth Matching ---")

# -----------------------------
# 1. Measured YTX observations
# -----------------------------

ytx_measured = df[
    (df["VARIABLE (abbreviation)"] == "YTX") &
    (df["<=>"] == "=")
].copy()

# -----------------------------
# 2. Environmental observations
# -----------------------------

environment_df = df[
    df["TEMPERATURE (°C)"].notna()
].copy()

# -----------------------------
# 3. Create exact matching keys
# -----------------------------

ytx_measured["DATE"] = (
    ytx_measured["UTC DATE (YYYYMMDD)"]
    .astype(str)
)

environment_df["DATE"] = (
    environment_df["UTC DATE (YYYYMMDD)"]
    .astype(str)
)

ytx_keys = ytx_measured[
    [
        "DATE",
        "STATION",
        "UTC TIME (hhmmss)",
        "SAMPLING DEPTH (m)"
    ]
].drop_duplicates()

environment_keys = environment_df[
    [
        "DATE",
        "STATION",
        "UTC TIME (hhmmss)",
        "SAMPLING DEPTH (m)"
    ]
].drop_duplicates()

# -----------------------------
# 4. Exact match
# -----------------------------

matched = ytx_keys.merge(
    environment_keys,
    on=[
        "DATE",
        "STATION",
        "UTC TIME (hhmmss)",
        "SAMPLING DEPTH (m)"
    ],
    how="inner"
)

print("\nTotal unique YTX sampling keys:")
print(len(ytx_keys))

print("\nExact DATE + STATION + TIME + DEPTH matches:")
print(len(matched))

print("\nExact matching percentage:")
print(
    round(
        len(matched) / len(ytx_keys) * 100,
        2
    ),
    "%"
)

# -----------------------------
# 5. Check which YTX samples
#    did not find exact matches
# -----------------------------

ytx_check = ytx_keys.merge(
    environment_keys,
    on=[
        "DATE",
        "STATION",
        "UTC TIME (hhmmss)",
        "SAMPLING DEPTH (m)"
    ],
    how="left",
    indicator=True
)

unmatched = ytx_check[
    ytx_check["_merge"] == "left_only"
]

print("\nUnmatched YTX sampling keys:")
print(len(unmatched))

print("\nFirst unmatched records:")
print(
    unmatched.head(20)
)

# -----------------------------
# 6. Check date + station + depth
#    without exact time
# -----------------------------

depth_matched = ytx_keys.merge(
    environment_keys[
        [
            "DATE",
            "STATION",
            "SAMPLING DEPTH (m)"
        ]
    ].drop_duplicates(),
    on=[
        "DATE",
        "STATION",
        "SAMPLING DEPTH (m)"
    ],
    how="inner"
)

print("\nDATE + STATION + DEPTH matches:")
print(len(depth_matched))

print("\nDATE + STATION + DEPTH matching percentage:")
print(
    round(
        len(depth_matched) / len(ytx_keys) * 100,
        2
    ),
    "%"
)

print("\n--- Step 3K: YTX Detection / Measurement Analysis ---")

# Select all YTX records
ytx_all = df[
    df["VARIABLE (abbreviation)"] == "YTX"
].copy()

print("\nTotal YTX records:")
print(len(ytx_all))

# -----------------------------
# 1. Detection status
# -----------------------------

print("\nYTX detection status:")
print(
    ytx_all["<=>"].value_counts(dropna=False)
)

measured_count = (ytx_all["<=>"] == "=").sum()
below_detection_count = (ytx_all["<=>"] == "<").sum()

print("\nMeasured YTX records:", measured_count)
print("Below-detection YTX records:", below_detection_count)

print(
    "Measured percentage:",
    round(measured_count / len(ytx_all) * 100, 2),
    "%"
)

print(
    "Below-detection percentage:",
    round(below_detection_count / len(ytx_all) * 100, 2),
    "%"
)


# -----------------------------
# 2. Measured YTX values
# -----------------------------

ytx_measured = ytx_all[
    ytx_all["<=>"] == "="
].copy()

print("\nMeasured YTX concentration summary:")

print(
    ytx_measured["VALUE"].describe()
)


# -----------------------------
# 3. Below-detection reported
#    values
# -----------------------------

ytx_below = ytx_all[
    ytx_all["<=>"] == "<"
].copy()

print("\nBelow-detection reported VALUE summary:")

print(
    ytx_below["VALUE"].describe()
)


# -----------------------------
# 4. Unique reported values
# -----------------------------

print("\nMost common measured YTX values:")

print(
    ytx_measured["VALUE"]
    .value_counts()
    .head(15)
)

print("\nMost common below-detection YTX values:")

print(
    ytx_below["VALUE"]
    .value_counts()
    .head(15)
)


# -----------------------------
# 5. Compare distributions
# -----------------------------

print("\nMeasured YTX range:")
print(
    "Minimum:",
    ytx_measured["VALUE"].min()
)

print(
    "Maximum:",
    ytx_measured["VALUE"].max()
)

print(
    "Median:",
    ytx_measured["VALUE"].median()
)

print("\nBelow-detection reported range:")
print(
    "Minimum:",
    ytx_below["VALUE"].min()
)

print(
    "Maximum:",
    ytx_below["VALUE"].max()
)

print(
    "Median:",
    ytx_below["VALUE"].median()
)


# -----------------------------
# 6. Check whether measured
#    values overlap with the
#    detection-limit values
# -----------------------------

measured_values = set(
    ytx_measured["VALUE"]
)

below_values = set(
    ytx_below["VALUE"]
)

overlap = measured_values.intersection(
    below_values
)

print("\nNumber of unique measured values:",
      len(measured_values))

print("Number of unique below-detection values:",
      len(below_values))

print("Values appearing in both groups:",
      len(overlap))

print("\nOverlapping values:")
print(sorted(overlap)[:20])


# -----------------------------
# 7. Check YTX units
# -----------------------------

print("\nYTX units:")
print(
    ytx_all["UNITS (abbreviation)"]
    .value_counts(dropna=False)
)


# Step 3L: Analyze Rare, High-Concentration YTX Events

print("\n--- Step 3L: Rare High-Concentration YTX Analysis ---")

# Keep all YTX records, including below-detection observations
ytx_all = df[df["VARIABLE (abbreviation)"] == "YTX"].copy()

# Only exact measured values are used to analyze the measured
# concentration distribution. Values marked "<" are censored.
ytx_measured = ytx_all[ytx_all["<=>"] == "="].copy()

print("\nTotal YTX records:", len(ytx_all))
print("Measured YTX records:", len(ytx_measured))
print("Below-detection records:", (ytx_all["<=>"] == "<").sum())

# Summarize the distribution of measured concentrations
print("\nMeasured YTX concentration percentiles:")
print(
    ytx_measured["VALUE"].quantile(
        [0.50, 0.75, 0.80, 0.90, 0.95, 0.99]
    )
)

# Inspect the highest measured concentrations
print("\nTop 20 measured YTX observations:")

columns_to_show = [
    "STATION",
    "UTC DATE (YYYYMMDD)",
    "UTC TIME (hhmmss)",
    "SAMPLING DEPTH (m)",
    "VALUE",
    "UNITS (abbreviation)"
]

print(
    ytx_measured
    .sort_values("VALUE", ascending=False)[columns_to_show]
    .head(20)
    .to_string(index=False)
)

# Count observations in the upper tail using exploratory percentiles.
# These are analysis cutoffs, NOT regulatory safety limits.
for percentile in [0.80, 0.90, 0.95]:
    cutoff = ytx_measured["VALUE"].quantile(percentile)
    high_count = (ytx_measured["VALUE"] >= cutoff).sum()
    percentage = high_count / len(ytx_measured) * 100

    print(f"\n{int(percentile * 100)}th percentile cutoff: {cutoff:.4f} mg/kg")
    print("Observations at or above cutoff:", high_count)
    print("Percentage of measured observations:", round(percentage, 2), "%")

# Check how many observations exceed selected exploratory cutoffs.
# These values help us inspect the distribution; they do not define danger.
print("\nCounts above selected concentration values:")

for cutoff in [0.5, 1.0, 1.5, 2.0, 3.0]:
    count = (ytx_measured["VALUE"] >= cutoff).sum()
    print(f"YTX >= {cutoff:.1f} mg/kg: {count} observations")

print("\nStep 3L analysis complete.")


# Step 3M: Inspect High-YTX Sampling Events

print("\n--- Step 3M: High-YTX Event Analysis ---")

ytx_measured = df[
    (df["VARIABLE (abbreviation)"] == "YTX") &
    (df["<=>"] == "=")
].copy()

# Identify the upper-tail samples for exploratory inspection.
# This cutoff is not a regulatory safety threshold.
high_cutoff = ytx_measured["VALUE"].quantile(0.90)
high_ytx = ytx_measured[
    ytx_measured["VALUE"] >= high_cutoff
].copy()

print("\nExploratory 90th-percentile cutoff:", high_cutoff, "mg/kg")
print("High-concentration rows:", len(high_ytx))

# Count how many distinct dates and sampling events are represented.
high_ytx["DATE"] = high_ytx["UTC DATE (YYYYMMDD)"].astype(str)

print("\nUnique high-YTX dates:", high_ytx["DATE"].nunique())

event_columns = [
    "STATION",
    "UTC DATE (YYYYMMDD)",
    "UTC TIME (hhmmss)"
]

print("\nHigh-YTX rows by sampling event:")
print(
    high_ytx.groupby(event_columns, dropna=False)
    .agg(
        sample_count=("VALUE", "size"),
        maximum_y_tx=("VALUE", "max"),
        depths=("SAMPLING DEPTH (m)", "nunique")
    )
    .sort_values("maximum_y_tx", ascending=False)
    .to_string()
)

print("\nHigh-YTX rows by station:")
print(high_ytx["STATION"].value_counts(dropna=False))

print("\nHigh-YTX concentration summary:")
print(high_ytx["VALUE"].describe())

print("\nStep 3M analysis complete.")


# Step 3N: Inspect Environmental Conditions During High-YTX Events

print("\n--- Step 3N: High-YTX Environmental Analysis ---")

# Keep measured YTX observations
ytx_measured = df[
    (df["VARIABLE (abbreviation)"] == "YTX") &
    (df["<=>"] == "=")
].copy()

# Exploratory cutoff only; not a regulatory danger threshold
high_cutoff = ytx_measured["VALUE"].quantile(0.90)

high_ytx = ytx_measured[
    ytx_measured["VALUE"] >= high_cutoff
].copy()

# Identify environmental columns
environment_columns = [
    "PRESSURE (dbar)",
    "TEMPERATURE (°C)",
    "SALINITY",
    "SIGMA THETA (kg m-3)",
    "LT (%)",
    "PAR (µmol m-2 s-1)",
    "CHLOROPHYLL-a (µg l-1)",
    "DISSOLVED OXYGEN (ml l-1)",
    "OXYGEN SATURATION (%)",
    "SAMPLING DEPTH (m)"
]

print("\nEnvironmental variable availability:")
for col in environment_columns:
    if col in high_ytx.columns:
        print(
            f"{col}: "
            f"{high_ytx[col].notna().sum()}/{len(high_ytx)} non-missing"
        )

# Summarize conditions associated with high observations
print("\nEnvironmental summary for high-YTX rows:")
print(high_ytx[environment_columns].describe().T)

# Compare high-YTX observations with all measured YTX observations
print("\nComparison: high-YTX vs all measured YTX")

for col in environment_columns:
    if col not in high_ytx.columns:
        continue

    high_mean = high_ytx[col].mean()
    all_mean = ytx_measured[col].mean()

    print(f"\n{col}")
    print("High-YTX mean:", round(high_mean, 4) if pd.notna(high_mean) else "N/A")
    print("All measured YTX mean:", round(all_mean, 4) if pd.notna(all_mean) else "N/A")

print("\nStep 3N analysis complete.")


# Step 3O: Build and Validate the YTX Modeling Dataset

print("\n--- Step 3O: Build Matched YTX Dataset ---")

# Prepare YTX records
ytx_data = df[
    df["VARIABLE (abbreviation)"] == "YTX"
].copy()

ytx_data["DATE_KEY"] = (
    ytx_data["UTC DATE (YYYYMMDD)"].astype("string")
)

ytx_data["TIME_KEY"] = (
    ytx_data["UTC TIME (hhmmss)"].astype("string")
)

ytx_data["DEPTH_KEY"] = ytx_data["SAMPLING DEPTH (m)"]

# Keep the environmental columns only
environment_columns = [
    "STATION",
    "UTC DATE (YYYYMMDD)",
    "UTC TIME (hhmmss)",
    "SAMPLING DEPTH (m)",
    "PRESSURE (dbar)",
    "TEMPERATURE (°C)",
    "SALINITY",
    "SIGMA THETA (kg m-3)",
    "LT (%)",
    "PAR (µmol m-2 s-1)",
    "CHLOROPHYLL-a (µg l-1)",
    "DISSOLVED OXYGEN (ml l-1)",
    "OXYGEN SATURATION (%)"
]

environment = df[environment_columns].copy()

# Environmental rows contain toxin metadata in the original file.
# Retain one environmental observation per exact sampling key.
environment = environment.drop_duplicates(
    subset=[
        "STATION",
        "UTC DATE (YYYYMMDD)",
        "UTC TIME (hhmmss)",
        "SAMPLING DEPTH (m)"
    ]
)

environment["DATE_KEY"] = (
    environment["UTC DATE (YYYYMMDD)"].astype("string")
)

environment["TIME_KEY"] = (
    environment["UTC TIME (hhmmss)"].astype("string")
)

environment["DEPTH_KEY"] = environment["SAMPLING DEPTH (m)"]


# Merge using exact sampling keys without duplicate environmental columns

# Keep the toxin records and add only the environmental columns
# from the environment DataFrame.
model_data = ytx_data.drop(
    columns=[
        col for col in environment_columns
        if col != "STATION" and col in ytx_data.columns
    ],
    errors="ignore"
).merge(
    environment[
        [
            "STATION",
            "DATE_KEY",
            "TIME_KEY",
            "DEPTH_KEY",
            "PRESSURE (dbar)",
            "TEMPERATURE (°C)",
            "SALINITY",
            "SIGMA THETA (kg m-3)",
            "LT (%)",
            "PAR (µmol m-2 s-1)",
            "CHLOROPHYLL-a (µg l-1)",
            "DISSOLVED OXYGEN (ml l-1)",
            "OXYGEN SATURATION (%)"
        ]
    ],
    on=["STATION", "DATE_KEY", "TIME_KEY", "DEPTH_KEY"],
    how="left",
    validate="many_to_one",
    indicator=True
)

print("\nOriginal YTX rows:", len(ytx_data))
print("Matched dataset rows:", len(model_data))

print("\nEnvironmental match status:")
print(model_data["_merge"].value_counts(dropna=False))

print("\nDetection status in matched dataset:")
print(model_data["<=>"].value_counts(dropna=False))

feature_columns = [
    "PRESSURE (dbar)",
    "TEMPERATURE (°C)",
    "SALINITY",
    "SIGMA THETA (kg m-3)",
    "LT (%)",
    "PAR (µmol m-2 s-1)",
    "CHLOROPHYLL-a (µg l-1)",
    "DISSOLVED OXYGEN (ml l-1)",
    "OXYGEN SATURATION (%)"
]

print("\nMissing environmental values:")
print(model_data[feature_columns].isna().sum())

print("\nEnvironmental columns found:")
print([col for col in feature_columns if col in model_data.columns])

print("\nStep 3O validation complete.")


# Step 3P: Check repeated YTX sampling keys

sampling_keys = [
    "STATION",
    "DATE_KEY",
    "TIME_KEY",
    "DEPTH_KEY"
]

print("\n--- Step 3P: Check Repeated Sampling Keys ---")

# Count how many distinct sampling keys exist
unique_keys = model_data[sampling_keys].drop_duplicates()

print("Total YTX rows:", len(model_data))
print("Unique sampling keys:", len(unique_keys))
print("Rows beyond unique keys:", len(model_data) - len(unique_keys))

# Identify keys that occur more than once
key_counts = (
    model_data.groupby(sampling_keys, dropna=False)
    .size()
    .reset_index(name="row_count")
)

repeated_keys = key_counts[key_counts["row_count"] > 1]

print("\nNumber of repeated sampling keys:", len(repeated_keys))
print("\nRepeated sampling key examples:")
print(repeated_keys.head(10).to_string(index=False))

print("\nStep 3P validation complete.")


# Step 4A: Prepare data for the baseline ML model

print("\n--- Step 4A: Prepare Modeling Dataset ---")

# Keep only measured YTX concentrations
modeling_data = model_data[
    model_data["<=>"] == "="
].copy()

# Environmental features used as model inputs
feature_columns = [
    "PRESSURE (dbar)",
    "TEMPERATURE (°C)",
    "SALINITY",
    "SIGMA THETA (kg m-3)",
    "LT (%)",
    "PAR (µmol m-2 s-1)",
    "CHLOROPHYLL-a (µg l-1)",
    "DISSOLVED OXYGEN (ml l-1)",
    "OXYGEN SATURATION (%)",
]

# Target: measured YTX concentration
target_column = "VALUE"

# Keep only the model inputs and target
modeling_data = modeling_data[
    feature_columns + [target_column]
].copy()

print("Modeling dataset shape:", modeling_data.shape)
print("Number of features:", len(feature_columns))
print("Target column:", target_column)

print("\nMissing values:")
print(modeling_data.isna().sum())

print("\nTarget concentration summary:")
print(modeling_data[target_column].describe())

print("\nStep 4A complete.")


# Step 4B: Check ML libraries

print("\n--- Step 4B: Check ML Libraries ---")

import sklearn

print("scikit-learn version:", sklearn.__version__)
print("Ready to build a baseline regression model.")


# Step 4C: Prepare a group-aware train/test split

from sklearn.model_selection import GroupShuffleSplit

print("\n--- Step 4C: Train/Test Split ---")

# Use the measured YTX records from Step 4A
X = modeling_data[feature_columns].copy()
y = modeling_data["VALUE"].copy()

# Identify sampling events so related samples stay together
groups = model_data.loc[modeling_data.index, [
    "STATION",
    "DATE_KEY",
    "TIME_KEY"
]].astype(str).agg("_".join, axis=1)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training sampling events:", groups.iloc[train_idx].nunique())
print("Testing sampling events:", groups.iloc[test_idx].nunique())
print(
    "Shared events:",
    len(
        set(groups.iloc[train_idx])
        & set(groups.iloc[test_idx])
    )
)

print("\nStep 4C complete.")


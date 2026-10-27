
# ============================================================
# TOXIGUARD: COMPLETE MARINE DATA PREPROCESSING
# Clean ALL records first, then extract DOMOIC ACID records.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# -------------------- 1. SETTINGS ---------------------------

INPUT_FILE = Path("Hydrographic_Datasett.csv")
OUTPUT_DIR = Path("toxiguard_output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"CSV not found: {INPUT_FILE.resolve()}"
    )

# -------------------- 2. LOAD FULL DATASET ------------------

df = pd.read_csv(
    INPUT_FILE,
    encoding="latin1",
    low_memory=False
)

print("Original dataset shape:", df.shape)

# Clean column names, including the BOM found in this CSV.
df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

# Remove accidental encoding artifacts from headers.
df.columns = df.columns.str.replace("Â", "", regex=False)

# Rename the station column if it has a BOM artifact.
station_col = next(
    (c for c in df.columns if c.endswith("STATION")),
    None
)

if station_col and station_col != "STATION":
    df = df.rename(columns={station_col: "STATION"})

# -------------------- 3. STANDARDIZE MISSING VALUES ---------

# Convert common text placeholders into real missing values.
missing_tokens = [
    "", "NA", "N/A", "NULL", "NONE",
    "NAN", "MISSING", "-", "--"
]

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace(missing_tokens, pd.NA)

# Numeric columns: convert invalid text to NaN.
numeric_columns = [
    "LATITUDE (degrees North)",
    "LONGITUDE (degrees East)",
    "UTC DATE (YYYYMMDD)",
    "UTC TIME (hhmmss)",
    "PRESSURE (dbar)",
    "TEMPERATURE (°C)",
    "SALINITY",
    "SIGMA THETA (kg m-3)",
    "LT (%)",
    "PAR (µmol m-2 s-1)",
    "CHLOROPHYLL-a (µg l-1)",
    "DISSOLVED OXYGEN (ml l-1)",
    "OXYGEN SATURATION (%)",
    "SAMPLING DEPTH (m)",
    "VALUE"
]

# Match column names even if special-character encoding differs.
def find_column(fragment):
    matches = [
        c for c in df.columns
        if fragment.casefold() in c.casefold()
    ]
    return matches[0] if matches else None


# Use known column names or identify columns by stable fragments.
column_aliases = {
    "TEMPERATURE (°C)": ["TEMPERATURE"],
    "PAR (µmol m-2 s-1)": ["PAR"],
    "CHLOROPHYLL-a (µg l-1)": ["CHLOROPHYLL"],
}

for expected in numeric_columns:
    actual = expected if expected in df.columns else None

    if actual is None:
        for fragment in column_aliases.get(expected, []):
            actual = find_column(fragment)
            if actual:
                break

    if actual is not None:
        df[actual] = pd.to_numeric(df[actual], errors="coerce")

# Treat -999 as a missing code in environmental measurements,
# not as a missing code for the toxin VALUE column.
environment_fragments = [
    "LATITUDE", "LONGITUDE", "PRESSURE", "TEMPERATURE",
    "SALINITY", "SIGMA THETA", "LT (%)", "PAR",
    "CHLOROPHYLL", "DISSOLVED OXYGEN",
    "OXYGEN SATURATION", "SAMPLING DEPTH"
]

environment_columns = [
    c for c in df.columns
    if any(f.casefold() in c.casefold()
           for f in environment_fragments)
]

for col in environment_columns:
    df.loc[df[col] == -999, col] = np.nan

# -------------------- 4. CLEAN DATES AND TIMES --------------

date_col = "UTC DATE (YYYYMMDD)"
time_col = "UTC TIME (hhmmss)"

if date_col in df.columns:
    df["SAMPLE_DATE"] = pd.to_datetime(
        df[date_col].astype("Int64").astype("string"),
        format="%Y%m%d",
        errors="coerce"
    )

if time_col in df.columns:
    time_num = pd.to_numeric(df[time_col], errors="coerce")
    time_text = time_num.map(
        lambda x: f"{int(x):06d}"
        if pd.notna(x) and float(x).is_integer()
        and 0 <= x <= 235959
        else None
    )

    parsed_time = pd.to_datetime(
        time_text, format="%H%M%S", errors="coerce"
    )

    df["SAMPLE_TIME"] = parsed_time.dt.strftime("%H:%M:%S")
    df.loc[parsed_time.isna(), "SAMPLE_TIME"] = pd.NA

# -------------------- 5. CLEAN CATEGORICAL COLUMNS ----------

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns.tolist()

# Preserve missing category information explicitly rather than
# inventing a station, species, toxin or laboratory.
for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

# -------------------- 6. VALIDATE NUMERIC DATA --------------

# Negative toxin concentrations are invalid. Missing toxin values
# are not filled with an average because that invents measurements.
value_col = "VALUE"

if value_col in df.columns:
    df[value_col] = pd.to_numeric(
        df[value_col], errors="coerce"
    )

    invalid_toxin = (
        df[value_col].notna()
        & (
            ~np.isfinite(df[value_col])
            | (df[value_col] < 0)
        )
    )

    print("Invalid negative/infinite toxin values:",
          int(invalid_toxin.sum()))

    df.loc[invalid_toxin, value_col] = np.nan

# -------------------- 7. REMOVE EXACT DUPLICATES ------------

before = len(df)
df = df.drop_duplicates().copy()

print("Exact duplicate rows removed:", before - len(df))

# -------------------- 8. HANDLE MISSING ENVIRONMENT VALUES --

# Impute environmental numeric features using their MEDIAN.
# The median is less sensitive to extreme environmental values
# than the mean. Entirely missing columns remain missing.
#
# For rigorous ML evaluation, learn these medians on training
# data only. Here this step creates a cleaned analysis dataset.

for col in environment_columns:
    if col in df.columns and df[col].notna().any():
        df[col] = df[col].fillna(df[col].median())

# Important:
# - Do not impute VALUE, the toxin measurement.
# - Do not invent missing sample dates or times.
# - "Unknown" identifies missing categorical information.

# -------------------- 9. SAVE CLEANED FULL DATASET -----------

full_output = OUTPUT_DIR / "marine_dataset_cleaned_all_toxins.csv"
df.to_csv(full_output, index=False)

print("\nCleaned full dataset shape:", df.shape)
print("Saved:", full_output.resolve())

# -------------------- 10. FILTER DOMOIC ACID ONLY ------------

# Filtering happens only after the cleaning steps above.
abbr_col = "VARIABLE (abbreviation)"
name_col = "VARIABLE (long)"

if abbr_col not in df.columns or name_col not in df.columns:
    raise ValueError(
        "Could not find toxin abbreviation/name columns. "
        f"Available columns: {df.columns.tolist()}"
    )

abbr = df[abbr_col].astype("string").str.strip().str.upper()
name = df[name_col].astype("string").str.strip().str.casefold()

domoic_acid = df[
    abbr.eq("DA") | name.eq("domoic acid")
].copy()

# Keep only the expected shellfish-tissue unit for this project.
unit_col = "UNITS (abbreviation)"

if unit_col in domoic_acid.columns:
    units = (
        domoic_acid[unit_col]
        .astype("string")
        .str.strip()
        .str.casefold()
        .str.replace(" ", "", regex=False)
    )

    # Do not silently convert other units.
    expected_unit = "mgda/kg"
    wrong_units = ~units.eq(expected_unit)

    print(
        "\nDomoic acid rows with unexpected/missing units:",
        int(wrong_units.sum())
    )

    domoic_acid = domoic_acid[~wrong_units].copy()

# Rows without a toxin result cannot be used as concentration targets.
before_missing_target = len(domoic_acid)

domoic_acid = domoic_acid[
    domoic_acid[value_col].notna()
].copy()

print(
    "Domoic acid rows removed due to missing toxin value:",
    before_missing_target - len(domoic_acid)
)

# Preserve the < or = qualifier: these are scientifically different.
qualifier_col = "<=>"

if qualifier_col in domoic_acid.columns:
    domoic_acid[qualifier_col] = (
        domoic_acid[qualifier_col]
        .astype("string").str.strip()
    )

# -------------------- 11. EXPORT FINAL DA DATASET ------------

da_output = OUTPUT_DIR / "toxiguard_domoic_acid_cleaned.csv"
domoic_acid.to_csv(da_output, index=False)

print("\n" + "=" * 60)
print("TOXIGUARD PREPROCESSING SUMMARY")
print("=" * 60)
print("Full cleaned rows:", len(df))
print("Final domoic acid rows:", len(domoic_acid))

if qualifier_col in domoic_acid.columns:
    print("\nDomoic acid measurement qualifiers:")
    print(domoic_acid[qualifier_col].value_counts(dropna=False))

print("\nRemaining missing values in final DA dataset:")
print(domoic_acid.isna().sum().to_string())

print("\nSaved final domoic acid dataset:")
print(da_output.resolve())

print("\nPreprocessing complete.")

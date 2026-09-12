import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\NERProject"

TRAINING_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "landslide_ml_dataset.csv"
)

LIVE_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "live_road_features.csv"
)

LANDSLIDE_MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "lgbm_landslide.pkl"
)

FLOOD_TRAINING_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "flood_ml_dataset_2023.csv"
)


# ============================================================
# LANDSLIDE FEATURES
# ============================================================

LANDSLIDE_FEATURES = [

    "rain_event_day",

    "rain_3d",

    "rain_7d",

    "rain_10d",

    "max_rain_3d",

    "max_rain_7d",

    "heavy_rain_days",

    "elevation_m",

    "slope_deg"

]


# ============================================================
# LIVE FEATURE MAPPING
# ============================================================

LIVE_MAPPING = {

    "rain_event_day":
        "rain_event_day",

    "rain_3d":
        "rain_3d",

    "rain_7d":
        "rain_7d",

    "rain_10d":
        "rain_10d",

    "max_rain_3d":
        "max_rain_3d",

    "max_rain_7d":
        "max_rain_7d",

    "heavy_rain_days":
        "heavy_rain_days",

    "elevation_m":
        "elevation_m",

    "slope_deg":
        "slope_deg"

}


# ============================================================
# LOAD FILES
# ============================================================

print("\n==========================================")
print("STEP 28.5: Live vs training diagnostics")
print("==========================================")


print(
    "\nLoading historical landslide dataset..."
)

train_df = pd.read_csv(
    TRAINING_FILE
)


print(
    f"Training rows: {len(train_df)}"
)


print(
    "\nLoading live road features..."
)

live_df = pd.read_csv(
    LIVE_FILE
)


print(
    f"Live road rows: {len(live_df)}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

for col in LANDSLIDE_FEATURES:

    if col not in train_df.columns:

        raise RuntimeError(
            f"Training data missing: {col}"
        )


for col in LIVE_MAPPING.values():

    if col not in live_df.columns:

        raise RuntimeError(
            f"Live data missing: {col}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading landslide model..."
)


model_package = joblib.load(
    LANDSLIDE_MODEL_FILE
)


if isinstance(
    model_package,
    dict
):

    model = model_package["model"]

else:

    model = model_package


# ============================================================
# COMPARE FEATURE DISTRIBUTIONS
# ============================================================

print(
    "\n=========================================="
)

print(
    "FEATURE DISTRIBUTION COMPARISON"
)

print(
    "=========================================="
)


rows = []


for feature in LANDSLIDE_FEATURES:

    live_column = LIVE_MAPPING[
        feature
    ]


    train_values = pd.to_numeric(
        train_df[feature],
        errors="coerce"
    ).dropna()


    live_values = pd.to_numeric(
        live_df[live_column],
        errors="coerce"
    ).dropna()


    rows.append({

        "feature": feature,

        "train_min":
            train_values.min(),

        "train_p25":
            train_values.quantile(0.25),

        "train_median":
            train_values.median(),

        "train_mean":
            train_values.mean(),

        "train_p75":
            train_values.quantile(0.75),

        "train_max":
            train_values.max(),

        "live_min":
            live_values.min(),

        "live_p25":
            live_values.quantile(0.25),

        "live_median":
            live_values.median(),

        "live_mean":
            live_values.mean(),

        "live_p75":
            live_values.quantile(0.75),

        "live_max":
            live_values.max()

    })


comparison = pd.DataFrame(
    rows
)


pd.set_option(
    "display.max_columns",
    None
)


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# FIND OUT-OF-RANGE LIVE VALUES
# ============================================================

print(
    "\n=========================================="
)

print(
    "LIVE VALUES OUTSIDE TRAINING RANGE"
)

print(
    "=========================================="
)


for feature in LANDSLIDE_FEATURES:

    live_column = LIVE_MAPPING[
        feature
    ]


    train_values = pd.to_numeric(
        train_df[feature],
        errors="coerce"
    ).dropna()


    live_values = pd.to_numeric(
        live_df[live_column],
        errors="coerce"
    )


    train_min = train_values.min()

    train_max = train_values.max()


    below = (
        live_values < train_min
    ).sum()


    above = (
        live_values > train_max
    ).sum()


    total = len(live_values)


    print(
        f"\n{feature}"
    )


    print(
        f"Training range: "
        f"{train_min:.4f} to {train_max:.4f}"
    )


    print(
        f"Below training minimum: "
        f"{below} / {total}"
    )


    print(
        f"Above training maximum: "
        f"{above} / {total}"
    )


# ============================================================
# SPECIAL TERRAIN CHECK
# ============================================================

print(
    "\n=========================================="
)

print(
    "TERRAIN SANITY CHECK"
)

print(
    "=========================================="
)


print(
    "\nHistorical elevation:"
)


print(
    train_df[
        "elevation_m"
    ].describe()
)


print(
    "\nLive elevation:"
)


print(
    live_df[
        "elevation_m"
    ].describe()
)


print(
    "\nHistorical slope:"
)


print(
    train_df[
        "slope_deg"
    ].describe()
)


print(
    "\nLive slope:"
)


print(
    live_df[
        "slope_deg"
    ].describe()
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print(
    "\n=========================================="
)

print(
    "CURRENT LIVE PREDICTION DISTRIBUTION"
)

print(
    "=========================================="
)


live_X = live_df[
    LANDSLIDE_FEATURES
].copy()


for col in live_X.columns:

    live_X[col] = pd.to_numeric(
        live_X[col],
        errors="coerce"
    )


live_probability = (
    model.predict_proba(
        live_X
    )[:, 1]
)


print(
    pd.Series(
        live_probability
    ).describe()
)


print(
    "\nProbability bins:"
)


bins = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0
]


probability_bins = pd.cut(
    live_probability,
    bins=bins,
    include_lowest=True
)

probability_counts = (
    probability_bins
    .value_counts()
    .sort_index()
)

print(
    probability_counts.to_string()
)


# ============================================================
# CHECK IDENTICAL PROBABILITIES
# ============================================================

print(
    "\n=========================================="
)

print(
    "MOST COMMON LANDSLIDE PROBABILITIES"
)

print(
    "=========================================="
)


probability_counts = (

    pd.Series(
        live_probability
    )

    .round(6)

    .value_counts()

    .head(20)

)


print(
    probability_counts.to_string()
)


# ============================================================
# SAVE DIAGNOSTIC REPORT
# ============================================================

diagnostic_file = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "live_vs_training_diagnostics.csv"
)


comparison.to_csv(
    diagnostic_file,
    index=False
)


print(
    f"\nSaved diagnostic report:"
)

print(
    diagnostic_file
)


# ============================================================
# FINAL
# ============================================================

print(
    "\n=========================================="
)

print(
    "STEP 28.5 COMPLETE"
)

print(
    "=========================================="
)

print(
    "\nDo NOT change the model yet."
)

print(
    "First inspect the training/live ranges above."
)

import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

LANDSLIDE_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_landslide_master.csv"
)

CONTROL_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_control_features.csv"
)

OUTPUT_CSV = (
    r"C:\NERProject\Data\processed"
    r"\landslide_ml_dataset.csv"
)


# ============================================================
# 2. Features used by the first landslide model
# ============================================================

FEATURE_COLUMNS = [
    "rain_event_day",
    "rain_1d",
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
# 3. Read data
# ============================================================

print("Reading historical landslides...")

landslides = pd.read_csv(
    LANDSLIDE_CSV
)

print(
    "Landslide records:",
    len(landslides)
)


print("\nReading control samples...")

controls = pd.read_csv(
    CONTROL_CSV
)

print(
    "Control records:",
    len(controls)
)


# ============================================================
# 4. Prepare positive samples
# ============================================================

positive_columns = [
    "event_id",
    "event_date",
    "longitude",
    "latitude"
] + FEATURE_COLUMNS


positives = landslides[
    positive_columns
].copy()


positives["sample_type"] = "landslide"

positives["label"] = 1


# ============================================================
# 5. Prepare control samples
# ============================================================

control_columns = [
    "control_id",
    "event_date",
    "longitude",
    "latitude"
] + FEATURE_COLUMNS


negatives = controls[
    control_columns
].copy()


negatives = negatives.rename(
    columns={
        "control_id": "original_id"
    }
)


negatives["sample_type"] = "control"

negatives["label"] = 0


# ============================================================
# 6. Create a common ID column
# ============================================================

positives = positives.rename(
    columns={
        "event_id": "original_id"
    }
)


# ============================================================
# 7. Combine datasets
# ============================================================

print("\nCombining positive and control samples...")

dataset = pd.concat(
    [
        positives,
        negatives
    ],
    ignore_index=True
)


# ============================================================
# 8. Create unique sample ID
# ============================================================

dataset.insert(
    0,
    "sample_id",
    range(
        1,
        len(dataset) + 1
    )
)


# ============================================================
# 9. Convert date
# ============================================================

dataset["event_date"] = pd.to_datetime(
    dataset["event_date"]
)


# ============================================================
# 10. Sort chronologically
# ============================================================

dataset = dataset.sort_values(
    "event_date"
).reset_index(
    drop=True
)


# Re-create sample IDs after sorting.

dataset["sample_id"] = range(
    1,
    len(dataset) + 1
)


# ============================================================
# 11. Check missing ML features
# ============================================================

print("\nChecking missing feature values...")

missing = dataset[
    FEATURE_COLUMNS
].isna().sum()


print(
    missing
)


# Remove rows only if a required ML feature is missing.

before_rows = len(dataset)


dataset = dataset.dropna(
    subset=FEATURE_COLUMNS
).reset_index(
    drop=True
)


after_rows = len(dataset)


# Re-create IDs after dropping rows.

dataset["sample_id"] = range(
    1,
    len(dataset) + 1
)


print(
    "\nRows before cleaning:",
    before_rows
)

print(
    "Rows after cleaning:",
    after_rows
)

print(
    "Rows removed:",
    before_rows - after_rows
)


# ============================================================
# 12. Check duplicate coordinates
# ============================================================

duplicate_coordinates = dataset.duplicated(
    subset=[
        "longitude",
        "latitude"
    ]
).sum()


print(
    "\nDuplicate coordinates:",
    duplicate_coordinates
)


# ============================================================
# 13. Check unique sample IDs
# ============================================================

duplicate_sample_ids = dataset.duplicated(
    subset=[
        "sample_id"
    ]
).sum()


print(
    "Duplicate sample IDs:",
    duplicate_sample_ids
)


# ============================================================
# 14. Class distribution
# ============================================================

print(
    "\nClass distribution:"
)

print(
    dataset["label"].value_counts()
)


print(
    "\nClass percentages:"
)

print(
    (
        dataset["label"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# 15. Sample type distribution
# ============================================================

print(
    "\nSample type distribution:"
)

print(
    dataset["sample_type"].value_counts()
)


# ============================================================
# 16. Check invalid numeric values
# ============================================================

print(
    "\nChecking invalid numeric values..."
)

invalid_found = False


for column in FEATURE_COLUMNS:

    invalid_count = (
        ~np.isfinite(
            dataset[column]
        )
    ).sum()


    if invalid_count > 0:

        invalid_found = True

        print(
            column,
            "=>",
            invalid_count,
            "invalid values"
        )


if not invalid_found:

    print(
        "No invalid numeric values found."
    )


# ============================================================
# 17. Feature statistics by class
# ============================================================

print(
    "\nFeature means by class:"
)

class_means = (
    dataset
    .groupby("label")[
        FEATURE_COLUMNS
    ]
    .mean()
    .T
)


class_means.columns = [
    "control_mean",
    "landslide_mean"
]


print(
    class_means.round(3).to_string()
)


# ============================================================
# 18. Correlation matrix
# ============================================================

print(
    "\nFeature correlation matrix:"
)

correlation = (
    dataset[
        FEATURE_COLUMNS
    ]
    .corr()
)


print(
    correlation.round(2).to_string()
)


# ============================================================
# 19. Final columns
# ============================================================

FINAL_COLUMNS = [
    "sample_id",
    "original_id",
    "sample_type",
    "event_date",
    "longitude",
    "latitude"
] + FEATURE_COLUMNS + [
    "label"
]


dataset = dataset[
    FINAL_COLUMNS
]


# ============================================================
# 20. Save
# ============================================================

print(
    "\nSaving final ML dataset..."
)

dataset.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 21. Final validation
# ============================================================

print(
    "\n========================================"
)

print(
    "FINAL ML DATASET"
)

print(
    "========================================"
)

print(
    "Total samples:",
    len(dataset)
)

print(
    "Positive samples:",
    (
        dataset["label"] == 1
    ).sum()
)

print(
    "Control samples:",
    (
        dataset["label"] == 0
    ).sum()
)

print(
    "Features:",
    len(FEATURE_COLUMNS)
)

print(
    "Columns:",
    len(dataset.columns)
)

print(
    "Duplicate sample IDs:",
    dataset["sample_id"].duplicated().sum()
)

print(
    "Missing values:",
    dataset.isna().sum().sum()
)

print(
    "Output:",
    OUTPUT_CSV
)


# ============================================================
# 22. First 10 rows
# ============================================================

print(
    "\nFirst 10 rows:"
)

print(
    dataset.head(10).to_string(
        index=False
    )
)


# ============================================================
# 23. Final pass/fail
# ============================================================

if (
    len(dataset) == 382
    and
    dataset["sample_id"].duplicated().sum() == 0
    and
    dataset.isna().sum().sum() == 0
):

    print(
        "\n========================================"
    )

    print(
        "DATASET VALIDATION: PASSED"
    )

    print(
        "========================================"
    )

else:

    print(
        "\n========================================"
    )

    print(
        "DATASET VALIDATION: CHECK REQUIRED"
    )

    print(
        "========================================"
    )


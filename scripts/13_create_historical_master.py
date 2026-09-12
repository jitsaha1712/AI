
import os

import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

LANDSLIDE_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_landslide_rainfall_features.csv"
)

TERRAIN_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_landslide_terrain_features.csv"
)

FLOOD_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_landslide_flood_features.csv"
)

OUTPUT_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_landslide_master.csv"
)


# ============================================================
# 2. Read datasets
# ============================================================

print("Reading rainfall features...")

rainfall = pd.read_csv(LANDSLIDE_CSV)

print(
    "Rainfall records:",
    len(rainfall)
)


print("\nReading terrain features...")

terrain = pd.read_csv(TERRAIN_CSV)

print(
    "Terrain records:",
    len(terrain)
)


print("\nReading flood features...")

flood = pd.read_csv(FLOOD_CSV)

print(
    "Flood records:",
    len(flood)
)


# ============================================================
# 3. Keep only required columns
# ============================================================

terrain = terrain[
    [
        "event_id",
        "elevation_m",
        "slope_deg"
    ]
]


flood = flood[
    [
        "event_id",
        "flood_risk"
    ]
]


# ============================================================
# 4. Rename flood feature
# ============================================================

flood = flood.rename(
    columns={
        "flood_risk": "flood_exposure_2023"
    }
)


# ============================================================
# 5. Merge rainfall + terrain
# ============================================================

print("\nMerging rainfall and terrain...")

master = rainfall.merge(
    terrain,
    on="event_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 6. Merge flood
# ============================================================

print("Merging flood exposure...")

master = master.merge(
    flood,
    on="event_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 7. Sort by event date
# ============================================================

master["event_date"] = pd.to_datetime(
    master["event_date"]
)

master = master.sort_values(
    "event_date"
).reset_index(drop=True)


# ============================================================
# 8. Save master dataset
# ============================================================

print("\nSaving master dataset...")

master.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 9. Final information
# ============================================================

print("\n========================================")
print("Historical master dataset created!")
print("========================================")

print(
    "Total records:",
    len(master)
)

print(
    "Total columns:",
    len(master.columns)
)

print(
    "Output:",
    OUTPUT_CSV
)


# ============================================================
# 10. Show columns
# ============================================================

print("\nColumns:")

for column in master.columns:
    print("-", column)


# ============================================================
# 11. Missing values
# ============================================================

print("\nMissing values:")

print(
    master.isna().sum()
)


# ============================================================
# 12. Duplicate check
# ============================================================

print("\nDuplicate event IDs:")

print(
    master["event_id"].duplicated().sum()
)


# ============================================================
# 13. Numeric feature summary
# ============================================================

print("\nNumeric feature summary:")

print(
    master.describe()
)


# ============================================================
# 14. Show first 5 records
# ============================================================

print("\nFirst 5 records:")

print(
    master.head()
)


# ============================================================
# 15. Show flood-missing events
# ============================================================

missing_flood = master[
    master["flood_exposure_2023"].isna()
]

print("\nEvents without 2023 flood coverage:")

if len(missing_flood) == 0:

    print("None")

else:

    print(
        missing_flood[
            [
                "event_id",
                "event_date",
                "longitude",
                "latitude"
            ]
        ].to_string(index=False)
    )


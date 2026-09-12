
import os

import joblib
import numpy as np
import pandas as pd
import psycopg2


# ============================================================
# 1. Paths
# ============================================================

FLOOD_DATASET_CSV = (
    r"C:\NERProject\Data\processed"
    r"\flood_ml_dataset_2023.csv"
)

LANDSLIDE_MODEL_PATH = (
    r"C:\NERProject\models"
    r"\lgbm_landslide.pkl"
)

FLOOD_MODEL_PATH = (
    r"C:\NERProject\models"
    r"\lgbm_flood.pkl"
)

OUTPUT_CSV = (
    r"C:\NERProject\Data\processed"
    r"\road_risk_predictions.csv"
)


# ============================================================
# 2. Database configuration
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "assam_routing",
    "user": "postgres",
    "password": "postgres"
}


# ============================================================
# 3. Helper
# ============================================================

def print_section(title):

    print(
        "\n"
        + "=" * 55
    )

    print(title)

    print(
        "=" * 55
    )


# ============================================================
# 4. Load landslide model
# ============================================================

print_section(
    "LOADING LANDSLIDE MODEL"
)

if not os.path.exists(
    LANDSLIDE_MODEL_PATH
):

    raise FileNotFoundError(
        "Landslide model not found:\n"
        + LANDSLIDE_MODEL_PATH
    )


landslide_package = joblib.load(
    LANDSLIDE_MODEL_PATH
)

landslide_model = (
    landslide_package["model"]
)

landslide_features = (
    landslide_package["features"]
)

landslide_threshold = (
    landslide_package.get(
        "selected_threshold",
        0.20
    )
)


print(
    "Features:",
    landslide_features
)

print(
    "Selected threshold:",
    landslide_threshold
)


# ============================================================
# 5. Load flood model
# ============================================================

print_section(
    "LOADING FLOOD MODEL"
)

if not os.path.exists(
    FLOOD_MODEL_PATH
):

    raise FileNotFoundError(
        "Flood model not found:\n"
        + FLOOD_MODEL_PATH
    )


flood_package = joblib.load(
    FLOOD_MODEL_PATH
)

flood_model = (
    flood_package["model"]
)

flood_features = (
    flood_package["features"]
)


print(
    "Features:",
    flood_features
)


# ============================================================
# 6. Read flood-model feature dataset
# ============================================================

print_section(
    "READING FLOOD DATASET"
)

flood_dataset = pd.read_csv(
    FLOOD_DATASET_CSV
)


print(
    "Rows:",
    len(flood_dataset)
)


required_flood_columns = [
    "edge_id"
] + flood_features


missing_flood_columns = [
    column
    for column in required_flood_columns
    if column not in flood_dataset.columns
]


if missing_flood_columns:

    raise ValueError(
        "Missing flood dataset columns:\n"
        + str(missing_flood_columns)
    )


# ------------------------------------------------------------
# IMPORTANT:
#
# We only need the actual features for the flood model.
# Duplicated terrain columns will be deliberately renamed
# later so pandas cannot create _x / _y confusion.
# ------------------------------------------------------------

flood_features_df = flood_dataset[
    required_flood_columns
].copy()


# Check duplicate edge IDs.

duplicate_edges = (
    flood_features_df["edge_id"]
    .duplicated()
    .sum()
)


if duplicate_edges > 0:

    raise ValueError(
        "Duplicate edge IDs in flood dataset: "
        + str(duplicate_edges)
    )


# ============================================================
# 7. Connect PostgreSQL
# ============================================================

print_section(
    "CONNECTING TO POSTGRESQL"
)

conn = psycopg2.connect(
    **DB_CONFIG
)

print(
    "Database connected."
)


# ============================================================
# 8. Read routing roads
# ============================================================

print_section(
    "READING ROUTING ROADS"
)


# Historical road rainfall features are joined here.
#
# IMPORTANT:
# road_rainfall_features.edge_id corresponds directly to
# routing_noded.edge_id.
#

roads_query = """
SELECT
    r.edge_id,

    f.rain_mean_mm_day,
    f.rain_max_mm_day,
    f.rain_7d_max_mm,
    f.rain_30d_max_mm,
    f.heavy_rain_days,

    r.elevation_m,
    r.slope_deg,
    r.distance_to_river_m

FROM routing_noded AS r

LEFT JOIN road_rainfall_features AS f
    ON r.edge_id = f.edge_id

ORDER BY r.edge_id;
"""


cur = conn.cursor()

cur.execute(
    roads_query
)

rows = cur.fetchall()

column_names = [
    description[0]
    for description in cur.description
]

cur.close()


roads = pd.DataFrame(
    rows,
    columns=column_names
)


print(
    "Routing road segments:",
    len(roads)
)


# ============================================================
# 9. Validate road rainfall
# ============================================================

print(
    "\nRoads missing rainfall rows:",
    roads[
        "rain_mean_mm_day"
    ].isna().sum()
)


if roads[
    [
        "rain_mean_mm_day",
        "rain_max_mm_day",
        "rain_7d_max_mm",
        "rain_30d_max_mm",
        "heavy_rain_days"
    ]
].isna().any().any():

    raise ValueError(
        "Some road segments have missing "
        "historical rainfall features."
    )


# ============================================================
# 10. Validate static road features
# ============================================================

static_features = [
    "elevation_m",
    "slope_deg",
    "distance_to_river_m"
]


print(
    "\nMissing static features:"
)

print(
    roads[
        static_features
    ].isna().sum()
)


if roads[
    static_features
].isna().any().any():

    raise ValueError(
        "Some road segments have missing "
        "terrain/river features."
    )


# ============================================================
# 11. Prepare landslide-model proxy features
# ============================================================
#
# IMPORTANT:
#
# The historical landslide model was trained using:
#
# rain_event_day
# rain_3d
# rain_7d
# rain_10d
# max_rain_3d
# max_rain_7d
# heavy_rain_days
# elevation_m
# slope_deg
#
# Current road rainfall features are:
#
# rain_max_mm_day
# rain_7d_max_mm
# rain_30d_max_mm
#
# These are not identical definitions.
#
# Therefore the mapping below is explicitly a PROTOTYPE
# approximation.
#
# The final dynamic system will replace these with live
# weather-derived features.
#


roads["rain_event_day"] = (
    roads["rain_max_mm_day"]
)

roads["rain_3d"] = (
    roads["rain_max_mm_day"]
)

roads["rain_7d"] = (
    roads["rain_7d_max_mm"]
)

roads["rain_10d"] = (
    roads["rain_30d_max_mm"]
)

roads["max_rain_3d"] = (
    roads["rain_max_mm_day"]
)

roads["max_rain_7d"] = (
    roads["rain_7d_max_mm"]
)


# ============================================================
# 12. Validate landslide input matrix
# ============================================================

print(
    "\nMissing landslide-model features:"
)

print(
    roads[
        landslide_features
    ].isna().sum()
)


if roads[
    landslide_features
].isna().any().any():

    raise ValueError(
        "Missing landslide-model features."
    )


if not np.isfinite(
    roads[
        landslide_features
    ].to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "Invalid landslide-model feature values."
    )


# ============================================================
# 13. Predict P(landslide)
# ============================================================

print_section(
    "PREDICTING LANDSLIDE PROBABILITY"
)


roads["p_landslide"] = (
    landslide_model.predict_proba(
        roads[
            landslide_features
        ]
    )[:, 1]
)


print(
    roads[
        "p_landslide"
    ].describe()
)


# ============================================================
# 14. Prepare flood feature dataframe
# ============================================================
#
# Since roads already contains elevation_m, slope_deg and
# distance_to_river_m, we DON'T need duplicate versions
# from the flood CSV.
#
# The flood model needs them, so we simply use the road-table
# values directly.
#

flood_required_from_csv = [
    "rain_1d_2023",
    "rain_3d_2023",
    "rain_7d_2023",
    "max_rain_3d_2023",
    "max_rain_7d_2023"
]


missing_rain_features = [
    column
    for column in flood_required_from_csv
    if column not in flood_features_df.columns
]


if missing_rain_features:

    raise ValueError(
        "Missing flood rainfall features:\n"
        + str(missing_rain_features)
    )


# ============================================================
# 15. Join only flood rainfall features
# ============================================================

print_section(
    "JOINING FLOOD FEATURES"
)


flood_rain_df = flood_features_df[
    [
        "edge_id",
        "rain_1d_2023",
        "rain_3d_2023",
        "rain_7d_2023",
        "max_rain_3d_2023",
        "max_rain_7d_2023"
    ]
].copy()


roads = roads.merge(
    flood_rain_df,
    on="edge_id",
    how="left",
    validate="one_to_one"
)


print(
    "Rows after flood join:",
    len(roads)
)


if len(roads) != 63736:

    raise ValueError(
        "Road count changed after flood join."
    )


# ============================================================
# 16. Build exact flood-model feature matrix
# ============================================================

#
# The flood model expects:
#
# rain_1d_2023
# rain_3d_2023
# rain_7d_2023
# max_rain_3d_2023
# max_rain_7d_2023
# elevation_m
# slope_deg
# distance_to_river_m
#
# The last 3 are taken directly from routing_noded.
#

roads["flood_elevation_m"] = (
    roads["elevation_m"]
)

roads["flood_slope_deg"] = (
    roads["slope_deg"]
)

roads["flood_distance_to_river_m"] = (
    roads["distance_to_river_m"]
)


# Build a dataframe using exactly the model's expected names.

flood_input = pd.DataFrame(
    {
        "rain_1d_2023":
            roads["rain_1d_2023"],

        "rain_3d_2023":
            roads["rain_3d_2023"],

        "rain_7d_2023":
            roads["rain_7d_2023"],

        "max_rain_3d_2023":
            roads["max_rain_3d_2023"],

        "max_rain_7d_2023":
            roads["max_rain_7d_2023"],

        "elevation_m":
            roads["flood_elevation_m"],

        "slope_deg":
            roads["flood_slope_deg"],

        "distance_to_river_m":
            roads["flood_distance_to_river_m"]
    }
)


# Reorder exactly according to saved model.

flood_input = flood_input[
    flood_features
]


# ============================================================
# 17. Validate flood matrix
# ============================================================

print(
    "\nMissing flood-model features:"
)

print(
    flood_input.isna().sum()
)


if flood_input.isna().any().any():

    raise ValueError(
        "Missing flood-model feature values."
    )


if not np.isfinite(
    flood_input.to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "Invalid flood-model feature values."
    )


# ============================================================
# 18. Predict P(flood)
# ============================================================

print_section(
    "PREDICTING FLOOD PROBABILITY"
)


roads["p_flood"] = (
    flood_model.predict_proba(
        flood_input
    )[:, 1]
)


print(
    roads[
        "p_flood"
    ].describe()
)


# ============================================================
# 19. Calculate combined hazard
# ============================================================

print_section(
    "CALCULATING COMBINED HAZARD"
)


roads["p_hazard"] = (
    1.0
    -
    (
        1.0
        -
        roads["p_landslide"]
    )
    *
    (
        1.0
        -
        roads["p_flood"]
    )
)


roads["p_landslide"] = (
    roads["p_landslide"]
    .clip(0.0, 1.0)
)

roads["p_flood"] = (
    roads["p_flood"]
    .clip(0.0, 1.0)
)

roads["p_hazard"] = (
    roads["p_hazard"]
    .clip(0.0, 1.0)
)


# ============================================================
# 20. Prototype classifications
# ============================================================

roads["landslide_alert"] = (
    roads["p_landslide"]
    >= landslide_threshold
).astype(int)


roads["hazard_level"] = pd.cut(
    roads["p_hazard"],
    bins=[
        -np.inf,
        0.20,
        0.40,
        0.60,
        np.inf
    ],
    labels=[
        "low",
        "moderate",
        "high",
        "very_high"
    ]
)


# ============================================================
# 21. Print risk statistics
# ============================================================

print_section(
    "ROAD RISK SUMMARY"
)


print(
    "Road segments:",
    len(roads)
)


print(
    "\nP(landslide):"
)

print(
    roads[
        "p_landslide"
    ].describe()
)


print(
    "\nP(flood):"
)

print(
    roads[
        "p_flood"
    ].describe()
)


print(
    "\nP(hazard):"
)

print(
    roads[
        "p_hazard"
    ].describe()
)


print(
    "\nLandslide alert distribution:"
)

print(
    roads[
        "landslide_alert"
    ].value_counts()
)


print(
    "\nHazard level distribution:"
)

print(
    roads[
        "hazard_level"
    ].value_counts(
        dropna=False
    )
)


# ============================================================
# 22. Top 20 risk segments
# ============================================================

print_section(
    "TOP 20 HIGHEST-RISK ROAD SEGMENTS"
)


top_risk = roads[
    [
        "edge_id",
        "p_landslide",
        "p_flood",
        "p_hazard",
        "landslide_alert",
        "hazard_level"
    ]
].sort_values(
    "p_hazard",
    ascending=False
).head(20)


print(
    top_risk.to_string(
        index=False
    )
)


# ============================================================
# 23. Add database columns
# ============================================================

print_section(
    "UPDATING ROUTING_NODES"
)


cur = conn.cursor()


cur.execute(
    """
    ALTER TABLE routing_noded

    ADD COLUMN IF NOT EXISTS
        p_landslide double precision,

    ADD COLUMN IF NOT EXISTS
        p_flood double precision,

    ADD COLUMN IF NOT EXISTS
        p_hazard double precision,

    ADD COLUMN IF NOT EXISTS
        landslide_alert integer,

    ADD COLUMN IF NOT EXISTS
        hazard_level text;
    """
)


conn.commit()


# ============================================================
# 24. Update database
# ============================================================

print(
    "Writing predictions to PostgreSQL..."
)


update_query = """
UPDATE routing_noded

SET
    p_landslide = %s,
    p_flood = %s,
    p_hazard = %s,
    landslide_alert = %s,
    hazard_level = %s

WHERE edge_id = %s;
"""


records = []


for _, row in roads.iterrows():

    records.append(
        (
            float(
                row["p_landslide"]
            ),

            float(
                row["p_flood"]
            ),

            float(
                row["p_hazard"]
            ),

            int(
                row["landslide_alert"]
            ),

            str(
                row["hazard_level"]
            ),

            int(
                row["edge_id"]
            )
        )
    )


cur.executemany(
    update_query,
    records
)


conn.commit()


print(
    "Predictions written."
)


# ============================================================
# 25. Verify database
# ============================================================

print_section(
    "VERIFYING POSTGRESQL"
)


cur.execute(
    """
    SELECT
        COUNT(*) AS total,
        COUNT(p_landslide) AS landslide_count,
        COUNT(p_flood) AS flood_count,
        COUNT(p_hazard) AS hazard_count
    FROM routing_noded;
    """
)


total, landslide_count, flood_count, hazard_count = (
    cur.fetchone()
)


print(
    "Total roads:",
    total
)

print(
    "P(landslide):",
    landslide_count
)

print(
    "P(flood):",
    flood_count
)

print(
    "P(hazard):",
    hazard_count
)


if not (
    total == 63736
    and
    landslide_count == 63736
    and
    flood_count == 63736
    and
    hazard_count == 63736
):

    raise RuntimeError(
        "Database prediction verification failed."
    )


# ============================================================
# 26. Save CSV
# ============================================================

roads[
    [
        "edge_id",
        "p_landslide",
        "p_flood",
        "p_hazard",
        "landslide_alert",
        "hazard_level"
    ]
].to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 27. Close database
# ============================================================

cur.close()

conn.close()


# ============================================================
# 28. Final message
# ============================================================

print_section(
    "ROAD RISK PREDICTION COMPLETED"
)


print(
    "Roads processed:",
    len(roads)
)

print(
    "Prediction CSV:",
    OUTPUT_CSV
)


print(
    "\nPostGIS columns created:"
)

print(
    "p_landslide"
)

print(
    "p_flood"
)

print(
    "p_hazard"
)

print(
    "landslide_alert"
)

print(
    "hazard_level"
)


print(
    "\nExisting columns NOT modified:"
)

print(
    "hazard_risk"
)

print(
    "dynamic_weight"
)


print(
    "\nIMPORTANT:"
)

print(
    "Landslide road rainfall inputs currently use "
    "prototype proxy mappings."
)

print(
    "They will be replaced by live/current and "
    "previous-7-day weather features in the dynamic stage."
)



import os

import joblib

import numpy as np

import pandas as pd

import psycopg2

from psycopg2.extras import execute_values


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\NERProject"


ROAD_FEATURE_FILE = os.path.join(
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


FLOOD_MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "lgbm_flood.pkl"
)


OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "Data",
    "processed",
    "live_road_risk_predictions.csv"
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {

    "host": "localhost",

    "port": 5432,

    "dbname": "assam_routing",

    "user": "postgres",

    "password": "postgres"   # use your existing PostgreSQL password

}


# ============================================================
# SETTINGS
# ============================================================

# Threshold selected previously during grouped OOF tuning.

LANDSLIDE_DEFAULT_THRESHOLD = 0.20


# Hazard penalty used for routing.

RISK_MULTIPLIER = 5.0


# ============================================================
# HELPER
# LOAD JOBLIB MODEL PACKAGE
# ============================================================

def load_model_package(path):

    print(
        f"\nLoading model: {path}"
    )


    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model file not found:\n{path}"
        )


    # The models were saved using joblib.dump(),
    # so they must be loaded using joblib.load().

    package = joblib.load(
        path
    )


    print(
        "Loaded object type:",
        type(package)
    )


    return package


# ============================================================
# HELPER
# EXTRACT MODEL FROM PACKAGE
# ============================================================

def extract_model(package):


    # --------------------------------------------------------
    # Expected package:
    #
    # {
    #     "model": final_model,
    #     "features": [...]
    # }
    # --------------------------------------------------------

    if isinstance(
        package,
        dict
    ):


        possible_keys = [

            "model",

            "classifier",

            "lgbm",

            "estimator"

        ]


        for key in possible_keys:

            if key in package:

                obj = package[key]


                if hasattr(
                    obj,
                    "predict_proba"
                ):

                    return obj


        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        for key, value in package.items():

            if hasattr(
                value,
                "predict_proba"
            ):

                return value


    # --------------------------------------------------------
    # Raw model
    # --------------------------------------------------------

    if hasattr(
        package,
        "predict_proba"
    ):

        return package


    raise RuntimeError(

        "Could not find a predict_proba model "
        "inside the model package."

    )


# ============================================================
# HELPER
# EXTRACT FEATURE NAMES
# ============================================================

def extract_features(

    package,

    model,

    default_features

):


    # --------------------------------------------------------
    # First check saved package metadata.
    # --------------------------------------------------------

    if isinstance(
        package,
        dict
    ):


        possible_keys = [

            "features",

            "feature_names",

            "feature_cols",

            "columns",

            "input_features"

        ]


        for key in possible_keys:

            if key in package:

                value = package[key]


                if isinstance(

                    value,

                    (

                        list,

                        tuple,

                        np.ndarray

                    )

                ):

                    return list(value)


    # --------------------------------------------------------
    # LightGBM feature names
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_name_"
    ):

        names = model.feature_name_


        if names:

            return list(names)


    # --------------------------------------------------------
    # sklearn feature names
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        return list(
            model.feature_names_in_
        )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return list(
        default_features
    )


# ============================================================
# HELPER
# EXTRACT LANDSLIDE THRESHOLD
# ============================================================

def extract_threshold(

    package,

    default_threshold

):


    if isinstance(
        package,
        dict
    ):


        possible_keys = [

            "threshold",

            "decision_threshold",

            "best_threshold"

        ]


        for key in possible_keys:

            if key in package:

                try:

                    return float(
                        package[key]
                    )

                except (

                    TypeError,

                    ValueError

                ):

                    pass


    return float(
        default_threshold
    )


# ============================================================
# STEP 1
# LOAD LIVE ROAD FEATURES
# ============================================================

print("\n==========================================")
print("STEP 1: Loading live road features")
print("==========================================")


if not os.path.exists(
    ROAD_FEATURE_FILE
):

    raise FileNotFoundError(
        ROAD_FEATURE_FILE
    )


roads = pd.read_csv(
    ROAD_FEATURE_FILE
)


print(
    f"Road rows: {len(roads)}"
)


if len(roads) == 0:

    raise RuntimeError(
        "live_road_features.csv is empty."
    )


# ------------------------------------------------------------
# Validate edge_id
# ------------------------------------------------------------

if "edge_id" not in roads.columns:

    raise RuntimeError(
        "live_road_features.csv does not contain edge_id."
    )


# ------------------------------------------------------------
# Check duplicate edge IDs
# ------------------------------------------------------------

duplicate_edges = (
    roads["edge_id"]
    .duplicated()
    .sum()
)


print(
    f"Duplicate edge IDs: {duplicate_edges}"
)


if duplicate_edges > 0:

    raise RuntimeError(
        "Duplicate edge_id values found."
    )


# ============================================================
# STEP 2
# LOAD LANDSLIDE MODEL
# ============================================================

print("\n==========================================")
print("STEP 2: Loading landslide model")
print("==========================================")


landslide_package = load_model_package(
    LANDSLIDE_MODEL_FILE
)


landslide_model = extract_model(
    landslide_package
)


landslide_default_features = [

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


landslide_features = extract_features(

    landslide_package,

    landslide_model,

    landslide_default_features

)


landslide_threshold = extract_threshold(

    landslide_package,

    LANDSLIDE_DEFAULT_THRESHOLD

)


print(
    "\nLandslide model features:"
)


for feature in landslide_features:

    print(
        f"  - {feature}"
    )


print(
    f"\nLandslide threshold: "
    f"{landslide_threshold:.4f}"
)


# ============================================================
# CHECK LANDSLIDE FEATURES
# ============================================================

missing_land = [

    col

    for col in landslide_features

    if col not in roads.columns

]


if len(missing_land) > 0:

    raise RuntimeError(

        "Missing landslide model features:\n"

        +

        "\n".join(
            missing_land
        )

    )


# ============================================================
# STEP 3
# LOAD FLOOD MODEL
# ============================================================

print("\n==========================================")
print("STEP 3: Loading flood model")
print("==========================================")


flood_package = load_model_package(
    FLOOD_MODEL_FILE
)


flood_model = extract_model(
    flood_package
)


flood_default_features = [

    "rain_1d_2023",

    "rain_3d_2023",

    "rain_7d_2023",

    "max_rain_3d_2023",

    "max_rain_7d_2023",

    "elevation_m",

    "slope_deg",

    "distance_to_river_m"

]


flood_features = extract_features(

    flood_package,

    flood_model,

    flood_default_features

)


print(
    "\nFlood model features:"
)


for feature in flood_features:

    print(
        f"  - {feature}"
    )


# ============================================================
# FLOOD MODEL FEATURE MAPPING
# ============================================================

flood_feature_mapping = {

    "rain_1d_2023":
        "rain_1d_live",

    "rain_3d_2023":
        "rain_3d_live",

    "rain_7d_2023":
        "rain_7d_live",

    "max_rain_3d_2023":
        "max_rain_3d_live",

    "max_rain_7d_2023":
        "max_rain_7d_live",

    "elevation_m":
        "elevation_m",

    "slope_deg":
        "slope_deg",

    "distance_to_river_m":
        "distance_to_river_m"

}


flood_input_columns = []


for feature in flood_features:


    if feature in flood_feature_mapping:

        actual_column = (
            flood_feature_mapping[feature]
        )

    else:

        actual_column = feature


    if actual_column not in roads.columns:

        raise RuntimeError(

            f"Flood model requires '{feature}', "
            f"but the live road data does not contain "
            f"'{actual_column}'."

        )


    flood_input_columns.append(
        actual_column
    )


print(
    "\nFlood input columns:"
)


for feature in flood_input_columns:

    print(
        f"  - {feature}"
    )


# ============================================================
# STEP 4
# CHECK TERRAIN DATA
# ============================================================

print("\n==========================================")
print("STEP 4: Checking terrain features")
print("==========================================")


terrain_columns = [

    "elevation_m",

    "slope_deg",

    "distance_to_river_m"

]


for col in terrain_columns:

    if col not in roads.columns:

        raise RuntimeError(
            f"Missing terrain column: {col}"
        )


print(
    roads[
        terrain_columns
    ].describe()
)


print(
    "\nElevation range:"
)


print(
    f"Minimum : "
    f"{roads['elevation_m'].min():.3f} m"
)


print(
    f"Median  : "
    f"{roads['elevation_m'].median():.3f} m"
)


print(
    f"Maximum : "
    f"{roads['elevation_m'].max():.3f} m"
)


# ------------------------------------------------------------
# Warning only.
# We do not modify elevation automatically.
# ------------------------------------------------------------

if roads[
    "elevation_m"
].min() < -5:

    print(
        "\nWARNING:"
    )

    print(
        "Some elevation values are below -5 m."
    )

    print(
        "Current values will be passed to the model unchanged."
    )


# ============================================================
# STEP 5
# PREPARE MODEL INPUTS
# ============================================================

print("\n==========================================")
print("STEP 5: Preparing model inputs")
print("==========================================")


land_X = roads[
    landslide_features
].copy()


flood_X = roads[
    flood_input_columns
].copy()


# ------------------------------------------------------------
# Convert to numeric
# ------------------------------------------------------------

for col in land_X.columns:

    land_X[col] = pd.to_numeric(
        land_X[col],
        errors="coerce"
    )


for col in flood_X.columns:

    flood_X[col] = pd.to_numeric(
        flood_X[col],
        errors="coerce"
    )


# ------------------------------------------------------------
# Check NaN values
# ------------------------------------------------------------

land_missing = (
    land_X.isna().sum()
)


if land_missing.sum() > 0:

    print(
        "\nLandslide missing values:"
    )

    print(
        land_missing
    )

    raise RuntimeError(
        "Missing landslide model inputs."
    )


flood_missing = (
    flood_X.isna().sum()
)


if flood_missing.sum() > 0:

    print(
        "\nFlood missing values:"
    )

    print(
        flood_missing
    )

    raise RuntimeError(
        "Missing flood model inputs."
    )


# ------------------------------------------------------------
# Check infinite values
# ------------------------------------------------------------

if not np.isfinite(

    land_X.to_numpy(
        dtype=float
    )

).all():

    raise RuntimeError(
        "Invalid or infinite landslide model inputs."
    )


if not np.isfinite(

    flood_X.to_numpy(
        dtype=float
    )

).all():

    raise RuntimeError(
        "Invalid or infinite flood model inputs."
    )


# ============================================================
# STEP 6
# LANDSLIDE PREDICTION
# ============================================================

print("\n==========================================")
print("STEP 6: Predicting landslide probability")
print("==========================================")


land_probs = (

    landslide_model
    .predict_proba(
        land_X
    )[:, 1]

)


land_probs = np.clip(

    land_probs,

    0.0,

    1.0

)


roads[
    "p_landslide"
] = land_probs


roads[
    "landslide_alert"
] = (

    roads[
        "p_landslide"
    ]

    >= landslide_threshold

).astype(
    int
)


print(
    "\nLandslide probability summary:"
)


print(
    roads[
        "p_landslide"
    ].describe()
)


print(
    "\nLandslide alerts:",
    int(
        roads[
            "landslide_alert"
        ].sum()
    )
)


# ============================================================
# STEP 7
# FLOOD PREDICTION
# ============================================================

print("\n==========================================")
print("STEP 7: Predicting flood probability")
print("==========================================")


flood_probs = (

    flood_model
    .predict_proba(
        flood_X
    )[:, 1]

)


flood_probs = np.clip(

    flood_probs,

    0.0,

    1.0

)


roads[
    "p_flood"
] = flood_probs


print(
    "\nFlood probability summary:"
)


print(
    roads[
        "p_flood"
    ].describe()
)


# ============================================================
# STEP 8
# COMBINE HAZARDS
# ============================================================

print("\n==========================================")
print("STEP 8: Combining hazards")
print("==========================================")


# ------------------------------------------------------------
# P_hazard =
#
# 1 - (1 - P_landslide)(1 - P_flood)
# ------------------------------------------------------------

roads[
    "p_hazard"
] = (

    1.0

    -

    (

        (1.0 - roads[
            "p_landslide"
        ])

        *

        (1.0 - roads[
            "p_flood"
        ])

    )

)


roads[
    "p_hazard"
] = np.clip(

    roads[
        "p_hazard"
    ],

    0.0,

    1.0

)


# ============================================================
# HAZARD LEVEL
# ============================================================

def get_hazard_level(p):

    if p < 0.20:

        return "low"

    elif p < 0.40:

        return "medium"

    elif p < 0.60:

        return "high"

    else:

        return "very_high"


roads[
    "hazard_level"
] = (

    roads[
        "p_hazard"
    ]

    .apply(
        get_hazard_level
    )

)


# ============================================================
# STEP 9
# DYNAMIC ROUTING WEIGHT
# ============================================================

print("\n==========================================")
print("STEP 9: Creating dynamic routing weight")
print("==========================================")


# ------------------------------------------------------------
# weight =
#
# length_km *
# (1 + RISK_MULTIPLIER * p_hazard)
# ------------------------------------------------------------

roads[
    "dynamic_weight"
] = (

    roads[
        "length_km"
    ]

    *

    (

        1.0

        +

        RISK_MULTIPLIER

        *

        roads[
            "p_hazard"
        ]

    )

)


roads[
    "dynamic_weight"
] = (

    roads[
        "dynamic_weight"
    ]

    .clip(
        lower=0.000001
    )

)


# ============================================================
# STEP 10
# RISK SUMMARY
# ============================================================

print("\n==========================================")
print("STEP 10: Risk summary")
print("==========================================")


print(
    "\nProbability summary:"
)


print(
    roads[
        [
            "p_landslide",
            "p_flood",
            "p_hazard"
        ]
    ].describe()
)


print(
    "\nHazard levels:"
)


print(
    roads[
        "hazard_level"
    ].value_counts()
)


print(
    "\nLandslide alerts:"
)


print(
    roads[
        "landslide_alert"
    ].value_counts()
)


print(
    "\nDynamic weight summary:"
)


print(
    roads[
        "dynamic_weight"
    ].describe()
)


# ============================================================
# STEP 11
# SAVE PREDICTION CSV
# ============================================================

print("\n==========================================")
print("STEP 11: Saving prediction CSV")
print("==========================================")


output_columns = [

    "edge_id",

    "grid_id",

    "longitude",

    "latitude",

    "length_km",

    "elevation_m",

    "slope_deg",

    "distance_to_river_m",

    "current_rain_mm",

    "rain_24h_mm",

    "rain_3d_mm",

    "rain_7d_mm",

    "rain_10d_mm",

    "max_rain_3d_mm",

    "max_rain_7d_mm",

    "heavy_rain_days",

    "current_temperature_c",

    "current_visibility_m",

    "current_wind_kmh",

    "current_wind_gust_kmh",

    "p_landslide",

    "p_flood",

    "p_hazard",

    "landslide_alert",

    "hazard_level",

    "dynamic_weight"

]


output_columns = [

    col

    for col in output_columns

    if col in roads.columns

]


roads[
    output_columns
].to_csv(

    OUTPUT_FILE,

    index=False

)


print(
    f"Saved:\n{OUTPUT_FILE}"
)


# ============================================================
# STEP 12
# UPDATE POSTGRESQL
# ============================================================

print("\n==========================================")
print("STEP 12: Updating PostgreSQL")
print("==========================================")


# ------------------------------------------------------------
# These are candidate columns.
#
# We will update ONLY columns that exist in both:
#
# 1. roads DataFrame
# 2. dynamic_edge_state
# ------------------------------------------------------------

candidate_update_columns = [

    "edge_id",

    "p_landslide",

    "p_flood",

    "p_hazard",

    "dynamic_weight",

    "rain_24h_mm",

    "rain_1h_mm",

    "visibility_m",

    "wind_speed_kmh"

]


# ============================================================
# STEP 12A
# CONNECT TO DATABASE
# ============================================================

try:

    conn = psycopg2.connect(
        **DB_CONFIG
    )

except Exception as e:

    raise RuntimeError(
        f"Could not connect to PostgreSQL:\n{e}"
    )


try:

    # ========================================================
    # STEP 12B
    # READ DATABASE COLUMNS
    # ========================================================

    with conn.cursor() as cur:

        cur.execute(

            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'dynamic_edge_state'
            ORDER BY ordinal_position;
            """

        )

        db_columns = {

            row[0]

            for row in cur.fetchall()

        }


    print(
        "\ndynamic_edge_state columns:"
    )


    print(
        sorted(
            db_columns
        )
    )


    # ========================================================
    # STEP 12C
    # FIND COMMON COLUMNS
    # ========================================================

    road_available_columns = [

        col

        for col in candidate_update_columns

        if col in roads.columns

    ]


    print(
        "\nColumns available in road data:"
    )


    print(
        road_available_columns
    )


    available_updates = [

        col

        for col in road_available_columns

        if col in db_columns

    ]


    print(
        "\nColumns that will be updated:"
    )


    print(
        available_updates
    )


    # --------------------------------------------------------
    # edge_id is required.
    # --------------------------------------------------------

    if "edge_id" not in available_updates:

        raise RuntimeError(

            "edge_id is not available in both "
            "road data and dynamic_edge_state."

        )


    if len(available_updates) <= 1:

        raise RuntimeError(
            "No risk columns are available for update."
        )


    # ========================================================
    # STEP 12D
    # CREATE TEMPORARY TABLE
    # ========================================================

    temp_column_definitions = []


    for col in available_updates:

        if col == "edge_id":

            temp_column_definitions.append(
                '"edge_id" BIGINT'
            )

        else:

            temp_column_definitions.append(
                f'"{col}" DOUBLE PRECISION'
            )


    with conn.cursor() as cur:

        cur.execute(
            "DROP TABLE IF EXISTS temp_live_risk;"
        )


        cur.execute(

            f"""
            CREATE TEMP TABLE temp_live_risk (

                {", ".join(
                    temp_column_definitions
                )}

            )
            ON COMMIT DROP;
            """

        )


    # ========================================================
    # STEP 12E
    # PREPARE RECORDS
    # ========================================================

    records = []


    for _, row in roads[
        available_updates
    ].iterrows():


        values = []


        for col in available_updates:

            value = row[col]


            # ------------------------------------------------
            # Convert NumPy / pandas values into native
            # Python values before psycopg2 receives them.
            # ------------------------------------------------

            if pd.isna(value):

                values.append(
                    None
                )

            elif col == "edge_id":

                values.append(
                    int(value)
                )

            else:

                values.append(
                    float(value)
                )


        records.append(
            tuple(values)
        )


    print(
        f"\nPrepared rows: {len(records)}"
    )


    if len(records) != len(roads):

        raise RuntimeError(

            "Prepared row count does not match "
            "road row count."

        )


    # ========================================================
    # STEP 12F
    # INSERT TEMPORARY DATA
    # ========================================================

    column_sql = ", ".join(

        f'"{col}"'

        for col in available_updates

    )


    with conn.cursor() as cur:

        execute_values(

            cur,

            f"""
            INSERT INTO temp_live_risk
            ({column_sql})
            VALUES %s
            """,

            records,

            page_size=5000

        )


    print(
        "Temporary risk table populated."
    )


    # ========================================================
    # STEP 12G
    # BUILD UPDATE SET CLAUSE
    # ========================================================

    set_columns = [

        col

        for col in available_updates

        if col != "edge_id"

    ]


    # IMPORTANT:
    #
    # Do NOT write:
    #
    # d."p_hazard" = t."p_hazard"
    #
    # PostgreSQL does not allow the target alias on the
    # left side of SET.
    #
    # Correct:
    #
    # "p_hazard" = t."p_hazard"

    set_sql = ", ".join(

        f'"{col}" = t."{col}"'

        for col in set_columns

    )


    # ========================================================
    # STEP 12H
    # UPDATE DATABASE
    # ========================================================

    with conn.cursor() as cur:

        cur.execute(

            f"""
            UPDATE dynamic_edge_state AS d

            SET

                {set_sql}

            FROM temp_live_risk AS t

            WHERE d.edge_id = t.edge_id;
            """

        )

        updated_rows = cur.rowcount


    print(
        f"\nUpdated dynamic_edge_state rows: "
        f"{updated_rows}"
    )


    # ========================================================
    # STEP 12I
    # VERIFY UPDATE COUNT
    # ========================================================

    if updated_rows != len(roads):

        raise RuntimeError(

            f"Database update incomplete. "
            f"Expected {len(roads)} rows, "
            f"but updated {updated_rows} rows."

        )


    # ========================================================
    # STEP 12J
    # COMMIT
    # ========================================================

    conn.commit()


    print(
        "\nPostgreSQL update committed successfully."
    )


finally:

    conn.close()


# ============================================================
# FINAL
# ============================================================

print("\n==========================================")
print("STEP 27 COMPLETE")
print("==========================================")


print(
    "\nLive prediction file:"
)


print(
    OUTPUT_FILE
)


print(
    "\nUpdated PostgreSQL table:"
)


print(
    "dynamic_edge_state"
)


print(
    "\nUpdated columns:"
)


for col in available_updates:

    print(
        f"  - {col}"
    )


print(
    "\nNext step:"
)


print(
    "Step 28: Dynamic pgRouting using dynamic_weight"
)


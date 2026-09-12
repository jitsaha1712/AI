
import random

import pandas as pd
import psycopg2


# ============================================================
# 1. Configuration
# ============================================================

LANDSLIDE_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_landslide_master.csv"
)

OUTPUT_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_control_points.csv"
)

NUM_CONTROLS = 300

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


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
# 3. Read historical landslides
# ============================================================

print("Reading historical landslides...")

landslides = pd.read_csv(LANDSLIDE_CSV)

print(
    "Landslide events:",
    len(landslides)
)


# ============================================================
# 4. Calculate study area
# ============================================================

# Convert NumPy values to normal Python floats.
# This prevents psycopg2 from sending values like np.float64(...)

min_lon = float(landslides["longitude"].min())
max_lon = float(landslides["longitude"].max())

min_lat = float(landslides["latitude"].min())
max_lat = float(landslides["latitude"].max())


print("\nStudy area:")

print(
    "Longitude:",
    min_lon,
    "to",
    max_lon
)

print(
    "Latitude:",
    min_lat,
    "to",
    max_lat
)


# ============================================================
# 5. Connect to PostgreSQL
# ============================================================

print("\nConnecting to PostgreSQL...")

conn = psycopg2.connect(**DB_CONFIG)

cur = conn.cursor()

print("Database connected!")


# ============================================================
# 6. Generate control points
# ============================================================

print("\nGenerating control points...")

controls = []

attempts = 0

max_attempts = NUM_CONTROLS * 100


while len(controls) < NUM_CONTROLS and attempts < max_attempts:

    attempts += 1


    # --------------------------------------------------------
    # Generate random longitude and latitude
    # --------------------------------------------------------

    lon = float(
        random.uniform(
            min_lon,
            max_lon
        )
    )

    lat = float(
        random.uniform(
            min_lat,
            max_lat
        )
    )


    # --------------------------------------------------------
    # Check distance from historical landslides
    # --------------------------------------------------------
    #
    # The generated point must be at least 5 km away from
    # every known historical landslide.
    #
    # 5000 metres = 5 km
    #

    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM historical_landslides
            WHERE ST_DWithin(
                geometry,
                ST_Transform(
                    ST_SetSRID(
                        ST_MakePoint(%s, %s),
                        4326
                    ),
                    32645
                ),
                5000
            )
        );
        """,
        (
            lon,
            lat
        )
    )


    too_close = cur.fetchone()[0]


    # --------------------------------------------------------
    # Reject point if it is too close
    # --------------------------------------------------------

    if too_close:

        continue


    # --------------------------------------------------------
    # Accept control point
    # --------------------------------------------------------

    controls.append(
        {
            "control_id": len(controls) + 1,
            "longitude": lon,
            "latitude": lat,
            "label": 0
        }
    )


# ============================================================
# 7. Close database connection
# ============================================================

cur.close()

conn.close()


# ============================================================
# 8. Convert to DataFrame
# ============================================================

controls_df = pd.DataFrame(
    controls
)


# ============================================================
# 9. Save control points
# ============================================================

controls_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 10. Results
# ============================================================

print("\n========================================")
print("Control point generation completed!")
print("========================================")

print(
    "Control points generated:",
    len(controls_df)
)

print(
    "Attempts:",
    attempts
)

print(
    "Output:",
    OUTPUT_CSV
)


# ============================================================
# 11. Check whether target was reached
# ============================================================

if len(controls_df) < NUM_CONTROLS:

    print("\nWARNING:")

    print(
        "Only",
        len(controls_df),
        "control points were generated."
    )

    print(
        "The maximum number of attempts was reached."
    )

else:

    print(
        "\nSuccessfully generated all",
        NUM_CONTROLS,
        "control points."
    )


# ============================================================
# 12. Show first 10 control points
# ============================================================

print("\nFirst 10 control points:")

print(
    controls_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 13. Check labels
# ============================================================

print("\nLabel distribution:")

print(
    controls_df["label"].value_counts()
)


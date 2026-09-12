
import os
import random

import numpy as np
import pandas as pd
import psycopg2
import rasterio


# ============================================================
# 1. Paths
# ============================================================

LANDSLIDE_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_landslide_master.csv"
)

DEM_DIR = (
    r"C:\NERProject\Data\dem"
)

OUTPUT_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_control_points.csv"
)


# ============================================================
# 2. Configuration
# ============================================================

NUM_CONTROLS = 300

RANDOM_SEED = 42

# Minimum distance from a known historical landslide
MIN_DISTANCE_FROM_LANDSLIDE_METERS = 5000

# Minimum distance between two control points
MIN_DISTANCE_BETWEEN_CONTROLS_METERS = 1000

# Number of random attempts allowed
MAX_ATTEMPTS_MULTIPLIER = 3000


random.seed(RANDOM_SEED)


# ============================================================
# 3. Database configuration
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "assam_routing",
    "user": "postgres",
    "password": "postgres"
}


# ============================================================
# 4. Read historical landslides
# ============================================================

print("Reading historical landslides...")

landslides = pd.read_csv(
    LANDSLIDE_CSV
)

print(
    "Landslide events:",
    len(landslides)
)


# ============================================================
# 5. Historical study area
# ============================================================

# We keep the controls inside the same geographical area
# covered by the historical landslide observations.

study_min_lon = float(
    landslides["longitude"].min()
)

study_max_lon = float(
    landslides["longitude"].max()
)

study_min_lat = float(
    landslides["latitude"].min()
)

study_max_lat = float(
    landslides["latitude"].max()
)


print("\nHistorical study area:")

print(
    "Longitude:",
    study_min_lon,
    "to",
    study_max_lon
)

print(
    "Latitude:",
    study_min_lat,
    "to",
    study_max_lat
)


# ============================================================
# 6. Scan DEM tiles
# ============================================================

print("\nScanning DEM tiles...")

dem_tiles = []


for filename in os.listdir(DEM_DIR):

    if not filename.lower().endswith(".tif"):
        continue


    path = os.path.join(
        DEM_DIR,
        filename
    )


    try:

        with rasterio.open(path) as src:

            dem_tiles.append(
                {
                    "filename": filename,
                    "path": path,
                    "left": float(src.bounds.left),
                    "right": float(src.bounds.right),
                    "bottom": float(src.bounds.bottom),
                    "top": float(src.bounds.top)
                }
            )


            print(
                filename,
                "=>",
                src.bounds
            )


    except Exception as e:

        print(
            "Could not read:",
            filename,
            "|",
            e
        )


print(
    "\nDEM tiles found:",
    len(dem_tiles)
)


if len(dem_tiles) == 0:

    raise RuntimeError(
        "No valid DEM tiles were found."
    )


# ============================================================
# 7. Find DEM tile containing a coordinate
# ============================================================

def find_dem_tile(
    longitude,
    latitude
):

    for tile in dem_tiles:

        if (
            tile["left"] <= longitude <= tile["right"]
            and
            tile["bottom"] <= latitude <= tile["top"]
        ):

            return tile


    return None


# ============================================================
# 8. Check whether DEM pixel is valid
# ============================================================

def valid_dem_point(
    longitude,
    latitude
):

    tile = find_dem_tile(
        longitude,
        latitude
    )


    if tile is None:

        return False


    try:

        with rasterio.open(
            tile["path"]
        ) as src:

            row, col = src.index(
                longitude,
                latitude
            )


            # ------------------------------------------------
            # Check pixel boundaries
            # ------------------------------------------------

            if (
                row < 0
                or row >= src.height
                or col < 0
                or col >= src.width
            ):

                return False


            # ------------------------------------------------
            # Read center pixel
            # ------------------------------------------------

            value = src.read(
                1,
                window=rasterio.windows.Window(
                    col,
                    row,
                    1,
                    1
                )
            )[0, 0]


            # ------------------------------------------------
            # Reject NoData
            # ------------------------------------------------

            if src.nodata is not None:

                if value == src.nodata:

                    return False


            # ------------------------------------------------
            # Reject NaN / infinite values
            # ------------------------------------------------

            if not np.isfinite(value):

                return False


            # ------------------------------------------------
            # Check a small neighbourhood too.
            #
            # This helps avoid points where the center pixel
            # is technically valid but the surrounding DEM
            # contains mostly NoData.
            # ------------------------------------------------

            row_start = max(
                0,
                row - 1
            )

            row_end = min(
                src.height,
                row + 2
            )

            col_start = max(
                0,
                col - 1
            )

            col_end = min(
                src.width,
                col + 2
            )


            window = src.read(
                1,
                window=rasterio.windows.Window(
                    col_start,
                    row_start,
                    col_end - col_start,
                    row_end - row_start
                )
            ).astype(float)


            if src.nodata is not None:

                window[
                    window == src.nodata
                ] = np.nan


            valid_count = np.isfinite(
                window
            ).sum()


            # Need at least 3 valid cells to calculate
            # a sensible local slope later.

            if valid_count < 3:

                return False


            return True


    except Exception:

        return False


# ============================================================
# 9. Connect to PostgreSQL
# ============================================================

print("\nConnecting to PostgreSQL...")

conn = psycopg2.connect(
    **DB_CONFIG
)

cur = conn.cursor()

print(
    "Database connected!"
)


# ============================================================
# 10. Generate control points
# ============================================================

print(
    "\nGenerating valid DEM-covered control points..."
)

controls = []

attempts = 0

max_attempts = (
    NUM_CONTROLS
    *
    MAX_ATTEMPTS_MULTIPLIER
)


while (
    len(controls) < NUM_CONTROLS
    and
    attempts < max_attempts
):

    attempts += 1


    # --------------------------------------------------------
    # Random point within historical study area
    # --------------------------------------------------------

    longitude = float(
        random.uniform(
            study_min_lon,
            study_max_lon
        )
    )

    latitude = float(
        random.uniform(
            study_min_lat,
            study_max_lat
        )
    )


    # --------------------------------------------------------
    # Check actual DEM pixel
    # --------------------------------------------------------

    if not valid_dem_point(
        longitude,
        latitude
    ):

        continue


    # --------------------------------------------------------
    # Check distance from historical landslides
    # --------------------------------------------------------

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
                %s
            )
        );
        """,
        (
            longitude,
            latitude,
            MIN_DISTANCE_FROM_LANDSLIDE_METERS
        )
    )


    too_close_to_landslide = (
        cur.fetchone()[0]
    )


    if too_close_to_landslide:

        continue


    # --------------------------------------------------------
    # Check distance from existing control points
    # --------------------------------------------------------

    if len(controls) > 0:

        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM unnest(%s::double precision[]) AS x
                JOIN unnest(%s::double precision[]) AS y
                ON ST_DWithin(
                    ST_Transform(
                        ST_SetSRID(
                            ST_MakePoint(x, y),
                            4326
                        ),
                        32645
                    ),
                    ST_Transform(
                        ST_SetSRID(
                            ST_MakePoint(%s, %s),
                            4326
                        ),
                        32645
                    ),
                    %s
                )
            );
            """,
            (
                [p["longitude"] for p in controls],
                [p["latitude"] for p in controls],
                longitude,
                latitude,
                MIN_DISTANCE_BETWEEN_CONTROLS_METERS
            )
        )


        too_close_to_control = (
            cur.fetchone()[0]
        )


        if too_close_to_control:

            continue


    # --------------------------------------------------------
    # Accept control point
    # --------------------------------------------------------

    controls.append(
        {
            "control_id": len(controls) + 1,
            "longitude": longitude,
            "latitude": latitude,
            "label": 0
        }
    )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if len(controls) % 25 == 0:

        print(
            "Controls generated:",
            len(controls),
            "/",
            NUM_CONTROLS,
            "| Attempts:",
            attempts
        )


# ============================================================
# 11. Close PostgreSQL
# ============================================================

cur.close()

conn.close()


# ============================================================
# 12. Convert to DataFrame
# ============================================================

controls_df = pd.DataFrame(
    controls
)


# ============================================================
# 13. Save
# ============================================================

controls_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 14. Final generation result
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
# 15. Check whether target was reached
# ============================================================

if len(controls_df) < NUM_CONTROLS:

    print(
        "\nWARNING:"
    )

    print(
        "Only",
        len(controls_df),
        "control points were generated."
    )

    print(
        "Maximum attempts were reached."
    )

else:

    print(
        "\nSuccessfully generated all",
        NUM_CONTROLS,
        "control points."
    )


# ============================================================
# 16. Verify DEM validity
# ============================================================

print(
    "\nVerifying DEM validity..."
)

invalid_dem_count = 0


for _, row in controls_df.iterrows():

    longitude = float(
        row["longitude"]
    )

    latitude = float(
        row["latitude"]
    )


    if not valid_dem_point(
        longitude,
        latitude
    ):

        invalid_dem_count += 1


print(
    "Controls with invalid DEM:",
    invalid_dem_count
)


# ============================================================
# 17. Verify study area
# ============================================================

outside_study_area = 0


for _, row in controls_df.iterrows():

    longitude = float(
        row["longitude"]
    )

    latitude = float(
        row["latitude"]
    )


    if not (
        study_min_lon <= longitude <= study_max_lon
        and
        study_min_lat <= latitude <= study_max_lat
    ):

        outside_study_area += 1


print(
    "Controls outside historical study area:",
    outside_study_area
)


# ============================================================
# 18. Show first 10 controls
# ============================================================

print(
    "\nFirst 10 control points:"
)

print(
    controls_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 19. Label distribution
# ============================================================

print(
    "\nLabel distribution:"
)

print(
    controls_df["label"].value_counts()
)


# ============================================================
# 20. Coordinate summary
# ============================================================

print(
    "\nControl coordinate summary:"
)

print(
    controls_df[
        [
            "longitude",
            "latitude"
        ]
    ].describe()
)


# ============================================================
# 21. Final validation
# ============================================================

if (
    len(controls_df) == NUM_CONTROLS
    and
    invalid_dem_count == 0
    and
    outside_study_area == 0
):

    print(
        "\n========================================"
    )

    print(
        "FINAL VALIDATION: PASSED"
    )

    print(
        "All control points have valid DEM data."
    )

    print(
        "All control points are inside the "
        "historical study area."
    )

    print(
        "========================================"
    )

else:

    print(
        "\n========================================"
    )

    print(
        "FINAL VALIDATION: CHECK REQUIRED"
    )

    print(
        "========================================"
    )


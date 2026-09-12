
import os

import psycopg2
import pandas as pd
import rasterio
from rasterio.windows import Window


# ============================================================
# 1. Paths
# ============================================================

FLOOD_RASTER = r"C:\NERProject\Data\processed\flood_2023_utm45.tif"

OUTPUT_CSV = r"C:\NERProject\Data\processed\historical_landslide_flood_features.csv"


# ============================================================
# 2. PostgreSQL connection
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "assam_routing",
    "user": "postgres",
    "password": "postgres"
}


# ============================================================
# 3. Read historical landslide points
# ============================================================

print("Reading historical landslide points...")

conn = psycopg2.connect(**DB_CONFIG)

query = """
SELECT
    event_id,
    event_date,
    longitude,
    latitude,
    ST_X(geometry) AS x_utm,
    ST_Y(geometry) AS y_utm
FROM historical_landslides
ORDER BY event_date;
"""

df = pd.read_sql(query, conn)

print("Landslide events found:", len(df))


# ============================================================
# 4. Open flood raster
# ============================================================

print("\nChecking flood raster...")

with rasterio.open(FLOOD_RASTER) as src:

    print("Flood CRS:", src.crs)

    print(
        "Flood size:",
        src.width,
        "x",
        src.height
    )

    print(
        "Flood resolution:",
        src.res
    )

    print(
        "Flood NoData:",
        src.nodata
    )

    print(
        "Flood bounds:",
        src.bounds
    )


# ============================================================
# 5. Extract flood exposure
# ============================================================

print("\nExtracting flood exposure...")

flood_risk_values = []

outside_events = []

with rasterio.open(FLOOD_RASTER) as src:

    # --------------------------------------------------------
    # Flood raster bounds
    # --------------------------------------------------------

    left = src.bounds.left
    right = src.bounds.right
    bottom = src.bounds.bottom
    top = src.bounds.top

    # --------------------------------------------------------
    # Approximately 100 metre radius
    # --------------------------------------------------------

    radius_m = 100

    pixel_size = src.res[0]

    radius_pixels = int(
        radius_m / pixel_size
    )

    print(
        "Flood analysis radius:",
        radius_m,
        "metres"
    )

    print(
        "Radius in pixels:",
        radius_pixels
    )

    # --------------------------------------------------------
    # Process every landslide
    # --------------------------------------------------------

    for _, row in df.iterrows():

        event_id = row["event_id"]

        x = row["x_utm"]
        y = row["y_utm"]

        # ----------------------------------------------------
        # Check whether point is inside raster
        # ----------------------------------------------------

        if not (
            left <= x <= right
            and
            bottom <= y <= top
        ):

            print(
                f"Point outside flood raster: "
                f"event {event_id} "
                f"at ({row['longitude']}, "
                f"{row['latitude']})"
            )

            flood_risk_values.append(
                float("nan")
            )

            outside_events.append(
                event_id
            )

            continue

        # ----------------------------------------------------
        # Convert coordinate to raster row/column
        # ----------------------------------------------------

        raster_row, raster_col = src.index(
            x,
            y
        )

        # ----------------------------------------------------
        # Create safe raster window
        # ----------------------------------------------------

        row_start = max(
            0,
            raster_row - radius_pixels
        )

        row_end = min(
            src.height,
            raster_row + radius_pixels + 1
        )

        col_start = max(
            0,
            raster_col - radius_pixels
        )

        col_end = min(
            src.width,
            raster_col + radius_pixels + 1
        )

        # ----------------------------------------------------
        # Make sure window is valid
        # ----------------------------------------------------

        window_width = col_end - col_start
        window_height = row_end - row_start

        if (
            window_width <= 0
            or
            window_height <= 0
        ):

            print(
                f"Invalid raster window for "
                f"event {event_id}"
            )

            flood_risk_values.append(
                float("nan")
            )

            outside_events.append(
                event_id
            )

            continue

        # ----------------------------------------------------
        # Read flood pixels
        # ----------------------------------------------------

        window = Window(
            col_start,
            row_start,
            window_width,
            window_height
        )

        flood_data = src.read(
            1,
            window=window
        )

        # ----------------------------------------------------
        # Calculate flood exposure
        # ----------------------------------------------------

        total_pixels = flood_data.size

        if total_pixels == 0:

            flood_risk = float("nan")

        else:

            flooded_pixels = (
                flood_data == 1
            ).sum()

            flood_risk = (
                flooded_pixels /
                total_pixels
            )

        flood_risk_values.append(
            float(flood_risk)
        )


# ============================================================
# 6. Add flood feature
# ============================================================

df["flood_risk"] = flood_risk_values


# ============================================================
# 7. Save CSV
# ============================================================

print("\nSaving flood features...")

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 8. Final result
# ============================================================

print("\n====================================")
print("Flood extraction completed!")
print("Landslide events:", len(df))

print(
    "Valid flood values:",
    df["flood_risk"].notna().sum()
)

print(
    "Missing flood values:",
    df["flood_risk"].isna().sum()
)

print(
    "Points outside raster:",
    len(outside_events)
)

if outside_events:

    print(
        "Outside event IDs:",
        outside_events
    )

print("Output:")
print(OUTPUT_CSV)

print("====================================")


# ============================================================
# 9. Flood summary
# ============================================================

print("\nFlood risk summary:")

print(
    df["flood_risk"].describe()
)


# ============================================================
# 10. Flood risk distribution
# ============================================================

print("\nFlood risk distribution:")

print(
    df["flood_risk"].value_counts(
        bins=5,
        sort=False
    )
)


# ============================================================
# 11. Show highest flood exposure
# ============================================================

print("\nHighest flood exposure events:")

print(
    df[
        [
            "event_id",
            "event_date",
            "longitude",
            "latitude",
            "flood_risk"
        ]
    ]
    .sort_values(
        "flood_risk",
        ascending=False
    )
    .head(10)
)


# ============================================================
# 12. Close database
# ============================================================

conn.close()


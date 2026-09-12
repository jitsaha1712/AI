import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import psycopg2
from shapely import wkb


# ============================================================
# SETTINGS
# ============================================================

RAIN_DIR = r"C:\NERProject\Data\rainfall"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "assam_routing",
    "user": "postgres",
    "password": "postgres"
}


# ============================================================
# 1. CONNECT TO POSTGRESQL
# ============================================================

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

print("Connected to PostgreSQL")


# ============================================================
# 2. GET ROAD MIDPOINTS
# ============================================================

cur.execute("""
    SELECT
        edge_id,
        ST_AsBinary(
            ST_LineInterpolatePoint(geometry, 0.5)
        )
    FROM routing_noded
    WHERE geometry IS NOT NULL
""")

road_rows = cur.fetchall()

print("Road segments:", len(road_rows))


road_ids = []
road_lats = []
road_lons = []

for edge_id, geom_binary in road_rows:

    point = wkb.loads(bytes(geom_binary))

    road_ids.append(edge_id)
    road_lats.append(point.y)
    road_lons.append(point.x)


road_ids = np.array(road_ids)
road_lats = np.array(road_lats)
road_lons = np.array(road_lons)

print("Road points created:", len(road_ids))


# ============================================================
# 3. CONVERT ROAD COORDINATES
# ============================================================
#
# routing_noded is EPSG:32645 (meters).
# GPM uses latitude/longitude.
#
# Therefore convert road points from UTM 45N
# to WGS84.
# ============================================================

from pyproj import Transformer

transformer = Transformer.from_crs(
    "EPSG:32645",
    "EPSG:4326",
    always_xy=True
)

road_lons, road_lats = transformer.transform(
    road_lons,
    road_lats
)

print("Road coordinates converted to WGS84")


# ============================================================
# 4. LOAD GPM FILE LIST
# ============================================================

files = sorted(
    glob.glob(
        os.path.join(RAIN_DIR, "*.nc4")
    )
)

print("GPM files:", len(files))


# ============================================================
# 5. STORAGE
# ============================================================

rainfall_history = {
    edge_id: []
    for edge_id in road_ids
}


# ============================================================
# 6. PROCESS EACH GPM FILE
# ============================================================

for count, file_path in enumerate(files, start=1):

    try:

        ds = xr.open_dataset(
            file_path,
            engine="netcdf4"
        )

        rain = ds["precipitation"].isel(time=0)

        # ----------------------------------------------------
        # Select Assam / NER region
        # ----------------------------------------------------

        rain = rain.sel(
            lat=slice(24, 30),
            lon=slice(89, 98)
        )

        # ----------------------------------------------------
        # Get coordinates
        # ----------------------------------------------------

        lats = rain["lat"].values
        lons = rain["lon"].values

        values = rain.values

        # ----------------------------------------------------
        # Find nearest latitude for every road
        # ----------------------------------------------------

        lat_indices = np.abs(
            lats[:, None] - road_lats[None, :]
        ).argmin(axis=0)

        # ----------------------------------------------------
        # Find nearest longitude for every road
        # ----------------------------------------------------

        lon_indices = np.abs(
            lons[:, None] - road_lons[None, :]
        ).argmin(axis=0)

        # ----------------------------------------------------
        # Extract rainfall for every road
        #
        # values shape:
        #       latitude × longitude
        # ----------------------------------------------------

        road_rain = values[
            lat_indices,
            lon_indices
        ]

        # ----------------------------------------------------
        # Store rainfall
        # ----------------------------------------------------

        for i, edge_id in enumerate(road_ids):

            value = road_rain[i]

            if np.isfinite(value):

                rainfall_history[edge_id].append(
                    float(value)
                )

        ds.close()

    except Exception as e:

        print(
            f"Error processing "
            f"{os.path.basename(file_path)}:"
        )

        print(e)

        continue


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if count % 50 == 0:

        print(
            f"Processed {count}/{len(files)} rainfall files"
        )


# ============================================================
# 7. CREATE RAINFALL FEATURES
# ============================================================

print("Creating rainfall features...")


feature_rows = []


for edge_id in road_ids:

    values = rainfall_history[edge_id]

    if len(values) == 0:
        continue

    values = np.array(values)


    # --------------------------------------------------------
    # Average daily rainfall
    # --------------------------------------------------------

    rain_mean = float(
        np.mean(values)
    )


    # --------------------------------------------------------
    # Maximum daily rainfall
    # --------------------------------------------------------

    rain_max = float(
        np.max(values)
    )


    # --------------------------------------------------------
    # Maximum 7-day rainfall
    # --------------------------------------------------------

    if len(values) >= 7:

        rolling_7 = (
            pd.Series(values)
            .rolling(7)
            .sum()
        )

        rain_7d_max = float(
            rolling_7.max()
        )

    else:

        rain_7d_max = float(
            np.sum(values)
        )


    # --------------------------------------------------------
    # Maximum 30-day rainfall
    # --------------------------------------------------------

    if len(values) >= 30:

        rolling_30 = (
            pd.Series(values)
            .rolling(30)
            .sum()
        )

        rain_30d_max = float(
            rolling_30.max()
        )

    else:

        rain_30d_max = float(
            np.sum(values)
        )


    # --------------------------------------------------------
    # Number of heavy rainfall days
    #
    # Initial threshold = 50 mm/day
    # --------------------------------------------------------

    heavy_rain_days = int(
        np.sum(values >= 50)
    )


    feature_rows.append(
        (
            int(edge_id),
            rain_mean,
            rain_max,
            rain_7d_max,
            rain_30d_max,
            heavy_rain_days
        )
    )


# ============================================================
# 8. SAVE TO POSTGRESQL
# ============================================================

print("Saving rainfall features to PostgreSQL...")


insert_query = """
    INSERT INTO road_rainfall_features (
        edge_id,
        rain_mean_mm_day,
        rain_max_mm_day,
        rain_7d_max_mm,
        rain_30d_max_mm,
        heavy_rain_days
    )
    VALUES (%s, %s, %s, %s, %s, %s)

    ON CONFLICT (edge_id)
    DO UPDATE SET

        rain_mean_mm_day =
            EXCLUDED.rain_mean_mm_day,

        rain_max_mm_day =
            EXCLUDED.rain_max_mm_day,

        rain_7d_max_mm =
            EXCLUDED.rain_7d_max_mm,

        rain_30d_max_mm =
            EXCLUDED.rain_30d_max_mm,

        heavy_rain_days =
            EXCLUDED.heavy_rain_days
"""


cur.executemany(
    insert_query,
    feature_rows
)

conn.commit()


# ============================================================
# 9. CLOSE DATABASE
# ============================================================

cur.close()
conn.close()


# ============================================================
# 10. FINISHED
# ============================================================

print("===================================")
print("Rainfall feature calculation done!")
print("Roads processed:", len(feature_rows))
print("===================================")
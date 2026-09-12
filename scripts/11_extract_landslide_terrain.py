
import os

import psycopg2
import pandas as pd
import rasterio
import numpy as np


# ============================================================
# 1. Paths
# ============================================================

ELEVATION_RASTER = r"C:\NERProject\Data\processed\elevation_utm.tif"
SLOPE_RASTER = r"C:\NERProject\Data\processed\slope_utm.tif"

DEM_DIR = r"C:\NERProject\Data\dem"

OUTPUT_CSV = r"C:\NERProject\Data\processed\historical_landslide_terrain_features.csv"


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
# 3. Function to find original DEM tile
# ============================================================

def find_dem_tile(longitude, latitude):

    for filename in os.listdir(DEM_DIR):

        if not filename.lower().endswith(".tif"):
            continue

        path = os.path.join(DEM_DIR, filename)

        try:

            with rasterio.open(path) as src:

                bounds = src.bounds

                if (
                    bounds.left <= longitude <= bounds.right
                    and
                    bounds.bottom <= latitude <= bounds.top
                ):
                    return path

        except Exception:
            continue

    return None


# ============================================================
# 4. Function to extract from original DEM
# ============================================================

def extract_from_original_dem(longitude, latitude):

    dem_path = find_dem_tile(longitude, latitude)

    if dem_path is None:
        return None, None

    with rasterio.open(dem_path) as src:

        row, col = src.index(longitude, latitude)

        elevation = src.read(1)[row, col]

        nodata = src.nodata

        # --------------------------------------------
        # Check elevation
        # --------------------------------------------

        if nodata is not None and elevation == nodata:
            return None, None

        if not np.isfinite(elevation):
            return None, None

        # --------------------------------------------
        # Get small neighborhood
        # --------------------------------------------

        row_start = max(0, row - 1)
        row_end = min(src.height, row + 2)

        col_start = max(0, col - 1)
        col_end = min(src.width, col + 2)

        window = src.read(
            1,
            window=rasterio.windows.Window(
                col_start,
                row_start,
                col_end - col_start,
                row_end - row_start
            )
        ).astype(float)

        # --------------------------------------------
        # Replace NoData with NaN
        # --------------------------------------------

        if nodata is not None:
            window[window == nodata] = np.nan

        # --------------------------------------------
        # Calculate slope
        # --------------------------------------------

        valid_values = window[np.isfinite(window)]

        if len(valid_values) < 3:
            slope = None

        else:

            # DEM resolution in degrees
            dx_deg = src.res[0]
            dy_deg = src.res[1]

            # Convert degree distance to metres
            lat_rad = np.radians(latitude)

            meters_per_degree_lat = 111320.0
            meters_per_degree_lon = 111320.0 * np.cos(lat_rad)

            dx = dx_deg * meters_per_degree_lon
            dy = dy_deg * meters_per_degree_lat

            # Fill small NaN gaps using nearest valid value
            if np.isnan(window).any():
                mean_value = np.nanmean(window)
                window = np.where(
                    np.isnan(window),
                    mean_value,
                    window
                )

            # np.gradient handles edge pixels using one-sided
            # differences, which is exactly what we need here.
            gradient_y, gradient_x = np.gradient(
                window,
                dy,
                dx
            )

            center_row = row - row_start
            center_col = col - col_start

            dz_dx = gradient_x[center_row, center_col]
            dz_dy = gradient_y[center_row, center_col]

            slope = np.degrees(
                np.arctan(
                    np.sqrt(
                        dz_dx ** 2 +
                        dz_dy ** 2
                    )
                )
            )

    return float(elevation), float(slope)


# ============================================================
# 5. Read landslide points
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
# 6. Open processed rasters
# ============================================================

print("\nChecking elevation raster...")

elevation_src = rasterio.open(ELEVATION_RASTER)

print("Elevation CRS:", elevation_src.crs)
print(
    "Elevation size:",
    elevation_src.width,
    "x",
    elevation_src.height
)
print("Elevation resolution:", elevation_src.res)
print("Elevation NoData:", elevation_src.nodata)


print("\nChecking slope raster...")

slope_src = rasterio.open(SLOPE_RASTER)

print("Slope CRS:", slope_src.crs)
print(
    "Slope size:",
    slope_src.width,
    "x",
    slope_src.height
)
print("Slope resolution:", slope_src.res)
print("Slope NoData:", slope_src.nodata)


# ============================================================
# 7. Extract elevation and slope
# ============================================================

print("\nExtracting terrain values...")

elevation_values = []
slope_values = []

fallback_count = 0


for _, row in df.iterrows():

    x = row["x_utm"]
    y = row["y_utm"]

    longitude = row["longitude"]
    latitude = row["latitude"]

    # --------------------------------------------
    # Elevation from processed raster
    # --------------------------------------------

    elevation_value = list(
        elevation_src.sample([(x, y)])
    )[0][0]

    # --------------------------------------------
    # Slope from processed raster
    # --------------------------------------------

    slope_value = list(
        slope_src.sample([(x, y)])
    )[0][0]

    # --------------------------------------------
    # Handle NoData
    # --------------------------------------------

    if elevation_src.nodata is not None:
        if elevation_value == elevation_src.nodata:
            elevation_value = None

    if slope_src.nodata is not None:
        if slope_value == slope_src.nodata:
            slope_value = None

    # --------------------------------------------
    # Fallback to original DEM
    # --------------------------------------------

    if elevation_value is None or slope_value is None:

        print(
            f"Fallback DEM used for event "
            f"{row['event_id']} "
            f"at ({longitude}, {latitude})"
        )

        original_elevation, original_slope = (
            extract_from_original_dem(
                longitude,
                latitude
            )
        )

        if elevation_value is None:
            elevation_value = original_elevation

        if slope_value is None:
            slope_value = original_slope

        fallback_count += 1

    elevation_values.append(elevation_value)
    slope_values.append(slope_value)


# Close rasters
elevation_src.close()
slope_src.close()


# ============================================================
# 8. Add terrain features
# ============================================================

df["elevation_m"] = elevation_values
df["slope_deg"] = slope_values


# ============================================================
# 9. Save CSV
# ============================================================

print("\nSaving terrain features...")

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 10. Final result
# ============================================================

print("\n====================================")
print("Terrain extraction completed!")
print("Landslide events:", len(df))
print("Fallback DEM used:", fallback_count)
print("Output:")
print(OUTPUT_CSV)
print("====================================")


# ============================================================
# 11. Terrain summary
# ============================================================

print("\nTerrain summary:")

print(
    df[
        [
            "elevation_m",
            "slope_deg"
        ]
    ].describe()
)


print("\nMissing terrain values:")

print(
    df[
        [
            "elevation_m",
            "slope_deg"
        ]
    ].isna().sum()
)


# ============================================================
# 12. Close database
# ============================================================

conn.close()


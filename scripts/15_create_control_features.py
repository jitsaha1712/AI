
import os
import random
import re

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer
import psycopg2


# ============================================================
# 1. Paths
# ============================================================

CONTROL_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_control_points.csv"
)

LANDSLIDE_MASTER_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_landslide_master.csv"
)

RAINFALL_DIR = (
    r"C:\NERProject\Data\historical_rainfall"
)

ELEVATION_RASTER = (
    r"C:\NERProject\Data\processed"
    r"\elevation_utm.tif"
)

SLOPE_RASTER = (
    r"C:\NERProject\Data\processed"
    r"\slope_utm.tif"
)

DEM_DIR = (
    r"C:\NERProject\Data\dem"
)

FLOOD_RASTER = (
    r"C:\NERProject\Data\processed"
    r"\flood_2023_utm45.tif"
)

OUTPUT_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_control_features.csv"
)


# ============================================================
# 2. Configuration
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

FLOOD_RADIUS_METERS = 100


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
# 4. Read control points
# ============================================================

print("Reading control points...")

controls = pd.read_csv(
    CONTROL_CSV
)

print(
    "Control points:",
    len(controls)
)


# ============================================================
# 5. Read historical landslide dates
# ============================================================

print("\nReading historical landslide dates...")

landslides = pd.read_csv(
    LANDSLIDE_MASTER_CSV
)

landslides["event_date"] = pd.to_datetime(
    landslides["event_date"]
)

historical_dates = (
    landslides["event_date"]
    .dt.strftime("%Y-%m-%d")
    .tolist()
)

print(
    "Available historical dates:",
    len(historical_dates)
)


# ============================================================
# 6. Assign historical dates to controls
# ============================================================

print("\nAssigning historical dates...")

controls["event_date"] = [
    random.choice(historical_dates)
    for _ in range(len(controls))
]

controls["event_date"] = pd.to_datetime(
    controls["event_date"]
)


# ============================================================
# 7. Find rainfall files
# ============================================================

print("\nScanning rainfall files...")

rainfall_files = {}

for filename in os.listdir(RAINFALL_DIR):

    if not filename.lower().endswith(".nc4"):
        continue

    match = re.match(
        r"IMERG_(\d{8})\.nc4$",
        filename
    )

    if match is None:
        continue

    date_text = match.group(1)

    formatted_date = (
        f"{date_text[:4]}-"
        f"{date_text[4:6]}-"
        f"{date_text[6:8]}"
    )

    rainfall_files[formatted_date] = os.path.join(
        RAINFALL_DIR,
        filename
    )


print(
    "Rainfall files found:",
    len(rainfall_files)
)


# ============================================================
# 8. Rainfall cache
# ============================================================

rainfall_cache = {}


def load_rainfall(date_text):

    """
    Load one rainfall file.

    Returns:
        latitudes
        longitudes
        rainfall_values
    """

    if date_text in rainfall_cache:

        return rainfall_cache[date_text]


    if date_text not in rainfall_files:

        return None


    path = rainfall_files[date_text]


    with xr.open_dataset(path) as ds:

        rain = (
            ds["precipitation"]
            .isel(time=0)
            .sel(
                lat=slice(24, 28),
                lon=slice(89, 96)
            )
            .transpose("lat", "lon")
            .values
        )

        latitudes = (
            ds["lat"]
            .sel(lat=slice(24, 28))
            .values
        )

        longitudes = (
            ds["lon"]
            .sel(lon=slice(89, 96))
            .values
        )


    result = (
        latitudes,
        longitudes,
        rain
    )


    rainfall_cache[date_text] = result

    return result


# ============================================================
# 9. Rainfall features
# ============================================================

def get_rainfall_features(
    longitude,
    latitude,
    event_date
):

    event_date = pd.Timestamp(
        event_date
    )


    daily_values = []


    # --------------------------------------------------------
    # Event day + previous 10 days
    # --------------------------------------------------------

    for days_back in range(10, -1, -1):

        current_date = (
            event_date
            - pd.Timedelta(days=days_back)
        )

        date_text = current_date.strftime(
            "%Y-%m-%d"
        )


        result = load_rainfall(
            date_text
        )


        if result is None:

            continue


        latitudes, longitudes, rain = result


        # ----------------------------------------------------
        # Nearest IMERG cell
        # ----------------------------------------------------

        lat_index = np.abs(
            latitudes - latitude
        ).argmin()

        lon_index = np.abs(
            longitudes - longitude
        ).argmin()


        value = rain[
            lat_index,
            lon_index
        ]


        if np.isfinite(value):

            daily_values.append(
                float(value)
            )


    # --------------------------------------------------------
    # No rainfall data
    # --------------------------------------------------------

    if len(daily_values) == 0:

        return {
            "rain_event_day": np.nan,
            "rain_1d": np.nan,
            "rain_3d": np.nan,
            "rain_7d": np.nan,
            "rain_10d": np.nan,
            "max_rain_3d": np.nan,
            "max_rain_7d": np.nan,
            "heavy_rain_days": np.nan,
            "valid_rainfall_days": 0
        }


    # --------------------------------------------------------
    # The list is oldest → newest
    # Therefore the last value is the event day.
    # --------------------------------------------------------

    event_day = daily_values[-1]


    rain_1d = sum(
        daily_values[-1:]
    )


    rain_3d = sum(
        daily_values[-3:]
    )


    rain_7d = sum(
        daily_values[-7:]
    )


    rain_10d = sum(
        daily_values[-10:]
    )


    # --------------------------------------------------------
    # Maximum rolling rainfall
    # --------------------------------------------------------

    max_rain_3d = 0.0

    max_rain_7d = 0.0


    for i in range(
        len(daily_values)
    ):

        start_3 = max(
            0,
            i - 2
        )

        start_7 = max(
            0,
            i - 6
        )


        total_3 = sum(
            daily_values[start_3:i + 1]
        )


        total_7 = sum(
            daily_values[start_7:i + 1]
        )


        max_rain_3d = max(
            max_rain_3d,
            total_3
        )


        max_rain_7d = max(
            max_rain_7d,
            total_7
        )


    # --------------------------------------------------------
    # Heavy rainfall days
    # --------------------------------------------------------

    heavy_rain_days = sum(
        value >= 50
        for value in daily_values
    )


    return {
        "rain_event_day": event_day,
        "rain_1d": rain_1d,
        "rain_3d": rain_3d,
        "rain_7d": rain_7d,
        "rain_10d": rain_10d,
        "max_rain_3d": max_rain_3d,
        "max_rain_7d": max_rain_7d,
        "heavy_rain_days": heavy_rain_days,
        "valid_rainfall_days": len(
            daily_values
        )
    }


# ============================================================
# 10. Extract rainfall
# ============================================================

print("\nExtracting rainfall features...")

rainfall_results = []


for index, row in controls.iterrows():

    features = get_rainfall_features(
        row["longitude"],
        row["latitude"],
        row["event_date"]
    )

    rainfall_results.append(
        features
    )


    if (
        (index + 1) % 50 == 0
        or index == len(controls) - 1
    ):

        print(
            "Processed rainfall:",
            index + 1,
            "/",
            len(controls)
        )


rainfall_df = pd.DataFrame(
    rainfall_results
)


controls = pd.concat(
    [
        controls.reset_index(drop=True),
        rainfall_df.reset_index(drop=True)
    ],
    axis=1
)


# ============================================================
# 11. Find original DEM tile
# ============================================================

def find_dem_tile(
    longitude,
    latitude
):

    for filename in os.listdir(DEM_DIR):

        if not filename.lower().endswith(".tif"):
            continue


        path = os.path.join(
            DEM_DIR,
            filename
        )


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
# 12. Extract terrain from original DEM
# ============================================================

def extract_from_original_dem(
    longitude,
    latitude
):

    dem_path = find_dem_tile(
        longitude,
        latitude
    )


    if dem_path is None:

        return np.nan, np.nan


    with rasterio.open(
        dem_path
    ) as src:

        row, col = src.index(
            longitude,
            latitude
        )


        if (
            row < 0
            or row >= src.height
            or col < 0
            or col >= src.width
        ):

            return np.nan, np.nan


        elevation = src.read(
            1
        )[row, col]


        nodata = src.nodata


        if (
            nodata is not None
            and
            elevation == nodata
        ):

            return np.nan, np.nan


        if not np.isfinite(elevation):

            return np.nan, np.nan


        # ----------------------------------------------------
        # Small neighborhood for slope
        # ----------------------------------------------------

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


        if nodata is not None:

            window[
                window == nodata
            ] = np.nan


        valid_values = window[
            np.isfinite(window)
        ]


        if len(valid_values) < 3:

            return (
                float(elevation),
                np.nan
            )


        # ----------------------------------------------------
        # Convert geographic pixel size to metres
        # ----------------------------------------------------

        dx_deg = src.res[0]
        dy_deg = src.res[1]

        lat_rad = np.radians(
            latitude
        )

        meters_per_degree_lat = 111320.0

        meters_per_degree_lon = (
            111320.0
            * np.cos(lat_rad)
        )


        dx = (
            dx_deg
            * meters_per_degree_lon
        )

        dy = (
            dy_deg
            * meters_per_degree_lat
        )


        # ----------------------------------------------------
        # Fill tiny NoData holes only inside the neighborhood
        # ----------------------------------------------------

        if np.isnan(window).any():

            mean_value = np.nanmean(
                window
            )

            window = np.where(
                np.isnan(window),
                mean_value,
                window
            )


        gradient_y, gradient_x = np.gradient(
            window,
            dy,
            dx
        )


        center_row = (
            row - row_start
        )

        center_col = (
            col - col_start
        )


        dz_dx = gradient_x[
            center_row,
            center_col
        ]

        dz_dy = gradient_y[
            center_row,
            center_col
        ]


        slope = np.degrees(
            np.arctan(
                np.sqrt(
                    dz_dx ** 2
                    +
                    dz_dy ** 2
                )
            )
        )


    return (
        float(elevation),
        float(slope)
    )


# ============================================================
# 13. Extract terrain
# ============================================================

print("\nExtracting terrain features...")

elevations = []

slopes = []

terrain_fallback_count = 0


# Create coordinate transformer once
terrain_transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32645",
    always_xy=True
)


for index, row in controls.iterrows():

    longitude = float(
        row["longitude"]
    )

    latitude = float(
        row["latitude"]
    )


    elevation = np.nan

    slope = np.nan


    # --------------------------------------------------------
    # First try processed UTM rasters
    # --------------------------------------------------------

    with rasterio.open(
        ELEVATION_RASTER
    ) as elevation_src:

        x, y = terrain_transformer.transform(
            longitude,
            latitude
        )


        if (
            elevation_src.bounds.left <= x <= elevation_src.bounds.right
            and
            elevation_src.bounds.bottom <= y <= elevation_src.bounds.top
        ):

            row_num, col_num = elevation_src.index(
                x,
                y
            )


            if (
                0 <= row_num < elevation_src.height
                and
                0 <= col_num < elevation_src.width
            ):

                value = elevation_src.read(
                    1,
                    window=rasterio.windows.Window(
                        col_num,
                        row_num,
                        1,
                        1
                    )
                )[0, 0]


                if (
                    elevation_src.nodata is None
                    or value != elevation_src.nodata
                ):

                    if np.isfinite(value):

                        elevation = float(value)


    with rasterio.open(
        SLOPE_RASTER
    ) as slope_src:

        x, y = terrain_transformer.transform(
            longitude,
            latitude
        )


        if (
            slope_src.bounds.left <= x <= slope_src.bounds.right
            and
            slope_src.bounds.bottom <= y <= slope_src.bounds.top
        ):

            row_num, col_num = slope_src.index(
                x,
                y
            )


            if (
                0 <= row_num < slope_src.height
                and
                0 <= col_num < slope_src.width
            ):

                value = slope_src.read(
                    1,
                    window=rasterio.windows.Window(
                        col_num,
                        row_num,
                        1,
                        1
                    )
                )[0, 0]


                if (
                    slope_src.nodata is None
                    or value != slope_src.nodata
                ):

                    if np.isfinite(value):

                        slope = float(value)


    # --------------------------------------------------------
    # Fallback to original DEM tile
    # --------------------------------------------------------

    if (
        not np.isfinite(elevation)
        or
        not np.isfinite(slope)
    ):

        fallback_elevation, fallback_slope = (
            extract_from_original_dem(
                longitude,
                latitude
            )
        )


        if not np.isfinite(elevation):

            elevation = fallback_elevation


        if not np.isfinite(slope):

            slope = fallback_slope


        if (
            np.isfinite(fallback_elevation)
            or
            np.isfinite(fallback_slope)
        ):

            terrain_fallback_count += 1


    elevations.append(
        elevation
    )

    slopes.append(
        slope
    )


    if (
        (index + 1) % 50 == 0
        or index == len(controls) - 1
    ):

        print(
            "Processed terrain:",
            index + 1,
            "/",
            len(controls)
        )


controls["elevation_m"] = elevations

controls["slope_deg"] = slopes


# ============================================================
# 14. Extract 2023 flood exposure
# ============================================================

print("\nExtracting 2023 flood exposure...")

flood_values = []


with rasterio.open(
    FLOOD_RASTER
) as flood_src:

    flood_transformer = Transformer.from_crs(
        "EPSG:4326",
        flood_src.crs,
        always_xy=True
    )


    radius = int(
        FLOOD_RADIUS_METERS
        / flood_src.res[0]
    )


    for index, row in controls.iterrows():

        longitude = float(
            row["longitude"]
        )

        latitude = float(
            row["latitude"]
        )


        x, y = flood_transformer.transform(
            longitude,
            latitude
        )


        # ----------------------------------------------------
        # Outside raster
        # ----------------------------------------------------

        if not (
            flood_src.bounds.left <= x <= flood_src.bounds.right
            and
            flood_src.bounds.bottom <= y <= flood_src.bounds.top
        ):

            flood_values.append(
                np.nan
            )

            if (
                (index + 1) % 50 == 0
                or index == len(controls) - 1
            ):

                print(
                    "Processed flood:",
                    index + 1,
                    "/",
                    len(controls)
                )

            continue


        row_num, col_num = flood_src.index(
            x,
            y
        )


        row_start = max(
            0,
            row_num - radius
        )

        row_end = min(
            flood_src.height,
            row_num + radius + 1
        )

        col_start = max(
            0,
            col_num - radius
        )

        col_end = min(
            flood_src.width,
            col_num + radius + 1
        )


        if (
            row_start >= row_end
            or
            col_start >= col_end
        ):

            flood_values.append(
                np.nan
            )

            continue


        window = flood_src.read(
            1,
            window=rasterio.windows.Window(
                col_start,
                row_start,
                col_end - col_start,
                row_end - row_start
            )
        )


        if window.size == 0:

            flood_values.append(
                np.nan
            )

            continue


        valid = window[
            np.isfinite(window)
        ]


        if len(valid) == 0:

            flood_values.append(
                np.nan
            )

            continue


        flood_pixels = np.sum(
            valid == 1
        )


        flood_risk = (
            flood_pixels
            /
            len(valid)
        )


        flood_values.append(
            float(flood_risk)
        )


        if (
            (index + 1) % 50 == 0
            or index == len(controls) - 1
        ):

            print(
                "Processed flood:",
                index + 1,
                "/",
                len(controls)
            )


controls["flood_exposure_2023"] = (
    flood_values
)


# ============================================================
# 15. Add label
# ============================================================

controls["label"] = 0


# ============================================================
# 16. Select final columns
# ============================================================

final_columns = [
    "control_id",
    "event_date",
    "longitude",
    "latitude",

    "rain_event_day",
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_10d",
    "max_rain_3d",
    "max_rain_7d",
    "heavy_rain_days",
    "valid_rainfall_days",

    "elevation_m",
    "slope_deg",

    "flood_exposure_2023",

    "label"
]


controls = controls[
    final_columns
]


# ============================================================
# 17. Save
# ============================================================

print("\nSaving control features...")

controls.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 18. Final summary
# ============================================================

print("\n========================================")
print("Control feature extraction completed!")
print("========================================")

print(
    "Control records:",
    len(controls)
)

print(
    "Output:",
    OUTPUT_CSV
)

print(
    "Terrain fallback used:",
    terrain_fallback_count
)


# ============================================================
# 19. Missing values
# ============================================================

print("\nMissing values:")

print(
    controls.isna().sum()
)


# ============================================================
# 20. Rainfall validity check
# ============================================================

print("\nRainfall validity:")

print(
    "Controls with rainfall:",
    controls["rain_7d"].notna().sum()
)

print(
    "Controls without rainfall:",
    controls["rain_7d"].isna().sum()
)


# ============================================================
# 21. Numeric summary
# ============================================================

print("\nNumeric feature summary:")

print(
    controls.describe()
)


# ============================================================
# 22. First 10 records
# ============================================================

print("\nFirst 10 control records:")

print(
    controls.head(10).to_string(
        index=False
    )
)



import os

import numpy as np
import pandas as pd
import psycopg2
import xarray as xr


# ============================================================
# 1. Paths
# ============================================================

OUTPUT_CSV = (
    r"C:\NERProject\Data\processed"
    r"\flood_ml_dataset_2023.csv"
)

RAINFALL_DIR = (
    r"C:\NERProject\Data\rainfall"
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
# 3. Flood label threshold
# ============================================================
#
# flood_risk is the fraction of flooded pixels around a road.
#
# Example:
# 0.10 = 10% of the sampled pixels are flooded.
#
# This is a prototype labeling rule, not a universal
# flood-definition standard.
#

FLOOD_LABEL_THRESHOLD = 0.10


# ============================================================
# 4. Connect to PostgreSQL
# ============================================================

print("Connecting to PostgreSQL...")

conn = psycopg2.connect(
    **DB_CONFIG
)

cur = conn.cursor()

print("Database connected!")


# ============================================================
# 5. Read road-level static features
# ============================================================

print("\nReading routing road features...")

cur.execute(
    """
    SELECT
        edge_id,
        elevation_m,
        slope_deg,
        distance_to_river_m,
        flood_risk
    FROM routing_noded
    WHERE
        elevation_m IS NOT NULL
        AND slope_deg IS NOT NULL
        AND distance_to_river_m IS NOT NULL
        AND flood_risk IS NOT NULL
    ORDER BY edge_id;
    """
)


rows = cur.fetchall()


columns = [
    "edge_id",
    "elevation_m",
    "slope_deg",
    "distance_to_river_m",
    "flood_risk"
]


roads = pd.DataFrame(
    rows,
    columns=columns
)


print(
    "Road segments:",
    len(roads)
)


# ============================================================
# 6. Close cursor/connection
# ============================================================

cur.close()

conn.close()


# ============================================================
# 7. Create flood label
# ============================================================

roads["flood_label"] = (
    roads["flood_risk"]
    >= FLOOD_LABEL_THRESHOLD
).astype(int)


# ============================================================
# 8. Flood label statistics
# ============================================================

print(
    "\nFlood label distribution:"
)

print(
    roads["flood_label"].value_counts()
)


print(
    "\nFlood label percentages:"
)

print(
    (
        roads["flood_label"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


print(
    "\nFlood exposure statistics:"
)

print(
    roads["flood_risk"].describe()
)


# ============================================================
# 9. Scan 2023 rainfall files
# ============================================================

print(
    "\nScanning rainfall directory..."
)

rainfall_files = []


for filename in os.listdir(
    RAINFALL_DIR
):

    if not filename.lower().endswith(
        ".nc4"
    ):
        continue


    # Expected format:
    #
    # 3B-DAY.MS.MRG.3IMERG.20230101-S000000-E235959.V07B.nc4
    #
    # We extract the first 8-digit date after "3IMERG."

    parts = filename.split(
        "3IMERG."
    )


    if len(parts) != 2:
        continue


    date_text = parts[1][:8]


    if not date_text.isdigit():
        continue


    year = date_text[:4]


    if year != "2023":
        continue


    rainfall_files.append(
        (
            date_text,
            os.path.join(
                RAINFALL_DIR,
                filename
            )
        )
    )


rainfall_files.sort()


print(
    "2023 rainfall files found:",
    len(rainfall_files)
)


if len(rainfall_files) == 0:

    raise RuntimeError(
        "No 2023 rainfall files were found."
    )


# ============================================================
# 10. Extract road coordinates from PostGIS
# ============================================================

print(
    "\nReading road coordinates..."
)

conn = psycopg2.connect(
    **DB_CONFIG
)

cur = conn.cursor()


cur.execute(
    """
    SELECT
        edge_id,
        ST_X(
            ST_Transform(
                ST_StartPoint(geometry),
                4326
            )
        ) AS longitude,
        ST_Y(
            ST_Transform(
                ST_StartPoint(geometry),
                4326
            )
        ) AS latitude
    FROM routing_noded
    WHERE edge_id = ANY(%s);
    """,
    (
        roads["edge_id"]
        .astype(int)
        .tolist(),
    )
)


coordinate_rows = cur.fetchall()


cur.close()

conn.close()


coordinates = pd.DataFrame(
    coordinate_rows,
    columns=[
        "edge_id",
        "longitude",
        "latitude"
    ]
)


print(
    "Road coordinates:",
    len(coordinates)
)


# ============================================================
# 11. Merge coordinates
# ============================================================

roads = roads.merge(
    coordinates,
    on="edge_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 12. Prepare rainfall storage
# ============================================================

#
# We create daily rainfall values for each road.
#
# Shape:
#
#     rows    = roads
#     columns = days in 2023
#
# Later we calculate:
#
# rain_1d
# rain_3d
# rain_7d
# max_rain_3d
# max_rain_7d
#

daily_rainfall = []


# ============================================================
# 13. Process 2023 rainfall
# ============================================================

print(
    "\nExtracting 2023 rainfall..."
)


for file_index, (
    date_text,
    rainfall_path
) in enumerate(
    rainfall_files,
    start=1
):

    try:

        with xr.open_dataset(
            rainfall_path
        ) as ds:

            rain = (
                ds["precipitation"]
                .isel(time=0)
                .sel(
                    lat=slice(24, 28),
                    lon=slice(89, 96)
                )
                .transpose(
                    "lat",
                    "lon"
                )
                .values
            )


            latitudes = (
                ds["lat"]
                .sel(
                    lat=slice(24, 28)
                )
                .values
            )


            longitudes = (
                ds["lon"]
                .sel(
                    lon=slice(89, 96)
                )
                .values
            )


        # ----------------------------------------------------
        # Convert road coordinates into nearest IMERG cells
        # ----------------------------------------------------

        lat_indices = np.abs(
            latitudes[:, None]
            -
            roads["latitude"].to_numpy()[None, :]
        ).argmin(
            axis=0
        )


        lon_indices = np.abs(
            longitudes[:, None]
            -
            roads["longitude"].to_numpy()[None, :]
        ).argmin(
            axis=0
        )


        values = rain[
            lat_indices,
            lon_indices
        ]


        values = values.astype(
            float
        )


        values[
            ~np.isfinite(values)
        ] = np.nan


        daily_rainfall.append(
            values
        )


    except Exception as e:

        print(
            "\nWARNING:"
        )

        print(
            "Could not process:",
            os.path.basename(
                rainfall_path
            )
        )

        print(
            "Error:",
            e
        )

        # Keep a missing day rather than crashing.
        daily_rainfall.append(
            np.full(
                len(roads),
                np.nan
            )
        )


    if (
        file_index % 25 == 0
        or
        file_index == len(rainfall_files)
    ):

        print(
            "Processed rainfall:",
            file_index,
            "/",
            len(rainfall_files)
        )


# ============================================================
# 14. Convert rainfall matrix
# ============================================================

daily_rainfall = np.array(
    daily_rainfall
)


print(
    "\nRainfall matrix shape:",
    daily_rainfall.shape
)


# ============================================================
# 15. Calculate 2023 rainfall features
# ============================================================

print(
    "\nCalculating 2023 rainfall features..."
)


road_count = len(roads)


rain_1d = np.full(
    road_count,
    np.nan
)

rain_3d = np.full(
    road_count,
    np.nan
)

rain_7d = np.full(
    road_count,
    np.nan
)

max_rain_3d = np.full(
    road_count,
    np.nan
)

max_rain_7d = np.full(
    road_count,
    np.nan
)


for road_index in range(
    road_count
):

    values = daily_rainfall[
        :,
        road_index
    ]


    valid = values[
        np.isfinite(values)
    ]


    if len(valid) == 0:
        continue


    # --------------------------------------------------------
    # Most recent 2023 rainfall day
    # --------------------------------------------------------

    rain_1d[road_index] = (
        values[-1]
        if np.isfinite(values[-1])
        else np.nan
    )


    # --------------------------------------------------------
    # 3-day and 7-day totals at the end of 2023
    # --------------------------------------------------------

    valid_3 = values[
        -3:
    ]

    valid_7 = values[
        -7:
    ]


    if np.isfinite(valid_3).any():

        rain_3d[road_index] = np.nansum(
            valid_3
        )


    if np.isfinite(valid_7).any():

        rain_7d[road_index] = np.nansum(
            valid_7
        )


    # --------------------------------------------------------
    # Maximum rolling 3-day / 7-day rainfall during 2023
    # --------------------------------------------------------

    for day_index in range(
        len(values)
    ):

        window_3 = values[
            max(
                0,
                day_index - 2
            ):
            day_index + 1
        ]


        window_7 = values[
            max(
                0,
                day_index - 6
            ):
            day_index + 1
        ]


        if np.isfinite(
            window_3
        ).any():

            total_3 = np.nansum(
                window_3
            )


            if (
                np.isnan(
                    max_rain_3d[
                        road_index
                    ]
                )
                or
                total_3
                >
                max_rain_3d[
                    road_index
                ]
            ):

                max_rain_3d[
                    road_index
                ] = total_3


        if np.isfinite(
            window_7
        ).any():

            total_7 = np.nansum(
                window_7
            )


            if (
                np.isnan(
                    max_rain_7d[
                        road_index
                    ]
                )
                or
                total_7
                >
                max_rain_7d[
                    road_index
                ]
            ):

                max_rain_7d[
                    road_index
                ] = total_7


    if (
        (road_index + 1) % 5000 == 0
    ):

        print(
            "Calculated rainfall features:",
            road_index + 1,
            "/",
            road_count
        )


# ============================================================
# 16. Add rainfall features
# ============================================================

roads["rain_1d_2023"] = rain_1d

roads["rain_3d_2023"] = rain_3d

roads["rain_7d_2023"] = rain_7d

roads["max_rain_3d_2023"] = (
    max_rain_3d
)

roads["max_rain_7d_2023"] = (
    max_rain_7d
)


# ============================================================
# 17. Remove incomplete rainfall records
# ============================================================

print(
    "\nChecking rainfall completeness..."
)

print(
    "Missing rain_7d:",
    roads["rain_7d_2023"].isna().sum()
)


roads = roads.dropna(
    subset=[
        "rain_1d_2023",
        "rain_3d_2023",
        "rain_7d_2023",
        "max_rain_3d_2023",
        "max_rain_7d_2023",
        "elevation_m",
        "slope_deg",
        "distance_to_river_m"
    ]
).reset_index(
    drop=True
)


# ============================================================
# 18. Select ML features
# ============================================================

final_columns = [
    "edge_id",

    "rain_1d_2023",
    "rain_3d_2023",
    "rain_7d_2023",
    "max_rain_3d_2023",
    "max_rain_7d_2023",

    "elevation_m",
    "slope_deg",
    "distance_to_river_m",

    "flood_risk",
    "flood_label"
]


dataset = roads[
    final_columns
].copy()


# ============================================================
# 19. Validate dataset
# ============================================================

print(
    "\n========================================"
)

print(
    "FLOOD ML DATASET"
)

print(
    "========================================"
)

print(
    "Total samples:",
    len(dataset)
)

print(
    "Flooded samples:",
    (
        dataset["flood_label"] == 1
    ).sum()
)

print(
    "Non-flooded samples:",
    (
        dataset["flood_label"] == 0
    ).sum()
)


print(
    "\nMissing values:"
)

print(
    dataset.isna().sum()
)


print(
    "\nFeature summary:"
)

print(
    dataset[
        [
            "rain_1d_2023",
            "rain_3d_2023",
            "rain_7d_2023",
            "max_rain_3d_2023",
            "max_rain_7d_2023",
            "elevation_m",
            "slope_deg",
            "distance_to_river_m"
        ]
    ].describe()
)


# ============================================================
# 20. Save
# ============================================================

dataset.to_csv(
    OUTPUT_CSV,
    index=False
)


print(
    "\n========================================"
)

print(
    "Flood ML dataset created!"
)

print(
    "========================================"
)

print(
    "Output:",
    OUTPUT_CSV
)

print(
    "Label threshold:",
    FLOOD_LABEL_THRESHOLD
)

print(
    "========================================"
)


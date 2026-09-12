
import pandas as pd
import rasterio


CONTROL_CSV = (
    r"C:\NERProject\Data\processed"
    r"\historical_control_features.csv"
)

DEM_DIR = (
    r"C:\NERProject\Data\dem"
)


# ============================================================
# 1. Read control features
# ============================================================

controls = pd.read_csv(
    CONTROL_CSV
)


# ============================================================
# 2. Find controls with missing terrain
# ============================================================

missing = controls[
    controls["elevation_m"].isna()
    |
    controls["slope_deg"].isna()
].copy()


print("Missing terrain controls:")
print("----------------------------------------")

print(
    missing[
        [
            "control_id",
            "event_date",
            "longitude",
            "latitude",
            "elevation_m",
            "slope_deg"
        ]
    ].to_string(index=False)
)


print("\nTotal missing terrain controls:")
print(
    len(missing)
)


# ============================================================
# 3. Scan DEM tiles
# ============================================================

dem_tiles = []

print("\nScanning DEM tiles...")

for filename in __import__("os").listdir(DEM_DIR):

    if not filename.lower().endswith(".tif"):
        continue

    path = __import__("os").path.join(
        DEM_DIR,
        filename
    )

    try:

        with rasterio.open(path) as src:

            dem_tiles.append(
                (
                    filename,
                    path,
                    src.bounds,
                    src.crs
                )
            )

    except Exception:
        pass


# ============================================================
# 4. Check every missing point against DEM tiles
# ============================================================

print("\nChecking DEM coverage...")

for _, row in missing.iterrows():

    lon = float(
        row["longitude"]
    )

    lat = float(
        row["latitude"]
    )

    print(
        f"\nControl {int(row['control_id'])}: "
        f"({lon}, {lat})"
    )


    found_tile = False


    for filename, path, bounds, crs in dem_tiles:

        if (
            bounds.left <= lon <= bounds.right
            and
            bounds.bottom <= lat <= bounds.top
        ):

            found_tile = True

            print(
                "  DEM tile:",
                filename
            )

            print(
                "  CRS:",
                crs
            )

            print(
                "  Bounds:",
                bounds
            )


            try:

                with rasterio.open(path) as src:

                    row_num, col_num = src.index(
                        lon,
                        lat
                    )


                    if (
                        0 <= row_num < src.height
                        and
                        0 <= col_num < src.width
                    ):

                        value = src.read(
                            1
                        )[row_num, col_num]


                        print(
                            "  Pixel row:",
                            row_num
                        )

                        print(
                            "  Pixel col:",
                            col_num
                        )

                        print(
                            "  DEM value:",
                            value
                        )

                        print(
                            "  NoData:",
                            src.nodata
                        )

            except Exception as e:

                print(
                    "  Error reading pixel:",
                    e
                )


    if not found_tile:

        print(
            "  No DEM tile contains this point."
        )


print("\n========================================")
print("Terrain inspection completed!")
print("========================================")


import os
import glob
import numpy as np
import xarray as xr
import pandas as pd

# ============================================================
# PATHS
# ============================================================

RAINFALL_DIR = r"C:\NERProject\Data\rainfall"
OUTPUT_DIR = r"C:\NERProject\Data\processed\rainfall"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND ALL NETCDF FILES
# ============================================================

files = sorted(glob.glob(os.path.join(RAINFALL_DIR, "*.nc4")))

print(f"Found {len(files)} rainfall files.")

if len(files) == 0:
    print("ERROR: No .nc4 files found.")
    exit()


# ============================================================
# PROCESS EACH FILE
# ============================================================

records = []

for i, file_path in enumerate(files, start=1):

    print(f"[{i}/{len(files)}] Processing: {os.path.basename(file_path)}")

    try:
        ds = xr.open_dataset(file_path)

        # ----------------------------------------------------
        # Find rainfall variable
        # ----------------------------------------------------

        if "precipitation" in ds:
            rain = ds["precipitation"]
        else:
            print("  ERROR: precipitation variable not found.")
            ds.close()
            continue

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        data = rain.values

        # Remove time dimension if present
        data = np.squeeze(data)

        # ----------------------------------------------------
        # Replace invalid values
        # ----------------------------------------------------

        data = np.where(np.isfinite(data), data, np.nan)

        # ----------------------------------------------------
        # Calculate statistics
        # ----------------------------------------------------

        mean_rainfall = float(np.nanmean(data))
        max_rainfall = float(np.nanmax(data))
        min_rainfall = float(np.nanmin(data))

        # ----------------------------------------------------
        # Get date from filename / dataset
        # ----------------------------------------------------

        date_value = None

        if "time" in ds.coords:
            try:
                date_value = pd.to_datetime(ds["time"].values[0]).date()
            except Exception:
                pass

        if date_value is None:
            # Example filename:
            # 3B-DAY.MS.MRG.3IMERG.20190102-S000000-E235959.V07B.nc4

            filename = os.path.basename(file_path)

            parts = filename.split(".")

            for part in parts:
                if len(part) == 8 and part.isdigit():
                    try:
                        date_value = pd.to_datetime(part, format="%Y%m%d").date()
                        break
                    except Exception:
                        pass

        if date_value is None:
            print("  WARNING: Could not determine date.")
            ds.close()
            continue

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        records.append({
            "date": date_value,
            "mean_rainfall_mm_day": mean_rainfall,
            "max_rainfall_mm_day": max_rainfall,
            "min_rainfall_mm_day": min_rainfall
        })

        ds.close()

    except Exception as e:

        print(f"  ERROR: {e}")

        try:
            ds.close()
        except Exception:
            pass


# ============================================================
# CREATE DATAFRAME
# ============================================================

if len(records) == 0:
    print("ERROR: No rainfall data was successfully processed.")
    exit()

df = pd.DataFrame(records)

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# CALCULATE 7-DAY ACCUMULATED RAINFALL
# ============================================================

df["rainfall_7d_mm"] = (
    df["mean_rainfall_mm_day"]
    .rolling(window=7, min_periods=1)
    .sum()
)


# ============================================================
# SAVE RESULT
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "daily_rainfall_summary.csv"
)

df.to_csv(output_file, index=False)


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("=" * 60)
print("RAINFALL PROCESSING COMPLETE")
print("=" * 60)

print(f"Total days processed : {len(df)}")
print(f"First date           : {df['date'].min()}")
print(f"Last date            : {df['date'].max()}")

print()
print("Output:")
print(output_file)

print()
print("First 5 rows:")
print(df.head())

print()
print("Last 5 rows:")
print(df.tail())
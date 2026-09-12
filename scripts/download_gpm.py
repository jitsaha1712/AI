import re
import time
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

import earthaccess
from netCDF4 import Dataset


PROJECT_DIR = Path(r"C:\NERProject")
LINKS_FILE = PROJECT_DIR / "links.txt"
OUTPUT_DIR = PROJECT_DIR / "Data" / "rainfall"

START_DATE = datetime(2019, 1, 1)
END_DATE = datetime(2024, 12, 31)

MAX_WORKERS = 4
MAX_RETRIES = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_date(url):
    match = re.search(r"3IMERG\.(\d{8})-", url)

    if not match:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return None


def extract_filename(url):
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    if "FILENAME" not in params:
        return None

    path = unquote(params["FILENAME"][0])
    return Path(path).name


def valid_netcdf(path):
    try:
        with Dataset(path, "r") as ds:

            required = [
                "precipitation",
                "lat",
                "lon",
                "time"
            ]

            return all(x in ds.variables for x in required)

    except Exception:
        return False


def download_one(item):
    file_date, url = item

    filename = extract_filename(url)

    if filename is None:
        return file_date, False, "Could not extract filename"

    output_file = OUTPUT_DIR / filename

    # Already downloaded
    if output_file.exists():

        size = output_file.stat().st_size

        if size > 100000 and valid_netcdf(output_file):
            return file_date, True, "Already exists"

        output_file.unlink(missing_ok=True)

    # Search exact NASA granule
    try:

        results = earthaccess.search_data(
            short_name="GPM_3IMERGDF",
            version="07",
            temporal=(
                file_date.strftime("%Y-%m-%d"),
                file_date.strftime("%Y-%m-%d")
            ),
            bounding_box=(88, 21, 98, 30)
        )

        if not results:
            return file_date, False, "NASA granule not found"

        granule = results[0]

    except Exception as e:
        return file_date, False, f"Search error: {e}"

    # Download with retries
    for attempt in range(1, MAX_RETRIES + 1):

        try:

            files = earthaccess.download(
                [granule],
                local_path=str(OUTPUT_DIR)
            )

            downloaded = None

            if files:
                for f in files:
                    p = Path(str(f))

                    if p.exists():
                        downloaded = p
                        break

            if downloaded is None and output_file.exists():
                downloaded = output_file

            if downloaded is None:
                raise RuntimeError("Downloaded file not found")

            size = downloaded.stat().st_size

            if size < 100000:
                raise RuntimeError(
                    f"File too small: {size} bytes"
                )

            if not valid_netcdf(downloaded):
                raise RuntimeError(
                    "Invalid NetCDF file"
                )

            if downloaded != output_file:

                if output_file.exists():
                    output_file.unlink()

                downloaded.rename(output_file)

            return file_date, True, f"{size:,} bytes"

        except Exception as e:

            if attempt < MAX_RETRIES:
                time.sleep(5)
            else:
                return file_date, False, str(e)

    return file_date, False, "Unknown error"


# ==========================================================
# READ LINKS
# ==========================================================

with open(LINKS_FILE, "r", encoding="utf-8") as f:
    links = [
        line.strip()
        for line in f
        if line.strip()
    ]


items = []

for url in links:

    if "SUB.nc4" not in url:
        continue

    d = extract_date(url)

    if d is None:
        continue

    if START_DATE <= d <= END_DATE:
        items.append((d, url))


items.sort()


print("=" * 70)
print("FAST GPM IMERG DOWNLOADER")
print("=" * 70)

print(f"Total requested : {len(items)}")
print(f"Parallel workers: {MAX_WORKERS}")
print()


# ==========================================================
# LOGIN
# ==========================================================

print("Logging into NASA Earthdata...")

earthaccess.login(strategy="netrc")

print("Login successful.")
print()


# ==========================================================
# REMOVE ALREADY DOWNLOADED FILES
# ==========================================================

remaining = []

for item in items:

    d, url = item
    filename = extract_filename(url)

    if filename is None:
        remaining.append(item)
        continue

    path = OUTPUT_DIR / filename

    if path.exists():

        if (
            path.stat().st_size > 100000
            and valid_netcdf(path)
        ):
            continue

    remaining.append(item)


print(f"Already valid : {len(items) - len(remaining)}")
print(f"Remaining     : {len(remaining)}")
print()


if not remaining:

    print("Everything is already downloaded.")
    raise SystemExit(0)


# ==========================================================
# PARALLEL DOWNLOAD
# ==========================================================

success = 0
failed = 0

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = {
        executor.submit(download_one, item): item
        for item in remaining
    }

    for number, future in enumerate(
        as_completed(futures),
        start=1
    ):

        try:
            d, ok, message = future.result()

        except Exception as e:

            item = futures[future]
            d = item[0]
            ok = False
            message = str(e)

        if ok:

            success += 1

            print(
                f"[{number}/{len(remaining)}] "
                f"{d.date()}  SUCCESS  {message}"
            )

        else:

            failed += 1

            print(
                f"[{number}/{len(remaining)}] "
                f"{d.date()}  FAILED   {message}"
            )


# ==========================================================
# FINAL
# ==========================================================

print()
print("=" * 70)
print("DOWNLOAD FINISHED")
print("=" * 70)

print(f"Downloaded successfully : {success}")
print(f"Failed                   : {failed}")

print()
print(f"Folder: {OUTPUT_DIR}")
print("=" * 70)
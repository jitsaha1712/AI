import re
import requests
import psycopg2

from pathlib import Path
from datetime import timedelta


# ==========================================
# 1. Paths
# ==========================================

LINK_FILE = Path(
    r"C:\NERProject\Data\landslide\rainfall_history_links2007_to_2016.txt"
)

OUTPUT_DIR = Path(
    r"C:\NERProject\Data\historical_rainfall"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. PostgreSQL connection
# ==========================================

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="assam_routing",
    user="postgres",
    password="postgres"
)

cur = conn.cursor()


# ==========================================
# 3. Get landslide dates
# ==========================================

print("Reading landslide dates...")

cur.execute("""
    SELECT DISTINCT event_date
    FROM historical_landslides
    WHERE event_date IS NOT NULL
    ORDER BY event_date;
""")

event_dates = [row[0] for row in cur.fetchall()]

print("Unique landslide dates:", len(event_dates))


# ==========================================
# 4. Generate required dates
# ==========================================

required_dates = set()

for event_date in event_dates:

    for days_before in range(0, 11):

        required_date = event_date - timedelta(days=days_before)

        required_dates.add(required_date)


required_dates = sorted(required_dates)

print("Total unique rainfall dates needed:", len(required_dates))


# ==========================================
# 5. Read NASA links
# ==========================================

print()
print("Reading NASA IMERG links...")

with open(LINK_FILE, "r", encoding="utf-8") as f:

    links = [
        line.strip()
        for line in f
        if line.strip().startswith("http")
    ]

print("Total links found:", len(links))


# ==========================================
# 6. Match links with required dates
# ==========================================

date_to_url = {}

for url in links:

    match = re.search(
        r"3IMERG\.(\d{8})-",
        url
    )

    if not match:
        continue

    date_string = match.group(1)

    year = int(date_string[0:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])

    from datetime import date

    url_date = date(year, month, day)

    date_to_url[url_date] = url


print("Usable dated links:", len(date_to_url))


# ==========================================
# 7. Find missing links
# ==========================================

missing_dates = [
    d
    for d in required_dates
    if d not in date_to_url
]


print("Missing required dates:", len(missing_dates))


if missing_dates:

    print()
    print("First missing dates:")

    for d in missing_dates[:20]:
        print(d)


# ==========================================
# 8. Select only required URLs
# ==========================================

download_list = [
    (d, date_to_url[d])
    for d in required_dates
    if d in date_to_url
]


print()
print("Files that can be downloaded:", len(download_list))


# ==========================================
# 9. NASA Earthdata login
# ==========================================

username = input("\nNASA Earthdata username: ")
password = input("NASA Earthdata password: ")


# ==========================================
# 10. Download
# ==========================================

session = requests.Session()

session.auth = (username, password)


success = 0
skipped = 0
failed = 0


print()
print("==========================================")
print("Starting historical rainfall download")
print("==========================================")


for index, (rain_date, url) in enumerate(download_list, start=1):

    filename = (
        f"IMERG_{rain_date.strftime('%Y%m%d')}.nc4"
    )

    output_file = OUTPUT_DIR / filename


    # --------------------------------------
    # Skip already downloaded files
    # --------------------------------------

    if output_file.exists() and output_file.stat().st_size > 1000:

        skipped += 1

        print(
            f"[{index}/{len(download_list)}] "
            f"{rain_date} ALREADY EXISTS"
        )

        continue


    print(
        f"[{index}/{len(download_list)}] "
        f"Downloading {rain_date}..."
    )


    try:

        response = session.get(
            url,
            stream=True,
            timeout=180
        )


        if response.status_code != 200:

            print(
                f"    FAILED HTTP {response.status_code}"
            )

            failed += 1

            continue


        with open(output_file, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:

                    f.write(chunk)


        file_size = output_file.stat().st_size


        if file_size > 1000:

            success += 1

            print(
                f"    SUCCESS "
                f"({file_size:,} bytes)"
            )

        else:

            failed += 1

            print("    FAILED: file too small")

            output_file.unlink(missing_ok=True)


    except Exception as e:

        failed += 1

        print(
            f"    ERROR: {e}"
        )


# ==========================================
# 11. Finish
# ==========================================

print()
print("==========================================")
print("Historical rainfall download completed!")
print("==========================================")

print("Required dates :", len(required_dates))
print("Downloaded      :", success)
print("Already existed  :", skipped)
print("Failed           :", failed)
print("Output directory :", OUTPUT_DIR)
print("==========================================")


cur.close()
conn.close()
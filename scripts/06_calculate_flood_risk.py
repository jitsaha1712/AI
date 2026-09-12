import rasterio
import numpy as np
import psycopg2
from rasterio.mask import mask
from shapely.geometry import shape
import json


# ============================================================
# 1. FLOOD RASTER PATH
# ============================================================

RASTER_PATH = r"C:\NERProject\Data\processed\flood_2023_utm45.tif"


# ============================================================
# 2. DATABASE CONNECTION
# ============================================================

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="assam_routing",
    user="postgres",
    password="postgres"
)

cur = conn.cursor()


# ============================================================
# 3. OPEN FLOOD RASTER
# ============================================================

src = rasterio.open(RASTER_PATH)

print("Flood raster CRS:", src.crs)
print("Starting flood-risk calculation...")


# ============================================================
# 4. GET ALL ROAD SEGMENTS
# ============================================================

cur.execute("""
    SELECT
        edge_id,
        ST_AsGeoJSON(geometry)
    FROM routing_noded
    WHERE geometry IS NOT NULL
""")

roads = cur.fetchall()

print("Road segments:", len(roads))


# ============================================================
# 5. CALCULATE FLOOD RISK FOR EACH ROAD
# ============================================================

for count, (edge_id, geom_json) in enumerate(roads, start=1):

    # Convert GeoJSON text into Shapely geometry
    geom = shape(json.loads(geom_json))

    # --------------------------------------------------------
    # Create 30 meter buffer around the road
    # --------------------------------------------------------
    buffered = geom.buffer(30)

    try:

        # ----------------------------------------------------
        # Extract only raster pixels around this road
        # ----------------------------------------------------
        out_image, _ = mask(
            src,
            [buffered],
            crop=True,
            filled=True,
            nodata=0
        )

        data = out_image[0]

        # ----------------------------------------------------
        # Count pixels
        # ----------------------------------------------------
        total_pixels = data.size

        flooded_pixels = np.count_nonzero(data == 1)

        # ----------------------------------------------------
        # Calculate flood risk
        # ----------------------------------------------------
        if total_pixels > 0:

            flood_risk = flooded_pixels / total_pixels

        else:

            flood_risk = 0.0

    except Exception:

        flood_risk = 0.0


    # ========================================================
    # 6. SAVE FLOOD RISK INTO POSTGRESQL
    # ========================================================

    cur.execute(
        """
        UPDATE routing_noded
        SET flood_risk = %s
        WHERE edge_id = %s
        """,
        (float(flood_risk), edge_id)
    )


    # ========================================================
    # 7. COMMIT EVERY 1000 ROADS
    # ========================================================

    if count % 1000 == 0:

        conn.commit()

        print(
            f"Processed {count}/{len(roads)}"
        )


# ============================================================
# 8. FINAL COMMIT
# ============================================================

conn.commit()


# ============================================================
# 9. CLOSE EVERYTHING
# ============================================================

src.close()

cur.close()

conn.close()


# ============================================================
# 10. FINISHED
# ============================================================

print("===================================")
print("Flood-risk calculation completed!")
print("===================================")
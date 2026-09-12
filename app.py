# ============================================================
# C:\NERProject\app.py
# NER Smart Logistics & Accessibility Intelligence Platform
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import json
import psycopg2
from psycopg2.extras import RealDictCursor


# ============================================================
# DATABASE CONFIG
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "assam_routing",
    "user": "postgres",
    "password": "postgres",  # <-- PUT YOUR LOCAL POSTGRES PASSWORD HERE
}


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_SPACING_KM = 6.0


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="NER Smart Logistics Routing API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class Point(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class RouteRequest(BaseModel):
    source: Point
    destination: Point


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    try:

        return psycopg2.connect(
            **DB_CONFIG
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )


# ============================================================
# FIND NEAREST NODE
# ============================================================

def get_nearest_node(
    cur,
    lat,
    lon
):

    sql = """
        SELECT
            node_id,

            ST_Distance(
                geometry,

                ST_Transform(
                    ST_SetSRID(
                        ST_MakePoint(%s, %s),
                        4326
                    ),
                    ST_SRID(geometry)
                )

            ) AS distance_m

        FROM routing_vertices

        ORDER BY
            geometry <->

            ST_Transform(
                ST_SetSRID(
                    ST_MakePoint(%s, %s),
                    4326
                ),
                ST_SRID(geometry)
            )

        LIMIT 1;
    """

    cur.execute(
        sql,
        (
            lon,
            lat,
            lon,
            lat
        )
    )

    row = cur.fetchone()

    if not row:

        raise HTTPException(
            status_code=404,
            detail="Could not find nearest routing node."
        )

    return int(
        row["node_id"]
    )


# ============================================================
# RUN DYNAMIC pgRouting
#
# VERY IMPORTANT:
#
# The route cost MUST come from:
#
# dynamic_edge_state.dynamic_weight
#
# NOT:
#
# routing_noded.dynamic_weight
#
# because dynamic_edge_state contains the CURRENT
# live prediction.
# ============================================================

def get_route_rows(
    cur,
    source_node,
    destination_node
):

    sql = """
        SELECT

            dijkstra.seq,

            dijkstra.path_seq,

            dijkstra.node,

            dijkstra.edge,

            dijkstra.cost,

            dijkstra.agg_cost

        FROM pgr_dijkstra(

            $$

            SELECT

                r.edge_id AS id,

                r.source,

                r.target,

                COALESCE(
                    d.dynamic_weight,
                    r.length_km
                ) AS cost,

                COALESCE(
                    d.dynamic_weight,
                    r.length_km
                ) AS reverse_cost

            FROM routing_noded r

            LEFT JOIN dynamic_edge_state d

                ON d.edge_id = r.edge_id

            WHERE
                COALESCE(
                    d.dynamic_weight,
                    r.length_km
                ) IS NOT NULL

            $$,

            %s,
            %s,

            directed := TRUE
        )

        AS dijkstra

        ORDER BY
            dijkstra.seq;
    """

    cur.execute(
        sql,
        (
            source_node,
            destination_node
        )
    )

    rows = cur.fetchall()

    if not rows:

        raise HTTPException(
            status_code=404,
            detail="No route found between source and destination."
        )

    return rows


# ============================================================
# GET EXACT EDGE IDS IN ROUTE ORDER
# ============================================================

def get_edge_ids(
    route_rows
):

    edge_ids = []

    for row in route_rows:

        edge = row["edge"]

        if edge is None:
            continue

        edge = int(edge)

        if edge > 0:

            edge_ids.append(
                edge
            )

    return edge_ids


# ============================================================
# GET EXACT ROUTE DISTANCE
#
# Uses routing_noded.length_km.
#
# This is the reliable road-network distance.
# ============================================================

def get_route_distance(
    cur,
    edge_ids
):

    sql = """
        SELECT

            COALESCE(
                SUM(length_km),
                0
            ) AS distance_km

        FROM routing_noded

        WHERE edge_id = ANY(%s);
    """

    cur.execute(
        sql,
        (
            edge_ids,
        )
    )

    row = cur.fetchone()

    return float(
        row["distance_km"] or 0
    )


# ============================================================
# BUILD ROUTE GEOMETRY
#
# Uses the exact route edges.
#
# For the frontend, a single LineString is convenient.
# ST_UnaryUnion + ST_LineMerge gives the connected route.
#
# Then we explicitly orient it SOURCE -> DESTINATION.
# ============================================================

def build_route_geometry(
    cur,
    edge_ids,
    source_node,
    destination_node
):

    sql = """
        WITH route_data AS (

            SELECT

                ST_LineMerge(
                    ST_UnaryUnion(
                        ST_Collect(
                            r.geometry
                        )
                    )
                ) AS route_geom

            FROM routing_noded r

            WHERE r.edge_id = ANY(%s)
        ),

        points AS (

            SELECT

                route_geom,

                (
                    SELECT geometry
                    FROM routing_vertices
                    WHERE node_id = %s
                ) AS source_geom,

                (
                    SELECT geometry
                    FROM routing_vertices
                    WHERE node_id = %s
                ) AS destination_geom

            FROM route_data
        )

        SELECT

            CASE

                WHEN ST_Distance(
                    ST_StartPoint(route_geom),
                    source_geom
                )
                <=
                ST_Distance(
                    ST_EndPoint(route_geom),
                    source_geom
                )

                THEN route_geom

                ELSE ST_Reverse(
                    route_geom
                )

            END AS route_geom

        FROM points;
    """

    cur.execute(
        sql,
        (
            edge_ids,
            source_node,
            destination_node
        )
    )

    row = cur.fetchone()

    if not row or row["route_geom"] is None:

        raise HTTPException(
            status_code=500,
            detail="Could not construct route geometry."
        )

    return row["route_geom"]


# ============================================================
# GET RISK STATISTICS
#
# CURRENT LIVE values come from dynamic_edge_state.
# ============================================================

def get_route_risk_stats(
    cur,
    edge_ids
):

    sql = """
        SELECT

            AVG(
                COALESCE(
                    d.p_landslide,
                    r.p_landslide,
                    0
                )
            ) AS avg_landslide,

            AVG(
                COALESCE(
                    d.p_flood,
                    r.p_flood,
                    0
                )
            ) AS avg_flood,

            AVG(
                COALESCE(
                    d.p_hazard,
                    r.p_hazard,
                    0
                )
            ) AS avg_hazard,

            MAX(
                COALESCE(
                    d.p_hazard,
                    r.p_hazard,
                    0
                )
            ) AS max_hazard,

            COUNT(
                CASE
                    WHEN COALESCE(
                        d.p_hazard,
                        r.p_hazard,
                        0
                    ) >= 0.60
                    THEN 1
                END
            ) AS high_risk_segments,

            COUNT(
                CASE
                    WHEN COALESCE(
                        d.p_hazard,
                        r.p_hazard,
                        0
                    ) >= 0.80
                    THEN 1
                END
            ) AS very_high_risk_segments

        FROM routing_noded r

        LEFT JOIN dynamic_edge_state d
            ON d.edge_id = r.edge_id

        WHERE r.edge_id = ANY(%s);
    """

    cur.execute(
        sql,
        (
            edge_ids,
        )
    )

    row = cur.fetchone()

    return {

        "avg_landslide":
            float(
                row["avg_landslide"] or 0
            ),

        "avg_flood":
            float(
                row["avg_flood"] or 0
            ),

        "avg_hazard":
            float(
                row["avg_hazard"] or 0
            ),

        "max_hazard":
            float(
                row["max_hazard"] or 0
            ),

        "high_risk_segments":
            int(
                row["high_risk_segments"] or 0
            ),

        "very_high_risk_segments":
            int(
                row["very_high_risk_segments"] or 0
            )
    }


# ============================================================
# HAZARD LEVEL
# ============================================================

def get_hazard_level(
    p_hazard
):

    if p_hazard >= 0.80:

        return "very_high"

    elif p_hazard >= 0.60:

        return "high"

    elif p_hazard >= 0.40:

        return "moderate"

    else:

        return "low"


# ============================================================
# GET RISK FOR CHECKPOINT
# ============================================================

def get_checkpoint_risk(
    cur,
    checkpoint_geom,
    edge_ids
):

    sql = """
        SELECT

            r.edge_id,

            COALESCE(
                d.p_landslide,
                r.p_landslide,
                0
            ) AS p_landslide,

            COALESCE(
                d.p_flood,
                r.p_flood,
                0
            ) AS p_flood,

            COALESCE(
                d.p_hazard,
                r.p_hazard,
                0
            ) AS p_hazard,

            COALESCE(
                d.dynamic_weight,
                r.length_km
            ) AS dynamic_weight

        FROM routing_noded r

        LEFT JOIN dynamic_edge_state d
            ON d.edge_id = r.edge_id

        WHERE r.edge_id = ANY(%s)

        ORDER BY
            r.geometry <-> %s

        LIMIT 1;
    """

    cur.execute(
        sql,
        (
            edge_ids,
            checkpoint_geom
        )
    )

    row = cur.fetchone()

    if not row:

        return {

            "edge_id": None,

            "p_landslide": 0.0,

            "p_flood": 0.0,

            "p_hazard": 0.0,

            "dynamic_weight": 0.0,

            "hazard_level": "unknown"
        }

    p_landslide = float(
        row["p_landslide"] or 0
    )

    p_flood = float(
        row["p_flood"] or 0
    )

    p_hazard = float(
        row["p_hazard"] or 0
    )

    dynamic_weight = float(
        row["dynamic_weight"] or 0
    )

    return {

        "edge_id":
            int(
                row["edge_id"]
            ),

        "p_landslide":
            p_landslide,

        "p_flood":
            p_flood,

        "p_hazard":
            p_hazard,

        "dynamic_weight":
            dynamic_weight,

        "hazard_level":
            get_hazard_level(
                p_hazard
            )
    }


# ============================================================
# CREATE CHECKPOINT DISTANCES
# ============================================================

def create_checkpoint_distances(
    total_distance_km
):

    distances = []

    current = 0.0

    while current < total_distance_km:

        distances.append(
            current
        )

        current += CHECKPOINT_SPACING_KM

    # Always include final destination.
    if (
        not distances
        or
        abs(
            distances[-1] -
            total_distance_km
        ) > 0.001
    ):

        distances.append(
            total_distance_km
        )

    return distances


# ============================================================
# CREATE CHECKPOINTS
#
# IMPORTANT:
#
# The route geometry is oriented source -> destination,
# so checkpoint order follows the correct direction.
# ============================================================

def create_checkpoints(
    cur,
    route_geom,
    edge_ids,
    total_distance_km,
    source_lat,
    source_lon,
    destination_lat,
    destination_lon
):

    checkpoint_distances = (
        create_checkpoint_distances(
            total_distance_km
        )
    )

    checkpoints = []

    total_checkpoints = len(
        checkpoint_distances
    )

    for index, distance_km in enumerate(
        checkpoint_distances
    ):

        fraction = (
            distance_km /
            total_distance_km
        )

        fraction = max(
            0.0,
            min(
                1.0,
                fraction
            )
        )

        # ----------------------------------------------------
        # Get route point
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT

                ST_LineInterpolatePoint(
                    %s,
                    %s
                ) AS point;
            """,
            (
                route_geom,
                fraction
            )
        )

        point_row = cur.fetchone()

        if not point_row:

            continue

        checkpoint_geom = (
            point_row["point"]
        )

        # ----------------------------------------------------
        # WGS84 coordinates
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT

                ST_X(
                    ST_Transform(
                        %s,
                        4326
                    )
                ) AS longitude,

                ST_Y(
                    ST_Transform(
                        %s,
                        4326
                    )
                ) AS latitude;
            """,
            (
                checkpoint_geom,
                checkpoint_geom
            )
        )

        coord_row = cur.fetchone()

        longitude = float(
            coord_row["longitude"]
        )

        latitude = float(
            coord_row["latitude"]
        )

        actual_distance = float(
            distance_km
        )

        # ----------------------------------------------------
        # EXACT SOURCE
        # ----------------------------------------------------

        if index == 0:

            longitude = float(
                source_lon
            )

            latitude = float(
                source_lat
            )

            actual_distance = 0.0

        # ----------------------------------------------------
        # EXACT DESTINATION
        # ----------------------------------------------------

        elif index == total_checkpoints - 1:

            longitude = float(
                destination_lon
            )

            latitude = float(
                destination_lat
            )

            actual_distance = (
                total_distance_km
            )

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk = get_checkpoint_risk(
            cur,
            checkpoint_geom,
            edge_ids
        )

        checkpoints.append(
            {

                "checkpoint_id":
                    f"CP-{index + 1:03d}",

                "distance_from_start_km":
                    round(
                        actual_distance,
                        3
                    ),

                "longitude":
                    round(
                        longitude,
                        6
                    ),

                "latitude":
                    round(
                        latitude,
                        6
                    ),

                "edge_id":
                    risk["edge_id"],

                "p_landslide":
                    round(
                        risk["p_landslide"],
                        6
                    ),

                "p_flood":
                    round(
                        risk["p_flood"],
                        6
                    ),

                "p_hazard":
                    round(
                        risk["p_hazard"],
                        6
                    ),

                "dynamic_weight":
                    round(
                        risk["dynamic_weight"],
                        6
                    ),

                "hazard_level":
                    risk["hazard_level"]
            }
        )

    return checkpoints


# ============================================================
# MAIN ROUTE ENDPOINT
# ============================================================

@app.post("/route")
def calculate_route(
    request: RouteRequest
):

    source_lat = float(
        request.source.lat
    )

    source_lon = float(
        request.source.lon
    )

    destination_lat = float(
        request.destination.lat
    )

    destination_lon = float(
        request.destination.lon
    )

    conn = None

    try:

        conn = get_connection()

        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:

            # =================================================
            # 1. SOURCE NODE
            # =================================================

            source_node = (
                get_nearest_node(
                    cur,
                    source_lat,
                    source_lon
                )
            )

            # =================================================
            # 2. DESTINATION NODE
            # =================================================

            destination_node = (
                get_nearest_node(
                    cur,
                    destination_lat,
                    destination_lon
                )
            )

            # =================================================
            # 3. DYNAMIC ROUTING
            # =================================================

            route_rows = (
                get_route_rows(
                    cur,
                    source_node,
                    destination_node
                )
            )

            # =================================================
            # 4. EXACT ROUTE EDGES
            # =================================================

            edge_ids = (
                get_edge_ids(
                    route_rows
                )
            )

            if not edge_ids:

                raise HTTPException(
                    status_code=404,
                    detail="No valid road edges found."
                )

            # =================================================
            # 5. EDGE COUNT
            # =================================================

            edge_count = len(
                edge_ids
            )

            # =================================================
            # 6. DYNAMIC COST
            # =================================================

            dynamic_cost = float(
                route_rows[-1]["agg_cost"]
            )

            # =================================================
            # 7. ROUTE DISTANCE
            #
            # IMPORTANT:
            # Use SUM(length_km), not ST_Length(ST_MakeLine()).
            # =================================================

            distance_km = (
                get_route_distance(
                    cur,
                    edge_ids
                )
            )

            if distance_km <= 0:

                raise HTTPException(
                    status_code=500,
                    detail="Route distance is zero."
                )

            # =================================================
            # 8. ROUTE GEOMETRY
            # =================================================

            route_geom = (
                build_route_geometry(
                    cur,
                    edge_ids,
                    source_node,
                    destination_node
                )
            )

            # =================================================
            # 9. RISK STATISTICS
            # =================================================

            risk_stats = (
                get_route_risk_stats(
                    cur,
                    edge_ids
                )
            )

            # =================================================
            # 10. CHECKPOINTS
            # =================================================

            checkpoints = (
                create_checkpoints(
                    cur,
                    route_geom,
                    edge_ids,
                    distance_km,
                    source_lat,
                    source_lon,
                    destination_lat,
                    destination_lon
                )
            )

            # =================================================
            # 11. ROUTE GEOJSON
            # =================================================

            cur.execute(
                """
                SELECT

                    ST_AsGeoJSON(
                        ST_Transform(
                            %s,
                            4326
                        )
                    ) AS geojson;
                """,
                (
                    route_geom,
                )
            )

            geojson_row = cur.fetchone()

            if not geojson_row:

                raise HTTPException(
                    status_code=500,
                    detail="Could not convert route to GeoJSON."
                )

            route_geojson = json.loads(
                geojson_row["geojson"]
            )

            # =================================================
            # 12. RESPONSE
            # =================================================

            return {

                "source": {

                    "lat":
                        source_lat,

                    "lon":
                        source_lon,

                    "node_id":
                        source_node
                },

                "destination": {

                    "lat":
                        destination_lat,

                    "lon":
                        destination_lon,

                    "node_id":
                        destination_node
                },

                "route": {

                    "type":
                        "Feature",

                    "properties": {

                        "edge_count":
                            edge_count,

                        "distance_km":
                            round(
                                distance_km,
                                3
                            ),

                        "dynamic_cost":
                            round(
                                dynamic_cost,
                                3
                            ),

                        "avg_landslide":
                            round(
                                risk_stats[
                                    "avg_landslide"
                                ],
                                6
                            ),

                        "avg_flood":
                            round(
                                risk_stats[
                                    "avg_flood"
                                ],
                                6
                            ),

                        "avg_hazard":
                            round(
                                risk_stats[
                                    "avg_hazard"
                                ],
                                6
                            ),

                        "max_hazard":
                            round(
                                risk_stats[
                                    "max_hazard"
                                ],
                                6
                            )
                    },

                    "geometry":
                        route_geojson
                },

                "summary": {

                    "edge_count":
                        edge_count,

                    "distance_km":
                        round(
                            distance_km,
                            3
                        ),

                    "dynamic_cost":
                        round(
                            dynamic_cost,
                            3
                        ),

                    "average_landslide_probability":
                        round(
                            risk_stats[
                                "avg_landslide"
                            ],
                            6
                        ),

                    "average_flood_probability":
                        round(
                            risk_stats[
                                "avg_flood"
                            ],
                            6
                        ),

                    "average_hazard_probability":
                        round(
                            risk_stats[
                                "avg_hazard"
                            ],
                            6
                        ),

                    "max_hazard_probability":
                        round(
                            risk_stats[
                                "max_hazard"
                            ],
                            6
                        ),

                    "high_risk_segments":
                        risk_stats[
                            "high_risk_segments"
                        ],

                    "very_high_risk_segments":
                        risk_stats[
                            "very_high_risk_segments"
                        ],

                    "checkpoint_spacing_km":
                        CHECKPOINT_SPACING_KM
                },

                "checkpoints":
                    checkpoints
            }

    except HTTPException:

        raise

    except Exception as e:

        print(
            "ROUTE ERROR:",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Route calculation failed: {str(e)}"
        )

    finally:

        if conn is not None:

            conn.close()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "message":
            "NER Smart Logistics Routing API is running",

        "endpoint":
            "POST /route"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    conn = None

    try:

        conn = get_connection()

        with conn.cursor() as cur:

            cur.execute(
                "SELECT 1;"
            )

            cur.fetchone()

        return {

            "status":
                "ok",

            "database":
                "connected"
        }

    except Exception as e:

        return {

            "status":
                "error",

            "database":
                "disconnected",

            "message":
                str(e)
        }

    finally:

        if conn is not None:

            conn.close()
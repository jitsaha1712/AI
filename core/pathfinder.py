import heapq
import math
from numbers import Integral

from sqlalchemy import create_engine, text

from shapely import wkb
from shapely.geometry import Point
from shapely.strtree import STRtree
from pyproj import Transformer


class DynamicPathfinder:

    def __init__(self, db_url: str):

        self.engine = create_engine(db_url)

        # Hazard penalty.
        #
        # Example:
        # p_hazard = 0.0  -> normal cost
        # p_hazard = 0.5  -> 3.5x cost
        # p_hazard = 1.0  -> 6x cost
        self.alpha = 5.0

        # Two road endpoints are considered connected
        # when they are within 40 metres.
        self.connection_tolerance_m = 40.0

        # Roads within this distance from the direct
        # source -> destination line are loaded.
        self.corridor_distance_m = 25000.0

        # Database geometries are EPSG:32645.
        # Frontend needs EPSG:4326.
        self.to_wgs84 = Transformer.from_crs(
            32645,
            4326,
            always_xy=True
        ).transform

    # =========================================================
    # PUBLIC FUNCTION
    # =========================================================

    def get_safe_route(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float
    ) -> dict:

        route = self._find_route(
            start_lat,
            start_lng,
            dest_lat,
            dest_lng,
            use_hazard=True
        )

        if route is None:
            return self._empty_route(
                start_lat,
                start_lng,
                dest_lat,
                dest_lng,
                "No connected road route was found."
            )

        return route

    # =========================================================
    # FIND ROUTE
    # =========================================================

    def _find_route(
        self,
        start_lat,
        start_lng,
        dest_lat,
        dest_lng,
        use_hazard=True
    ):

        # -----------------------------------------------------
        # 1. Load roads inside a 25 km corridor
        # -----------------------------------------------------

        corridor_query = text("""
            WITH direct_line AS (

                SELECT ST_Transform(
                    ST_MakeLine(

                        ST_SetSRID(
                            ST_MakePoint(
                                :start_lng,
                                :start_lat
                            ),
                            4326
                        ),

                        ST_SetSRID(
                            ST_MakePoint(
                                :dest_lng,
                                :dest_lat
                            ),
                            4326
                        )

                    ),
                    32645
                ) AS geom
            )

            SELECT
                r.edge_id,
                r.length_km,

                COALESCE(
                    d.p_hazard,
                    0.0
                ) AS p_hazard,

                ST_AsBinary(
                    ST_Force2D(
                        r.geometry
                    )
                ) AS geometry_wkb

            FROM road_edges r

            CROSS JOIN direct_line

            LEFT JOIN dynamic_edge_state d
                ON r.edge_id = d.edge_id

            WHERE ST_DWithin(
                r.geometry,
                direct_line.geom,
                :corridor_distance
            );
        """)

        try:

            with self.engine.connect() as conn:

                rows = conn.execute(
                    corridor_query,
                    {
                        "start_lat": start_lat,
                        "start_lng": start_lng,
                        "dest_lat": dest_lat,
                        "dest_lng": dest_lng,
                        "corridor_distance":
                            self.corridor_distance_m
                    }
                ).fetchall()

        except Exception as e:

            print(
                f"Database error while loading roads: {e}"
            )

            return None

        if not rows:

            print(
                "No road edges found inside corridor."
            )

            return None

        print(
            f"Routing corridor contains "
            f"{len(rows)} road edges."
        )

        # -----------------------------------------------------
        # 2. Convert PostGIS geometries to Shapely
        # -----------------------------------------------------

        edges = []

        for row in rows:

            try:

                geometry = wkb.loads(
                    bytes(row.geometry_wkb)
                )

            except Exception as e:

                print(
                    f"Skipping invalid geometry "
                    f"for edge {row.edge_id}: {e}"
                )

                continue

            if geometry.is_empty:
                continue

            if geometry.geom_type != "LineString":
                continue

            length_km = float(
                row.length_km or 0.0
            )

            p_hazard = float(
                row.p_hazard or 0.0
            )

            # Keep probability inside [0, 1].
            p_hazard = max(
                0.0,
                min(1.0, p_hazard)
            )

            # -------------------------------------------------
            # Edge cost
            # -------------------------------------------------

            if use_hazard:

                weight = (
                    length_km
                    * (
                        1.0
                        + self.alpha * p_hazard
                    )
                )

            else:

                weight = length_km

            edges.append(
                {
                    "edge_id": row.edge_id,
                    "length_km": length_km,
                    "p_hazard": p_hazard,
                    "geometry": geometry,
                    "weight": weight
                }
            )

        if not edges:

            print(
                "No usable LineString road edges."
            )

            return None

        print(
            f"Usable road edges: {len(edges)}"
        )

        # -----------------------------------------------------
        # 3. Build spatial index
        # -----------------------------------------------------

        geometries = [
            edge["geometry"]
            for edge in edges
        ]

        tree = STRtree(geometries)

        # -----------------------------------------------------
        # 4. Convert source/destination to UTM
        # -----------------------------------------------------

        try:

            start_point = self._get_projected_point(
                start_lat,
                start_lng
            )

            dest_point = self._get_projected_point(
                dest_lat,
                dest_lng
            )

        except Exception as e:

            print(
                f"Coordinate conversion error: {e}"
            )

            return None

        # -----------------------------------------------------
        # 5. Find nearest road edges
        # -----------------------------------------------------

        start_idx = self._nearest_edge(
            tree,
            start_point
        )

        dest_idx = self._nearest_edge(
            tree,
            dest_point
        )

        if start_idx is None:

            print(
                "Could not find road near source."
            )

            return None

        if dest_idx is None:

            print(
                "Could not find road near destination."
            )

            return None

        print(
            f"Start edge: "
            f"{edges[start_idx]['edge_id']}"
        )

        print(
            f"Destination edge: "
            f"{edges[dest_idx]['edge_id']}"
        )

        # -----------------------------------------------------
        # 6. Dijkstra search
        # -----------------------------------------------------

        distances = {
            start_idx: 0.0
        }

        previous = {}

        priority_queue = [
            (0.0, start_idx)
        ]

        visited = set()

        found = False

        while priority_queue:

            current_cost, current_idx = (
                heapq.heappop(
                    priority_queue
                )
            )

            if current_idx in visited:
                continue

            visited.add(current_idx)

            # Destination reached
            if current_idx == dest_idx:

                found = True
                break

            current_geometry = (
                edges[current_idx]["geometry"]
            )

            # -------------------------------------------------
            # Current road's two endpoints
            # -------------------------------------------------

            endpoints = [

                Point(
                    current_geometry.coords[0]
                ),

                Point(
                    current_geometry.coords[-1]
                )
            ]

            candidate_indexes = set()

            # -------------------------------------------------
            # Search roads near both endpoints
            #
            # Shapely 2.1 returns integer indexes.
            # -------------------------------------------------

            for endpoint in endpoints:

                nearby_indexes = tree.query(
                    endpoint,
                    predicate="dwithin",
                    distance=self.connection_tolerance_m
                )

                for item in nearby_indexes:

                    if isinstance(
                        item,
                        Integral
                    ):

                        candidate_indexes.add(
                            int(item)
                        )

            # -------------------------------------------------
            # Examine neighboring roads
            # -------------------------------------------------

            for next_idx in candidate_indexes:

                if next_idx == current_idx:
                    continue

                if next_idx in visited:
                    continue

                next_geometry = (
                    edges[next_idx]["geometry"]
                )

                # Make sure the actual endpoints
                # are connected.
                if not self._roads_connected(
                    current_geometry,
                    next_geometry
                ):
                    continue

                new_cost = (
                    current_cost
                    + edges[next_idx]["weight"]
                )

                old_cost = distances.get(
                    next_idx,
                    float("inf")
                )

                if new_cost < old_cost:

                    distances[next_idx] = (
                        new_cost
                    )

                    previous[next_idx] = (
                        current_idx
                    )

                    heapq.heappush(
                        priority_queue,
                        (
                            new_cost,
                            next_idx
                        )
                    )

        # -----------------------------------------------------
        # 7. Route not found
        # -----------------------------------------------------

        if not found:

            print(
                "No connected road route found."
            )

            return None

        # -----------------------------------------------------
        # 8. Reconstruct edge path
        # -----------------------------------------------------

        route_indexes = []

        current = dest_idx

        while current != start_idx:

            route_indexes.append(
                current
            )

            if current not in previous:

                print(
                    "Could not reconstruct route."
                )

                return None

            current = previous[current]

        route_indexes.append(
            start_idx
        )

        route_indexes.reverse()

        print(
            f"Route contains "
            f"{len(route_indexes)} road edges."
        )

        # -----------------------------------------------------
        # 9. Build route geometry
        # -----------------------------------------------------

        route_coords = []

        previous_point = start_point

        for position, idx in enumerate(
            route_indexes
        ):

            geometry = edges[idx]["geometry"]

            coords = list(
                geometry.coords
            )

            if len(coords) < 2:
                continue

            # -------------------------------------------------
            # First edge
            # -------------------------------------------------

            if position == 0:

                d_start = previous_point.distance(
                    Point(coords[0])
                )

                d_end = previous_point.distance(
                    Point(coords[-1])
                )

                if d_end < d_start:

                    coords.reverse()

                route_coords.append(
                    (
                        previous_point.x,
                        previous_point.y
                    )
                )

            # -------------------------------------------------
            # Remaining edges
            # -------------------------------------------------

            else:

                last_point = Point(
                    route_coords[-1]
                )

                d_start = last_point.distance(
                    Point(coords[0])
                )

                d_end = last_point.distance(
                    Point(coords[-1])
                )

                if d_end < d_start:

                    coords.reverse()

            # -------------------------------------------------
            # Add coordinates
            # -------------------------------------------------

            for x, y in coords:

                if not route_coords:

                    route_coords.append(
                        (x, y)
                    )

                    continue

                last_x, last_y = (
                    route_coords[-1]
                )

                distance = math.hypot(
                    x - last_x,
                    y - last_y
                )

                # Ignore duplicate coordinates.
                if distance > 0.5:

                    route_coords.append(
                        (x, y)
                    )

        # -----------------------------------------------------
        # 10. Add destination
        # -----------------------------------------------------

        if route_coords:

            last_x, last_y = (
                route_coords[-1]
            )

            destination_gap = math.hypot(
                dest_point.x - last_x,
                dest_point.y - last_y
            )

            if destination_gap > 0.5:

                route_coords.append(
                    (
                        dest_point.x,
                        dest_point.y
                    )
                )

        # -----------------------------------------------------
        # 11. Convert UTM -> WGS84
        # -----------------------------------------------------

        lat_lng_coords = []

        for x, y in route_coords:

            lng, lat = self.to_wgs84(
                x,
                y
            )

            lat_lng_coords.append(
                [
                    lat,
                    lng
                ]
            )

        # -----------------------------------------------------
        # 12. Prepare edge information
        # -----------------------------------------------------

        edge_details = []

        total_length_km = 0.0

        for idx in route_indexes:

            edge = edges[idx]

            total_length_km += (
                edge["length_km"]
            )

            edge_details.append(
                {
                    "edge_id":
                        edge["edge_id"],

                    "length_km":
                        round(
                            edge["length_km"],
                            3
                        ),

                    "p_hazard":
                        round(
                            edge["p_hazard"],
                            4
                        ),

                    "weight":
                        round(
                            edge["weight"],
                            4
                        )
                }
            )

        # -----------------------------------------------------
        # 13. Return route
        # -----------------------------------------------------

        return {
            "route_found": True,

            "edge_ids": [
                edges[idx]["edge_id"]
                for idx in route_indexes
            ],

            "geometry": lat_lng_coords,

            "total_length_km": round(
                total_length_km,
                2
            ),

            "edge_details": edge_details
        }

    # =========================================================
    # CONVERT LAT/LNG -> UTM
    # =========================================================

    def _get_projected_point(
        self,
        lat,
        lng
    ):

        query = text("""
            SELECT ST_AsBinary(
                ST_Transform(
                    ST_SetSRID(
                        ST_MakePoint(
                            :lng,
                            :lat
                        ),
                        4326
                    ),
                    32645
                )
            );
        """)

        with self.engine.connect() as conn:

            result = conn.execute(
                query,
                {
                    "lat": lat,
                    "lng": lng
                }
            ).scalar()

        if result is None:

            raise ValueError(
                "PostGIS returned NULL geometry."
            )

        # PostgreSQL BYTEA may arrive as bytes.
        if isinstance(result, bytes):

            binary_data = result

        # Some drivers return memoryview.
        elif isinstance(result, memoryview):

            binary_data = result.tobytes()

        # Some configurations can return hex text.
        elif isinstance(result, str):

            binary_data = bytes.fromhex(
                result
            )

        else:

            binary_data = bytes(result)

        return wkb.loads(
            binary_data
        )

    # =========================================================
    # FIND NEAREST EDGE
    # =========================================================

    def _nearest_edge(
        self,
        tree,
        point
    ):

        try:

            result = tree.nearest(
                point
            )

            # Shapely 2.x normally returns
            # numpy.int64 / numpy integer.
            if isinstance(
                result,
                Integral
            ):

                return int(result)

            # Extra compatibility.
            if hasattr(
                result,
                "item"
            ):

                return int(
                    result.item()
                )

            return None

        except Exception as e:

            print(
                f"Nearest edge error: {e}"
            )

            return None

    # =========================================================
    # CHECK ROAD CONNECTION
    # =========================================================

    def _roads_connected(
        self,
        road_a,
        road_b
    ):

        a_start = Point(
            road_a.coords[0]
        )

        a_end = Point(
            road_a.coords[-1]
        )

        b_start = Point(
            road_b.coords[0]
        )

        b_end = Point(
            road_b.coords[-1]
        )

        distances = [

            a_start.distance(
                b_start
            ),

            a_start.distance(
                b_end
            ),

            a_end.distance(
                b_start
            ),

            a_end.distance(
                b_end
            )
        ]

        return (
            min(distances)
            <= self.connection_tolerance_m
        )

    # =========================================================
    # SAFEST + SHORTEST
    # =========================================================

    def get_dual_route_comparison(
        self,
        start_lat,
        start_lng,
        dest_lat,
        dest_lng
    ):

        # -----------------------------------------------------
        # SAFEST
        # -----------------------------------------------------

        safest = self._find_route(
            start_lat,
            start_lng,
            dest_lat,
            dest_lng,
            use_hazard=True
        )

        # -----------------------------------------------------
        # SHORTEST
        # -----------------------------------------------------

        shortest = self._find_route(
            start_lat,
            start_lng,
            dest_lat,
            dest_lng,
            use_hazard=False
        )

        # -----------------------------------------------------
        # Prepare response
        # -----------------------------------------------------

        if safest is None:

            safest = self._empty_route(
                start_lat,
                start_lng,
                dest_lat,
                dest_lng,
                "Safest route could not be found."
            )

        if shortest is None:

            shortest = self._empty_route(
                start_lat,
                start_lng,
                dest_lat,
                dest_lng,
                "Shortest route could not be found."
            )

        return {
            "safest_route": safest,

            "shortest_route": shortest,

            "tradeoff_summary": (
                f"Safest route: "
                f"{safest['total_length_km']} km. "
                f"Shortest route: "
                f"{shortest['total_length_km']} km."
            )
        }

    # =========================================================
    # EMPTY ROUTE
    # =========================================================

    def _empty_route(
        self,
        start_lat,
        start_lng,
        dest_lat,
        dest_lng,
        message
    ):

        return {
            "route_found": False,

            "edge_ids": [],

            "geometry": [],

            "total_length_km": 0.0,

            "edge_details": [],

            "message": message
        }
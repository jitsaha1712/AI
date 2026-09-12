# scripts/09_pathfinder.py
import os
import networkx as nx
from shapely.geometry import Point
from scripts.routing_graph import build_routing_graph

def find_nearest_node_manual(graph, lon, lat):
    """Finds the closest graph node to a given (lon, lat) using node 'x' and 'y' attributes."""
    target_point = Point(lon, lat)
    nearest_node = None
    min_dist = float('inf')
    
    for node, data in graph.nodes(data=True):
        if 'x' in data and 'y' in data:
            # Distance using graph node coordinates
            node_point = Point(data['x'], data['y'])
            dist = node_point.distance(target_point)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                
    return nearest_node

def calculate_safe_route(start_coords, end_coords, roads_path, model_path):
    """
    Calculates the safest route between start and end coordinates 
    avoiding high-risk zones using A* algorithm.
    start_coords & end_coords format: (longitude, latitude)
    """
    # 1. Build or retrieve the live risk-weighted graph
    G, df = build_routing_graph(roads_path, model_path)
    
    lon1, lat1 = start_coords
    lon2, lat2 = end_coords
    
    print(f"\n--- PATHFINDER DEBUG ---")
    print(f"Target Start (Lon, Lat): {start_coords}")
    print(f"Target End (Lon, Lat): {end_coords}")
    
    print("Snapping coordinates using manual node attribute lookup...")
    start_node = find_nearest_node_manual(G, lon1, lat1)
    end_node = find_nearest_node_manual(G, lon2, lat2)
    
    print(f"Snapped Start Node ID: {start_node}")
    print(f"Snapped End Node ID: {end_node}")
    
    if start_node is None or end_node is None:
        print("Error: Could not snap coordinates to any graph node.")
        return None

    print("Computing optimal safe path using A* algorithm...")
    try:
        # Define Euclidean heuristic using graph node attributes 'x' and 'y'
        def spatial_heuristic(u, v):
            node_u = G.nodes[u]
            node_v = G.nodes[v]
            if 'x' in node_u and 'y' in node_u and 'x' in node_v and 'y' in node_v:
                return ((node_u['x'] - node_v['x']) ** 2 + (node_u['y'] - node_v['y']) ** 2) ** 0.5
            return 0

        # Calculate A* path using risk-adjusted 'weight'
        path_nodes = nx.astar_path(
            G, 
            source=start_node, 
            target=end_node, 
            heuristic=spatial_heuristic, 
            weight='weight'
        )
        
        print(f"Route calculated successfully! Total nodes in path: {len(path_nodes)}")
        return path_nodes
        
    except nx.NetworkXNoPath:
        print("Error: NetworkXNoPath - The graph's LCC filter is too small or disconnected between these points.")
        return None
    except Exception as e:
        print(f"Error during A* computation: {e}")
        return None
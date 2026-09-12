import os
import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from models.disaster_model import load_trained_model

def build_routing_graph(roads_gpkg_path, model_path):
    print("Loading labeled road network and trained model...")
    df = gpd.read_file(roads_gpkg_path)
    model = load_trained_model(model_path)
    
    if df.crs != "EPSG:4326":
        df = df.to_crs("EPSG:4326")
        
    print("Predicting risk probabilities for all road segments...")
    feature_cols = ['mean_elevation', 'max_elevation', 'mean_slope', 'max_slope']
    X = df[feature_cols]
    df['risk_probability'] = model.predict_proba(X)[:, 1]
    
    print("Constructing NetworkX graph from road segments...")
    G = nx.Graph()
    
    for idx, row in df.iterrows():
        geom = row.geometry
        if geom and geom.geom_type == 'LineString':
            coords = list(geom.coords)
            
            # Snap coordinates to 5 decimal places
            start_node = (round(coords[0][0], 5), round(coords[0][1], 5))
            end_node = (round(coords[-1][0], 5), round(coords[-1][1], 5))
            
            if start_node == end_node:
                continue
            
            G.add_node(start_node, x=start_node[0], y=start_node[1])
            G.add_node(end_node, x=end_node[0], y=end_node[1])
            
            # Balanced Risk Penalty: 2.5 multiplier
            effective_weight = geom.length * (1.0 + (row['risk_probability'] * 2.5))
            
            if row.get('authority_blocked', 0) == 1:
                effective_weight = float('inf')
                
            if G.has_edge(start_node, end_node):
                if effective_weight < G[start_node][end_node]['weight']:
                    G.add_edge(start_node, end_node, weight=effective_weight, geometry=geom, risk=row['risk_probability'])
            else:
                G.add_edge(start_node, end_node, weight=effective_weight, geometry=geom, risk=row['risk_probability'])
            
    print(f"Initial graph built with {G.number_of_nodes():,} nodes.")
    
    # --- GUARANTEED AUTO-BRIDGING (NO NODES DELETED) ---
    print("Stitching missing bridges and connecting all regions...")
    components = list(nx.connected_components(G))
    
    if len(components) > 1:
        # Sort by size so we connect everything to the largest main map chunk
        components.sort(key=len, reverse=True)
        main_nodes = list(components[0])
        main_tree = cKDTree(np.array(main_nodes))
        
        added_links = 0
        for comp in components[1:]:
            comp_nodes = list(comp)
            
            # Find the absolute closest point between this broken region and the main map
            distances, indices = main_tree.query(np.array(comp_nodes), k=1)
            
            min_idx = np.argmin(distances)
            u = comp_nodes[min_idx]
            v = main_nodes[indices[min_idx]]
            dist = float(distances[min_idx])
            
            # Draw a synthetic bridge across the gap
            G.add_edge(u, v, weight=dist * 2.0, risk=0.0)
            added_links += 1
            
        print(f"Successfully stitched {added_links:,} disconnected regions/missing bridges.")

    # We NO LONGER delete the smaller components. The map is now 100% connected.
    print(f"Final Graph ready for routing: {G.number_of_nodes():,} total nodes.")
        
    return G, df

if __name__ == "__main__":
    roads_path = "Data/processed/assam_roads_final_labeled.gpkg"
    model_file = "models/disaster_risk_model.pkl"
    
    if os.path.exists(roads_path) and os.path.exists(model_file):
        graph, road_df = build_routing_graph(roads_path, model_file)
    else:
        print("Files missing.")
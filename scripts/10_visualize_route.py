# scripts/10_visualize_route.py
import os
import folium
from shapely.geometry import LineString
from scripts.pathfinder import calculate_safe_route

def visualize_route():
    roads_gpkg = "Data/processed/assam_roads_final_labeled.gpkg"
    model_pkl = "models/disaster_risk_model.pkl"
    
    # Define your start and end coordinates (Longitude, Latitude)
    start_point = (91.7362, 26.1445)
    end_point = (92.9376, 26.6380)
    
    print("Calculating safe route for map visualization...")
    path_nodes = calculate_safe_route(start_point, end_point, roads_gpkg, model_pkl)
    
    if not path_nodes:
        print("No path to visualize.")
        return
        
    print("Generating interactive Folium map...")
    # Center map around the midpoint of the start and end coordinates
    center_lat = (start_point[1] + end_point[1]) / 2
    center_lon = (start_point[0] + end_point[0]) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap")
    
    # Add Start and End markers
    folium.Marker(
        location=[start_point[1], start_point[0]],
        popup="Start Location",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)
    
    folium.Marker(
        location=[end_point[1], end_point[0]],
        popup="Destination",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)
    
    # Extract line segments from the path nodes and draw them on the map
    route_coords = [[lat, lon] for lon, lat in path_nodes] # Folium uses [lat, lon]
    
    folium.PolyLine(
        locations=route_coords,
        color="#2b8cbe",
        weight=5,
        opacity=0.8,
        tooltip="Optimized Safe Route"
    ).add_to(m)
    
    # Save output map
    output_path = "safe_route_map.html"
    m.save(output_path)
    print(f"\nMap successfully saved to {os.path.abspath(output_path)}")
    print("Open 'safe_route_map.html' in your browser to view your dynamic safe route!")

if __name__ == "__main__":
    visualize_route()
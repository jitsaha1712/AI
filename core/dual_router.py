# core/dual_router.py
import networkx as nx # Assuming we are using a graph library, or custom adjacency lists
from core.predictive_math import calculate_edge_weight  # Placeholder for your math logic

class CheckpointRouter:
    def __init__(self, graph):
        self.graph = graph

    def segment_checkpoints(self, checkpoints):
        """
        Breaks a list of checkpoints [A, B, C, D] into segments: 
        [(A, B), (B, C), (C, D)]
        """
        if len(checkpoints) < 2:
            raise ValueError("At least two checkpoints are required.")
        return [(checkpoints[i], checkpoints[i+1]) for i in range(len(checkpoints)-1)]

    def compute_dual_route(self, start_node, end_node):
        """
        Calculates two distinct paths between a single segment.
        Path 1: Primary (e.g., shortest distance)
        Path 2: Secondary/Alternative (e.g., least resistance/predicted traffic)
        """
        # Example using standard shortest path for Path 1
        path_primary = nx.shortest_path(self.graph, start=start_node, target=end_node, weight='distance')
        
        # Example using predictive math for Path 2's weights
        # We would apply your predictive_math functions to the edge attributes here
        path_secondary = nx.shortest_path(self.graph, start=start_node, target=end_node, weight='predicted_cost')
        
        return {
            "primary": path_primary,
            "secondary": path_secondary
        }

    def route_full_journey(self, checkpoints):
        """
        Executes dual routing across all checkpoint segments and aggregates the results.
        """
        segments = self.segment_checkpoints(checkpoints)
        journey_plan = []

        for start, end in segments:
            segment_routes = self.compute_dual_route(start, end)
            journey_plan.append({
                "segment": f"{start} -> {end}",
                "routes": segment_routes
            })
            
        return journey_plan
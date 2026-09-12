# File: core/shap_explainer.py

def generate_shap_explanation(hazard_type: str, features: dict) -> str:
    """
    Translates raw terrain and weather feature values into human-readable 
    SHAP explanation sentences for the UI side panel.
    
    Args:
        hazard_type: 'landslide' or 'flood'
        features: dict containing 'rain_24h_mm', 'slope_deg', 'ari_7d', 'distance_to_river_m'
        
    Returns:
        str: Professional natural-language explanation string.
    """
    explanations = []
    
    rain_24h = features.get("rain_24h_mm", 0)
    slope = features.get("slope_deg", 0)
    ari = features.get("ari_7d", 0)
    river_dist = features.get("distance_to_river_m", 500)
    
    if hazard_type == "landslide":
        if rain_24h > 80:
            explanations.append(f"24-hour rainfall ({rain_24h:.1f}mm) is significantly above local historical landslide triggers.")
        if slope > 20:
            explanations.append(f"Terrain slope ({slope:.1f}°) exceeds safe stability thresholds with high soil vulnerability.")
        if ari > 75:
            explanations.append("Antecedent soil saturation index (ARI) indicates deep soil saturation and high failure probability.")
            
    elif hazard_type == "flood":
        if rain_24h > 60:
            explanations.append(f"Heavy short-term precipitation ({rain_24h:.1f}mm) is causing rapid surface water pooling.")
        if river_dist < 300:
            explanations.append(f"Short distance to river basin ({river_dist:.0f}m) increases vulnerability to overflowing banks.")
        if ari > 70:
            explanations.append("Saturated surrounding watershed is accelerating runoff velocity toward this road segment.")

    # Fallback if specific thresholds didn't trip
    if not explanations:
        explanations.append("Combined cumulative weather metrics and local topography indicate elevated hazard risk.")
        
    return " ".join(explanations)
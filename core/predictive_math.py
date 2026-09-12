# File: core/predictive_math.py
import math
from datetime import datetime, timedelta

def calculate_7d_ari(daily_rainfall_mm: list, decay_constant: float = 0.8) -> float:
    """
    Calculates the 7-Day Antecedent Rainfall Index (ARI) to measure soil saturation.
    """
    ari_score = 0.0
    # Formula: ARI = sum( (decay_constant^t) * Rain_t )
    for t, rain in enumerate(daily_rainfall_mm, start=1):
        ari_score += (decay_constant ** t) * rain
        
    return round(ari_score, 3)


def get_weather_for_eta(current_time: datetime, eta_minutes: int, hourly_forecast_data: dict) -> dict:
    """
    Finds the correct weather forecast based on when the vehicle will actually 
    arrive at a downstream checkpoint.
    """
    # 1. Calculate precise arrival time
    arrival_time = current_time + timedelta(minutes=eta_minutes)
    
    # 2. Round to the nearest hour (since forecasts are usually hourly)
    if arrival_time.minute >= 30:
        arrival_hour = arrival_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        arrival_hour = arrival_time.replace(minute=0, second=0, microsecond=0)
        
    arrival_iso = arrival_hour.isoformat()
    
    # 3. Fetch future conditions (Fallback to latest if forecast doesn't extend that far)
    if arrival_iso in hourly_forecast_data:
        return hourly_forecast_data[arrival_iso]
    else:
        # Fallback for unexpected ETA bounds
        latest_available = list(hourly_forecast_data.keys())[-1]
        return hourly_forecast_data[latest_available]
import pandas as pd
import numpy as np
from datetime import timedelta
import os

def generate_custom_route_data(route_id, base_time, output_file='traffic_data.csv'):
    print(f"Generating 6-month synthetic data for new route {route_id} with base time {base_time}m...")
    np.random.seed(abs(hash(route_id)) % (10 ** 8)) # Stable seed per route
    
    start_date = pd.to_datetime('2025-07-01')
    num_days = 180 # 6 months
    
    data = []
    total_hours = num_days * 24
    time_series = [start_date + timedelta(hours=i) for i in range(total_hours)]
    
    for dt in time_series:
        hour = dt.hour
        day_of_week = dt.weekday()
        is_weekend = day_of_week >= 5
        
        # Rush hour travel time multipliers
        if 7 <= hour <= 9: # Morning rush
            hour_mult = 2.0
        elif 16 <= hour <= 19: # Evening rush
            hour_mult = 2.5
        elif 0 <= hour <= 5: # Night (empty roads)
            hour_mult = 0.8
        else: # Regular day
            hour_mult = 1.0
            
        weekend_mult = 0.7 if is_weekend else 1.0
        
        noise = np.random.normal(0, base_time * 0.1)
        w_mult = 1.3 if not is_weekend else 0.9 # Generic custom route multiplier
        
        travel_time = (base_time * hour_mult * weekend_mult * w_mult) + noise
        travel_time = max(base_time * 0.7, int(travel_time)) 
        
        data.append({
            'timestamp': dt,
            'route_id': route_id,
            'travel_time': travel_time
        })
        
    new_df = pd.DataFrame(data)
    
    # Append or create
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        # Drop old records for this route if they exist
        existing_df = existing_df[existing_df['route_id'] != route_id]
        combined = pd.concat([existing_df, new_df])
        combined.to_csv(output_file, index=False)
    else:
        new_df.to_csv(output_file, index=False)
        
    print(f"Generated route data with {len(new_df)} rows.")
    return True

if __name__ == "__main__":
    pass

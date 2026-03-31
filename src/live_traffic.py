import os
import json
import requests
from datetime import datetime

ROUTES_FILE = r'src\routes.json'

def load_routes():
    if os.path.exists(ROUTES_FILE):
        with open(ROUTES_FILE, 'r') as f:
            return json.load(f)
    # Default fallback
    default = {
        'Downtown_to_Airport': {
            'origin': 'Times Square, New York, NY',
            'destination': 'JFK Airport, New York, NY',
            'origin_coords': [40.7580, -73.9855],
            'dest_coords': [40.6413, -73.7781],
            'base_time_mins': 35
        }
    }
    save_routes(default)
    return default

def save_routes(routes):
    with open(ROUTES_FILE, 'w') as f:
        json.dump(routes, f, indent=4)

def geocode_osm(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': address, 'format': 'json', 'limit': 1}
    headers = {'User-Agent': 'SmartTrafficPredictor/1.0'}
    
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200 and len(response.json()) > 0:
        data = response.json()[0]
        return {
            'lat': float(data['lat']),
            'lon': float(data['lon']),
            'name': data.get('display_name', address)
        }
    raise ValueError(f"Could not geocode {address} using Nominatim API.")

def get_route_details(origin_str, dest_str):
    """
    Geocodes the origin and dest via OSM Nominatim and fetches
    baseline routing duration via OSRM.
    """
    org_geo = geocode_osm(origin_str)
    dest_geo = geocode_osm(dest_str)
    
    # OSRM expects: {lon},{lat}
    lon1, lat1 = org_geo['lon'], org_geo['lat']
    lon2, lat2 = dest_geo['lon'], dest_geo['lat']
    
    # Ask OSRM for full GeoJSON geometries to sketch the exact road layout
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    response = requests.get(osrm_url)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('code') == 'Ok' and len(data.get('routes', [])) > 0:
            route_data = data['routes'][0]
            duration_sec = route_data['duration']
            base_duration = int(duration_sec / 60)
            
            # OSRM returns GeoJSON coordinates as [lon, lat]. 
            # Folium requires [lat, lon], so we reverse them cleanly here.
            geometry_path = []
            if 'geometry' in route_data and 'coordinates' in route_data['geometry']:
                geometry_path = [[c[1], c[0]] for c in route_data['geometry']['coordinates']]
            
            return {
                'origin': org_geo['name'],
                'destination': dest_geo['name'],
                'base_time_mins': base_duration,
                'origin_coords': [lat1, lon1],
                'dest_coords': [lat2, lon2],
                'geometry_path': geometry_path
            }
            
    raise ValueError("Could not find a driving route using OSRM.")

def get_live_travel_time(route_info):
    """
    Because OSM's public routing engines (OSRM) do not track live "traffic volume",
    this function applies a mathematical simulation of current traffic conditions 
    based on the current hour of the week to emulate a live 'duration_in_traffic' query.
    """
    base_time = route_info['base_time_mins']
    now = datetime.now()
    hour = now.hour
    is_weekend = now.weekday() >= 5
    
    if 7 <= hour <= 9:   # Morning rush
        multiplier = 2.0
    elif 16 <= hour <= 19: # Evening rush
        multiplier = 2.5
    elif 0 <= hour <= 5: # Night
        multiplier = 0.8
    else:
        multiplier = 1.0
        
    if is_weekend:
        multiplier = multiplier * 0.7 if multiplier > 1.0 else multiplier
        
    return int(max(base_time * 0.8, base_time * multiplier))

def auto_locate_ip():
    """
    Pings a free IP-locator to get the approximate City and State.
    This provides a clean, concise string for Nominatim spatial routing ('Hospital near X').
    """
    try:
        ip_res = requests.get('http://ip-api.com/json', timeout=5)
        if ip_res.status_code == 200:
            data = ip_res.json()
            city = data.get('city')
            region = data.get('regionName')
            country = data.get('country')
            
            if city and region:
                return f"{city}, {region}, {country}"
                
    except Exception as e:
        print(f"Auto-locate failed: {e}")
        
    return "Los Angeles, CA"

def get_ip_coords():
    """Fetches raw GPS coordinates for Folium Map plotting."""
    try:
        ip_res = requests.get('http://ip-api.com/json', timeout=5)
        if ip_res.status_code == 200:
            data = ip_res.json()
            lat, lon = data.get('lat'), data.get('lon')
            if lat and lon:
                return [lat, lon]
    except:
        pass
    return None

def search_osm_live(searchterm: str):
    """
    Live asynchronous keystroke handler. Resolves partial typing queries instantly
    via the Nominatim spatial index and returns a list of formatted location names.
    """
    if not searchterm or len(searchterm) < 3:
        return []
        
    url = "https://nominatim.openstreetmap.org/search"
    params = {'q': searchterm, 'format': 'json', 'limit': 5}
    headers = {'User-Agent': 'SmartTrafficPredictor/1.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return [str(d.get('display_name')) for d in data]
    except Exception:
        pass
    return []

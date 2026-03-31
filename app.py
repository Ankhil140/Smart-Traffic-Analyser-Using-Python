import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import tensorflow as tf
import os
import sys
import joblib

sys.path.append('src')
from live_traffic import get_live_travel_time, load_routes, save_routes, get_route_details, auto_locate_ip, get_ip_coords, search_osm_live
from data_generator import generate_custom_route_data
from model_trainer import train_custom_model
from streamlit_searchbox import st_searchbox

st.set_page_config(page_title="Smart Traffic Analyser", layout="wide", page_icon="🌍")

# --- Premium Custom CSS Injection ---
st.markdown("""
<style>
    /* Scrub Native Streamlit Branding and Headers */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Expand the layout constraints and tighten top padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }

    /* Apply Glassmorphism Aesthetics to all KPI Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 34, 45, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(0, 210, 255, 0.4);
        box-shadow: 0 10px 40px 0 rgba(0, 210, 255, 0.15);
        background: rgba(30, 34, 45, 0.7);
    }

    /* Overwrite all Generic Streamlit Buttons to Hyper-Modern Gradient Pulsing Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.4) !important;
        background: linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%) !important;
    }
    
    /* Elegant Expander Background Overrides */
    div[data-testid="stExpander"] {
        background: rgba(27, 30, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


st.title("🌍 Smart Traffic Analyser")

# Load configurations
ROUTES = load_routes()

@st.cache_data(ttl=60)
def load_data():
    if not os.path.exists('traffic_data.csv'):
        return None
    df = pd.read_csv('traffic_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_resource
def load_models_and_scalers(routes_dict):
    models = {}
    scalers = {}
    for route in routes_dict.keys():
        model_path = f"models/model_{route}.keras"
        scaler_path = f"models/scaler_{route}.pkl"
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            models[route] = tf.keras.models.load_model(model_path)
            scalers[route] = joblib.load(scaler_path)
    return models, scalers

df = load_data()
route_ids = list(ROUTES.keys())

# --- Sidebar UI ---
st.sidebar.header("🗺️ Route Explorer")

if not route_ids:
    st.sidebar.warning("No routes stored. Create one on the main dashboard!")
    selected_route = None
else:
    # Ensure active_route syncs with session state
    if 'active_route' not in st.session_state or st.session_state.active_route not in route_ids:
        st.session_state.active_route = route_ids[0]
        
    selected_route = st.sidebar.selectbox(
        "Select Active Route", 
        route_ids, 
        index=route_ids.index(st.session_state.active_route)
    )
    st.session_state.active_route = selected_route
    
    if selected_route:
        st.sidebar.markdown(f"**Origin:** {ROUTES[selected_route]['origin']}")
        st.sidebar.markdown(f"**Destination:** {ROUTES[selected_route]['destination']}")
        st.sidebar.markdown(f"**Base Travel Time:** {ROUTES[selected_route]['base_time_mins']} mins")


# --- Main Header Expander ---
if 'orig_addr' not in st.session_state:
    st.session_state.orig_addr = ""

if 'dest_addr' not in st.session_state:
    st.session_state.dest_addr = ""

with st.expander("🔍 Search & Build New Route", expanded=(not route_ids)):
    st.markdown("Build a custom Deep Learning model tracking any Origin -> Destination path globally.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        locate_clicked = st.button("📍 Use My Current Location")
        if locate_clicked:
            with st.spinner("Triangulating your coordinates..."):
                st.session_state.orig_addr = auto_locate_ip()
                st.rerun()
                
        st.markdown("**📍 Origin Address**")
        selected_orig = st_searchbox(
            search_osm_live,
            key="search_orig",
            default=st.session_state.get('orig_addr', "") if st.session_state.get('orig_addr', "") else None,
            placeholder="Type a location..."
        )
        if selected_orig:
            st.session_state.orig_addr = selected_orig
                
    with col2:
        bc1, bc2, bc3 = st.columns(3)
        if bc1.button("✈️ Airport"):
            st.session_state.dest_addr = "Airport near " + st.session_state.orig_addr
            st.rerun()
        if bc2.button("🏥 Hospital"):
            st.session_state.dest_addr = "Hospital near " + st.session_state.orig_addr
            st.rerun()
        if bc3.button("🏛️ Downtown"):
            st.session_state.dest_addr = "Downtown near " + st.session_state.get('orig_addr', "")
            st.rerun()
            
        st.markdown("**🏁 Destination Address**")
        selected_dest = st_searchbox(
            search_osm_live,
            key="search_dest",
            default=st.session_state.get('dest_addr', "") if st.session_state.get('dest_addr', "") else None,
            placeholder="Type a destination..."
        )
        if selected_dest:
            st.session_state.dest_addr = selected_dest
        
    st.markdown(" ") # Spacer
    if st.button("⚙️ Analyze & Train Model", type="primary", use_container_width=True):
        if not st.session_state.orig_addr or not st.session_state.dest_addr:
            st.error("Please enter both an Origin and a Destination before training!")
        else:
            orig_val = st.session_state.orig_addr
            
            # Auto name generation algorithm
            n1 = ''.join(e for e in orig_val.split(',')[0].strip() if e.isalnum())
            n2 = ''.join(e for e in st.session_state.dest_addr.split(',')[0].strip() if e.isalnum())
            new_name = f"{n1}_to_{n2}"
            
            if new_name in ROUTES:
                st.warning(f"Route '{new_name}' already exists! Selecting it now.")
                st.session_state.active_route = new_name
                st.rerun()
            else:
                try:
                    with st.spinner(f"1/3 Mapping coordinates for '{new_name}' via Nominatim Geocoder..."):
                        details = get_route_details(orig_val, st.session_state.dest_addr)
                
                    with st.spinner("2/3 Synthesizing 6-months of timeline data..."):
                        generate_custom_route_data(new_name, details['base_time_mins'])
                        
                    with st.spinner("3/3 Training new LSTM architecture (Takes ~10s)..."):
                        train_custom_model(new_name)
                    
                    # Update config
                    ROUTES[new_name] = details
                    save_routes(ROUTES)
                    st.success(f"Successfully trained {new_name}!")
                    st.session_state.active_route = new_name
                    
                    # Clear caches
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error building route: {e}")


# --- Dashboard Visuals ---
if selected_route and df is not None:
    models, scalers = load_models_and_scalers(ROUTES)
    route_info = ROUTES[selected_route]
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Forecast", "Interactive Map"])
    
    with tab1:
        st.subheader(f"Historical Status ({selected_route})")
        col1, col2 = st.columns(2)
        
        latest_data = df.sort_values('timestamp').groupby('route_id').tail(1)
        
        try:
            time_mins = latest_data[latest_data['route_id'] == selected_route]['travel_time'].values[0]
        except IndexError:
            time_mins = 0
            
        col1.metric("Current Simulated Route Traffic", f"{int(time_mins)} mins")
        
        hi_df = df[df['route_id'] == selected_route].tail(168)
        st.line_chart(hi_df.set_index('timestamp')['travel_time'])
        
    with tab2:
        st.subheader(f"24-Hour Deep Learning Forecast")
        
        if selected_route in models and selected_route in scalers:
            model = models[selected_route]
            scaler = scalers[selected_route]
            
            sub_df = df[df['route_id'] == selected_route].sort_values('timestamp').tail(24)
            if not sub_df.empty:
                last_times = sub_df['travel_time'].values.reshape(-1, 1)
                scaled_last = scaler.transform(last_times)
                
                input_seq = np.reshape(scaled_last, (1, 24, 1))
                predictions = []
                current_input = input_seq.copy()
                
                for _ in range(24):
                    pred = model.predict(current_input, verbose=0)
                    predictions.append(pred[0, 0])
                    new_input = np.zeros((1, 24, 1))
                    new_input[0, 0:23, 0] = current_input[0, 1:24, 0]
                    new_input[0, 23, 0] = pred[0, 0]
                    current_input = new_input
                    
                predictions = np.array(predictions).reshape(-1, 1)
                inv_predictions = scaler.inverse_transform(predictions)
                
                last_timestamp = pd.to_datetime(sub_df['timestamp'].iloc[-1])
                future_timestamps = [last_timestamp + pd.Timedelta(hours=i+1) for i in range(24)]
                
                forecast_df = pd.DataFrame({
                    'timestamp': future_timestamps,
                    'predicted_travel_time_mins': inv_predictions.flatten()
                })
                
                forecast_df['baseline'] = route_info['base_time_mins']
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.line_chart(forecast_df.set_index('timestamp')[['predicted_travel_time_mins', 'baseline']])
                    
                with col2:
                    st.markdown("### 🔥 High Congestion Hours")
                    forecast_df['delay'] = forecast_df['predicted_travel_time_mins'] - forecast_df['baseline']
                    peak_hours = forecast_df.nlargest(3, 'delay')
                    for _, row in peak_hours.iterrows():
                        if row['delay'] > 0:
                            st.warning(f"**{row['timestamp'].strftime('%Y-%m-%d %H:%M')}**\n\nExpected: {int(row['predicted_travel_time_mins'])} mins (+{int(row['delay'])}m delay)")
            else:
                st.info("Insufficient data for forecasting.")
        else:
            st.info("Model not trained yet.")
            
    with tab3:
        st.subheader("Global Live Mapper")
        
        col_map, col_live = st.columns([2, 1])
        
        with col_live:
            st.markdown("### 📡 Live API Request")
            st.write(f"Analyze the live `{selected_route}` driving duration.")
            
            if st.button("Fetch Real-Time Traffic"):
                with st.spinner("Simulating OSRM live routing conditions..."):
                    live_time = get_live_travel_time(route_info)
                    
                    if live_time is None:
                        st.error("API Call Failed. OSRM public router might be down.")
                    else:
                        base = route_info['base_time_mins']
                        delay = live_time - base
                        color = "normal" if delay <= 5 else ("inverse" if delay > 15 else "off")
                        st.metric("Live Current Travel Time", f"{live_time} mins", delta=f"{delay} min vs baseline", delta_color=color)
        
        with col_map:
            st.markdown("### 🗺️ Route Visualization")
            show_location = st.checkbox("📍 Use My Current Location")
            
            center_lat = (route_info['origin_coords'][0] + route_info['dest_coords'][0]) / 2
            center_lon = (route_info['origin_coords'][1] + route_info['dest_coords'][1]) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
            
            if show_location:
                user_coords = get_ip_coords()
                if user_coords:
                    folium.Marker(
                        location=user_coords,
                        popup="📡 You Are Here",
                        icon=folium.Icon(color="blue", icon="user")
                    ).add_to(m)
            
            if 'geometry_path' in route_info and len(route_info['geometry_path']) > 0:
                points = route_info['geometry_path']
            else:
                points = [route_info['origin_coords'], route_info['dest_coords']]
            
            try:
                cur_time = latest_data[latest_data['route_id'] == selected_route]['travel_time'].values[0]
                ratio = cur_time / route_info['base_time_mins']
                color = "red" if ratio > 1.5 else ("orange" if ratio > 1.2 else "green")
            except:
                color = "blue"
                
            folium.PolyLine(
                locations=points,
                color=color,
                weight=5,
                opacity=0.8,
                popup=f"{selected_route}"
            ).add_to(m)
            
            # Place markers only at Origin and Dest, not at every geometry turn
            folium.Marker(route_info['origin_coords'], popup="Origin").add_to(m)
            folium.Marker(route_info['dest_coords'], popup="Destination").add_to(m)
            
            folium_static(m, width=600, height=400)

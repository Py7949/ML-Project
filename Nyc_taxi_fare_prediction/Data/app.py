import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic

# --- Page Configuration ---
st.set_page_config(
    page_title="NYC Taxi Fare Predictor 🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Gradient Background CSS ---
# --- NYC Background + Interactive Styling ---
st.markdown(
    """
    <style>
    /* Light background for entire app */
    .stApp {
        background-color: #f0f4f8;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar background & section styling */
    section[data-testid="stSidebar"] {
        background-color: #f0f4f8;
        padding: 1rem;
        border-right: 1px solid #ddd;
    }

    /* Sidebar headers */
    .sidebar-content h1, .sidebar-content h2, .sidebar-content h3 {
        color: #003366;
    }

    /* Number inputs and sliders in sidebar */
    .stNumberInput, .stSlider {
        background-color: #f0f5ff;
        padding: 0.3rem;
        border-radius: 8px;
        transition: 0.2s;
    }
    .stNumberInput:hover, .stSlider:hover {
        background-color: #e6f0ff;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1.2em;
        font-weight: bold;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #004b99;
        transform: scale(1.03);
    }

    /* Main column layout styling */
    .block-container .element-container:nth-child(4) > div {
        background: #e6f2ff;
        padding: 1.2em;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    .block-container .element-container:nth-child(5) > div {
        background: #f2f9ff;
        padding: 1.2em;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    /* Bullet list styling */
    .stMarkdown ul {
        color: #222;
        font-size: 1.05rem;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("🗽 NYC Taxi Fare Predictor")

# --- Sidebar Inputs ---
st.sidebar.header("🧾 Enter Trip Details")
pickup_lat = st.sidebar.number_input("📍 Pickup Latitude", value=40.761432)
pickup_lon = st.sidebar.number_input("📍 Pickup Longitude", value=-73.979815)
dropoff_lat = st.sidebar.number_input("📍 Dropoff Latitude", value=40.651311)
dropoff_lon = st.sidebar.number_input("📍 Dropoff Longitude", value=-73.880333)
passenger_count = st.sidebar.slider("🧍 Passengers", 1, 6, 1)
hour = st.sidebar.slider("⏰ Hour of Pickup", 0, 23, 14)
day_of_week = st.sidebar.selectbox("📅 Day of Week (0=Mon, 6=Sun)", list(range(7)))
is_weekend = 1 if day_of_week in [5, 6] else 0

# --- Calculate Distance ---
distance_km = geodesic((pickup_lat, pickup_lon), (dropoff_lat, dropoff_lon)).km

# --- Input DataFrame ---
input_data = {
    'passenger_count': passenger_count,
    'hour': hour,
    'day_of_week': day_of_week,
    'is_weekend': is_weekend,
    'distance_km': distance_km
}

# --- Try Loading Model ---
try:
    model = joblib.load("D:/Machine learning/ML Project/Nyc_taxi_fare_prediction/Data/taxi_fare_model.pkl")

    # --- Layout Split into Two Columns ---
    col1, col2 = st.columns([1, 1])

    # --- Fare Prediction Panel ---
    with col1:
        st.subheader("🚖 Predict Your Fare")

        if 'fare' not in st.session_state:
            st.session_state.fare = None

        if st.button("🔍 Predict Now"):
            prediction = model.predict(pd.DataFrame([input_data]))[0]
            st.session_state.fare = prediction

        if st.session_state.fare is not None:
            st.success(f"💵 **Estimated Fare: ${st.session_state.fare:.2f}**")

        st.markdown("---")
        st.markdown("### 📊 Trip Summary")
        st.markdown(f"- 🧍 **Passengers:** `{passenger_count}`")
        st.markdown(f"- 🗓️ **Day of Week:** `{day_of_week}` {'(Weekend)' if is_weekend else '(Weekday)' }")
        st.markdown(f"- ⏰ **Pickup Hour:** `{hour}`")
        st.markdown(f"- 🛣️ **Distance:** `{distance_km:.2f} km`")

    # --- Map Panel ---
    with col2:
        st.subheader("🗺️ Visualize Your Trip")
        m = folium.Map(location=[pickup_lat, pickup_lon], zoom_start=12)
        folium.Marker([pickup_lat, pickup_lon], tooltip="Pickup", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([dropoff_lat, dropoff_lon], tooltip="Dropoff", icon=folium.Icon(color='red')).add_to(m)
        folium.PolyLine([[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]], color="blue").add_to(m)
        st_folium(m, width=700, height=500)

except FileNotFoundError:
    st.error("❌ Model file 'taxi_fare_model.pkl' not found. Please make sure it's in the correct folder.")

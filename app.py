
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Define paths relative to the project root for Streamlit Cloud deployment
# For local development, assume app.py is in the project_name root.
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

# --- Page Configuration ---
st.set_page_config(
    page_title="Hotel Energy & Demand Prediction",
    page_icon="🏨",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Helper function to load models with error handling ---
def load_model_and_scaler(model_name):
    try:
        model_path = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, f'{model_name}_scaler.pkl')
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: {model_name} model or scaler files not found in {MODEL_DIR}.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {model_name} model components: {e}")
        st.stop()

# --- Load Models ---
energy_model, energy_scaler = load_model_and_scaler('energy')
hotel_demand_model, hotel_scaler = load_model_and_scaler('hotel_demand')

try:
    hotel_columns_path = os.path.join(MODEL_DIR, 'hotel_columns.pkl')
    hotel_feature_names = joblib.load(hotel_columns_path)
except FileNotFoundError:
    st.error(f"Error: hotel_columns.pkl not found in {MODEL_DIR}.")
    st.stop()
except Exception as e:
    st.error(f"Error loading hotel feature names: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("🏨 Hotel Energy System")
st.sidebar.info(
    "This application predicts hotel energy consumption and booking demand "
    "using machine learning models. Built for demonstration and deployment on Streamlit Cloud."
)
st.sidebar.markdown("**Tech Stack:** Streamlit, Pandas, NumPy, Scikit-learn, Joblib")
st.sidebar.markdown("This project uses sample data for demonstration purposes.")

# --- Main Application ---
st.title("Smart Hotel Predictions")
st.markdown("Welcome to the Hotel Energy System prediction interface.")

tab1, tab2 = st.tabs(["Energy Prediction", "Hotel Demand Prediction"])

with tab1:
    st.header("⚡ Energy Consumption Prediction")
    st.write("Enter the current environmental and occupancy conditions to predict energy usage.")

    # Input fields for Energy Model
    temp_energy = st.slider("Temperature (°C)", min_value=0.0, max_value=40.0, value=22.0, step=0.1)
    humidity_energy = st.slider("Humidity (%)", min_value=20.0, max_value=100.0, value=60.0, step=0.1)
    occupancy_energy = st.slider("Occupancy (Number of Guests)", min_value=1, max_value=500, value=150, step=1)

    if st.button("Predict Energy Consumption"):
        input_data_energy = pd.DataFrame([[temp_energy, humidity_energy, occupancy_energy]],
                                         columns=['temperature', 'humidity', 'occupancy'])
        input_scaled_energy = energy_scaler.transform(input_data_energy)
        prediction_energy = energy_model.predict(input_scaled_energy)[0]

        st.success("Energy Consumption Prediction:")
        st.metric(label="Predicted Energy (kWh)", value=f"{prediction_energy:.2f}")
        st.write("Disclaimer: This is a prediction based on a simplified model and sample data.")

with tab2:
    st.header("📈 Hotel Booking Demand Prediction")
    st.write("Enter details to predict the booking demand for a specific day.")

    # Input fields for Hotel Demand Model
    month_demand = st.selectbox("Month", list(range(1, 13)))
    day_demand = st.slider("Day of Month", min_value=1, max_value=31, value=15, step=1)
    special_event_demand = st.checkbox("Is there a special event?")
    temp_demand = st.slider("Outdoor Temperature (°C)", min_value=-10.0, max_value=45.0, value=20.0, step=0.1)

    if st.button("Predict Booking Demand"):
        special_event_val = 1 if special_event_demand else 0
        input_data_demand = pd.DataFrame([[month_demand, day_demand, special_event_val, temp_demand]],
                                         columns=hotel_feature_names)
        input_scaled_demand = hotel_scaler.transform(input_data_demand)
        prediction_demand = hotel_demand_model.predict(input_scaled_demand)[0]

        st.success("Hotel Booking Demand Prediction:")
        st.metric(label="Predicted Booking Demand (Rooms)", value=f"{prediction_demand:.0f}")
        st.write("Disclaimer: This is a prediction based on a simplified model and sample data.")

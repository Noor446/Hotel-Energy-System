import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Define paths relative to the project root
PROJECT_ROOT = os.path.join(os.getcwd(), 'HotelEnergySystem')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Energy Model Training ---
print("Training Energy Model...")
energy_df = pd.read_csv(os.path.join(DATA_DIR, 'energy_sample.csv'))

X_energy = energy_df[['temperature', 'humidity', 'occupancy']]
y_energy = energy_df['energy_consumption']

energy_scaler = StandardScaler()
X_energy_scaled = energy_scaler.fit_transform(X_energy)

energy_model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42)
energy_model.fit(X_energy_scaled, y_energy)

# Save Energy Model and Scaler
energy_model_path = os.path.join(MODEL_DIR, 'energy_model.pkl')
energy_scaler_path = os.path.join(MODEL_DIR, 'energy_scaler.pkl')

joblib.dump(energy_model, energy_model_path)
joblib.dump(energy_scaler, energy_scaler_path)
print(f"Energy Model and Scaler saved: {energy_model_path}, {energy_scaler_path}")

# --- Hotel Demand Model Training ---
print()
print("Training Hotel Demand Model...")
hotel_df = pd.read_csv(os.path.join(DATA_DIR, 'hotel_sample.csv'))

X_hotel = hotel_df[['month', 'day', 'special_event', 'temperature']]
y_hotel = hotel_df['booking_demand']

hotel_scaler = StandardScaler()
X_hotel_scaled = hotel_scaler.fit_transform(X_hotel)

hotel_model = RandomForestRegressor(n_estimators=12, max_depth=6, min_samples_split=15, min_samples_leaf=5, max_features=0.8, random_state=42)
hotel_model.fit(X_hotel_scaled, y_hotel)

# Save Hotel Demand Model, Scaler, and column names
hotel_model_path = os.path.join(MODEL_DIR, 'hotel_demand_model.pkl')
hotel_scaler_path = os.path.join(MODEL_DIR, 'hotel_scaler.pkl')
hotel_columns_path = os.path.join(MODEL_DIR, 'hotel_columns.pkl')

joblib.dump(hotel_model, hotel_model_path)
joblib.dump(hotel_scaler, hotel_scaler_path)
joblib.dump(X_hotel.columns.tolist(), hotel_columns_path)
print(f"Hotel Demand Model, Scaler, and Columns saved: {hotel_model_path}, {hotel_scaler_path}, {hotel_columns_path}")

print()
print("All models trained and saved successfully!")

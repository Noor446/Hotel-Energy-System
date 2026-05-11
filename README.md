
# HotelEnergySystem

## Smart Hotel Energy and Demand Prediction System

This project provides a machine learning-based system to predict energy consumption and hotel booking demand. It is designed to demonstrate a complete ML project pipeline, from data generation and model training to a live interactive application deployed on Streamlit Cloud. This project uses sample data for demonstration purposes.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)

## Features
- **Energy Consumption Prediction**: Predicts hotel energy usage based on temperature, humidity, and occupancy.
- **Hotel Booking Demand Prediction**: Forecasts booking demand using factors like month, day, special events, and temperature.
- **Interactive Web Interface**: A user-friendly Streamlit application for real-time predictions.
- **Machine Learning Models**: Utilizes `RandomForestRegressor` for both prediction tasks.
- **Scalable Data Preprocessing**: Employs `StandardScaler` for robust feature scaling.
- **Containerization Ready**: Designed with `requirements.txt` for easy environment setup.

## Tech Stack
- Python 3.9+
- Streamlit
- Pandas
- NumPy
- Scikit-learn (specifically `scikit-learn==1.3.2`)
- Joblib
- Matplotlib

## Project Structure
```
HotelEnergySystem/
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── model/                     # Trained models and scalers
│   ├── energy_model.pkl
│   ├── energy_scaler.pkl
│   ├── hotel_demand_model.pkl
│   ├── hotel_scaler.pkl
│   └── hotel_columns.pkl
├── notebook/                  # Jupyter notebooks or training scripts
│   └── training_pipeline.py   # Script to train and save models
├── data/                      # Sample datasets
│   ├── energy_sample.csv
│   └── hotel_sample.csv
└── images/                    # Placeholder for images/diagrams
    └── placeholder.txt
```

## How to Run Locally
Follow these steps to set up and run the project on your local machine.

1.  **Clone the repository (or download the ZIP file)**:
    ```bash
    git clone https://github.com/yourusername/HotelEnergySystem.git
    cd HotelEnergySystem
    ```

2.  **Create and activate a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    # On Windows
    .env\Scriptsctivate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare the data and train the models**:
    First, ensure the `data/` directory contains `energy_sample.csv` and `hotel_sample.csv`. If not, you might need to run a script to generate them (in this Colab environment, they were generated automatically).

    Then, run the training pipeline:
    ```bash
    python notebook/training_pipeline.py
    ```
    This script will train the machine learning models and save them, along with their respective scalers and feature names, into the `model/` directory.

5.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    This command will open the Streamlit application in your web browser, typically at `http://localhost:8501`.

## Streamlit Cloud Deployment
To deploy this application on Streamlit Cloud, follow these steps:

1.  **Push your project to a GitHub repository**: Ensure your repository contains `app.py`, `requirements.txt`, `training_pipeline.py`, the `data/` folder with CSVs, and the `model/` folder (or a mechanism to generate models on deployment).

2.  **Train models for deployment**: It's crucial that the `model/` directory contains all the `.pkl` files (trained models, scalers, and column names) before deployment. You can either:
    - Run `python notebook/training_pipeline.py` locally and commit the generated `model/` directory to your GitHub repository.
    - Modify your `app.py` or create a separate `startup.sh` script to run `python notebook/training_pipeline.py` as part of the Streamlit Cloud's build process. For this project, we assume models are committed.

3.  **Go to Streamlit Cloud**: Navigate to [share.streamlit.io](https://share.streamlit.io/).

4.  **Deploy a new app**: Click on 'Deploy an app' or 'New app'.

5.  **Connect to your GitHub repository**: Select your repository and the branch where your project code resides.

6.  **Specify the main file path**: Enter `app.py` as the main file path.

7.  **Click 'Deploy!'**: Streamlit Cloud will read your `requirements.txt`, install dependencies, and run `app.py`. Your application will then be live!


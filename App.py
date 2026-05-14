import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from sklearn.datasets import fetch_california_housing

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="California Housing Dashboard",
    page_icon="🏠",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

try:
    model = joblib.load("models/xgboost.pkl")
    scaler = joblib.load("models/scaler.pkl")

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# =========================
# LOAD DATASET
# =========================

try:
    housing = fetch_california_housing()

    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    df["Price"] = housing.target

except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

# =========================
# TITLE
# =========================

st.title("🏠 California Housing Prediction System")

st.markdown(
    "Machine Learning & Deep Learning Dashboard"
)

# =========================
# SIDEBAR
# =========================

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Home",
        "Data Insights",
        "Visualizations",
        "Prediction"
    ]
)

# =========================
# HOME
# =========================

if menu == "Home":

    st.header("Project Overview")

    st.write(
        "This project predicts California housing prices using Machine Learning algorithms."
    )

    st.dataframe(df.head())

    st.metric("Total Records", len(df))

# =========================
# DATA INSIGHTS
# =========================

elif menu == "Data Insights":

    st.header("Dataset Insights")

    st.write(df.describe())

    corr = df.corr()

    fig = px.imshow(
        corr,
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# VISUALIZATIONS
# =========================

elif menu == "Visualizations":

    st.header("Housing Visualizations")

    fig1 = px.scatter(
        df,
        x="MedInc",
        y="Price",
        color="HouseAge"
    )

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="Price"
    )

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(
        df,
        y="Price"
    )

    st.plotly_chart(fig3, use_container_width=True)

# =========================
# PREDICTION
# =========================

elif menu == "Prediction":

    st.header("Predict Housing Price")

    MedInc = st.number_input("Median Income")

    HouseAge = st.number_input("House Age")

    AveRooms = st.number_input("Average Rooms")

    AveBedrms = st.number_input("Average Bedrooms")

    Population = st.number_input("Population")

    AveOccup = st.number_input("Average Occupancy")

    Latitude = st.number_input("Latitude")

    Longitude = st.number_input("Longitude")

    if st.button("Predict"):

        try:

            data = np.array([
                [
                    MedInc,
                    HouseAge,
                    AveRooms,
                    AveBedrms,
                    Population,
                    AveOccup,
                    Latitude,
                    Longitude
                ]
            ])

            scaled_data = scaler.transform(data)

            prediction = model.predict(scaled_data)

            st.success(
                f"Predicted House Price: ${prediction[0]:.2f}"
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# =========================
# DEBUG MESSAGE
# =========================

st.write("App is running successfully ✅")
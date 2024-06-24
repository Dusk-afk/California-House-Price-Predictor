import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import skops.io as sio
import streamlit as st

loaded_model: RandomForestRegressor = sio.load("forest_model.skops")
cmap = "coolwarm"

df = pd.read_csv("housing.csv")
ocean_proximities = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

def onehot_encode(data: pd.DataFrame):
    data = data.copy()
    prefix = "ocean_proximity_"
    for ocean_proximity in ocean_proximities:
        data[prefix + ocean_proximity] = 0
    data[prefix + data["ocean_proximity"].iloc[0]] = 1
    data = data.drop("ocean_proximity", axis=1)
    return data

def feature_engineer(data: pd.DataFrame):
    data = data.copy()
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    return data

st.set_page_config(page_title="California Housing Prices Prediction", page_icon="🏠")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.selectbox("Choose an option:", ["Home", "Dataset", "Visualizations", "Model Evaluation", "Predict"])

if option == "Home":
    st.write("## Welcome to the California Housing Prices Dashboard!")
    st.write("This dashboard is to showcase the California Housing Prices dataset and the model built to predict the median house value based on various features.")
    st.write("The dataset used is the [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) from Kaggle. The model used is a Random Forest Regressor.")
    st.write("Use the sidebar to navigate through the different options.")

elif option == "Dataset":
    st.write("## Data set used:")
    st.write("[California Housing Prices (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices)")
    st.write(df)

elif option == "Visualizations":
    st.write("## Some visualizations:")
    st.write("### Median House Value by Longitude and Latitude:")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x="longitude", y="latitude", data=df, hue="median_house_value", palette=cmap, ax=ax)
    st.pyplot(fig)

    st.write("### Median House Value by Ocean Proximity:")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(x="ocean_proximity", y="median_house_value", data=df, hue="ocean_proximity", palette=cmap, ax=ax)
    st.pyplot(fig)

    st.write("### Correlation Matrix:")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap=cmap, ax=ax, fmt=".2f")
    st.pyplot(fig)

elif option == "Model Evaluation":
    st.write("## Model Evaluation:")
    st.write("#### R^2 Score:", 0.83)

elif option == "Predict":
    st.write("## Predict yourself:")
    longitude = st.number_input("Longitude", value=-122.25)
    latitude = st.number_input("Latitude", value=37.85)
    housing_median_age = st.number_input("Housing Median Age", value=41)
    total_rooms = st.number_input("Total Rooms", value=880)
    total_bedrooms = st.number_input("Total Bedrooms", value=129)
    population = st.number_input("Population", value=322)
    households = st.number_input("Households", value=126)
    median_income = st.number_input("Median Income", value=8.3252)
    ocean_proximity = st.selectbox("Ocean Proximity", ocean_proximities, key="ocean_proximity")

    data = {
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    }

    data = pd.DataFrame(data)

    data = onehot_encode(data)
    data = feature_engineer(data)

    st.write("### Predicted Price:")
    prediction = loaded_model.predict(data)
    st.write(prediction[0])
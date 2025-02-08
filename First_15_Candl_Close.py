

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import base64


# Function to encode image to base64
def get_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Provide the correct path to your local image
bg_image = get_base64(r"C:\Users\Admin\Downloads\Codes\Streamlit\Stocks\Stock-Dashboard\Bck_Stocks.jpg")


# Use Streamlit custom CSS to apply the background
st.markdown(
    f"""
    <style>
    /* Set background image for full page */
    .stApp {{
        background: url("data:image/jpg;base64,{bg_image}") no-repeat center center fixed;
        background-size: cover;
    }}

    /* Title Styling */
    .title {{
        color: black;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }}

    /* Make input labels bold and black */
    label {{
        font-weight: bold !important;
        color: black !important;
        font-size: 16px !important;
    }}

    /* Add border to input boxes */
    input[type="text"], input[type="number"], input[type="date"] {{
        border: 2px solid black !important;
        border-radius: 5px !important;
        padding: 8px !important;
        font-size: 14px !important;
        font-weight: bold !important;
    }}

    /* Add border to select dropdown */
    select {{
        border: 2px solid black !important;
        border-radius: 5px !important;
        padding: 8px !important;
        font-size: 14px !important;
        font-weight: bold !important;
    }}

    /* Add border to Streamlit input containers */
    .stTextInput, .stNumberInput, .stDateInput {{
        border: 2px solid black !important;
        border-radius: 5px !important;
        padding: 8px !important;
    }}
    
    .prediction-box {{
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: green;
            border: 2px solid black;
        }}
    
    </style>
    """,
    unsafe_allow_html=True
)



# Load model and transformer safely
try:
    model = joblib.load("RF_Gap_Prediction_01.joblib")
    transformer = joblib.load("Transformer_01_Gap_Train.joblib")
except Exception as e:
    st.error(f"Error loading model or transformer: {e}")
    st.stop()

def preprocess_input(open_price, close_price, date_str):
    """Preprocess user input for the model."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    gap_open = open_price - close_price
    week_of_year = date.isocalendar()[1]
    month_of_year = date.month
    day_of_week = date.weekday()

    # Convert to DataFrame for compatibility
    input_df = pd.DataFrame([[open_price, gap_open, week_of_year, month_of_year, day_of_week]], 
                            columns=['open', 'gap_open', 'week_of_year', 'month_of_year', 'day_of_week'])
    
    # Transform features
    return transformer.transform(input_df)

def predict_close(open_price, close_price, date_str):
    """Predicts the 15-minute candle close price."""
    input_features = preprocess_input(open_price, close_price, date_str)
    return float(model.predict(input_features))

def main():
    st.markdown(
        """
        <style>
        .title {
            color: black;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
        }
        </style>
        <h1 class="title">Nifty 50 Gap Prediction</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h2 class="title">Predict Nifty-50 First 15 Min Candle</h2>', unsafe_allow_html=True)

    # User Inputs
    open_price = st.number_input("Today's Open Price", min_value=0.0, format="%.2f")
    close_price = st.number_input("Yesterday's Close Price", min_value=0.0, format="%.2f")
    date = st.date_input("Select Date").strftime("%Y-%m-%d")

    if st.button("Predict the Closing of First 15 Min Candle"):
        if open_price > 0 and close_price > 0 and date.strip():
            try:
                prediction = predict_close(open_price, close_price, date)
                 # Display prediction with styled output
                st.markdown(
                    f'<div class="prediction-box">Predicted 15-min Candle Close Price: {prediction:.2f}</div>',
                    unsafe_allow_html=True
                )
                
#                 st.success(f"Predicted 15-min Candle Close Price: {prediction:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter valid inputs.")

if __name__ == '__main__':
    main()


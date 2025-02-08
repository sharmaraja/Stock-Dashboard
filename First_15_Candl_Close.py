import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Load model and transformer safely
try:
    model = joblib.load("RF_Gap_Prediction_01.joblib")
    transformer = joblib.load("Transformer_01_Gap_Train.joblib")
except Exception as e:
    st.error(f"Error loading model or transformer: {e}")
    st.stop()

def preprocess_input(open, close, date_str):
    """Preprocess user input for the model."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    gap_open = open - close
    week_of_year = date.isocalendar()[1]
    month_of_year = date.month
    day_of_week = date.weekday()

    # Convert to DataFrame for compatibility
    input_df = pd.DataFrame([[open, gap_open, week_of_year, month_of_year, day_of_week]], 
                            columns=['open', 'gap_open', 'week_of_year', 'month_of_year', 'day_of_week'])
    
    # Transform features
    return transformer.transform(input_df)

def predict_close(open, close, date_str):
    """Predicts the 15-minute candle close price."""
    input_features = preprocess_input(open, close, date_str)
    return float(model.predict(input_features))

def main():
    st.title("Nifty First 15 Min Candle Close Prediction")

    # App Header
    st.markdown("""
    <div style="background:#025246;padding:10px">
    <h2 style="color:white;text-align:center;"> ML-Based Nifty First 15 Min Close Price Prediction </h2>
    </div>
    """, unsafe_allow_html=True)

    # User Inputs
    open = st.number_input("Today's Open Price", min_value=0.0, format="%.2f")
    close = st.number_input("Yesterday's Close Price", min_value=0.0, format="%.2f")
    date = st.date_input("Select Date").strftime("%Y-%m-%d")

    if st.button("Predict the Closing of First 15 Min Candle"):
        if open > 0 and close > 0 and date.strip():
            try:
                prediction = predict_close(open, close, date)
                st.success(f"Predicted 15-min Candle Close Price: {prediction:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter valid inputs.")

if __name__ == '__main__':
    main()

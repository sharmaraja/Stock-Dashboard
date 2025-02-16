{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6484b20d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import joblib\n",
    "from datetime import datetime\n",
    "\n",
    "# Load model and transformer safely\n",
    "try:\n",
    "    model = joblib.load(\"RF_Gap_Prediction_01.joblib\")\n",
    "    transformer = joblib.load(\"Transformer_01_Gap_Train.joblib\")\n",
    "except Exception as e:\n",
    "    st.error(f\"Error loading model or transformer: {e}\")\n",
    "    st.stop()\n",
    "\n",
    "def preprocess_input(open_price, close_price, date_str):\n",
    "    \"\"\"Preprocess user input for the model.\"\"\"\n",
    "    date = datetime.strptime(date_str, \"%Y-%m-%d\")\n",
    "    gap_open = open_price - close_price\n",
    "    week_of_year = date.isocalendar()[1]\n",
    "    month_of_year = date.month\n",
    "    day_of_week = date.weekday()\n",
    "\n",
    "    # Convert to DataFrame for compatibility\n",
    "    input_df = pd.DataFrame([[open_price, gap_open, week_of_year, month_of_year, day_of_week]], \n",
    "                            columns=['open_price', 'gap_open', 'week_of_year', 'month_of_year', 'day_of_week'])\n",
    "    \n",
    "    # Transform features\n",
    "    return transformer.transform(input_df)\n",
    "\n",
    "def predict_close_price(open_price, close_price, date_str):\n",
    "    \"\"\"Predicts the 15-minute candle close price.\"\"\"\n",
    "    input_features = preprocess_input(open_price, close_price, date_str)\n",
    "    return float(model.predict(input_features))\n",
    "\n",
    "def main():\n",
    "    st.title(\"Nifty First 15 Min Candle Close Prediction\")\n",
    "\n",
    "    # App Header\n",
    "    st.markdown(\"\"\"\n",
    "    <div style=\"background:#025246;padding:10px\">\n",
    "    <h2 style=\"color:white;text-align:center;\"> ML-Based Nifty First 15 Min Close Price Prediction </h2>\n",
    "    </div>\n",
    "    \"\"\", unsafe_allow_html=True)\n",
    "\n",
    "    # User Inputs\n",
    "    open_price = st.number_input(\"Today's Open Price\", min_value=0.0, format=\"%.2f\")\n",
    "    close_price = st.number_input(\"Yesterday's Close Price\", min_value=0.0, format=\"%.2f\")\n",
    "    date = st.date_input(\"Select Date\").strftime(\"%Y-%m-%d\")\n",
    "\n",
    "    if st.button(\"Predict the Closing of First 15 Min Candle\"):\n",
    "        if open_price > 0 and close_price > 0 and date.strip():\n",
    "            try:\n",
    "                prediction = predict_close_price(open_price, close_price, date)\n",
    "                st.success(f\"Predicted 15-min Candle Close Price: {prediction:.2f}\")\n",
    "            except Exception as e:\n",
    "                st.error(f\"Error: {e}\")\n",
    "        else:\n",
    "            st.warning(\"Please enter valid inputs.\")\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

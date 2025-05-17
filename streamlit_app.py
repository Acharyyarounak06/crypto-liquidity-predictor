import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load model and scaler (for demonstration, will be created inside app)
@st.cache_data
def load_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    return model, scaler

# UI layout
st.title("Cryptocurrency Liquidity Predictor")
st.markdown("Upload a CSV file with historical crypto data to predict liquidity-related metrics.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df.head())

    # Basic preprocessing
    if 'price' in df.columns and 'volume' in df.columns:
        df['price_ma_7'] = df['price'].rolling(window=7).mean()
        df['volatility_7'] = df['price'].rolling(window=7).std()
        df = df.dropna()

        features = ['volume', 'price_ma_7', 'volatility_7']
        X = df[features]

        # Scaling
        model, scaler = load_model()
        X_scaled = scaler.fit_transform(X)

        # Dummy training (in production, use pre-trained model)
        model.fit(X_scaled, df['price'])

        # Predict
        df['predicted_price'] = model.predict(X_scaled)

        st.subheader("Predictions")
        st.write(df[['price', 'predicted_price']].tail())

        st.line_chart(df[['price', 'predicted_price']])
    else:
        st.error("Required columns 'price' and 'volume' not found in uploaded CSV.")
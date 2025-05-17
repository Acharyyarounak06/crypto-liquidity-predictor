import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Custom CSS and Google Fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap" rel="stylesheet">
<style>
body {
    background-color: #0f0f0f;
    color: #ffffff;
    font-family: 'Orbitron', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #1a1a1a, #0d0d0d);
    padding: 2rem;
    animation: fadeIn 1.5s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
h1 {
    color: #00ffcc;
    text-shadow: 0 0 12px #00ffcc;
    animation: glow 2s ease-in-out infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 5px #00ffcc; }
    to { text-shadow: 0 0 20px #00ffcc; }
}
.stFileUploader {
    background: rgba(0, 255, 204, 0.05);
    border: 2px dashed #00ffcc;
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
}
.stButton>button {
    background-color: #00cc99;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    background-color: #00ffcc;
    color: #000;
    transform: scale(1.05);
    box-shadow: 0 0 12px #00ffcc;
}
img.crypto-logo {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 120px;
    margin-bottom: 10px;
    filter: drop-shadow(0 0 10px #00ffcc);
}
.block-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 0 30px rgba(0, 255, 204, 0.1);
}
table {
    border-collapse: collapse;
    width: 100%;
}
table td, table th {
    border: 1px solid #00cc99;
    padding: 8px;
}
table tr:hover {
    background-color: rgba(0, 255, 204, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Crypto logo
st.markdown(
    '<img src="https://cryptologos.cc/logos/bitcoin-btc-logo.png" class="crypto-logo">',
    unsafe_allow_html=True
)

# Title
st.title("Cryptocurrency Liquidity Predictor")
st.markdown("Upload a CSV file with historical crypto data to predict liquidity-related metrics.")

# Glass effect container
st.markdown('<div class="block-container">', unsafe_allow_html=True)

# Load model and scaler
@st.cache_data
def load_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    return model, scaler

# File Upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df.head())

    if 'price' in df.columns and 'volume' in df.columns:
        # Preprocessing
        df['price_ma_7'] = df['price'].rolling(window=7).mean()
        df['volatility_7'] = df['price'].rolling(window=7).std()
        df = df.dropna()

        features = ['volume', 'price_ma_7', 'volatility_7']
        X = df[features]

        model, scaler = load_model()
        X_scaled = scaler.fit_transform(X)

        # Dummy training (in production, use pre-trained model)
        model.fit(X_scaled, df['price'])

        df['predicted_price'] = model.predict(X_scaled)

        st.subheader("Predictions")
        st.write(df[['price', 'predicted_price']].tail())

        st.line_chart(df[['price', 'predicted_price']])
    else:
        st.error("Required columns 'price' and 'volume' not found in uploaded CSV.")

# Close glass container
st.markdown('</div>', unsafe_allow_html=True)

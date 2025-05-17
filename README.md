# Crypto Liquidity Predictor
A machine learning tool for predicting cryptocurrency liquidity using historical price and volume data. This project applies time series analysis and advanced modeling techniques to forecast liquidity trends, aiding traders and analysts in making data-driven decisions.

# Table of Contents
1. Overview
2. Features
3. Installation
4. Usage
5. Modeling Approach
6. Evaluation
7. Deployment
8. Contributing
9. License

# Overview
Cryptocurrency markets are highly volatile and liquidity is a critical metric influencing trading strategies and market stability. This project leverages historical market data — including price and trading volume — to predict liquidity using machine learning models. The predictions help anticipate liquidity crunches or surges, providing actionable insights.

# Features
Data collection and preprocessing from cryptocurrency APIs or CSV files

Feature engineering on price and volume time series data

Support for various machine learning algorithms such as Random Forest, XGBoost, and LSTM

Model evaluation with metrics like RMSE, MAE, and R²

Interactive front-end deployment using Streamlit or Flask for live predictions


# Setup
Clone the repo:
git clone https://github.com/Acharyyarounak06/crypto-liquidity-predictor.git
cd crypto-liquidity-predictor

# Install dependencies:
pip install -r requirements.txt


## Usage

# To train the model:
python train.py
# To run the Streamlit app:
streamlit run app_streamlit.py
# To run the Flask app:
python app_flask.py

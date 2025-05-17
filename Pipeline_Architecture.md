# Pipeline Architecture

1. **Data Ingestion**
    - Load CSV file using `pandas`

2. **Data Preprocessing**
    - Handle missing values
    - Normalize features
    - Generate engineered features (e.g., moving averages, volatility)

3. **Modeling**
    - Random Forest Regression
    - Train/Test split
    - Evaluation using RMSE, MAE, R²

4. **Prediction Output**
    - Model makes predictions on unseen test data

5. **Deployment (optional)**
    - Simple Flask or Streamlit app for predictions
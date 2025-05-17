# Low-Level Design (LLD)

## Modules

### 1. Data Preprocessing
- Handle missing values
- Convert date columns
- Normalize features using `StandardScaler`
- Feature engineering (moving average, volatility)

### 2. EDA
- Line plots for price trends
- Heatmaps for correlation analysis

### 3. Modeling
- Use Random Forest Regressor (or switchable with other models)
- Train/test split (80/20)
- Performance metrics: RMSE, MAE, R²

### 4. Deployment (Future Work)
- Streamlit interface for uploading data and displaying predictions

## Dependencies
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
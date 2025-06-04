# Time Series Forecasting with Machine Learning

This repository demonstrates how to perform hourly energy consumption forecasting using Python, XGBoost, and time series feature engineering techniques. The dataset used is the PJME (PJM East) energy load data, and the project includes everything from preprocessing to future forecasting and model deployment.

---

## 📊 Project Overview

- **Data**: Hourly electricity consumption data from PJM East (PJME).
- **Goal**: Forecast hourly energy usage into the future using XGBoost.
- **Techniques**:
  - Time-based feature engineering (hour, month, day of year, etc.)
  - Lag features (values from previous years)
  - Outlier filtering
  - Time Series Cross Validation
  - Model persistence (saving and reloading)

---

## 🧪 Main Steps

1. **Data Preparation**
   - Loaded and cleaned PJME hourly data
   - Removed outliers below 19,000 MW
   - Resampled and interpolated for hourly gaps

2. **Train/Test Splitting**
   - Used time-based split (`TimeSeriesSplit`) for proper cross-validation

3. **Feature Engineering**
   - Added time-related features: `hour`, `dayofweek`, `month`, `year`, etc.
   - Added lag features: energy usage from 1, 2, and 3 years prior

4. **Model Training**
   - Used `XGBRegressor` with early stopping
   - Evaluated using RMSE across 5 CV folds

5. **Forecasting**
   - Trained final model on full data
   - Created a future hourly datetime range
   - Predicted future energy usage up to 1 year ahead

6. **Model Saving**
   - Saved the trained model as `model.json`
   - Reloaded it to ensure persistence works correctly

---

## 📈 Key Visualizations

- PJME Energy usage distribution
- Training vs Test set splits across folds
- Boxplots for seasonality (hourly and monthly)
- Future forecast visualization

---

## 📦 Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

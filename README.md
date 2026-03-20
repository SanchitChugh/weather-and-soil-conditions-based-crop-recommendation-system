# 🌾 Crop Recommendation & Weather Forecasting System

A Machine Learning + Deep Learning based system designed to help farmers make **data-driven agricultural decisions** by analyzing soil conditions and forecasting future weather.

---

## 🚀 Project Overview

Farmers often face challenges such as:

* Unpredictable weather 🌦️
* Lack of proper soil analysis 🧪
* Incorrect crop selection 🌱

To address these issues, this project introduces a **two-part intelligent system**:

### 🔹 1. Crop Recommendation Model

* Uses soil features to predict the most suitable crop

### 🔹 2. Weather Forecasting Model

* Predicts **next 30 days** of weather using historical data

📊 Both models are integrated into a **Streamlit dashboard** for easy visualization and decision-making.

---

## 📦 Datasets Used

### 1️⃣ Soil Features Dataset (Kaggle)

* Features:

  * N, P, K
  * pH
  * Moisture
  * Temperature
  * Humidity
* Purpose:

  * Analyze soil fertility
  * Determine crop suitability

---

### 2️⃣ Weather Dataset (Open-Meteo API)

* Districts:

  * Faridabad
  * Gurgaon
  * Palwal

* Features:

  * temp_max, temp_min, temp_mean
  * rainfall
  * humidity
  * month
  * day_of_year

* Purpose:

  * Train LSTM model for future weather prediction

---

## 🛠️ Models Used

### 🔹 Crop Recommendation (Tabular ML)

Tested Algorithms:

* Random Forest
* XGBoost
* LightGBM
* MLP
* CatBoost

#### 📊 Model Comparison

| Model        | Accuracy | Remarks                                               |
| ------------ | -------- | ----------------------------------------------------- |
| XGBoost      | 96–97%   | ✅ Selected (best stability & probability calibration) |
| RandomForest | 96–97%   | ✅ Selected (robust and interpretable)                 |
| LightGBM     | 96–97%   | ❌ Rejected (over-confident predictions)               |
| MLP          | 96–97%   | ❌ Rejected (poor probability distribution)            |
| CatBoost     | 96–97%   | ⚠️ Good but less stable than XGBoost                  |

---

### 🔹 Final Ensemble Model

➡️ Combination of:

* Random Forest
* XGBoost

---

### 🔹 Weather Forecasting (Time Series Deep Learning)

* Separate **LSTM models** for each district

#### Input:

* Last **90 days** of weather data

#### Output:

* Forecast for **next 30 days**

---

### 🧠 LSTM Architecture

```
LSTM(128, return_sequences=True)
Dropout(0.2)
LSTM(64)
Dropout(0.15)
Dense(64, activation='relu')
Dense(7) → Outputs 7 weather features
```

---

### 📉 Training Details

* Loss Function: Mean Squared Error (MSE)
* Evaluation Metric: Mean Absolute Error (MAE)

📊 Additional Metrics:

* RMSE
* MAPE
* R²

(Used for temperature, humidity, and rainfall evaluation)

---

## 📊 Dashboard (Streamlit)

The final application provides:

* 📈 30-day weather forecast graph
* 🌡️ Average temperature, humidity, rainfall
* 🧪 Soil health indicators
* 🌱 Top 3 recommended crops

---

## 📁 Project Structure

```
├── data/
│   ├── soil_dataset.csv
│   ├── faridabad_weather_2010_2025.csv
│   ├── gurgaon_weather_2010_2025.csv
│   ├── palwal_weather_2010_2025.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── faridabad_model.h5
│   ├── gurgaon_model.h5
│   ├── palwal_model.h5
│
├── streamlit_app.py
├── train_crop_model.py
├── train_weather_model.py
├── README.md
```

---

## 🧪 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train crop recommendation model
python train_crop_model.py

# Train weather forecasting model
python train_weather_model.py

# Run dashboard
streamlit run streamlit_app.py
```

---

## 🔮 Future Improvements

* 📊 Add market price analysis for profit-based crop selection
* 🌐 Integrate real-time sensor data from fields
* 📍 Expand system to more districts
* 💧 Implement irrigation and fertilizer planning

---

## 📌 Conclusion

This project combines:

* Machine Learning (for crop recommendation)
* Deep Learning (LSTM for weather forecasting)

to support Indian farmers with:

* Accurate weather predictions
* Reliable crop recommendations

✅ The use of **ensemble models + LSTM** ensures:

* Robustness
* Interpretability
* Practical real-world usability

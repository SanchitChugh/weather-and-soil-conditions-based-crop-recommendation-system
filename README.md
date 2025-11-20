🌾 Crop Recommendation & Weather Forecasting System

This project uses Machine Learning and Deep Learning to help farmers make data-driven agricultural decisions. The system predicts future weather conditions and recommends the most suitable crops based on soil and climate features.

🚀 Project Overview

Farmers often struggle with unpredictable weather, lack of soil analysis, and incorrect crop selection. To solve this, we built a two-part intelligent system:

Crop Recommendation Model – Uses soil features to predict the best crop.

Weather Forecasting Model – Predicts next 30 days of weather conditions using past data.

Both models are integrated into a Streamlit dashboard for easier visualization and decision support.

📦 Datasets Used
1️⃣ Soil Features Dataset (Kaggle)

Columns: N, P, K, pH, moisture, temperature, humidity, etc.

Used to determine soil fertility and crop suitability.

2️⃣ Weather Dataset (Open-Meteo API)

Districts: Faridabad, Gurgaon, Palwal

Columns: temp_max, temp_min, temp_mean, rain, humidity, month, day_of_year

Used to train LSTM model for forecasting future weather.

🛠️ Models Used
🔹 Crop Recommendation (Tabular ML)

We tested 5 algorithms:
Random Forest, XGBoost, LightGBM, MLP, CatBoost

Model	Accuracy	Remarks
XGBoost	96–97%	Selected ✔ (best stability & probability calibration)
RandomForest	96–97%	Selected ✔ (robust and interpretable)
LightGBM	96–97%	Rejected (over-confident 100% single prediction)
MLP	96–97%	Rejected (poor probability distribution)
CatBoost	96–97%	Good but less stable vs XGB

🔹 Final Ensemble:
➡️ Random Forest + XGBoost

🔹 Weather Forecasting (Time Series DL)

We built separate LSTM models for each district.

Input: Last 90 days of weather data  
Output: Prediction for next 30 days


LSTM Architecture

LSTM(128, return_sequences=True)
Dropout(0.2)
LSTM(64)
Dropout(0.15)
Dense(64, activation='relu')
Dense(7)  → Output all 7 weather features


Loss: Mean Squared Error (MSE)

Metric: Mean Absolute Error (MAE)

Accuracy measured using MAE, RMSE, MAPE, R² for temperature, humidity, rainfall.

📊 Dashboard (Streamlit)

The final application displays:

📈 Weather forecast graph (30 days)

🌡️ Average temperature, humidity, and rainfall

🧪 Soil health indicators

🌱 Top 3 recommended crops

📁 Project Structure
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

🧪 How to Run
# Install dependencies
pip install -r requirements.txt

# Train crop recommendation model
python train_crop_model.py

# Train weather forecasting model
python train_weather_model.py

# Run dashboard
streamlit run streamlit_app.py

🔮 Future Improvements

Add market price analysis for profitable crop selection

Integrate real-time sensor data from fields

Expand to more districts

Implement irrigation and fertilizer planning


📌 Conclusion

This project combines Machine Learning and Deep Learning to assist Indian farmers by providing accurate weather predictions and reliable crop recommendations. Using LSTM for forecasting and ensemble models for classification ensures robustness, interpretability, and practical usability in real-world farming.

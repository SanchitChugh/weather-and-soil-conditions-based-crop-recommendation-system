import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Crop model
from utils.ensemble import recommend_top3


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Smart Agriculture Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Weather Forecast", "Crop Recommendation"])


# ============================================================
#                PAGE 1 — WEATHER FORECAST
# ============================================================
if page == "Weather Forecast":

    st.markdown(
        "<h1 style='font-size:40px; font-weight:700;'> ☁️ 30-Day Weather Forecast </h1>",
        unsafe_allow_html=True
    )

    st.write("Forecasting next 30 days weather using LSTM model (5 features).")

    district_models = {
        "Faridabad": {
            "model": "weather_models/faridabad_model.h5",
            "scaler": "weather_models/faridabad_scaler.joblib",
            "data": "weather_models/faridabad_weather_2010_2025-11-15.csv"
        },
        "Gurgaon": {
            "model": "weather_models/gurgaon_model.h5",
            "scaler": "weather_models/gurgaon_scaler.joblib",
            "data": "weather_models/gurgaon_weather_2010_2025-11-15.csv"
        },
        "Palwal": {
            "model": "weather_models/palwal_model.h5",
            "scaler": "weather_models/palwal_scaler.joblib",
            "data": "weather_models/palwal_weather_2010_2025-11-15.csv"
        }
    }

    district = st.selectbox("Select District", list(district_models.keys()))
    info = district_models[district]

    df = pd.read_csv(info["data"], parse_dates=["date"])
    df = df.sort_values("date")

    model = load_model(info["model"])
    scaler = joblib.load(info["scaler"])

    features = ["temp_max", "temp_min", "temp_mean", "rain", "humidity"]
    WINDOW = 90
    FORECAST_DAYS = 30

    scaled_data = scaler.transform(df[features].values)
    last_seq = scaled_data[-WINDOW:].copy()

    pred_scaled = []
    for _ in range(FORECAST_DAYS):
        p = model.predict(last_seq.reshape(1, WINDOW, len(features)), verbose=0)[0]
        pred_scaled.append(p)
        last_seq = np.vstack([last_seq[1:], p])

    pred_scaled = np.array(pred_scaled)
    pred_original = scaler.inverse_transform(pred_scaled)

    future_dates = pd.date_range(
        df["date"].max() + pd.Timedelta(days=1),
        periods=FORECAST_DAYS
    )

    final_cols = ["temp_max", "temp_min", "temp_mean", "rain", "humidity"]
    forecast_df = pd.DataFrame(pred_original, columns=features)
    forecast_df["date"] = future_dates
    forecast_df = forecast_df[["date"] + final_cols]

    # Summary Metrics
    st.markdown("<h2>📊 Summary — Next 30 Days</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    col1.metric("🌡 Avg Temperature", f"{forecast_df['temp_mean'].mean():.2f} °C")
    col2.metric("💧 Avg Humidity", f"{forecast_df['humidity'].mean():.2f} %")
    col3.metric("🌧 Avg Rainfall", f"{forecast_df['rain'].mean():.2f} mm")

    # Forecast Table
    st.markdown("<h2>📄 Forecast Table</h2>", unsafe_allow_html=True)
    st.dataframe(forecast_df)

    # Rain Graph
    st.markdown("<h2>🌧 Rainfall Trend</h2>", unsafe_allow_html=True)
    plt.figure(figsize=(10, 5))
    plt.plot(forecast_df["date"], forecast_df["rain"], marker='o')
    plt.xlabel("Date")
    plt.ylabel("Rain (mm)")
    plt.grid(True)
    st.pyplot(plt)

    # Download CSV
    st.download_button(
        label="📥 Download Forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{district.lower()}_30day_forecast.csv",
        mime="text/csv"
    )


# ============================================================
#                PAGE 2 — CROP RECOMMENDATION
# ============================================================
elif page == "Crop Recommendation":

    st.markdown("<h1>🌱 Crop Recommendation System</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200, 50)
        P = st.number_input("Phosphorus (P)", 0, 200, 40)
        K = st.number_input("Potassium (K)", 0, 200, 50)

    with col2:
        temperature = st.number_input("Temperature (°C)", 0, 50, 25)
        humidity = st.number_input("Humidity (%)", 0, 100, 70)

    with col3:
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
        rainfall = st.number_input("Rainfall (mm)", 0, 500, 200)
        # --------------------------------------------------------
    # SINGLE SOIL HEALTH THERMOMETER (Above Predict Button)
    # --------------------------------------------------------
    st.markdown("<h2>🧪 Soil Health</h2>", unsafe_allow_html=True)

    # --- NORMALIZED VALUES (default = 100%) ---

    # 1) Organic Matter (NPK)
    organic_raw = N + P + K       # default = 140
    organic_matter = min(100, (organic_raw / 140) * 100)

    # 2) Soil Minerals (pH)
    # ideal = 6.5 to 7.0
    if 6.5 <= ph <= 7.0:
        mineral_balance = 100
    else:
        # deviation reduces score mildly
        mineral_balance = max(0, 100 - abs(ph - 6.75) * 25)

    # 3) Microbial Activity (humidity + rainfall)
    # default hum=70, rain=200 => 100%
    microbial = (
        (humidity / 70) * 50 +      # humidity contributes 50%
        (rainfall / 200) * 50       # rainfall contributes 50%
    )
    microbial = min(100, microbial)

    # ---- Combined Weighting ----
    soil_health_score = (
        organic_matter * 0.4 +
        mineral_balance * 0.3 +
        microbial * 0.3
    )

    soil_health_score = min(100, soil_health_score)

    # ---- Display ----
    st.markdown(f"### 🌡️ Soil Health: {soil_health_score:.1f}%")
    st.progress(int(soil_health_score))



    if st.button("Predict Best Crops"):

        farmer_input = {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        results = recommend_top3(farmer_input)

        st.markdown("<h2>🌾 Suggested Crops</h2>", unsafe_allow_html=True)

        # Crop Icons
        CROP_ICONS = {
            "rice": "🌾", "wheat": "🌿", "maize": "🌽", "corn": "🌽",
            "cotton": "🧵", "sugarcane": "🍭", "millet": "🌱",
            "pigeonpeas": "🟢", "kidneybeans": "🫘", "mothbeans": "🫘",
            "mungbean": "🟩", "blackgram": "⬛", "lentil": "🍲",
            "banana": "🍌", "mango": "🥭", "grapes": "🍇",
            "apple": "🍎", "orange": "🍊", "papaya": "🥭",
            "coconut": "🥥", "jute": "🧶", "coffee": "☕"
        }

        # Show Top 3 Crops with Icons
        for crop, prob in results:
            icon = CROP_ICONS.get(crop.lower(), "🌱")
            st.write(f"### {icon} {crop} ({prob*100:.1f}%)")

        

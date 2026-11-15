import datetime
import os
from collections import defaultdict

import joblib
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn as nn

st.set_page_config(page_title="Smart Irrigation Advisor", page_icon="💧", layout="centered")

INPUT_COLS = ["CROP TYPE", "SOIL TYPE", "REGION", "WEATHER CONDITION"]

# ---------------------------------------------------------------------------
# Load trained artifacts (cached so they load only once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    input_dim = len(encoder.get_feature_names_out())

    # Must match the architecture used in training exactly
    model = nn.Sequential(
        nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )
    model.load_state_dict(torch.load("water_requirement_model.pth", map_location="cpu"))
    model.eval()
    return encoder, scaler, model


try:
    encoder, scaler, model = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e.filename}. Put `water_requirement_model.pth`, "
        "`encoder.pkl` and `scaler.pkl` in the same folder as app.py."
    )
    st.stop()

# Dropdown options come straight from the encoder, so no dataset is needed here
categories = dict(zip(INPUT_COLS, encoder.categories_))

# ---------------------------------------------------------------------------
# Crop calendar / feasibility
# ---------------------------------------------------------------------------
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

CROP_CALENDAR = {
    "BANANA": ["Year-round"],
    "SOYABEAN": ["June-July", "September-October"],
    "CABBAGE": ["September-October", "December-March"],
    "POTATO": ["September-November", "January-March"],
    "RICE": ["June-July", "October-November"],
    "MELON": ["January-February", "April-June"],
    "MAIZE": ["June-July", "September-October"],
    "CITRUS": ["February-March", "November-December"],
    "BEAN": ["February-March", "April-May"],
    "WHEAT": ["October-November", "March-April"],
    "MUSTARD": ["October-November", "February-March"],
    "COTTON": ["April-May", "October-November"],
    "SUGARCANE": ["February-March", "December-January"],
    "TOMATO": ["June-August", "November-December", "October-December", "April-June"],
    "ONION": ["October-November", "May-June", "April-June"],
}


def month_in_period(month_idx: int, period: str) -> bool:
    """True if month index falls inside a 'Start-End' period (handles year wrap)."""
    period = period.split("(")[0].strip()
    start_name, end_name = [p.strip() for p in period.split("-")]
    start, end = MONTHS.index(start_name), MONTHS.index(end_name)
    if start <= end:
        return start <= month_idx <= end
    return month_idx >= start or month_idx <= end


def check_feasibility(crop: str):
    """Returns (status, message) where status is True / False / None (unknown)."""
    periods = CROP_CALENDAR.get(crop)
    if periods is None:
        return None, "Crop not found in the sowing calendar."
    if "Year-round" in periods:
        return True, "This crop can be grown year-round."
    month_idx = datetime.datetime.now().month - 1
    if any(month_in_period(month_idx, p) for p in periods):
        return True, "Feasible time to grow this crop."
    return False, (
        "Not a feasible time to grow this crop in most regions of India "
        f"(usual sowing: {', '.join(periods)}). Specific conditions may still allow cultivation."
    )


# ---------------------------------------------------------------------------
# Weather forecast
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    # 1) environment variable
    key = os.getenv("OPENWEATHER_API_KEY", "")
    if key:
        return key
    # 2) secrets.toml, but only touch st.secrets if the file exists
    #    (otherwise Streamlit shows a "No secrets files found" notice)
    secret_paths = [
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
        os.path.expanduser("~/.streamlit/secrets.toml"),
    ]
    if any(os.path.exists(p) for p in secret_paths):
        try:
            return st.secrets.get("OPENWEATHER_API_KEY", "")
        except Exception:
            pass
    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(city: str, api_key: str):
    url = "https://api.openweathermap.org/data/2.5/forecast"
    r = requests.get(url, params={"q": f"{city},IN", "appid": api_key, "units": "metric"}, timeout=10)
    data = r.json()
    if str(data.get("cod")) != "200":
        raise ValueError(data.get("message", "Could not fetch forecast"))

    # Group the 3-hourly entries by day
    days = defaultdict(lambda: {"temp": [], "humidity": [], "rain": 0.0, "desc": []})
    for item in data["list"]:
        d = item["dt_txt"].split()[0]
        days[d]["temp"].append(item["main"]["temp"])
        days[d]["humidity"].append(item["main"]["humidity"])
        days[d]["rain"] += item.get("rain", {}).get("3h", 0.0)
        days[d]["desc"].append(item["weather"][0]["description"])

    rows = []
    for d, v in days.items():
        rows.append({
            "Date": d,
            "Avg Temp (°C)": round(sum(v["temp"]) / len(v["temp"]), 1),
            "Avg Humidity (%)": round(sum(v["humidity"]) / len(v["humidity"])),
            "Rain (mm)": round(v["rain"], 1),
            "Conditions": max(set(v["desc"]), key=v["desc"].count),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_water(crop, soil, region, weather) -> float:
    X = pd.DataFrame([[crop, soil, region, weather]], columns=INPUT_COLS)
    X_enc = pd.DataFrame(encoder.transform(X), columns=encoder.get_feature_names_out())
    X_scaled = scaler.transform(X_enc)
    tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        return float(model(tensor).numpy().flatten()[0])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("💧 Smart Irrigation Advisor")
st.caption("Predicts crop water requirement and checks the 5-day forecast to advise on irrigation.")

with st.sidebar:
    st.header("Settings")
    rain_threshold = st.slider("Skip irrigation if daily rain exceeds (mm)", 1.0, 30.0, 5.0, 0.5)
    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("OpenWeatherMap API key", type="password")

with st.form("inputs"):
    city = st.text_input("City (India)", placeholder="e.g. Jalandhar")
    c1, c2 = st.columns(2)
    crop = c1.selectbox("Crop Type", categories["CROP TYPE"])
    soil = c2.selectbox("Soil Type", categories["SOIL TYPE"])
    region = c1.selectbox("Region", categories["REGION"])
    weather = c2.selectbox("Weather Condition", categories["WEATHER CONDITION"])
    submitted = st.form_submit_button("Get recommendation", type="primary", use_container_width=True)

if submitted:
    if not city.strip():
        st.warning("Please enter a city name.")
        st.stop()
    if not api_key:
        st.warning("Add your OpenWeatherMap API key in the sidebar (or in secrets).")
        st.stop()

    # 1. Feasibility
    st.subheader("🌱 Sowing time check")
    status, msg = check_feasibility(str(crop).upper())
    {True: st.success, False: st.warning, None: st.info}[status](msg)

    # 2. Forecast
    st.subheader(f"📅 5-day forecast for {city.strip().title()}")
    try:
        with st.spinner("Fetching forecast..."):
            forecast = fetch_forecast(city.strip(), api_key)
    except Exception as e:
        st.error(f"Weather lookup failed: {e}")
        st.stop()

    st.dataframe(forecast, hide_index=True, use_container_width=True)
    st.bar_chart(forecast.set_index("Date")["Rain (mm)"])
    needs_irrigation = not (forecast["Rain (mm)"] > rain_threshold).any()

    # 3. Prediction
    predicted = predict_water(crop, soil, region, weather)

    st.subheader("✅ Recommendation")
    m1, m2 = st.columns(2)
    m1.metric("Predicted water requirement", f"{predicted:.2f} mm/ha")
    m2.metric("Irrigate?", "Yes" if needs_irrigation else "No")

    if needs_irrigation:
        st.success(f"You SHOULD irrigate your field. Predicted water requirement: {predicted:.2f} mm/ha.")
    else:
        st.info("You do NOT need to irrigate: significant rainfall is expected in the forecast.")
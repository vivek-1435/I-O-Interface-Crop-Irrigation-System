# 💧 Smart Irrigation Advisor

A Streamlit web app that predicts a crop's water requirement with a trained PyTorch model and checks the 5-day weather forecast (OpenWeatherMap) to advise whether to irrigate.

## Features
- Select crop, soil type, region and weather condition
- Sowing-time check against a crop calendar for India
- 5-day forecast table with rainfall chart for any Indian city
- Predicted water requirement (mm/ha)
- Irrigation recommendation (skipped if heavy rain is forecast; threshold adjustable in the sidebar)

## Project Structure
```
CropIrrigationSystem/
├── app.py                         # Streamlit app
├── requirements.txt
├── water_requirement_model.pth    # trained PyTorch weights
├── encoder.pkl                    # fitted OneHotEncoder
└── scaler.pkl                     # fitted StandardScaler
```

## Setup
```bash
pip install -r requirements.txt
```

Get a free API key from [OpenWeatherMap](https://openweathermap.org/api) and provide it in one of three ways:

1. **Secrets file** `.streamlit/secrets.toml`:
   ```toml
   OPENWEATHER_API_KEY = "your_key_here"
   ```
2. **Environment variable:** `export OPENWEATHER_API_KEY="your_key_here"`
3. **Sidebar:** paste it into the app when prompted

## Run
```bash
streamlit run app.py
```

## Model
A feed-forward neural network (512 → 256 → 128 → 64 → 32 → 1) trained on one-hot encoded crop, soil, region and weather-condition features. Outliers in the target were removed with the IQR method, and inputs were standardised with `StandardScaler`.

## Deploy
Push the project to GitHub and deploy on [Streamlit Community Cloud](https://share.streamlit.io). Add `OPENWEATHER_API_KEY` under **Advanced settings → Secrets**.

> Never commit `.streamlit/secrets.toml`. Add it to `.gitignore`.

## Limitations
- The model uses only categorical inputs, so the same inputs always give the same prediction; the forecast only affects the yes/no irrigation advice.
- The crop calendar is a general guide for India and may not suit every local climate.

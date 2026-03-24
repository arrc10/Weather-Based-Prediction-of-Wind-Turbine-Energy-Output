"""
Flask/windApp.py
----------------
AeroPredict – Wind Energy Forecaster backend

Routes:
  GET  /                → serve index.html
  POST /weather         → fetch weather via OpenWeatherMap API (city or pincode)
  POST /predict         → predict energy output via ML model

Run:
    cd Flask
    python windApp.py
Then open http://127.0.0.1:5000
"""

import numpy as np
import pandas as pd
import joblib
import requests
from flask import Flask, request, jsonify, render_template

# ── App & model setup ─────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates', static_folder='static')
model = joblib.load("power_prediction.sav")

# ── ⚠️  Replace with your OpenWeatherMap API key ──────────────────────────────
OWM_API_KEY = "09ee9bdb1fbbc2979bdbdb77fa6d7f97"
OWM_BASE    = "https://api.openweathermap.org/data/2.5/weather"
OWM_GEO_BASE = "http://api.openweathermap.org/geo/1.0/direct"


# ── Helpers ───────────────────────────────────────────────────────────────────
def kelvin_to_celsius(k: float) -> float:
    return round(k - 273.15, 2)


def fetch_weather(query: str) -> dict:
    """
    query → city name  OR  6-digit Indian pincode
    Returns a dict with temp, humidity, pressure, wind_speed or an 'error' key.
    """
    # Determine if query is a pincode (all digits, 6 chars) or city name
    if query.strip().isdigit() and len(query.strip()) == 6:
        params = {"zip": f"{query.strip()},IN", "appid": OWM_API_KEY}
    else:
        params = {"q": query.strip(), "appid": OWM_API_KEY}

    try:
        resp = requests.get(OWM_BASE, params=params, timeout=8)
        data = resp.json()

        if resp.status_code != 200:
            return {"error": data.get("message", "City not found")}

        return {
            "city":        data.get("name", query),
            "temp":        kelvin_to_celsius(data["main"]["temp"]),
            "humidity":    data["main"]["humidity"],
            "pressure":    data["main"]["pressure"],
            "wind_speed":  round(data["wind"]["speed"], 2),
        }
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Check your internet connection."}
    except Exception as e:
        return {"error": str(e)}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    print("Home route called")
    return render_template("index.html")


@app.route("/search_city", methods=["POST"])
def search_city():
    body  = request.get_json(silent=True) or {}
    query = body.get("query", "").strip()

    if not query or len(query) < 2:
        return jsonify([])

    if query.isdigit() and len(query) == 6:
        return jsonify([])

    params = {"q": query, "limit": 5, "appid": OWM_API_KEY}
    try:
        resp = requests.get(OWM_GEO_BASE, params=params, timeout=4)
        if resp.status_code != 200:
            return jsonify([])
        
        data = resp.json()
        suggestions = []
        for item in data:
            name = item.get("name")
            state = item.get("state", "")
            country = item.get("country", "")
            
            label = name
            if state:
                label += f", {state}"
            if country:
                label += f", {country}"
                
            suggestions.append({"name": name, "label": label})
            
        return jsonify(suggestions)
    except Exception:
        return jsonify([])


@app.route("/weather", methods=["POST"])
def weather():
    """
    Expects JSON: { "query": "Nellore" }  or  { "query": "524001" }
    Returns JSON weather data or error.
    """
    body  = request.get_json(silent=True) or {}
    query = body.get("query", "").strip()

    if not query:
        return jsonify({"error": "Please enter a city name or pincode."}), 400

    result = fetch_weather(query)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "theoretical_power": 1500, "wind_speed": 4.12 }
    Returns JSON: { "prediction": 1380.97 }
    """
    body = request.get_json(silent=True) or {}

    try:
        theoretical_power = float(body.get("theoretical_power", 0))
        wind_speed        = float(body.get("wind_speed", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input values."}), 400

    if theoretical_power <= 0 or wind_speed < 0:
        return jsonify({"error": "Values must be positive numbers."}), 400

    features   = pd.DataFrame(
        [[theoretical_power, wind_speed]],
        columns=["Theoretical_Power_Curve(KWh)", "WindSpeed(m/s)"]
    )
    prediction = model.predict(features)[0]
    prediction = max(0.0, round(float(prediction), 2))   # no negative energy

    return jsonify({"prediction": prediction}), 200


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

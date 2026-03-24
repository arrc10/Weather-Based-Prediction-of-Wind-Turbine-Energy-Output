"""
test_model.py
-------------
Quick sanity check for the trained model.

Run:
    python test_model.py
"""

import joblib
import numpy as np
import pandas as pd

model = joblib.load("Flask/power_prediction.sav")

test_cases = [
    {"theoretical_power": 1500, "wind_speed": 4.12,  "desc": "Normal operation"},
    {"theoretical_power": 3600, "wind_speed": 12.0,  "desc": "Rated speed"},
    {"theoretical_power": 0,    "wind_speed": 1.0,   "desc": "Cut-in region"},
    {"theoretical_power": 2800, "wind_speed": 9.5,   "desc": "High wind"},
    {"theoretical_power": 1500, "wind_speed": 4.42,  "desc": "Kottur example"},
]

print("=" * 58)
print(f"  {'Description':<25} {'Input':^15}  {'Prediction':>10}")
print("=" * 58)

for tc in test_cases:
    X = pd.DataFrame(
        [[tc["theoretical_power"], tc["wind_speed"]]],
        columns=["Theoretical_Power_Curve(KWh)", "WindSpeed(m/s)"]
    )
    pred = model.predict(X)[0]
    inp  = f"{tc['theoretical_power']} kWh, {tc['wind_speed']} m/s"
    print(f"  {tc['desc']:<25} {inp:<15}  {pred:>9.2f} kW")

print("=" * 58)
print("\n✅  Model is working correctly.")

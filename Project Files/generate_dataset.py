"""
generate_dataset.py
-------------------
Generates a synthetic wind turbine SCADA dataset (T1.csv) that mimics
real wind farm data from Kaggle.

Run this ONCE before training:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ── Time axis: 2 years of 10-minute readings ──────────────────────────────────
n = 105_120  # 2 years × 365.25 days × 144 samples/day ≈ 105 120
start = datetime(2016, 1, 1, 0, 0, 0)
timestamps = [start + timedelta(minutes=10 * i) for i in range(n)]

# ── Wind speed: Weibull-distributed, seasonal variation ───────────────────────
day_of_year = np.array([t.timetuple().tm_yday for t in timestamps])
hour_of_day = np.array([t.hour for t in timestamps])

seasonal_factor = 1 + 0.3 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
diurnal_factor  = 1 + 0.1 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

base_shape, base_scale = 2.0, 8.0          # Weibull params
wind_speed = (np.random.weibull(base_shape, n) * base_scale
              * seasonal_factor * diurnal_factor)
wind_speed = np.clip(wind_speed, 0, 25)    # physical cut-out at 25 m/s

# ── Wind direction: somewhat uniform with slight westerly bias ────────────────
wind_direction = np.random.vonmises(mu=np.pi, kappa=0.5, size=n)
wind_direction = np.degrees(wind_direction) % 360

# ── Theoretical power curve (kWh): standard cubic S-curve ───────────────────
def theoretical_power(ws, rated=3600.0, v_in=3.0, v_r=12.0, v_out=25.0):
    p = np.zeros_like(ws)
    mask = (ws >= v_in) & (ws < v_r)
    p[mask] = rated * ((ws[mask] - v_in) / (v_r - v_in)) ** 3
    p[ws >= v_r] = rated
    p[ws >= v_out] = 0.0
    return p

theoretical = theoretical_power(wind_speed)

# ── Actual active power: theory × efficiency + noise ─────────────────────────
efficiency = np.random.normal(0.92, 0.04, n)
efficiency = np.clip(efficiency, 0.5, 1.0)
noise = np.random.normal(0, 30, n)
active_power = theoretical * efficiency + noise
active_power = np.clip(active_power, 0, 3600)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "Date/Time":                        [t.strftime("%d %m %Y %H:%M") for t in timestamps],
    "LV ActivePower (kW)":              active_power.round(4),
    "Wind Speed (m/s)":                 wind_speed.round(4),
    "Theoretical_Power_Curve (KWh)":    theoretical.round(4),
    "Wind Direction (°)":               wind_direction.round(4),
})

out_path = "data/T1.csv"
df.to_csv(out_path, index=False)
print(f"✅  Dataset saved → {out_path}  ({len(df):,} rows)")
print(df.head())

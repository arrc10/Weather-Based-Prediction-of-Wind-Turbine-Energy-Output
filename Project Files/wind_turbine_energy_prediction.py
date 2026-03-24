"""
wind_turbine_energy_prediction.py
----------------------------------
Full ML pipeline:
  1. Load & rename dataset
  2. EDA & visualisation (heatmap, pair plot, scatter)
  3. Feature engineering
  4. Train / test split (80 / 20)
  5. Random Forest Regressor
  6. Evaluate (MAE, R² score)
  7. Save model → Flask/power_prediction.sav

Run:
    python wind_turbine_energy_prediction.py
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, r2_score

# ── 2. Load dataset ───────────────────────────────────────────────────────────
path = "data/T1.csv"
df   = pd.read_csv(path)

df.rename(columns={
    "Date/Time":                      "Time",
    "LV ActivePower (kW)":            "ActivePower(kW)",
    "Wind Speed (m/s)":               "WindSpeed(m/s)",
    "Theoretical_Power_Curve (KWh)":  "Theoretical_Power_Curve(KWh)",
    "Wind Direction (°)":             "Wind_Direction",
}, inplace=True)

df["Time"] = pd.to_datetime(df["Time"], format="%d %m %Y %H:%M", errors="coerce")

print("── Dataset loaded ──────────────────────────────────────")
print(df.head())
print(f"\nShape : {df.shape}")
print(f"NaN   :\n{df.isnull().sum()}\n")

# ── 3. Handle missing values ──────────────────────────────────────────────────
df.dropna(inplace=True)
print(f"Shape after dropna : {df.shape}\n")

# ── 4. EDA / Visualisation ────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# 4a. Correlation heatmap
numeric_cols = ["ActivePower(kW)", "WindSpeed(m/s)",
                "Theoretical_Power_Curve(KWh)", "Wind_Direction"]
corr = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
ax = sns.heatmap(corr, vmin=-1, vmax=1, annot=True,
                 cmap="YlGnBu", fmt=".2f", linewidths=0.5)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150)
plt.close()
print("✅  plots/correlation_heatmap.png saved")

# 4b. Wind speed vs Active power scatter
plt.figure(figsize=(10, 6))
plt.scatter(df["WindSpeed(m/s)"], df["ActivePower(kW)"],
            alpha=0.3, s=2, c="#4a7c59", edgecolors="none")
plt.xlabel("Wind Speed (m/s)", fontsize=12)
plt.ylabel("Active Power (kW)", fontsize=12)
plt.title("Wind Speed vs Active Power Output", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/wind_vs_power.png", dpi=150)
plt.close()
print("✅  plots/wind_vs_power.png saved")

# 4c. Pair plot (sample 2000 rows for speed)
sample = df[numeric_cols].sample(2000, random_state=42)
pp = sns.pairplot(sample, plot_kws={"alpha": 0.3, "s": 5, "color": "#4a7c59"})
pp.fig.suptitle("Pair Plot – Wind Turbine Features", y=1.02, fontsize=13)
pp.savefig("plots/pairplot.png", dpi=100)
plt.close()
print("✅  plots/pairplot.png saved\n")

# ── 5. Feature selection ──────────────────────────────────────────────────────
# Heatmap shows Wind_Direction has near-zero correlation → drop it
df.drop(["Wind_Direction", "Time"], axis=1, inplace=True, errors="ignore")

X = df[["Theoretical_Power_Curve(KWh)", "WindSpeed(m/s)"]]
y = df["ActivePower(kW)"]

print(f"Features : {list(X.columns)}")
print(f"Target   : ActivePower(kW)\n")

# ── 6. Train / test split (80 / 20) ──────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=0
)
print(f"Train size : {len(X_train):,}   |   Val size : {len(X_val):,}\n")

# ── 7. Model – Random Forest Regressor ───────────────────────────────────────
print("Training Random Forest Regressor …")
forest_model = RandomForestRegressor(
    n_estimators=750,
    max_depth=4,
    max_leaf_nodes=500,
    random_state=1,
    n_jobs=-1,
)
forest_model.fit(X_train, y_train)
print("Training complete ✅\n")

# ── 8. Evaluation ─────────────────────────────────────────────────────────────
power_preds = forest_model.predict(X_val)

mae  = mean_absolute_error(y_val, power_preds)
r2   = r2_score(y_val, power_preds)

print("── Model Evaluation ───────────────────────────────────")
print(f"  MAE    : {mae:.4f} kW")
print(f"  R²     : {r2:.4f}")
print()

# Prediction vs actual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_val[:3000], power_preds[:3000],
            alpha=0.3, s=4, c="#4a7c59", edgecolors="none")
lims = [0, y_val.max()]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual Power (kW)", fontsize=12)
plt.ylabel("Predicted Power (kW)", fontsize=12)
plt.title(f"Actual vs Predicted  |  R² = {r2:.4f}", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png", dpi=150)
plt.close()
print("✅  plots/actual_vs_predicted.png saved\n")

# ── 9. Save model ─────────────────────────────────────────────────────────────
os.makedirs("Flask", exist_ok=True)
model_path = "Flask/power_prediction.sav"
joblib.dump(forest_model, model_path)
print(f"✅  Model saved → {model_path}\n")
print("Run `python Flask/windApp.py` to start the web app.")

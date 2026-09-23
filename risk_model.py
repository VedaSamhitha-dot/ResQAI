import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


np.random.seed(42)

models = {}
all_data = []


# ============================================================
# FLOOD MODEL
# ============================================================

n_samples = 600

rainfall = np.random.randint(20, 301, n_samples)
water_level = np.random.uniform(1, 10, n_samples)
wind_speed = np.random.randint(10, 151, n_samples)
population_density = np.random.randint(100, 1501, n_samples)
previous_disasters = np.random.randint(0, 8, n_samples)

score = (
    rainfall / 300 * 30
    + water_level / 10 * 25
    + wind_speed / 150 * 20
    + population_density / 1500 * 15
    + previous_disasters / 7 * 10
)

risk = np.where(
    score >= 60,
    "High",
    np.where(score >= 35, "Medium", "Low")
)

flood_data = pd.DataFrame({
    "disaster_type": "Flood",
    "rainfall": rainfall,
    "water_level": np.round(water_level, 2),
    "wind_speed": wind_speed,
    "atmospheric_pressure": np.nan,
    "magnitude": np.nan,
    "depth": np.nan,
    "seismic_intensity": np.nan,
    "population_density": population_density,
    "previous_disasters": previous_disasters,
    "risk": risk
})

X = flood_data[
    [
        "rainfall",
        "water_level",
        "wind_speed",
        "population_density",
        "previous_disasters"
    ]
]

y = flood_data["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

flood_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

flood_model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    flood_model.predict(X_test)
)

print(f"Flood Model Accuracy: {accuracy * 100:.2f}%")

models["Flood"] = flood_model
all_data.append(flood_data)


# ============================================================
# CYCLONE MODEL
# ============================================================

wind_speed = np.random.randint(30, 251, n_samples)
rainfall = np.random.randint(30, 501, n_samples)
atmospheric_pressure = np.random.randint(900, 1021, n_samples)
population_density = np.random.randint(100, 1501, n_samples)
previous_disasters = np.random.randint(0, 8, n_samples)

score = (
    wind_speed / 250 * 35
    + rainfall / 500 * 25
    + (1020 - atmospheric_pressure) / 120 * 20
    + population_density / 1500 * 10
    + previous_disasters / 7 * 10
)

risk = np.where(
    score >= 60,
    "High",
    np.where(score >= 35, "Medium", "Low")
)

cyclone_data = pd.DataFrame({
    "disaster_type": "Cyclone",
    "rainfall": rainfall,
    "water_level": np.nan,
    "wind_speed": wind_speed,
    "atmospheric_pressure": atmospheric_pressure,
    "magnitude": np.nan,
    "depth": np.nan,
    "seismic_intensity": np.nan,
    "population_density": population_density,
    "previous_disasters": previous_disasters,
    "risk": risk
})

X = cyclone_data[
    [
        "wind_speed",
        "rainfall",
        "atmospheric_pressure",
        "population_density",
        "previous_disasters"
    ]
]

y = cyclone_data["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

cyclone_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

cyclone_model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    cyclone_model.predict(X_test)
)

print(f"Cyclone Model Accuracy: {accuracy * 100:.2f}%")

models["Cyclone"] = cyclone_model
all_data.append(cyclone_data)


# ============================================================
# EARTHQUAKE MODEL
# ============================================================

magnitude = np.random.uniform(2.0, 8.0, n_samples)
depth = np.random.uniform(1, 100, n_samples)
seismic_intensity = np.random.uniform(1, 10, n_samples)
population_density = np.random.randint(100, 1501, n_samples)
previous_disasters = np.random.randint(0, 8, n_samples)

score = (
    magnitude / 8 * 35
    + (100 - depth) / 100 * 15
    + seismic_intensity / 10 * 25
    + population_density / 1500 * 15
    + previous_disasters / 7 * 10
)

risk = np.where(
    score >= 60,
    "High",
    np.where(score >= 35, "Medium", "Low")
)

earthquake_data = pd.DataFrame({
    "disaster_type": "Earthquake",
    "rainfall": np.nan,
    "water_level": np.nan,
    "wind_speed": np.nan,
    "atmospheric_pressure": np.nan,
    "magnitude": np.round(magnitude, 2),
    "depth": np.round(depth, 2),
    "seismic_intensity": np.round(seismic_intensity, 2),
    "population_density": population_density,
    "previous_disasters": previous_disasters,
    "risk": risk
})

X = earthquake_data[
    [
        "magnitude",
        "depth",
        "seismic_intensity",
        "population_density",
        "previous_disasters"
    ]
]

y = earthquake_data["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

earthquake_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

earthquake_model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    earthquake_model.predict(X_test)
)

print(f"Earthquake Model Accuracy: {accuracy * 100:.2f}%")

models["Earthquake"] = earthquake_model
all_data.append(earthquake_data)


# ============================================================
# SAVE DATASET
# ============================================================

data = pd.concat(
    all_data,
    ignore_index=True
)

data.to_csv(
    "data/disaster_data.csv",
    index=False
)

print("\nMulti-disaster dataset created successfully!")
print(f"Total records: {len(data)}")


# ============================================================
# SAVE ALL MODELS
# ============================================================

joblib.dump(
    models,
    "model/risk_model.pkl"
)

print("\nMulti-disaster models saved successfully!")
print("File: model/risk_model.pkl")


# ============================================================
# TEST SAMPLE PREDICTIONS
# ============================================================

print("\nSample Predictions")
print("------------------")


# Flood
flood_sample = pd.DataFrame({
    "rainfall": [180],
    "water_level": [7],
    "wind_speed": [80],
    "population_density": [800],
    "previous_disasters": [3]
})

prediction = flood_model.predict(flood_sample)[0]

print("Flood:", prediction)


# Cyclone
cyclone_sample = pd.DataFrame({
    "wind_speed": [140],
    "rainfall": [250],
    "atmospheric_pressure": [970],
    "population_density": [800],
    "previous_disasters": [3]
})

prediction = cyclone_model.predict(cyclone_sample)[0]

print("Cyclone:", prediction)


# Earthquake
earthquake_sample = pd.DataFrame({
    "magnitude": [6.5],
    "depth": [10],
    "seismic_intensity": [7],
    "population_density": [800],
    "previous_disasters": [3]
})

prediction = earthquake_model.predict(earthquake_sample)[0]

print("Earthquake:", prediction)
# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preprocessing: Load the weather dataset and handle missing values by replacing NaN entries with the mean of their respective columns.
2. Feature Selection: Define the input features (X) using environmental metrics and isolate the target variables (y) for pollution, temperature, and energy.M
3. Model Training: Split the data into training and testing sets, then fit a Decision Tree Regressor with a maximum depth of 5 for each target.
4. Evaluation: Generate predictions on the test set and calculate performance metrics, specifically RMSE and R^2 Score, to assess model accuracy.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Athul Krishna A V
RegisterNumber:  212225240017
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("weather.csv")


X = df[['hum', 'pressure', 'wind_speed', 'illumination', 'co2']]

X = X.fillna(X.mean())

# =========================
# 1. Pollution Prediction (PM2.5)
# =========================
y_pollution = df['pm2_5'].fillna(df['pm2_5'].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_pollution, test_size=0.2, random_state=42
)

pollution_model = DecisionTreeRegressor(random_state=42, max_depth=5)
pollution_model.fit(X_train, y_train)

pollution_pred = pollution_model.predict(X_test)

rmse_pollution = np.sqrt(mean_squared_error(y_test, pollution_pred))
r2_pollution = r2_score(y_test, pollution_pred)
accuracy_pollution = r2_pollution * 100

print("🏭 Pollution Prediction (PM2.5)")
print("Accuracy (%):", accuracy_pollution)
print("R2 Score:", r2_pollution)
print("RMSE:", rmse_pollution)


# =========================
# 2. Temperature Prediction
# =========================
y_temp = df['tem'].fillna(df['tem'].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_temp, test_size=0.2, random_state=42
)

temp_model = DecisionTreeRegressor(random_state=42, max_depth=5)
temp_model.fit(X_train, y_train)

temp_pred = temp_model.predict(X_test)

rmse_temp = np.sqrt(mean_squared_error(y_test, temp_pred))
r2_temp = r2_score(y_test, temp_pred)
accuracy_temp = r2_temp * 100
print("\n🌡️ Temperature Prediction")
print("Accuracy (%):", accuracy_temp)
print("RMSE:", rmse_temp)
print("R2 Score:", r2_temp)


# =========================
# 3. Energy Prediction (TSR)
# =========================
y_energy = df['tsr'].fillna(df['tsr'].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y_energy, test_size=0.2, random_state=42
)

energy_model = DecisionTreeRegressor(random_state=42, max_depth=5)
energy_model.fit(X_train, y_train)

energy_pred = energy_model.predict(X_test)

rmse_energy = np.sqrt(mean_squared_error(y_test, energy_pred))
r2_energy = r2_score(y_test, energy_pred)
accuracy_energy = r2_energy * 100
print("\n⚡ Energy Prediction (TSR)")
print("Accuracy (%):", accuracy_energy)


print("RMSE:", rmse_energy)
print("R2 Score:", r2_energy)
```

## Output:
<img width="334" height="304" alt="image" src="https://github.com/user-attachments/assets/0924eb93-37d1-4078-97e3-923fc768401c" />


## Result:
Thus the program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm is written and verified using python programming.

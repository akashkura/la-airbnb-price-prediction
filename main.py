import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =============================
# 1. Load and Clean the Dataset
# =============================
data = pd.read_csv('data/airbnb_listings.csv', encoding='utf-8')
print("Dataset loaded successfully!")

# Convert critical columns to numeric
data['price'] = data['price'].str.replace(r'[\$,]', '', regex=True).astype(float)
data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')

# Drop NaNs and invalid values
data.dropna(subset=['price', 'longitude', 'latitude'], inplace=True)
data = data[data['price'] > 0]

# Remove outliers
upper_limit = data['price'].quantile(0.99)
data = data[data['price'] <= upper_limit]

# Feature Engineering
data['amenity_count'] = data['amenities'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
data = pd.get_dummies(data, columns=['room_type'], drop_first=True)

# Train-test split
X = data[['longitude', 'latitude', 'amenity_count']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# Random Forest
# ======================
print("Starting Random Forest training...")
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully!")

print("Predicting and evaluating Random Forest model...")
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred_rf))
print("R² Score:", r2_score(y_test, y_pred_rf))

# ======================
# Linear Regression
# ======================
print("\nStarting Linear Regression training...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print("Linear Regression model trained successfully!")

print("Predicting and evaluating Linear Regression model...")
y_pred_lr = linear_model.predict(X_test)
print("\nLinear Regression Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred_lr))
print("R² Score:", r2_score(y_test, y_pred_lr))




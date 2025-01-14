import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =============================
# 1. Load and Clean the Dataset
# =============================

# Load dataset
data = pd.read_csv('data/airbnb_listings.csv', encoding='utf-8')
print("Dataset loaded successfully!")

# Inspect initial dataset size and columns
print(f"Initial dataset size: {len(data)} rows")
print("Columns in dataset:", data.columns)

# Convert critical columns to numeric
data['price'] = data['price'].str.replace('[\$,]', '', regex=True).astype(float)
data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')

# Missing values before cleaning
print("\nMissing values before cleaning:")
print(data[['price', 'longitude', 'latitude']].isnull().sum())

# Drop rows with missing or invalid values
data.dropna(subset=['price', 'longitude', 'latitude'], inplace=True)
print(f"Dataset size after removing NaNs: {len(data)} rows")

# Remove rows where price is less than or equal to 0
data = data[data['price'] > 0]
print(f"Dataset size after removing prices <= 0: {len(data)} rows")

# ===========================
# 2. Handle Outliers
# ===========================

# Remove listings with prices in the top 1% to reduce skewness
upper_limit = data['price'].quantile(0.99)  # Adjust threshold if needed
data = data[data['price'] <= upper_limit]
print(f"Dataset size after outlier removal: {len(data)} rows")

# ===========================
# 3. Exploratory Data Analysis
# ===========================

# Summary statistics for 'price'
print("\nSummary Statistics for 'price':")
print(data['price'].describe())

# Histogram for Price Distribution
plt.hist(data['price'], bins=50, edgecolor='black')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Geographic Scatter Plot
plt.scatter(data['longitude'], data['latitude'], c=data['price'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Price')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Price Distribution')
plt.show()

# =========================================
# 4. Feature Engineering for ML Prediction
# =========================================

# Create 'amenity_count' feature
data['amenity_count'] = data['amenities'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

# One-hot encode 'room_type' feature
data = pd.get_dummies(data, columns=['room_type'], drop_first=True)

# =============================
# 5. Prepare Data for Modeling
# =============================

# Select features and target
X = data[['longitude', 'latitude', 'amenity_count']]  # Add more features if desired
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================
# 6. Random Forest (Active)
# ======================

# Train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred_rf))
print("R² Score:", r2_score(y_test, y_pred_rf))

# ===========================
# 7. Visualize Results
# ===========================

# Random Forest Visualization
plt.scatter(y_test, y_pred_rf, alpha=0.6)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices (Random Forest)')
plt.show()

# ========================
# 8. Additional Visuals
# ========================

# Histogram for Price Distribution
plt.hist(data['price'], bins=20, edgecolor='black')  # Fewer bins
plt.title('Price Distribution (After Removing Outliers)')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.xlim(0, data['price'].max())  # Adjust x-axis range
plt.show()

# Geographic Scatter Plot
plt.scatter(data['longitude'], data['latitude'], c=data['price'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Price')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Price Distribution (After Removing Outliers)')
plt.show()

# Inspect cleaned data types
print("\nData Types for 'longitude', 'latitude', and 'price':")
print(data[['longitude', 'latitude', 'price']].dtypes)

# ======================
# Linear Regression
# ======================

from sklearn.linear_model import LinearRegression

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict and evaluate Linear Regression
y_pred_lr = linear_model.predict(X_test)
print("\nLinear Regression Performance:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred_lr))
print("R² Score:", r2_score(y_test, y_pred_lr))

# Visualize Results for Linear Regression
plt.scatter(y_test, y_pred_lr, alpha=0.6)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices (Linear Regression)')
plt.show()


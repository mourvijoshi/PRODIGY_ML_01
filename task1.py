import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt


# Load Cleaned Datasets

df_train = pd.read_csv('train_cleaned.csv')
df_test = pd.read_csv('test_cleaned.csv')

# Feature Selection

# Define Independent (X) and Dependent (y) Variables for Training
X_train = df_train[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr']]
y_train = df_train['SalePrice']

# Define Independent Variables for Testing
X_test = df_test[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr']]


# Handle Missing Values in Test Data

# Check for and handle missing values in X_test
if X_test.isnull().sum().any():
    print("Handling missing values in X_test...")
    X_test = X_test.fillna(X_test.median())


poly = PolynomialFeatures(degree=2, include_bias=False)  # Add interaction and squared terms
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)


reg = LinearRegression()
reg.fit(X_train_scaled, y_train)


y_train_pred = reg.predict(X_train_scaled)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print("Training Data Performance:")
print(f"Mean Squared Error (MSE): {mse_train:.4f}")
print(f"R-squared Score: {r2_train:.4f}")


y_test_pred = reg.predict(X_test_scaled)

# Reverse Log-Transformation for Predictions
y_test_pred_original = np.round(np.expm1(y_test_pred), 0)  # Round off to the nearest integer


if 'Id' in df_test.columns:
    df_output = df_test[['Id']].copy()
else:
    df_output = pd.DataFrame()

# Add predicted prices
df_output['SalePrice'] = y_test_pred_original

# Save the output file
df_output.to_csv('sample_submission.csv', index=False)

print("Predictions saved to 'sample_submission.csv'.")



# Scatter plot: Actual vs Predicted on Training Data
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_train_pred, alpha=0.7)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linewidth=2)  # Perfect prediction line
plt.xlabel("Actual SalePrice (log-transformed)")
plt.ylabel("Predicted SalePrice (log-transformed)")
plt.title("Actual vs Predicted SalePrice (Training Data)")
plt.show()

# Residual Plot: Errors on Training Data
residuals = y_train - y_train_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_train_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel("Predicted SalePrice (log-transformed)")
plt.ylabel("Residuals (log-transformed)")
plt.title("Residual Plot (Training Data)")
plt.show()

# Histogram: Distribution of Residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, alpha=0.7, color='blue')
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel("Residuals (log-transformed)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Training Data)")
plt.show()
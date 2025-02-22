# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ==========================
# DATA LOADING & PREPROCESSING
# ==========================

# Define file path (ensure this is correctly set to your local directory)
file_path = "World Population Data.csv" # Enter Your Directory !

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(file_path)

# Ensure column names are stripped of spaces for consistency
df.columns = df.columns.str.strip()

# Convert population column to numerical format (removing commas and spaces)
df['Population (2024)'] = df['Population (2024)'].replace({',': '', ' ': ''}, regex=True).astype(float)

# Convert land area column to numerical format (removing commas)
df['Land Area (Km²)'] = df['Land Area (Km²)'].replace({',': ''}, regex=True).astype(float)

# Calculate population density (Population per Km²)
df['Density'] = df['Population (2024)'] / df['Land Area (Km²)']

# ==========================
# DATA EXPLORATION & VISUALIZATION
# ==========================

# Compute correlation matrix for numerical variables
correlation = df[['Population (2024)', 'Fert. Rate', 'Med. Age', 'Density']].corr()

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Key Features')
plt.show()

# Extract the top 10 most populated countries
top_10 = df.nlargest(10, 'Population (2024)')

# FIX: Adjust Seaborn bar plot to remove warnings
plt.figure(figsize=(10, 6))
sns.barplot(x=top_10['Population (2024)'], y=top_10['Country'],
            hue=top_10['Country'], palette='viridis', dodge=False, legend=False)
plt.xlabel('Population (2024)')
plt.ylabel('Country')
plt.title('Top 10 Most Populated Countries in 2024')
plt.show()

# ==========================
# FEATURE ENGINEERING
# ==========================

# Convert Urban Population percentage to decimal (handling missing values)
df['Urban Pop %'] = df['Urban Pop %'].replace({'%': '', 'N.A.': None}, regex=True).astype(float) / 100

# Convert Net Change column to numerical format (removing commas and spaces)
df['Net Change'] = df['Net Change'].replace({',': '', ' ': ''}, regex=True).astype(float)

# Log-transform Population data to handle large variance (avoiding log(0) errors)
df['Log Population'] = np.log1p(df['Population (2024)'])

# ==========================
# MODEL PREPARATION
# ==========================

# Select relevant features for prediction
X = df[['Fert. Rate', 'Med. Age', 'Density', 'Urban Pop %', 'Net Change']].dropna()

# Target variable (log-transformed population)
y = df.loc[X.index, 'Log Population']

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# MODEL TRAINING
# ==========================

# Initialize models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ==========================
# MODEL PREDICTIONS & EVALUATION
# ==========================

# Generate predictions for test set
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Convert predictions back from log scale to original population scale
y_pred_linear = np.expm1(y_pred_linear)
y_pred_rf = np.expm1(y_pred_rf)
y_test_exp = np.expm1(y_test)

# Compute evaluation metrics for Linear Regression
mae_linear = mean_absolute_error(y_test_exp, y_pred_linear)
mse_linear = mean_squared_error(y_test_exp, y_pred_linear)
r2_linear = r2_score(y_test_exp, y_pred_linear)

# Compute evaluation metrics for Random Forest
mae_rf = mean_absolute_error(y_test_exp, y_pred_rf)
mse_rf = mean_squared_error(y_test_exp, y_pred_rf)
r2_rf = r2_score(y_test_exp, y_pred_rf)

# Display model performance
print("\n--- Updated Model Performance ---")
print(f'Linear Regression - MAE: {mae_linear:.2f}, MSE: {mse_linear:.2f}, R²: {r2_linear:.2f}')
print(f'Random Forest - MAE: {mae_rf:.2f}, MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}')

# ==========================
# VISUALIZATION: ACTUAL VS PREDICTED POPULATION
# ==========================

# Scatter plot comparing actual vs. predicted population values
plt.figure(figsize=(8, 6))
plt.scatter(y_test_exp, y_pred_linear, color='blue', alpha=0.6, label='Linear Regression Predictions')
plt.scatter(y_test_exp, y_pred_rf, color='green', alpha=0.6, label='Random Forest Predictions')

# Reference line (perfect prediction)
plt.plot([min(y_test_exp), max(y_test_exp)], [min(y_test_exp), max(y_test_exp)],
         color='red', linestyle='--', label='Ideal Prediction Line')

plt.xlabel('Actual Population (2024)')
plt.ylabel('Predicted Population (2024)')
plt.title('Actual vs Predicted Population (2024)')
plt.legend()
plt.show()

# ==========================
# INTERACTIVE VISUALIZATION WITH PLOTLY
# ==========================

# Scatter plot with Plotly: Fertility Rate vs. Population, colored by Median Age
fig = px.scatter(df, x='Fert. Rate', y='Population (2024)', color='Med. Age',
                 hover_data=['Country'], title='Fertility Rate vs Population')
fig.show()

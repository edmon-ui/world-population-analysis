# World Population Analysis

This project analyzes world population data, explores key demographic factors, and builds machine learning models to predict population.

## Workflow

### 1. Importing Necessary Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
```

Explanation:
pandas (pd): Handles structured (tabular) data.
matplotlib (plt): Basic plotting library.
seaborn (sns): Statistical visualization library.
plotly (px): Interactive visualizations.
sklearn (scikit-learn): Machine learning tools.
numpy (np): Numerical operations.

### 2. Loading & Preprocessing the Data
```python
# Define file path (ensure this is correctly set to your local directory)
file_path = "World Population Data.csv"  # Enter Your Directory !

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

# Convert Urban Population percentage to decimal (handling missing values)
df['Urban Pop %'] = df['Urban Pop %'].replace({'%': '', 'N.A.': None}, regex=True).astype(float) / 100

# Convert Net Change column to numerical format (removing commas and spaces)
df['Net Change'] = df['Net Change'].replace({',': '', ' ': ''}, regex=True).astype(float)

# Log-transform Population data to handle large variance (avoiding log(0) errors)
df['Log Population'] = np.log1p(df['Population (2024)'])
```

Explanation:
Reads the World Population dataset into a DataFrame (df).
Cleans the column names by removing spaces to prevent errors.
Converts Population (2024) and Land Area (Km²) columns from string format (with commas and spaces) to numerical format.
Computes Population Density using:
Density = Population(2024) / Land Area (Km²)
### 3. Data Exploration & Visualization
```python
correlation = df[['Population (2024)', 'Fert. Rate', 'Med. Age', 'Density']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Key Features')
plt.show()
``` 
Explanation:
Computes the correlation matrix between key numerical features.
Heatmap visualizes relationships between:
Population (2024)
Fertility Rate (number of births per woman)
Median Age (age distribution of population)
Density (people per unit area)
Interpretation: Darker or brighter colors indicate strong correlations.

```python
top_10 = df.nlargest(10, 'Population (2024)')

plt.figure(figsize=(10, 6))
sns.barplot(x=top_10['Population (2024)'], y=top_10['Country'],
            hue=top_10['Country'], palette='viridis', dodge=False, legend=False)
plt.xlabel('Population (2024)')
plt.ylabel('Country')
plt.title('Top 10 Most Populated Countries in 2024')
plt.show()
```
Explanation:
Extracts the top 10 most populated countries.
Uses Seaborn's bar plot to visualize them.
The hue parameter assigns each country a unique color

### 4. Feature Engineering
```python
df['Urban Pop %'] = df['Urban Pop %'].replace({'%': '', 'N.A.': None}, regex=True).astype(float) / 100

df['Net Change'] = df['Net Change'].replace({',': '', ' ': ''}, regex=True).astype(float)

df['Log Population'] = np.log1p(df['Population (2024)'])
```
Explanation:
Cleans and converts:
Urban Populatiopn % from a percentage string to a decimal.
Net Change (population growth/decline) into a numeric format.
Log transformation of popoulation to normalize large values:
Log Population = log(Population + 1)
This avoids errors with zero values and helps with model training.

### 5. Model Preparation
```python
X = df[['Fert. Rate', 'Med. Age', 'Density', 'Urban Pop %', 'Net Change']].dropna()
y = df.loc[X.index, 'Log Population']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Explanation:
Selects independent variables (X) and target variable (y).
Drops rows with missing values.
Splits dataset into:
80% training (X_train, y_train)
20% testing (X_test, y_test)
random_state=42 ensures reproducibility.

### 6. Model Training
```python
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
```
Explanation:
Linear Regression: A simple model assuming a straight-line relationship.
Random Forest Regressor: A more complex, ensemble model using multiple decision trees.
fit(X_train, y_train): Trains both models using the training data.

### 7. Model Predictions & Evaluation
```python
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

y_pred_linear = np.expm1(y_pred_linear)
y_pred_rf = np.expm1(y_pred_rf)
y_test_exp = np.expm1(y_test)
```
Explanation:
Generates predictions using both models.
Converts predictions back from the log scale to actual population values using:
exp^predicter - 1

```python
mae_linear = mean_absolute_error(y_test_exp, y_pred_linear)
mse_linear = mean_squared_error(y_test_exp, y_pred_linear)
r2_linear = r2_score(y_test_exp, y_pred_linear)

mae_rf = mean_absolute_error(y_test_exp, y_pred_rf)
mse_rf = mean_squared_error(y_test_exp, y_pred_rf)
r2_rf = r2_score(y_test_exp, y_pred_rf)

print("\n--- Updated Model Performance ---")
print(f'Linear Regression - MAE: {mae_linear:.2f}, MSE: {mse_linear:.2f}, R²: {r2_linear:.2f}')
print(f'Random Forest - MAE: {mae_rf:.2f}, MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}')
```
Explanation:
Computes:
MAE (Mean Absolute Error) → Measures average prediction error.
MSE (Mean Squared Error) → Penalizes large errors.
R² Score → Measures model performance (closer to 1 is better).
Prints model performance.

### 8. Visualization: Actual vs Predicted
```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test_exp, y_pred_linear, color='blue', alpha=0.6, label='Linear Regression Predictions')
plt.scatter(y_test_exp, y_pred_rf, color='green', alpha=0.6, label='Random Forest Predictions')

plt.plot([min(y_test_exp), max(y_test_exp)], [min(y_test_exp), max(y_test_exp)],
         color='red', linestyle='--', label='Ideal Prediction Line')

plt.xlabel('Actual Population (2024)')
plt.ylabel('Predicted Population (2024)')
plt.title('Actual vs Predicted Population (2024)')
plt.legend()
plt.show()
```
Explanation:
Scatter plot compares actual vs. predicted population.
Red dashed line represents perfect predictions.
If points are close to this line, the model is accurate.

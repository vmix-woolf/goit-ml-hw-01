#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso

#%%
california_housing = fetch_california_housing(as_frame=True)
df = california_housing['frame']
df['MedHouseValue'] = california_housing['target']

print(df.head())
print(df.info())
print(df.describe())

#%%
sns.set_theme()
target = df.pop('MedHouseVal')
melted = pd.concat([df, target], axis=1).melt()
g = sns.FacetGrid(
    melted,
    col='variable',
    col_wrap=3,
    sharex=False,
    sharey=False)
g.map(sns.histplot, 'value')
g.set_titles(col_template='{col_name}')
g.tight_layout()

#%%
columns_to_check = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
df[columns_to_check].describe()
z_scores = df[columns_to_check].apply(zscore)

outliers_mask = (np.abs(z_scores) > 3)
rows_to_remove = outliers_mask.any(axis=1)
print(f"Кількість викидів у кожній колонці: {df.shape[0]}")

df_cleaned = df[~rows_to_remove].reset_index(drop=True)
print(f"Видалено {rows_to_remove.sum()} рядків з аномальними значеннями.")
print(f"Новий розмір датасету: {df_cleaned.shape}")

plt.figure(figsize=(10, 8))
correlation_matrix = df_cleaned.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Кореляційна матриця ознак")
plt.show()

df_final = df_cleaned.drop(columns=['AveRooms'])
correlation_matrix = df_final.corr() # Кореляційна матриця

#%%
X = df_final.drop(columns=['MedHouseValue'])
y = df_final['MedHouseValue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # train і test

# Перевіримо розміри
print("Розмір навчальної вибірки:", X_train.shape)
print("Розмір тестової вибірки:", X_test.shape)

#%%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#%%
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

#%%
# Візуалізація результатів
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel("Реальні значення (y_test)")
plt.ylabel("Прогнозовані значення (y_pred)")
plt.title("Факт vs Прогноз (Linear Regression)")
plt.grid(True)
plt.show()

#%%
lasso_model = Lasso()
parameters = {'alpha': [0.1, 1, 10, 100, 1000]}

grid_search_lasso = GridSearchCV(lasso_model, parameters, cv=5, scoring='neg_mean_squared_error')
grid_search_lasso.fit(X_train_scaled, y_train)

best_alpha_lasso = grid_search_lasso.best_params_['alpha']
print("Кращий параметр alpha для Lasso:", best_alpha_lasso)

lasso_optimized = grid_search_lasso.best_estimator_
y_pred_lasso = lasso_optimized.predict(X_test_scaled)

mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"MAE (Lasso): {mae_lasso:.4f}")
print(f"MSE (Lasso): {mse_lasso:.4f}")
print(f"RMSE (Lasso): {rmse_lasso:.4f}")
print(f"R² (Lasso): {r2_lasso:.4f}")
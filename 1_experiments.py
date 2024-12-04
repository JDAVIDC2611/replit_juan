import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

# Cargar datos
data = pd.read_csv('/content/diamonds.csv')

# Codificación de variables categóricas con OneHotEncoder
encoder = OneHotEncoder(drop='first')
encoder.fit(data[['cut', 'color', 'clarity']])
data[encoder.get_feature_names_out()] = encoder.transform(data[['cut', 'color', 'clarity']]).toarray()
data.drop(['cut', 'color', 'clarity'], axis=1, inplace=True)

# Preparación de datos
scaler = StandardScaler()
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir parámetros para GridSearch
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Configurar MLflow
mlflow.set_experiment("Diamantes Gradient Boosting Experimento")

with mlflow.start_run():
    # Realizar GridSearch
    grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Mejor modelo
    best_model = grid_search.best_estimator_

    # Registrar hiperparámetros
    mlflow.log_params(grid_search.best_params_)

    # Entrenar modelo final
    best_model.fit(X_train, y_train)

    # Predicciones y métricas
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Registrar métricas
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2 Score", r2)

    # Crear gráfico de predicciones vs reales
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Precio actual")
    plt.ylabel("Precio predicho")
    plt.title("Actual vs Precio predicho")
    plt.savefig("actual_vs_predicho.png")
    mlflow.log_artifact("actual_vs_predicho.png")
    os.remove("actual_vs_predicho.png")

    # Registrar el modelo
    input_example = pd.DataFrame([X_train[0]], columns=data.columns[:-1])
    signature = mlflow.models.signature.infer_signature(X_train, best_model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model_diamonds",
        input_example=input_example,
        signature=signature
    )

    print("Experimento completado MLflow!")

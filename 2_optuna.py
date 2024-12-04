import os
import optuna
import mlflow
import pandas as pd
import mlflow.sklearn
import optuna.visualization as vis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

# Cargar datos
data = pd.read_csv('diamonds.csv')

# Codificación de variables categóricas
encoder = OneHotEncoder(drop='first')
encoder.fit(data[['cut', 'color', 'clarity']])
data[encoder.get_feature_names_out()] = encoder.transform(data[['cut', 'color', 'clarity']]).toarray()
data.drop(['cut', 'color', 'clarity'], axis=1, inplace=True)

# Escalamiento y partición de datos
scaler = StandardScaler()
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Configurar MLflow
mlflow.set_experiment("Optuna Diamantes Experimento")

# Definir la función objetivo para Optuna
def objective(trial):
    # Hiperparámetros sugeridos por Optuna
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
    max_depth = trial.suggest_int('max_depth', 3, 10)

    # Modelo Gradient Boosting
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predicción y métricas
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)

    # Registro en un nested run de MLflow
    run_name = f"Iteration {trial.number + 1}"
    with mlflow.start_run(nested=True, run_name=run_name):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

    return r2

# Estudio de optimización con Optuna
study = optuna.create_study(direction='maximize')

with mlflow.start_run(run_name="optuna_diamantes_experimento"):
    study.optimize(objective, n_trials=10)

    # Registrar los mejores parámetros y métricas del estudio
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_r2", study.best_value)

    # Entrenar modelo final con los mejores parámetros
    best_params = study.best_params
    final_model = GradientBoostingRegressor(
        **best_params,
        random_state=42
    )
    final_model.fit(X_train, y_train)
    y_pred_final = final_model.predict(X_test)

    # Métricas finales
    final_mae = mean_absolute_error(y_test, y_pred_final)
    final_rmse = mean_squared_error(y_test, y_pred_final)**0.5
    final_r2 = r2_score(y_test, y_pred_final)

    mlflow.log_metric("final_mae", final_mae)
    mlflow.log_metric("final_rmse", final_rmse)
    mlflow.log_metric("final_r2", final_r2)

    # Registrar modelo final en MLflow
    input_example = pd.DataFrame([X_train[0]], columns=data.drop('price', axis=1).columns)
    signature = infer_signature(X_train, final_model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model_diamonds",
        input_example=input_example,
        signature=signature
    )

    # Crear visualizaciones de Optuna
    opt_history_path = "optimization_history.png"
    opt_slice_path = "slice_plot.png"
    vis.plot_optimization_history(study).write_image(opt_history_path)
    vis.plot_slice(study).write_image(opt_slice_path)

    mlflow.log_artifact(opt_history_path)
    mlflow.log_artifact(opt_slice_path)

    # Limpiar archivos temporales
    os.system(f"rm *.png")

print("Mejores parametros:")
print(study.best_params)

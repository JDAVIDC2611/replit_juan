import os
import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.models.signature import infer_signature

import warnings
warnings.filterwarnings("ignore")

# Cargar datos
data = pd.read_csv('diamonds.csv')
X = data.drop('price', axis=1)
y = data['price']

# Identificar características categóricas y numéricas
categorical_features = ['cut', 'color', 'clarity']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Mejores parámetros (pueden ser ajustados en un flujo anterior)
best_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}

# Configurar MLflow
mlflow.set_experiment("Gradient Boosting Diamonds Pipeline")

with mlflow.start_run(run_name="final_model_pipeline"):
    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    # Modelo Gradient Boosting
    model = GradientBoostingRegressor(
        **best_params,
        random_state=42
    )

    # Crear pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Entrenar pipeline
    pipeline.fit(X, y)

    # Ejemplo de entrada y firma del modelo
    input_example = X[:1]
    signature = infer_signature(X, pipeline.predict(X))

    # Registrar modelo en MLflow
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="pipeline_diamonds",
        input_example=input_example,
        signature=signature
    )

print("Pipeline guardado con MLflow.")

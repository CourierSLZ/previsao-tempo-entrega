"""Treina e compara modelos, registrando tudo no MLflow."""
import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import ALVO, CATEGORICAS, COLUNAS, NUMERICAS, add_features


def preprocessador() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ]
    )


def modelos() -> dict:
    return {
        "baseline_media": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.08, max_iter=300, random_state=42
        ),
    }


def avaliar(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/entregas.csv")
    ap.add_argument("--experimento", default="tempo-entrega")
    a = ap.parse_args()

    df = add_features(pd.read_csv(a.data))
    X, y = df[COLUNAS], df[ALVO]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment(a.experimento)
    resultados = {}
    for nome, modelo in modelos().items():
        pipe = Pipeline([("prep", preprocessador()), ("modelo", modelo)])
        with mlflow.start_run(run_name=nome):
            pipe.fit(X_tr, y_tr)
            m = avaliar(y_te, pipe.predict(X_te))
            mlflow.log_param("modelo", nome)
            mlflow.log_metrics(m)
            resultados[nome] = m
            print(f"{nome:25s} MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.3f}")

    melhor = min(resultados, key=lambda k: resultados[k]["mae"])
    print(f"\nMelhor modelo (MAE): {melhor}")
    final = Pipeline([("prep", preprocessador()), ("modelo", modelos()[melhor])]).fit(X_tr, y_tr)
    Path("outputs").mkdir(exist_ok=True)
    joblib.dump(final, "outputs/modelo.joblib")


if __name__ == "__main__":
    main()

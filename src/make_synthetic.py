"""Gera um dataset sintético de entregas para o projeto rodar sem baixar nada.

Para usar um dataset real do Kaggle (ex.: 'Food Delivery Time Prediction'),
salve o CSV em data/entregas.csv com as mesmas colunas, ou ajuste
as colunas em src/features.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gerar(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    distancia = rng.gamma(shape=2.5, scale=2.0, size=n).clip(0.3, 25)
    preparo = rng.normal(18, 6, n).clip(5, 50)
    hora = rng.integers(10, 23, n)
    dia_semana = rng.integers(0, 7, n)
    transito = rng.choice(["baixo", "medio", "alto"], n, p=[0.4, 0.4, 0.2])
    clima = rng.choice(["sol", "nublado", "chuva"], n, p=[0.55, 0.3, 0.15])
    nota = rng.normal(4.6, 0.3, n).clip(3.0, 5.0)

    fator_transito = pd.Series(transito).map({"baixo": 3.0, "medio": 4.5, "alto": 7.0}).to_numpy()
    penal_clima = pd.Series(clima).map({"sol": 0, "nublado": 1, "chuva": 5}).to_numpy()
    rush = ((hora >= 11) & (hora <= 14)) | ((hora >= 18) & (hora <= 21))

    tempo = (
        preparo
        + distancia * fator_transito
        + penal_clima
        + rush * 4
        - (nota - 4.5) * 3
        + rng.normal(0, 3, n)
    ).clip(10, None)

    return pd.DataFrame(
        {
            "distancia_km": distancia.round(2),
            "tempo_preparo_min": preparo.round(1),
            "hora": hora,
            "dia_semana": dia_semana,
            "transito": transito,
            "clima": clima,
            "nota_entregador": nota.round(2),
            "tempo_entrega_min": tempo.round(1),
        }
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="data/entregas.csv")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    gerar(a.n).to_csv(a.out, index=False)
    print(f"Salvo em {a.out}")

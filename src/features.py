"""Feature engineering."""
import numpy as np
import pandas as pd

ALVO = "tempo_entrega_min"
NUMERICAS = [
    "distancia_km", "tempo_preparo_min", "nota_entregador",
    "hora_sin", "hora_cos", "rush", "fim_de_semana",
]
CATEGORICAS = ["transito", "clima"]
COLUNAS = NUMERICAS + CATEGORICAS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas. Não altera o DataFrame original."""
    out = df.copy()
    out["hora_sin"] = np.sin(2 * np.pi * out["hora"] / 24)
    out["hora_cos"] = np.cos(2 * np.pi * out["hora"] / 24)
    out["rush"] = (
        ((out["hora"] >= 11) & (out["hora"] <= 14)) | ((out["hora"] >= 18) & (out["hora"] <= 21))
    ).astype(int)
    out["fim_de_semana"] = (out["dia_semana"] >= 5).astype(int)
    return out

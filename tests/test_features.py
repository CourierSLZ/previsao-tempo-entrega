import pandas as pd

from src.features import COLUNAS, add_features
from src.make_synthetic import gerar


def test_add_features_nao_altera_original():
    df = gerar(50)
    antes = df.columns.tolist()
    add_features(df)
    assert df.columns.tolist() == antes


def test_rush_e_fim_de_semana():
    df = pd.DataFrame({"hora": [12, 15, 19], "dia_semana": [1, 5, 6]})
    out = add_features(df)
    assert out["rush"].tolist() == [1, 0, 1]
    assert out["fim_de_semana"].tolist() == [0, 1, 1]


def test_todas_colunas_existem():
    out = add_features(gerar(20))
    assert set(COLUNAS) <= set(out.columns)

# Previsão de tempo de entrega

Modelo de regressão para estimar o tempo de entrega (minutos) de um pedido,
comparando um baseline simples com modelos de ML e registrando os experimentos no **MLflow**.

## Dados
Por padrão o projeto gera dados **sintéticos** (`src/make_synthetic.py`) para rodar
sem dependências externas. Para usar um dataset real do Kaggle, salve-o em
`data/entregas.csv` e ajuste as colunas em `src/features.py`.

## Como rodar
```bash
pip install -r requirements.txt
python -m src.make_synthetic
python -m src.train
mlflow ui          # compara os experimentos em http://localhost:5000
pytest -q
```

## Modelos comparados
Média (baseline) · Ridge · Random Forest · HistGradientBoosting.

## Resultados (conjunto de teste)
| Modelo | MAE (min) | RMSE | R² |
|---|---|---|---|
| Baseline (média) | _preencher_ | _preencher_ | _preencher_ |
| Ridge | _preencher_ | _preencher_ | _preencher_ |
| Random Forest | _preencher_ | _preencher_ | _preencher_ |
| HistGradientBoosting | _preencher_ | _preencher_ | _preencher_ |

## Limitações
- Dados sintéticos refletem as regras que eu mesmo escrevi; com dados reais
  os ganhos e as features mais importantes mudam.
- Sem validação temporal (split aleatório).

## Próximos passos
- Importância de features (SHAP) e análise de erro por faixa de distância.
- Validação temporal e tuning com Optuna.

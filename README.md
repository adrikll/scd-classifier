Essa pipeline treina e avalia 4 modelos para classificação de 5 classes de batimentos cardíacos: 
- Normal (0)
- Supraventricular (1)
- Ventricular (2)
- Fusão (3)
- Desconhecido (4)

O dataset está disponivel em: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Os modelos utilizados são:
- Random Forest;
- XGBoost;
- MLP;
- CNN;
- Ensemble do Random Forest, XGBoost e CNN, usando como meta modelo a Regressão Logística.

*Os hiperparamêtros foram otimizados usando o RandomSearch;

Execução da Pipeline:
1. pip install requirements.txt
2. optimize.py (otimização)
3. train_evaluate.py (modelos individuais)
4. ensemble.py 

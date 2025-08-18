from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from . import config

def train_final_model(X_train, y_train, model_config, best_params, best_k, fit_params={}):
    """Treina o modelo final com os melhores parâmetros em todo o conjunto de treino."""
    print("Treinando o modelo final com os melhores parâmetros...")
    
    # Atualiza o estimador com os melhores parâmetros
    final_estimator = model_config['estimator'].set_params(**best_params)
    
    # Atualiza o seletor com o melhor k
    final_selector = config.SELECTOR
    if best_k is not None:
        final_selector.set_params(k=best_k)

    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('select', final_selector),
        (config.MODEL_PREFIX, final_estimator)
    ])
    
    final_pipeline.fit(X_train, y_train, **fit_params)
    print("Modelo final treinado com sucesso.")
    
    return final_pipeline
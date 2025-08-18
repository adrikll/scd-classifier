from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import src.config as config
from utils import apply_random_search, extract_params_and_k, gmean_scorer
from sklearn.model_selection import StratifiedKFold

def find_best_model_params(X_train, y_train, model_config, fit_params={}):
    """Executa o RandomizedSearchCV para encontrar os melhores hiperparâmetros."""
    print(f"Iniciando otimização com RandomizedSearch para: {model_config['name']}...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('select', config.SELECTOR),
        (config.MODEL_PREFIX, model_config['estimator'])
    ])

    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    best_raw_params = apply_random_search( 
        X=X_train, 
        y=y_train, 
        estimator=pipeline, 
        param_distributions=model_config['param_grid'], 
        scoring=gmean_scorer,
        cv=cv,
        n_iter=config.N_ITER_RANDOM_SEARCH, 
        fit_params=fit_params
    )

    # Extração (continua igual)
    best_model_params, best_k = extract_params_and_k(
        best_raw_params, 
        model_prefix=config.MODEL_PREFIX, 
        k_key=config.K_KEY
    )
    
    print(f"Melhores parâmetros encontrados: {best_model_params}")
    print(f"Melhor número de features (k): {best_k}")
    
    return best_model_params, best_k
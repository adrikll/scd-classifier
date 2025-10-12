from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .utils import apply_random_search, extract_params_and_k, gmean_scorer
from . import config
from sklearn.model_selection import StratifiedKFold
import numpy as np

def find_best_model_params(X_train, y_train, model_config, feature_names, fit_params={}):
    """Executes RandomizedSearchCV to find the best hyperparameters."""
    
    print(f"Iniciando otimização com RandomizedSearch para: {model_config['name']}...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('select', config.SELECTOR),
        (config.MODEL_PREFIX, model_config['estimator'])
    ])

    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    best_raw_params, best_pipeline = apply_random_search( 
        X=X_train, 
        y=y_train, 
        estimator=pipeline, 
        param_distributions=model_config['param_grid'], 
        scoring=gmean_scorer,
        cv=cv,
        n_iter=config.N_ITER_RANDOM_SEARCH, 
        fit_params=fit_params
    )

    best_model_params, best_k = extract_params_and_k(
        best_raw_params, 
        model_prefix=config.MODEL_PREFIX, 
        k_key=config.K_KEY
    )
    
    selector = best_pipeline.named_steps['select']
    mask = selector.get_support()
    feature_names_array = np.array(feature_names)
    selected_features = feature_names_array[mask].tolist()
    
    print(f"Melhores parâmetros encontrados: {best_model_params}")
    print(f"Melhor número de features (k): {best_k}")
    print(f"As {len(selected_features)} features selecionadas foram: {selected_features}")
    
    return best_model_params, best_k, selected_features
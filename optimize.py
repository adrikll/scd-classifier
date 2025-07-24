import os
import json
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
import config
from dataloader import load_and_prepare_data
from models.neural_networks import create_mlp, create_cnn # Assumindo que essas funções existem
from sklearn.utils import class_weight
import numpy as np
import pandas as pd

def optimize():
    """Executa o Random Search para encontrar os melhores hiperparâmetros e os salva em um arquivo JSON."""
    
    (X_train, y_train), _, _ = load_and_prepare_data()

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(weights))
    print(f"Pesos de classe calculados: {class_weights_dict}")

    models_and_params = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1), # n_jobs=-1 para usar todos os cores
            'params': {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10]}
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='logloss', n_jobs=-1), # Para binária, use 'logloss' e desative o encoder
            'params': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
        },
        'MLP': {
            'model': KerasClassifier(model=create_mlp, verbose=0, class_weight=class_weights_dict, loss="binary_crossentropy", # binária
                                     model__input_shape=config.NN_INPUT_SHAPE),
            'params': {'batch_size': [64, 128, 256], 'epochs': [30, 50, 70], 'optimizer': ['adam', 'rmsprop']}
        },
        'CNN': {
             'model': KerasClassifier(model=create_cnn, verbose=0, class_weight=class_weights_dict, loss="binary_crossentropy", # binária
                                      model__input_shape=config.NN_INPUT_SHAPE),
             'params': {'batch_size': [64, 128, 256], 'epochs': [20, 40, 60], 'optimizer': ['adam', 'rmsprop']}
        }
    }
    
    best_params_all_models = {}

    for model_name, mp in models_and_params.items():
        print(f"\nOtimizando {model_name}...")
        search = RandomizedSearchCV(estimator=mp['model'], param_distributions=mp['params'], n_iter=config.N_ITER_SEARCH, cv=config.CV_FOLDS, verbose=2, random_state=config.RANDOM_STATE, n_jobs=1) # n_jobs=1 para evitar problemas com KerasClassifier e GPU/CPU
        search.fit(X_train, y_train)
        
        # coleta os melhores parâmetros
        best_params_all_models[model_name] = search.best_params_
        print(f"Melhores parâmetros para {model_name}: {search.best_params_}")

        optimization_results_path = os.path.join(config.OUTPUT_DIR, f"{model_name}_optimization_results.csv")
        pd.DataFrame(search.cv_results_).to_csv(optimization_results_path, index=False)
        print(f"Resultados da otimização salvos em: {optimization_results_path}")

    print("\nOtimização concluída!")
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    with open(config.BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params_all_models, f, indent=4)
    
    print(f"Melhores parâmetros salvos em: {config.BEST_PARAMS_FILE}")

if __name__ == '__main__':
    optimize()
import os
import json
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from scikeras.wrappers import KerasClassifier
import config
from dataloader import load_and_prepare_data
from models.neural_networks import create_mlp, create_cnn
import numpy as np
import pandas as pd

def optimize():
    """Executa o Random Search para encontrar os melhores hiperparâmetros e os salva em um arquivo JSON."""
    
    (X_train, y_train), _, _ = load_and_prepare_data()

    my_manual_class_weights = {
        0: 1.0,  
        1: 1.0 
    }
    class_weights_dict = my_manual_class_weights
    print(f"Pesos de classe MANUAIS aplicados para MLP/CNN: {class_weights_dict}")
    
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight_value = neg_count / pos_count
    print(f"scale_pos_weight para XGBoost/LightGBM: {scale_pos_weight_value:.2f}")

    models_and_params = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='logloss', n_jobs=-1, scale_pos_weight=scale_pos_weight_value),
            'params': {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, class_weight='balanced', objective='binary'),
            'params': {
                'n_estimators': [100, 200, 300, 400, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [10, 20, 31, 40, 50],
                'max_depth': [3, 5, 7, -1]
            }
        },
        'SVM': {
            'model': SVC(random_state=config.RANDOM_STATE, probability=True, class_weight='balanced'),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
        },
        'MLP': {
            
            'model': KerasClassifier(model=create_mlp, verbose=0, class_weight=class_weights_dict, loss="binary_crossentropy",
                                     model__input_shape=config.NN_INPUT_SHAPE),
            'params': {
                'batch_size': [8, 16, 32, 64, 128],
                'epochs': [30, 50, 70, 90, 120],
                'optimizer': ['adam', 'rmsprop']
            }
        },
        'CNN': {

             'model': KerasClassifier(model=create_cnn, verbose=0, class_weight=class_weights_dict, loss="binary_crossentropy",
                                       model__input_shape=config.NN_INPUT_SHAPE),
             'params': {
                 'batch_size': [8, 16, 32, 64, 128],
                 'epochs': [20, 40, 60, 80, 100],
                 'optimizer': ['adam', 'rmsprop']
             }
        }
    }
    
    best_params_all_models = {}

    for model_name, mp in models_and_params.items():
        print(f"\nOtimizando {model_name}...")
        search = RandomizedSearchCV(estimator=mp['model'], param_distributions=mp['params'], n_iter=config.N_ITER_SEARCH, cv=config.CV_FOLDS, verbose=2, random_state=config.RANDOM_STATE, n_jobs=1, scoring='f1')
        search.fit(X_train, y_train)
        
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
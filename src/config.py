from sklearn.ensemble import (
    IsolationForest, 
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import randint, uniform

# --- Configurações Gerais ---
RANDOM_STATE = 42
CV_SPLITS = 5
TEST_SIZE = 0.2
N_ITER_RANDOM_SEARCH = 200
MODEL_PREFIX = "clf"
K_KEY = "select__k"

# --- Configurações de Paths ---
DATA_PATH = 'dados/chagas_all_features.xlsx'
RESULTS_PATH = 'results'

# --- Configurações do Dataset ---
TARGET_COLUMN = 'Obito_MS_FU-5 years'
DROP_COLUMNS = [
    'ID', 'Name', 'FE', 'Filename', 'Age', 'Obito_MS', 'Time', 
    'Date Holter', 'Sex', 'Nat', 'Event (FU-5 years)', 'Rassi Score', 
    'Rassi Points', 'Classe_FE', 'Obito_MS_FU-5 years'
]

# --- Configuração dos Modelos e Hiperparâmetros para Otimização ---
# Esta estrutura permite adicionar ou remover modelos facilmente.
MODELS_CONFIG = [
    {
        'name': 'RandomForest',
        'estimator': RandomForestClassifier(random_state=RANDOM_STATE),
        'param_grid': {
            'select__k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 180, 200, 220, 239],
            'clf__n_estimators': [100, 200, 300],
            'clf__max_features': ['sqrt', 0.6, 0.8],
            'clf__max_depth': [10, 20, None],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__criterion': ['gini', 'entropy'],
        }
    },
    {
        'name': 'GradientBoosting',
        'estimator': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'param_grid': {
            'select__k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 180, 200, 220, 239],
            'clf__n_estimators': [100, 200, 300],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__max_depth': [3, 5, 10],
            'clf__subsample': [0.8, 0.9, 1.0],
        }
    },
    {
        'name': 'XGBoost',
        'estimator': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
        'param_grid': {
            'select__k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 180, 200, 220, 239],
            'clf__n_estimators': [100, 200, 300],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__max_depth': [3, 5, 7],
            'clf__subsample': [0.8, 0.9, 1.0],
            'clf__colsample_bytree': [0.8, 0.9, 1.0],
        }
    },
    {
        'name': 'LightGBM',
        'estimator': LGBMClassifier(random_state=RANDOM_STATE),
        'param_grid': {
            'select__k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 180, 200, 220, 239],
            'clf__n_estimators': [100, 200, 300],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__num_leaves': [20, 31, 40], 
            'clf__max_depth': [-1, 10, 20], 
            'clf__subsample': [0.8, 0.9, 1.0],
            'clf__colsample_bytree': [0.8, 0.9, 1.0],
        }
    },
    {
        'name': 'SVM',
        'estimator': SVC(random_state=RANDOM_STATE, probability=True),
        'param_grid': {
            'select__k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 180, 200, 220, 239],
            'clf__C': [0.1, 1, 10, 100],
            'clf__kernel': ['rbf', 'linear'],
            'clf__gamma': ['scale', 'auto', 0.1, 0.01],
        }
    },
    {
        'name': 'MLPClassifier',
        'estimator': MLPClassifier(random_state=RANDOM_STATE, max_iter=2000),
        'param_grid': {
            'select__k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 180, 200, 220, 239],
            'clf__hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'clf__activation': ['tanh', 'relu'],
            'clf__solver': ['adam'],
            'clf__alpha': [0.0001, 0.001, 0.05],
            'clf__learning_rate': ['constant','adaptive'],
        }
    }
]

SELECTOR = SelectKBest(f_classif)
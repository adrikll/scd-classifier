import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV

# --- Funções de Métricas e Scoring ---

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-8)

def geometric_mean_score(y_true, y_pred):
    if -1 in np.unique(y_pred):
        y_pred = (y_pred == -1).astype(int)
    
    recall = recall_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    return np.sqrt(recall * specificity)

gmean_scorer = make_scorer(geometric_mean_score)

def calculate_metrics(y_true, y_pred, display=True):
    if -1 in np.unique(y_pred):
        y_pred = (y_pred == -1).astype(int)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError: # Acontece se o classificador prever apenas uma classe
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_pred)) == 1:
            if np.unique(y_pred)[0] == 1:
                tp = np.sum(y_true == 1)
                fp = np.sum(y_true == 0)
            else:
                tn = np.sum(y_true == 0)
                fn = np.sum(y_true == 1)

    epsilon = 1e-8
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    gmean = np.sqrt(specificity * recall)

    metrics = {
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "Specificity": specificity, "F1 Score": f1_score, "Geometric Mean": gmean
    }
    
    if display:
        print("\n--- Métricas de Avaliação ---")
        for name, value in metrics.items():
            print(f"{name}: {value*100:.2f}%")

    return metrics

# --- Funções de Plotagem e Display ---

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Previsto")
    plt.ylabel("Verdadeiro")
    if save_path:
        plt.savefig(save_path)
        print(f"\nMatriz de confusão salva em: {save_path}")
    plt.show()

def display_kfold_scores(metrics_list):
    mean_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
    std_metrics = {key: np.std([m[key] for m in metrics_list]) for key in metrics_list[0]}
    
    print("\n--- Resultados da Validação Cruzada (K-Fold) ---")
    for key in mean_metrics:
        mean_val = mean_metrics[key]
        std_val = std_metrics[key]
        print(f"{key}: {mean_val*100:.2f}% ± {std_val*100:.2f}%")

# --- Funções de Pipeline e Otimização ---

def apply_grid_search(X, y, estimator, param_grid, scoring, cv, fit_params={}):
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X, y, **fit_params)
    print(f"\nMelhor pontuação (G-Mean) no Grid Search: {grid_search.best_score_:.4f}")
    return grid_search.best_params_

def apply_random_search(X, y, estimator, param_distributions, scoring, cv, n_iter, fit_params={}):
    """Aplica o RandomizedSearchCV para encontrar os melhores hiperparâmetros."""
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions, # Nome do parâmetro oficial
        n_iter=n_iter,                           # Número de iterações a testar
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42                          # Garante que a busca aleatória seja reprodutível
    )
    
    random_search.fit(X, y, **fit_params)
    print(f"\nMelhor pontuação (G-Mean) no Random Search: {random_search.best_score_:.4f}")
    return random_search.best_params_

def extract_params_and_k(params, model_prefix, k_key):
    best_params = {
        k.split("__")[-1]: v for k, v in params.items() if k.startswith(model_prefix)
    }
    best_k = params.get(k_key, None)
    return best_params, best_k

# --- Classes e Funções de Pré-processamento ---

def preprocess(X_train, X_test, y_train, k, selector):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector.set_params(k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    return X_train_selected, X_test_selected


class CorrelationFeatureReducer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.cols_to_keep_ = None

    def fit(self, X, y=None):
        corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
        upper = np.triu(corr_matrix, k=1)
        
        to_drop = {column for column in range(upper.shape[1]) if any(upper[:, column] > self.threshold)}
        
        self.cols_to_keep_ = [i for i in range(X.shape[1]) if i not in to_drop]
        return self

    def transform(self, X):
        return X[:, self.cols_to_keep_]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
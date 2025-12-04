import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
import json
import os 

def specificity_score(y_true, y_pred):
    """Calculates the specificity score."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-8)

def geometric_mean_score(y_true, y_pred):
    """Calculates the geometric mean of recall and specificity."""
    if -1 in np.unique(y_pred):
        y_pred = (y_pred == -1).astype(int)
    
    recall = recall_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    return np.sqrt(recall * specificity)

gmean_scorer = make_scorer(geometric_mean_score)

def calculate_metrics(y_true, y_pred, display=True):
    """Calculates and optionally displays a dictionary of classification metrics."""
    if -1 in np.unique(y_pred):
        y_pred = (y_pred == -1).astype(int)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError: #se o classificador prever apenas uma classe
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

def find_optimal_threshold(y_true, y_pred_proba, metric_func):
    """
    Finds the optimal probability threshold that maximizes a given metric function.
    
    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities for the positive class.
        metric_func: The metric function to maximize (e.g., geometric_mean_score).
        
    Returns:
        The best threshold found.
    """
    thresholds = np.arange(0.01, 1.0, 0.01) 
    best_threshold = 0.5 
    best_score = 0.0

    for threshold in thresholds:
        # Converte probabilidades em classes com base no threshold atual
        y_pred_class = (y_pred_proba >= threshold).astype(int)
        score = metric_func(y_true, y_pred_class)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    print(f"\nMelhor threshold encontrado: {best_threshold:.2f} (com score de {best_score:.4f})")
    return best_threshold

def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusão", save_path=None):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    gradiente_azul = sns.light_palette("blue", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap=gradiente_azul, cbar=False, annot_kws={"size": 16})
    
    plt.title(title, fontsize=16)
    plt.xlabel("Previsto", fontsize=12)
    plt.ylabel("Verdadeiro", fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\nMatriz de confusão salva em: {save_path}")
    
    plt.show()

def display_kfold_scores(metrics_list):
    """Aggregates and displays the mean and std dev of metrics from a k-fold run."""
    mean_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
    std_metrics = {key: np.std([m[key] for m in metrics_list]) for key in metrics_list[0]}
    
    print("\n--- Resultados da Validação Cruzada (K-Fold) ---")
    for key in mean_metrics:
        mean_val = mean_metrics[key]
        std_val = std_metrics[key]
        print(f"{key}: {mean_val*100:.2f}% ± {std_val*100:.2f}%")


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
    """Applies RandomizedSearchCV to find the best hyperparameters."""
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions, 
        n_iter=n_iter,                          
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42                          
    )
    
    random_search.fit(X, y, **fit_params)
    print(f"\nMelhor pontuação (G-Mean) no Random Search: {random_search.best_score_:.4f}")
    return random_search.best_params_, random_search.best_estimator_

def extract_params_and_k(params, model_prefix, k_key):
    """Separates model-specific hyperparameters from the feature selection 'k' parameter."""
    best_params = {
        k.split("__")[-1]: v for k, v in params.items() if k.startswith(model_prefix)
    }
    best_k = params.get(k_key, None)
    return best_params, best_k

def save_model_summary(model_name, best_params, best_k, optimal_threshold, class_report_dict, selected_features, model_results_path):
    """Saves a summary of results (params, threshold, features, metrics) to a JSON file."""
    
    summary = {
        "model_name": model_name,
        "best_k_features": best_k,
        "optimal_threshold": f"{optimal_threshold:.4f}",
        "best_hyperparameters": best_params,
        "selected_features": selected_features,
        "classification_report_dict": class_report_dict
    }
        
    file_path = os.path.join(model_results_path, 'summary.json')
    
    with open(file_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Resumo do modelo salvo em: {file_path}")

def save_classification_report(class_report_str, model_results_path):
    """Saves the formatted classification report to a text file."""
    
    file_path = os.path.join(model_results_path, 'classification_report.txt')
    
    with open(file_path, 'w') as f:
        f.write("--- Relatório de Classificação ---\n\n")
        f.write(class_report_str)
        
    print(f"Relatório de classificação salvo em: {file_path}")


def preprocess(X_train, X_test, y_train, k, selector):
    """
    Applies standardization and feature selection to the training and test sets.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector.set_params(k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    return X_train_selected, X_test_selected

class CorrelationFeatureReducer(BaseEstimator, TransformerMixin):
    """A custom transformer to drop features with correlation above a given threshold."""
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
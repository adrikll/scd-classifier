import os
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.base import clone 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from utils import (
    calculate_metrics, 
    plot_confusion_matrix, 
    display_kfold_scores,
    find_optimal_threshold,
    geometric_mean_score 
)
import numpy as np
from . import config

def find_threshold_with_cv(pipeline, X_train, y_train, fit_params={}):
    """
    Usa validação cruzada MANUAL para obter previsões e encontrar o threshold ótimo,
    permitindo o uso de fit_params.
    """
    print("\nBuscando threshold ótimo com validação cruzada...")
    
    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    
    # Array para armazenar as probabilidades de validação de cada fold
    y_pred_proba_cv = np.zeros(len(y_train))

    for train_idx, val_idx in cv.split(X_train, y_train):
        # 1. Clona o pipeline para garantir que cada fold seja treinado do zero
        cloned_pipeline = clone(pipeline)
        
        # 2. Separa os dados e pesos para o fold atual
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold = y_train[train_idx]
        
        current_fit_params = {}
        if 'clf__sample_weight' in fit_params:
            sample_weights = fit_params['clf__sample_weight']
            current_fit_params['clf__sample_weight'] = sample_weights[train_idx]

        # 3. Treina o pipeline no fold de treino
        cloned_pipeline.fit(X_train_fold, y_train_fold, **current_fit_params)
        
        # 4. Faz a previsão de probabilidade no fold de validação
        y_pred_proba_fold = cloned_pipeline.predict_proba(X_val_fold)[:, 1]
        
        # 5. Armazena as previsões nas posições corretas
        y_pred_proba_cv[val_idx] = y_pred_proba_fold

    # Com as probabilidades de todos os folds, encontra o melhor threshold
    optimal_threshold = find_optimal_threshold(y_train, y_pred_proba_cv, geometric_mean_score)
    
    return optimal_threshold

def evaluate_on_test_set(pipeline, X_test, y_test, model_name, optimal_threshold, model_results_path):
    """
    Avalia o pipeline e retorna os relatórios de classificação (dict e string).
    Salva a matriz de confusão no diretório específico do modelo.
    """
    print(f"\n--- Avaliação no Conjunto de Teste ({model_name} com Threshold={optimal_threshold:.2f}) ---")
    
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Exibe as métricas no console
    calculate_metrics(y_test, y_pred, display=True)
    
    # --- Usa o novo caminho para salvar a matriz de confusão ---
    plot_path = os.path.join(model_results_path, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, title=f'Matriz de Confusão - {model_name}', save_path=plot_path)

    print("\n--- Relatório de Classificação ---")
    # Gera o relatório como dicionário (para o JSON)
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Gera o relatório como string (para o TXT e para imprimir)
    class_report_str = classification_report(y_test, y_pred)
    print(class_report_str)

    # Retorna ambos os formatos
    return class_report_dict, class_report_str

def perform_cross_validation(pipeline, X_train, y_train, fit_params={}):
    """Realiza a validação cruzada para uma estimativa robusta do desempenho."""
    cv = StratifiedKFold(n_splits=config.CV_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    
    metrics_list = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        current_fit_params = {}
        if 'clf__sample_weight' in fit_params:
            current_fit_params['clf__sample_weight'] = fit_params['clf__sample_weight'][train_idx]

        pipeline.fit(X_train_fold, y_train_fold, **current_fit_params)
        y_pred_fold = pipeline.predict(X_val_fold)
        
        fold_metrics = calculate_metrics(y_val_fold, y_pred_fold, display=False)
        metrics_list.append(fold_metrics)
        
    display_kfold_scores(metrics_list)
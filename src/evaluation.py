import os
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from utils import calculate_metrics, plot_confusion_matrix, display_kfold_scores
from . import config

def evaluate_on_test_set(pipeline, X_test, y_test, model_name):
    """Avalia o pipeline treinado no conjunto de teste."""
    print(f"\n--- Avaliação no Conjunto de Teste ({model_name}) ---")
    
    y_pred = pipeline.predict(X_test)
    
    calculate_metrics(y_test, y_pred, display=True)
    
    # Salva a matriz de confusão
    if not os.path.exists(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)
    
    plot_path = os.path.join(config.RESULTS_PATH, f'confusion_matrix_{model_name}.png')
    plot_confusion_matrix(y_test, y_pred, title=f'Matriz de Confusão - {model_name}', save_path=plot_path)

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
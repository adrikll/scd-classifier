import os
import time
import joblib
import json
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils import class_weight 

import config
from dataloader import load_and_prepare_data
from models.neural_networks import create_cnn, create_mlp
from utils import plot_confusion_matrix, plot_roc_curves
import utils

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def train_ensemble():
    """
    Treina e avalia um modelo de Stacking, combinando
    o Random Forest, XGBoost, LightGBM, SVM e CNN para classificação binária.
    """
    MODEL_NAME = 'Stacking_Ensemble_Diversificado_Chagas'

    print("\nCarregando dados e melhores hiperparâmetros...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_data()

    try:
        with open(config.BEST_PARAMS_FILE, 'r') as f:
            best_params = json.load(f)
        print("Hiperparâmetros otimizados carregados com sucesso.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{config.BEST_PARAMS_FILE}' não encontrado. Execute 'optimize.py' primeiro.")
        return

    print("\nConstruindo o modelo Stacking diverso...")

    my_manual_class_weights = {
        0: 1.0, 
        1: 1.0  
    }
    class_weights_dict = my_manual_class_weights
    print(f"Pesos de classe MANUAIS aplicados para MLP/CNN no Ensemble: {class_weights_dict}")
    # --- FIM DA ATRIBUIÇÃO MANUAL ---

    # Calcular scale_pos_weight para XGBoost e LightGBM
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight_value = neg_count / pos_count
    print(f"scale_pos_weight para XGBoost/LightGBM: {scale_pos_weight_value:.2f}")

    base_estimators = [
        #('rf', RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1, **best_params.get('RandomForest', {}))),
        #('xgb', XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='logloss', n_jobs=-1, scale_pos_weight=scale_pos_weight_value, **best_params.get('XGBoost', {}))),
        ('lgbm', LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, class_weight='balanced', objective='binary', **best_params.get('LightGBM', {}))),
        #('svm', SVC(random_state=config.RANDOM_STATE, probability=True, class_weight='balanced', **best_params.get('SVM', {}))),
        ('mlp', KerasClassifier( 
            model=create_mlp, 
            optimizer=best_params.get('MLP', {}).get('optimizer'),
            loss="binary_crossentropy",
            epochs=best_params.get('MLP', {}).get('epochs'),
            batch_size=best_params.get('MLP', {}).get('batch_size'),
            class_weight=class_weights_dict, 
            verbose=0,
            model__input_shape=config.NN_INPUT_SHAPE
        )),
        ('cnn', KerasClassifier( 
            model=create_cnn,
            optimizer=best_params.get('CNN', {}).get('optimizer'),
            loss="binary_crossentropy",
            epochs=best_params.get('CNN', {}).get('epochs'),
            batch_size=best_params.get('CNN', {}).get('batch_size'),
            class_weight=class_weights_dict, 
            verbose=0,
            model__input_shape=config.NN_INPUT_SHAPE
        ))
    ]

    #meta-modelo
    meta_model = LogisticRegression(max_iter=1000, n_jobs=1)

    # lassificador Stacking final
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=3,
        n_jobs=1,
        passthrough=True
    )
    print("Modelo Stacking construído.")

    print("\nTreinando o modelo Stacking... ")
    start_time = time.time()
    stacking_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Treinamento concluído em {(end_time - start_time):.2f} segundos.")

    print("\nAvaliando o modelo...")
    
    model_output_dir = os.path.join(config.OUTPUT_DIR, MODEL_NAME)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    y_val_pred_probs_ensemble = stacking_model.predict_proba(X_val)[:, 1]
    best_threshold_ensemble = utils.find_best_threshold(y_val, y_val_pred_probs_ensemble)
    print(f"Melhor limiar para Ensemble encontrado: {best_threshold_ensemble:.2f} (F1-Score no Val: {f1_score(y_val, (y_val_pred_probs_ensemble >= best_threshold_ensemble).astype(int)):.4f})") # Adicionado F1-score no print
    
    y_pred_probs = stacking_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_probs >= best_threshold_ensemble).astype(int) 

    plot_confusion_matrix(y_test, y_pred, MODEL_NAME, model_output_dir, CLASSES=utils.CLASSES)
    plot_roc_curves(y_test, y_pred_probs, MODEL_NAME, model_output_dir, CLASSES=utils.CLASSES)
    print(f"Gráficos de análise salvos em: {model_output_dir}")

    accuracy = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=utils.CLASSES)
    
    print(f"\nRelatório Final para {MODEL_NAME}----------------------------")
    print(f"Acurácia no Teste: {accuracy:.4f}")
    print(report_str)
    
    report_dict = classification_report(y_test, y_pred, target_names=utils.CLASSES, output_dict=True)
    report_json_path = os.path.join(model_output_dir, 'classification_report.json')
    report_txt_path = os.path.join(model_output_dir, 'classification_report.txt')
    
    with open(report_json_path, 'w') as f:
        json.dump(report_dict, f, cls=NpEncoder, indent=4)
    with open(report_txt_path, 'w') as f:
        f.write(f"Acurácia no Teste: {accuracy:.4f}\n\n")
        f.write(report_str)
    
    model_save_path = os.path.join(model_output_dir, "final_ensemble_model.joblib")
    joblib.dump(stacking_model, model_save_path)
    print(f"Modelo Ensemble final salvo em: {model_save_path}")

if __name__ == '__main__':
    train_ensemble()
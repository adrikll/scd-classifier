import os
import time
import joblib
import json
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

import config
from dataloader import load_and_prepare_data
from models.neural_networks import create_cnn # Assumindo que essa função existe
from utils import plot_confusion_matrix, plot_roc_curves # Removed CLASSES from import
import utils # Importa o módulo utils para acessar CLASSES.

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
    o Random Forest, XGBoost e CNN para classificação binária.
    """
    MODEL_NAME = 'Stacking_Ensemble_RF_XGB_CNN_Chagas'

    # carrega os dados e hiperparametros otimizados 
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

    params_rf = best_params.get('RandomForest')
    params_xgb = best_params.get('XGBoost')
    params_cnn = best_params.get('CNN')

    if not all([params_rf, params_xgb, params_cnn]):
        print("ERRO: Hiperparâmetros para RandomForest, XGBoost ou CNN não encontrados no arquivo JSON.")
        return

    # balanceamento das classes para a CNN no ensemble (se aplicável)
    # StackingClassifier lida com o class_weight para os modelos base se eles aceitarem
    # Para KerasClassifier no ensemble, podemos passar None para class_weight e usar loss='binary_crossentropy'
    # mas se o desbalanceamento for grande, calcular weights aqui para CNN pode ser útil.
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(weights)) # Para CNN, onde 0 e 1 são os índices de classe

    # cria a lista de estimadores de base
    base_estimators = [
        ('rf', RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1, **params_rf)),
        ('xgb', XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='logloss', n_jobs=-1, **params_xgb)),
        ('cnn', KerasClassifier(
            model=create_cnn,
            optimizer=params_cnn.get('optimizer'),
            loss="binary_crossentropy", # Alterado para binária
            epochs=params_cnn.get('epochs'),
            batch_size=params_cnn.get('batch_size'),
            class_weight=class_weights_dict, # Usar os pesos balanceados
            verbose=0,
            model__input_shape=config.NN_INPUT_SHAPE # Passa o input_shape
        ))
    ]

    # meta-modelo
    meta_model = LogisticRegression(max_iter=1000, n_jobs=1)

    # classificador Stacking final
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=3, # cv para StackingClassifier
        n_jobs=1, # n_jobs para o StackingClassifier em si
        passthrough=True # Passa features originais para o meta-modelo
    )
    print("Modelo Stacking construído.")

    # treinamento
    print("\nTreinando o modelo Stacking... ")
    start_time = time.time()
    stacking_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Treinamento concluído em {(end_time - start_time):.2f} segundos.")

    # avaliação
    print("\nAvaliando o modelo...")
    
    model_output_dir = os.path.join(config.OUTPUT_DIR, MODEL_NAME)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # predições
    # Para classificadores Scikit-learn, predict_proba retorna probabilidades para cada classe
    # Para binário, geralmente a segunda coluna é a probabilidade da classe positiva (1)
    y_val_pred_probs_ensemble = stacking_model.predict_proba(X_val)[:, 1]
    best_threshold_ensemble = utils.find_best_threshold(y_val, y_val_pred_probs_ensemble)
    
    y_pred_probs = stacking_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_probs >= best_threshold_ensemble).astype(int) 

    # plots
    plot_confusion_matrix(y_test, y_pred, MODEL_NAME, model_output_dir, CLASSES=utils.CLASSES) # Usa CLASSES do utils
    # y_test_one_hot não é necessário para binário, basta y_test para ROC
    plot_roc_curves(y_test, y_pred_probs, MODEL_NAME, model_output_dir, CLASSES=utils.CLASSES) # Usa CLASSES do utils
    print(f"Gráficos de análise salvos em: {model_output_dir}")

    accuracy = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=utils.CLASSES) # Usa CLASSES do utils
    
    print(f"\nRelatório Final para {MODEL_NAME}----------------------------")
    print(f"Acurácia no Teste: {accuracy:.4f}")
    print(report_str)
    
    report_dict = classification_report(y_test, y_pred, target_names=utils.CLASSES, output_dict=True) # Usa CLASSES do utils
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
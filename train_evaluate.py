import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping

import config
from dataloader import load_and_prepare_data
from models.neural_networks import create_mlp, create_cnn
from utils import plot_learning_curves, plot_confusion_matrix, plot_roc_curves
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

def train_and_evaluate():
    """Carrega os melhores hiperparâmetros, treina os modelos finais e gera uma análise detalhada."""
    print("Treinamento Final para Predição de Morte por Chagas")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_data()

    try:
        with open(config.BEST_PARAMS_FILE, 'r') as f:
            best_params = json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo de melhores parâmetros '{config.BEST_PARAMS_FILE}' não encontrado. Execute 'optimize.py' primeiro.")
        return

    my_manual_class_weights = {
        0: 1.0,  
        1: 3.0 
    }
    class_weights_dict = my_manual_class_weights
    print(f"Pesos de classe MANUAIS aplicados para MLP/CNN: {class_weights_dict}")

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight_value = neg_count / pos_count
    print(f"scale_pos_weight para XGBoost/LightGBM: {scale_pos_weight_value:.2f}")

    final_results = []

    for model_name, params in best_params.items():
        print(f"\nProcessando modelo: {model_name}-----------------------------------")
        
        model_output_dir = os.path.join(config.OUTPUT_DIR, model_name)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        model = None
        if model_name == 'RandomForest':
            model = RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1, **params)
            model.fit(X_train, y_train)
        
        elif model_name == 'XGBoost':
            model = XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='logloss', n_jobs=-1, scale_pos_weight=scale_pos_weight_value, **params)
            model.fit(X_train, y_train)

        elif model_name == 'LightGBM':
            model = LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, class_weight='balanced', objective='binary', **params)
            model.fit(X_train, y_train)

        elif model_name == 'SVM':
            model = SVC(random_state=config.RANDOM_STATE, probability=True, class_weight='balanced', **params)
            model.fit(X_train, y_train)
            
        elif model_name in ['MLP', 'CNN']:
            if model_name == 'MLP':
                model = create_mlp(optimizer_name=params.get('optimizer', 'adam'), 
                                   learning_rate=params.get('learning_rate', 0.001), 
                                   input_shape=config.NN_INPUT_SHAPE)
            else: # CNN
                model = create_cnn(optimizer_name=params.get('optimizer', 'adam'), 
                                   learning_rate=params.get('learning_rate', 0.001), 
                                   input_shape=config.NN_INPUT_SHAPE)
            
            model.compile(optimizer=model.optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=3, 
                verbose=1,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=params.get('epochs', config.NN_EPOCHS),
                batch_size=params.get('batch_size', config.NN_BATCH_SIZE),
                validation_data=(X_val, y_val),
                class_weight=class_weights_dict, # Usa os pesos MANUAIS aqui
                verbose=1,
                callbacks=[early_stopping]
            )
            plot_learning_curves(history, model_name, model_output_dir)

            history_path = os.path.join(model_output_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history.history, f, cls=NpEncoder, indent=4)
            print(f"Histórico de treinamento salvo em: {history_path}")

        if model is None:
            print(f"Erro: Modelo {model_name} não foi instanciado corretamente.")
            continue

        if model_name in ['MLP', 'CNN']:
            y_val_pred_probs = model.predict(X_val).flatten()
        else:
            y_val_pred_probs = model.predict_proba(X_val)[:, 1]
            
        best_threshold = utils.find_best_threshold(y_val, y_val_pred_probs)
        print(f"Melhor limiar encontrado: {best_threshold:.2f} (F1-Score no Val: {f1_score(y_val, (y_val_pred_probs >= best_threshold).astype(int)):.4f})")

        print("Realizando predições no conjunto de teste...")
        if model_name in ['MLP', 'CNN']:
            y_pred_probs = model.predict(X_test).flatten()
        else:
            y_pred_probs = model.predict_proba(X_test)[:, 1]

        y_pred = (y_pred_probs >= best_threshold).astype(int) 
        
        plot_confusion_matrix(y_test, y_pred, model_name, model_output_dir, CLASSES=utils.CLASSES)
        plot_roc_curves(y_test, y_pred_probs, model_name, model_output_dir, CLASSES=utils.CLASSES)

        accuracy = accuracy_score(y_test, y_pred)
        report_str = classification_report(y_test, y_pred, target_names=utils.CLASSES)
        print(f"Acurácia no Teste para {model_name}: {accuracy:.4f}")
        print(report_str)

        report_dict = classification_report(y_test, y_pred, target_names=utils.CLASSES, output_dict=True)
        report_json_path = os.path.join(model_output_dir, 'classification_report.json')
        report_txt_path = os.path.join(model_output_dir, 'classification_report.txt')
        with open(report_json_path, 'w') as f:
            json.dump(report_dict, f, indent=4)
        with open(report_txt_path, 'w') as f:
            f.write(f"Acurácia no Teste: {accuracy:.4f}\n\n")
            f.write(report_str)
        print(f"Relatório de classificação salvo em: {model_output_dir}")
        
        final_results.append({'Modelo': model_name, 'Acurácia Teste': accuracy})

    print("Pipeline concluída!")

if __name__ == '__main__':
    train_and_evaluate()
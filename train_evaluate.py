import os
import json
import numpy as np
import pandas as pd
# from tensorflow.keras.utils import to_categorical # Não necessário para classificação binária se o alvo for 0 ou 1
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.callbacks import EarlyStopping

import config
from dataloader import load_and_prepare_data
from models.neural_networks import create_mlp, create_cnn # Assumindo que essas funções existem
from utils import plot_learning_curves, plot_confusion_matrix, plot_roc_curves # Removed CLASSES from import
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

def train_and_evaluate():
    """Carrega os melhores hiperparâmetros, treina os modelos finais e gera uma análise detalhada."""
    print("Treinamento Final para Predição de Morte por Chagas")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_data()

    # melhores hiperparâmetros
    try:
        with open(config.BEST_PARAMS_FILE, 'r') as f:
            best_params = json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo de melhores parâmetros '{config.BEST_PARAMS_FILE}' não encontrado. Execute 'optimize.py' primeiro.")
        return

    # balanceamento das classes usando class_weights
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))
    
    final_results = []

    for model_name, params in best_params.items():
        print(f"\nProcessando modelo: {model_name}-----------------------------------")
        
        model_output_dir = os.path.join(config.OUTPUT_DIR, model_name)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        if model_name == 'RandomForest':
            model = RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1, **params)
            model.fit(X_train, y_train)
            
            # Predições para otimização de limiar no conjunto de validação
            y_val_pred_probs = model.predict_proba(X_val)[:, 1]
            best_threshold = utils.find_best_threshold(y_val, y_val_pred_probs) # Encontra limiar no val
        
        elif model_name == 'XGBoost':
            model = XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='logloss', n_jobs=-1, **params)
            model.fit(X_train, y_train)
            
            # Predições para otimização de limiar no conjunto de validação
            y_val_pred_probs = model.predict_proba(X_val)[:, 1]
            best_threshold = utils.find_best_threshold(y_val, y_val_pred_probs) # Encontra limiar no val
            
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
                class_weight=class_weights_dict,
                verbose=1,
                callbacks=[early_stopping]
            )
            plot_learning_curves(history, model_name, model_output_dir)

            history_path = os.path.join(model_output_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history.history, f, cls=NpEncoder, indent=4)
            print(f"Histórico de treinamento salvo em: {history_path}")

            # Predições para otimização de limiar no conjunto de validação
            y_val_pred_probs = model.predict(X_val).flatten()
            best_threshold = utils.find_best_threshold(y_val, y_val_pred_probs) # Encontra limiar no val
        
        # avaliação no conjunto de teste
        print("Realizando predições no conjunto de teste...")
        if model_name in ['MLP', 'CNN']:
            y_pred_probs = model.predict(X_test).flatten()
        else:
            y_pred_probs = model.predict_proba(X_test)[:, 1]

        # APLICA O MELHOR LIMIAR ENCONTRADO NO CONJUNTO DE VALIDAÇÃO PARA O CONJUNTO DE TESTE
        y_pred = (y_pred_probs >= best_threshold).astype(int) 
        
        # ... (Restante do código de avaliação e salvamento de relatórios permanece igual) ...

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
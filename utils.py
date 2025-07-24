import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os
import numpy as np
from sklearn.metrics import f1_score

# Classes para o problema de Chagas
CLASSES = ['Não-Óbito', 'Óbito'] # Alterado para as classes binárias do problema

def evaluate_and_compare_models(results, X_test, y_test):
    """
    Avalia os melhores modelos encontrados e gera um relatório comparativo.
    Esta função não está sendo usada na pipeline atual, mas é mantida por completude.
    """
    print("\nAvaliação Final no Conjunto de Teste -----------")
    
    summary = []
    
    for model_name, result in results.items():
        best_model = result['best_estimator']
        
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        # report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True) # removido para evitar erro se CLASSES for importado de outro lugar
        
        print(f"\n--- {model_name} ---")
        print(f"Acurácia no Teste: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASSES))
        
        summary.append({
            'Modelo': model_name,
            'Acurácia Teste': accuracy,
            'Melhores Parâmetros': result['best_params']
        })
        
    summary_df = pd.DataFrame(summary)
    print("\nResumo Comparativo dos Modelos ----------")
    print(summary_df.to_string())
    
    # Certifique-se de que o diretório de saída existe
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    summary_df.to_csv(f"{config.OUTPUT_DIR}resumo_comparativo.csv", index=False)
    print(f"\nRelatório comparativo salvo em: {config.OUTPUT_DIR}resumo_comparativo.csv")


def plot_learning_curves(history, model_name, save_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title(f'Curva de Acurácia - {model_name}')
    plt.xlabel('Época'); plt.ylabel('Acurácia'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treino')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title(f'Curva de Perda - {model_name}')
    plt.xlabel('Época'); plt.ylabel('Perda'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir, CLASSES): # Adicionado CLASSES como parâmetro
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Classe Prevista'); plt.ylabel('Classe Verdadeira')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curves(y_true, y_pred_probs, model_name, save_dir, CLASSES): # Simplificado para binário
    # y_true deve ser 0s e 1s, y_pred_probs deve ser a probabilidade da classe positiva (1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos'); plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - {model_name}'); plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
def find_best_threshold(y_true, y_pred_probs, classes_to_consider=None):
    """
    Encontra o limiar de classificação que maximiza o F1-Score.
    y_true: Rótulos verdadeiros.
    y_pred_probs: Probabilidades preditas para a classe positiva.
    classes_to_consider: Não usado para classificação binária.
    Retorna: O melhor limiar.
    """
    thresholds = np.linspace(0, 1, 100) 
    best_f1 = -1
    best_threshold = 0.5 
    
    for th in thresholds:
        y_pred = (y_pred_probs >= th).astype(int)
        # Calcula F1-score para a classe positiva (Óbito)
        # zero_division=0 evita warnings e trata casos onde não há predições para uma classe
        current_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0) 
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = th
            
    print(f"Melhor limiar encontrado: {best_threshold:.2f} com F1-Score: {best_f1:.4f}")
    return best_threshold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, recall_score, make_scorer
from sklearn.inspection import permutation_importance
import os
import joblib
import json
from upsetplot import from_contents, plot

from utils import gmean_scorer
import config

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = "BuPu"


def plot_feature_selection_scores(feature_names, selector, top_n=20, save_path=None):
    """
    Plots the scores of the most important features identified by
    the selector (e.g., SelectKBest with ANOVA F-test).
    """
    scores = selector.scores_
    
    df_scores = pd.DataFrame({
        'Feature': feature_names,
        'F-Score': scores
    }).sort_values(by='F-Score', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='F-Score', y='Feature', data=df_scores, palette=PALETTE)
    
    plt.title(f'Características Mais Relevantes (ANOVA F-test)', fontsize=16)
    plt.xlabel('F-Score', fontsize=12)
    plt.ylabel('Característica', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico de seleção de features salvo em: {save_path}")
    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """
    Creates a bar chart comparing model performance based on key metrics
    (G-Mean, Sensitivity, Specificity).
    """
    df_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    PALETTE = ["#2e8b57", "#90ee90", "#ffcc80"] 
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette=PALETTE)
    
    plt.title('Desempenho dos Modelos no Conjunto de Teste', fontsize=16)
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Pontuação', fontsize=12)
    plt.xticks(rotation=0, ha='right')
    plt.ylim(0, 1.0)
    plt.legend(title='Métrica')
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico de comparação de modelos salvo em: {save_path}")
    plt.show()


def plot_roc_curves(trained_pipelines, X_test, y_test, save_path=None):
    """
    Plots the ROC curves for multiple models on a single chart,
    using distinct colors and line styles for clarity.
    """
    plt.figure(figsize=(10, 8))

    colors = sns.color_palette("Set1", n_colors=len(trained_pipelines))
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]
    model_styles = {name: (color, style) for name, color, style in zip(trained_pipelines.keys(), colors, linestyles)}
    
    for model_name, pipeline in trained_pipelines.items():
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Usa a cor e o estilo definidos
        color, linestyle = model_styles[model_name]
        plt.plot(fpr, tpr, color=color, linestyle=linestyle, lw=2,
                 label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
    plt.title('Curvas ROC Comparativas', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico de Curvas ROC salvo em: {save_path}")
    plt.show()
    
def plot_roc_curves_faceted(trained_pipelines, X_test, y_test, save_path=None):
    """
    Plots the ROC curve for each model in a separate subplot (facet).
    """
    n_models = len(trained_pipelines)
    n_cols = 2
    n_rows = (n_models + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten() 
    colors = sns.color_palette("Set1", n_colors=n_models)
    
    for i, (model_name, pipeline) in enumerate(trained_pipelines.items()):
        ax = axes[i]
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        
        ax.set_title(model_name)
        ax.set_xlabel('Taxa de Falsos Positivos')
        ax.set_ylabel('Taxa de Verdadeiros Positivos')
        ax.legend(loc="lower right")
        ax.grid(True)
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle('Curvas ROC Comparativas (Visão Individual)', fontsize=18, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico de Curvas ROC em facetas salvo em: {save_path}")
    plt.show()

def plot_permutation_importance(classifier, X_test_transformed, y_test, feature_names, top_n=15, save_path=None):
    """
    Calculates and plots the permutation importance for the best model.
    Note: It receives the CLASSIFIER itself (not the full pipeline) and
    the ALREADY TRANSFORMED data.
    """
    print("Calculando importância por permutação...")
    result = permutation_importance(
        classifier, X_test_transformed, y_test, 
        n_repeats=10, 
        random_state=42, 
        n_jobs=-1,
        scoring=gmean_scorer
    )
    
    sorted_idx = result.importances_mean.argsort()

    df_importance = pd.DataFrame({
        'Feature': np.array(feature_names)[sorted_idx][-top_n:][::-1],
        'Importance': result.importances_mean[sorted_idx][-top_n:][::-1]
    })
    
    plt.figure(figsize=(10, 8))
    
    sns.barplot(x='Importance', y='Feature', data=df_importance, palette="Greens_r")
    
    plt.title(f'Características Mais Importantes (Permutação)', fontsize=16)
    plt.xlabel('Redução Média na Performance (G-Mean)', fontsize=12)
    plt.ylabel('Característica', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico de importância por permutação salvo em: {save_path}")
    plt.show()

def generate_all_plots_from_saved_artifacts():
    """
    Main function to load saved artifacts and generate all plots.
    """
    RESULTS_PATH = config.RESULTS_PATH
    print(f"Lendo artefatos do diretório: {RESULTS_PATH}")

    # 1. Carregar os dados de teste e nomes das features
    try:
        X_test = np.load(os.path.join(RESULTS_PATH, 'X_test.npy'))
        y_test = np.load(os.path.join(RESULTS_PATH, 'y_test.npy'))
        with open(os.path.join(RESULTS_PATH, 'feature_names.json'), 'r') as f:
            feature_names = json.load(f)
    except FileNotFoundError:
        print("Erro: Arquivos de artefatos (X_test.npy, y_test.npy, etc.) não encontrados.")
        print("Por favor, rode o script 'main.py' primeiro para gerar os resultados.")
        return
    
    models_to_compare_upset = ['MLPClassifier', 'XGBoost', 'LightGBM']
    # 2. Coletar os resultados e pipelines de cada modelo
    trained_pipelines = {}
    all_results = []
    
    model_folders = [f for f in os.listdir(RESULTS_PATH) if os.path.isdir(os.path.join(RESULTS_PATH, f))]

    for model_name in model_folders:
        model_path = os.path.join(RESULTS_PATH, model_name)
        pipeline_path = os.path.join(model_path, 'pipeline.joblib')
        summary_path = os.path.join(model_path, 'summary.json')

        if os.path.exists(pipeline_path) and os.path.exists(summary_path):
            print(f"Carregando resultados para o modelo: {model_name}")
            
            pipeline = joblib.load(pipeline_path)
            trained_pipelines[model_name] = pipeline
            
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            report = summary['classification_report_dict']
            all_results.append({
                'Model': model_name,
                'G-Mean': np.sqrt(float(report['1']['recall']) * float(report['0']['recall'])),
                'Sensibilidade': float(report['1']['recall']),
                'Especificidade': float(report['0']['recall'])
            })

    if not all_results:
        print("Nenhum resultado de modelo válido encontrado para gerar gráficos.")
        return
        
    # 3. Gerar os gráficos de comparação
    print("\nGerando gráficos finais de comparação de modelos...")
    results_df = pd.DataFrame(all_results)
    
    plot_model_comparison(results_df, save_path=os.path.join(RESULTS_PATH, 'model_comparison.png'))
    
    plot_roc_curves_faceted(
        trained_pipelines,
        X_test,
        y_test,
        save_path=os.path.join(RESULTS_PATH, 'roc_curves.png')
    )
    
    # 4. Gerar o gráfico de importância para o melhor modelo (MLP)
    best_model_name = 'MLPClassifier'
    best_model_pipeline = trained_pipelines.get(best_model_name)
    if best_model_pipeline:
        # Pega a máscara de features do seletor treinado dentro da pipeline
        mask = best_model_pipeline.named_steps['select'].get_support()
        # Filtra os nomes para obter apenas os que o modelo usou
        selected_feature_names = np.array(feature_names)[mask].tolist()
        
        # Transforma os dados de teste usando a pipeline (scaler + select)
        X_test_transformed = best_model_pipeline.named_steps['scaler'].transform(X_test)
        X_test_transformed = best_model_pipeline.named_steps['select'].transform(X_test_transformed)

        plot_permutation_importance(
            classifier=best_model_pipeline.named_steps['clf'],
            X_test_transformed=X_test_transformed,           
            y_test=y_test,
            feature_names=selected_feature_names,
            top_n=15,
            save_path=os.path.join(RESULTS_PATH, f'{best_model_name}_permutation_importance.png')
        )

if __name__ == "__main__":
    generate_all_plots_from_saved_artifacts()
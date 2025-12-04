import warnings
import os
import numpy as np
import json
import joblib
import matplotlib
import gc # Garbage Collector para limpar memória entre loops
matplotlib.use('Agg')

from src import dataloader, optimization, train, evaluation, config, plots, utils
from sklearn.base import clone

warnings.filterwarnings('ignore')

# Define o diretório raiz fixo para evitar o aninhamento (results/3anos/5anos...)
BASE_RESULTS_DIR = 'results'

def run_pipeline_for_horizon(horizon_years):
    """
    Executa todo o pipeline para um horizonte específico.
    """
    # 1. Construção Correta do Caminho
    # Define a pasta específica para este horizonte
    horizon_results_path = os.path.join(BASE_RESULTS_DIR, f'{horizon_years}anos')
    
    # Atualiza o config.RESULTS_PATH para que as funções de plot/utils salvem no lugar certo
    # Isso afeta apenas a execução atual, pois na próxima chamada recalculamos com BASE_RESULTS_DIR
    config.RESULTS_PATH = horizon_results_path
    
    # Garante que a pasta existe
    os.makedirs(horizon_results_path, exist_ok=True)

    print(f"----------->> HORIZONTE: {horizon_years} ANOS <<-----------")
    print(f"Salvando resultados em: {horizon_results_path}")

    # 2. Verificar Dataset
    dataset_name = f'dados/dataset_chagas_{horizon_years}anos.csv'
    if not os.path.exists(dataset_name):
        # Tenta procurar na pasta de análise se não estiver na raiz
        alt_path = os.path.join('analise de sensibilidade', dataset_name)
        if os.path.exists(alt_path):
            dataset_name = alt_path
        else:
            print(f"PULANDO: Arquivo {dataset_name} não encontrado.")
            return

    # 3. Carregar Dados
    try:
        X_train, X_test, y_train, y_test, feature_names = dataloader.load_and_split_data(
            path=dataset_name,
            target_col=config.TARGET_COLUMN,
            drop_cols=config.DROP_COLUMNS,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )
    except Exception as e:
        print(f"Erro crítico ao carregar dados para {horizon_years} anos: {e}")
        return

    # 4. Calcular Pesos (para desbalanceamento)
    sample_weights_train = dataloader.get_sample_weights(y_train)

    # 5. Loop de Modelos
    for model_config in config.MODELS_CONFIG:
        model_name = model_config['name']
        print(f"\n--- Processando: {model_name} ({horizon_years} anos) ---")
        
        # Pasta específica do modelo: results/5anos/XGBoost
        model_dir = os.path.join(horizon_results_path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Parâmetros de ajuste (sample_weight)
        fit_params = {}
        if model_name in ['XGBoost', 'LightGBM', 'GradientBoosting', 'RandomForest']:
             # Alguns sklearn classifiers usam 'sample_weight' diretamente no fit, 
             # mas via pipeline passamos como prefixo no optimization.py.
             # Aqui ajustamos para o dicionário que suas funções esperam.
             fit_params = {'clf__sample_weight': sample_weights_train}
        
        # 5.1 Otimização de Hiperparâmetros
        try:
            best_params, best_k, selected_features = optimization.find_best_model_params(
                X_train=X_train,
                y_train=y_train,
                model_config=model_config,
                feature_names=feature_names,
                fit_params=fit_params
            )
        except Exception as e:
            print(f"Erro na otimização do {model_name}: {e}")
            continue
        
        # 5.2 Treino do Modelo Final
        # O train.py deve usar clone() para garantir um modelo limpo
        final_pipeline = train.train_final_model(
            X_train=X_train,
            y_train=y_train,
            model_config=model_config,
            best_params=best_params,
            best_k=best_k,
            fit_params=fit_params
        )

        # Salvar Pipeline
        joblib.dump(final_pipeline, os.path.join(model_dir, 'pipeline.joblib'))
        
        # 5.3 Encontrar Threshold Ótimo
        optimal_threshold = evaluation.find_threshold_with_cv(
            pipeline=final_pipeline,
            X_train=X_train,
            y_train=y_train,
            fit_params=fit_params
        )
        
        # 5.4 Avaliação no Teste
        class_report_dict, class_report_str = evaluation.evaluate_on_test_set(
            pipeline=final_pipeline,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            optimal_threshold=optimal_threshold,
            model_results_path=model_dir
        )
        
        # 5.5 Salvar Relatórios
        utils.save_model_summary(
            model_name=model_name,
            best_params=best_params,
            best_k=best_k,
            optimal_threshold=optimal_threshold,
            class_report_dict=class_report_dict,
            selected_features=selected_features,
            model_results_path=model_dir
        )
        
        utils.save_classification_report(class_report_str, model_dir)
        
        # 5.6 Curva de Aprendizado (apenas para MLP)
        if model_name == 'MLPClassifier':
            try:
                train_loss, val_loss = train.train_mlp_and_get_history(
                    X_train=X_train, y_train=y_train,
                    model_config=model_config, best_params=best_params, best_k=best_k
                )
                plots.plot_learning_curve(train_loss, val_loss, save_path=os.path.join(model_dir, 'learning_curve.png'))
            except Exception as e:
                print(f"Aviso: Não foi possível gerar curva de aprendizado para MLP: {e}")

        # Limpeza de memória após cada modelo
        del final_pipeline
        gc.collect()

    # 6. Salvar Artefatos Globais do Horizonte (para plotagem comparativa posterior)
    print(f"\nSalvando artefatos globais para {horizon_years} anos...")
    np.save(os.path.join(horizon_results_path, 'X_test.npy'), X_test)
    np.save(os.path.join(horizon_results_path, 'y_test.npy'), y_test)
    with open(os.path.join(horizon_results_path, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    # 7. Gerar Gráficos Comparativos (ROC, etc) para este horizonte
    print(f"Gerando gráficos comparativos para o horizonte {horizon_years} anos...")
    try:
        # A função usa config.RESULTS_PATH, que já atualizamos no início desta função
        plots.generate_all_plots_from_saved_artifacts()
    except Exception as e:
        print(f"Erro ao gerar gráficos comparativos: {e}")

    print(f"--- Finalizado horizonte {horizon_years} anos ---\n")

def main():
    # Lista de horizontes para processar
    horizons = [3, 5, 7, 10]
    
    print(f"Iniciando execução para horizontes: {horizons}")
    print(f"Diretório base de resultados: {os.path.abspath(BASE_RESULTS_DIR)}")
    
    for h in horizons:
        run_pipeline_for_horizon(h)
        # Limpeza profunda entre horizontes
        gc.collect()

if __name__ == "__main__":
    main()
import warnings
from src import dataloader, optimization, train, evaluation, config, plots, utils
import matplotlib
import joblib
from sklearn.preprocessing import StandardScaler
matplotlib.use('Agg')
import os
import numpy as np
import json

warnings.filterwarnings('ignore')

def main():
    """
    The main function that orchestrates the entire machine learning pipeline.
    """
    # Garante que o diretório principal de resultados seja criado no início
    print(f"Garantindo que o diretório de resultados exista: {config.RESULTS_PATH}")
    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    # 1. Carregar e Dividir os Dados
    X_train, X_test, y_train, y_test, feature_names = dataloader.load_and_split_data(
        path=config.DATA_PATH,
        target_col=config.TARGET_COLUMN,
        drop_cols=config.DROP_COLUMNS,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    print("\nGerando gráfico de pontuação das características (ANOVA F-test)...")
    temp_scaler = StandardScaler()
    X_train_scaled = temp_scaler.fit_transform(X_train)
    temp_selector = config.SELECTOR.fit(X_train_scaled, y_train)
    plots.plot_feature_selection_scores(
        feature_names=feature_names,
        selector=temp_selector,
        top_n=20,
        save_path=os.path.join(config.RESULTS_PATH, 'anova_f_scores.png')
    )

    # 2. Calcular Pesos para Desbalanceamento
    sample_weights_train = dataloader.get_sample_weights(y_train)

    # 3. Loop através dos modelos configurados
    for model_config in config.MODELS_CONFIG:
        model_name = model_config['name']
        print(f"\n{'='*20} Processando Modelo: {model_name} {'='*20}")
        
        model_results_path = os.path.join(config.RESULTS_PATH, model_name)
        os.makedirs(model_results_path, exist_ok=True)

        fit_params = {}
        if model_name in ['XGBoost', 'LightGBM']:
              fit_params = {'clf__sample_weight': sample_weights_train}
        
        # 3.1. Otimização de Hiperparâmetros
        # AGORA a otimização acontece ANTES do bloco que usa seus resultados
        best_params, best_k, selected_features = optimization.find_best_model_params(
            X_train=X_train,
            y_train=y_train,
            model_config=model_config,
            feature_names=feature_names,
            fit_params=fit_params
        )
        
        # 3.2. Treino do Modelo Final
        final_pipeline = train.train_final_model(
            X_train=X_train,
            y_train=y_train,
            model_config=model_config,
            best_params=best_params,
            best_k=best_k,
            fit_params=fit_params
        )

        joblib.dump(final_pipeline, os.path.join(model_results_path, 'pipeline.joblib'))
        print(f"Pipeline do modelo {model_name} salvo em: {model_results_path}")
        
        # 3.3. Encontrar o Threshold Ótimo
        optimal_threshold = evaluation.find_threshold_with_cv(
            pipeline=final_pipeline,
            X_train=X_train,
            y_train=y_train,
            fit_params=fit_params
        )
        
        # 3.4. Avaliação no Conjunto de Teste
        class_report_dict, class_report_str = evaluation.evaluate_on_test_set(
            pipeline=final_pipeline,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            optimal_threshold=optimal_threshold,
            model_results_path=model_results_path
        )
        
        utils.save_model_summary(
            model_name=model_name,
            best_params=best_params,
            best_k=best_k,
            optimal_threshold=optimal_threshold,
            class_report_dict=class_report_dict,
            selected_features=selected_features,
            model_results_path=model_results_path
        )
        
        utils.save_classification_report(
            class_report_str=class_report_str,
            model_results_path=model_results_path
        )
        
        if model_name == 'MLPClassifier':
            train_loss, val_loss = train.train_mlp_and_get_history(
                X_train=X_train,
                y_train=y_train,
                model_config=model_config,
                best_params=best_params,
                best_k=best_k
            )
            plots.plot_learning_curve(
                train_loss, 
                val_loss, 
                save_path=os.path.join(model_results_path, 'learning_curve.png')
            )
        
        # 3.5. Validação Cruzada para robustez
        evaluation.perform_cross_validation(
            pipeline=final_pipeline,
            X_train=X_train,
            y_train=y_train,
            fit_params=fit_params
        )

    print("\nSalvando artefatos de dados para regeneração de gráficos...")
    np.save(os.path.join(config.RESULTS_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(config.RESULTS_PATH, 'y_test.npy'), y_test)
    with open(os.path.join(config.RESULTS_PATH, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    print("Artefatos de dados salvos com sucesso.")


if __name__ == "__main__":
    main()
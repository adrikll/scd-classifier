import warnings
from src import dataloader, optimization, train, evaluation
import src.config as config

warnings.filterwarnings('ignore')

def main():
    """
    Função principal que orquestra toda a pipeline de machine learning.
    """
    # 1. Carregar e Dividir os Dados
    X_train, X_test, y_train, y_test = dataloader.load_and_split_data(
        path=config.DATA_PATH,
        target_col=config.TARGET_COLUMN,
        drop_cols=config.DROP_COLUMNS,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    # 2. Calcular Pesos para Desbalanceamento
    sample_weights_train = dataloader.get_sample_weights(y_train)

    # 3. Loop através dos modelos configurados
    for model_config in config.MODELS_CONFIG:
        model_name = model_config['name']
        print(f"\n{'='*20} Processando Modelo: {model_name} {'='*20}")

        # Definir parâmetros de fit específicos para cada modelo, se necessário
        fit_params = {}
        if model_name in ['GradientBoosting', 'XGBoost', 'LightGBM']:
             fit_params = {'clf__sample_weight': sample_weights_train}
        
        # 3.1. Otimização de Hiperparâmetros
        best_params, best_k = optimization.find_best_model_params(
            X_train=X_train,
            y_train=y_train,
            model_config=model_config,
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

        # 3.3. Avaliação no Conjunto de Teste
        evaluation.evaluate_on_test_set(
            pipeline=final_pipeline,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name
        )
        
        # 3.4. Validação Cruzada para robustez
        evaluation.perform_cross_validation(
            pipeline=final_pipeline,
            X_train=X_train,
            y_train=y_train,
            fit_params=fit_params
        )

if __name__ == "__main__":
    main()
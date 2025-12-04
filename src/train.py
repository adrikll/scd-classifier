from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import clone  # <--- Importação essencial adicionada
from . import config
import numpy as np

def train_final_model(X_train, y_train, model_config, best_params, best_k, fit_params={}):
    """Trains the final model on the entire training set using the best-found hyperparameters."""
    print("Treinando o modelo final com os melhores parâmetros...")
    
    # CLONE: Cria uma cópia limpa do estimador para evitar contaminação entre loops (horizontes)
    final_estimator = clone(model_config['estimator'])
    final_estimator.set_params(**best_params)
    
    final_selector = config.SELECTOR
    if best_k is not None:
        final_selector.set_params(k=best_k)

    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('select', final_selector),
        (config.MODEL_PREFIX, final_estimator)
    ])
    
    final_pipeline.fit(X_train, y_train, **fit_params)
    print("Modelo final treinado com sucesso.")
    
    return final_pipeline

def train_mlp_and_get_history(X_train, y_train, model_config, best_params, best_k):
    """
    Trains the final MLP model on a subset of the training data while validating on another
    to generate learning curves (train loss vs. validation loss).
    """
    print("\nTreinando MLP final para capturar histórico de aprendizado...")

    # 1. Prepara o estimador (USANDO CLONE para garantir que comece do zero)
    final_estimator = clone(model_config['estimator'])
    final_estimator.set_params(**best_params)
    
    # Adiciona warm_start=True APENAS nesta instância clonada local
    final_estimator.set_params(warm_start=True, verbose=False)

    # 2. Prepara o pipeline de pré-processamento (scaler + select)
    final_selector = config.SELECTOR.set_params(k=best_k)
    preprocessing_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('select', final_selector)
    ])

    # 3. Transforma todo o conjunto de treino
    X_train_transformed = preprocessing_pipeline.fit_transform(X_train, y_train)

    # 4. Cria uma divisão interna de treino/validação para a curva de aprendizado
    # 25% dos dados de treino como um conjunto de validação "interno"
    X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
        X_train_transformed, y_train, test_size=0.25, stratify=y_train, random_state=config.RANDOM_STATE
    )

    # 5. Treina o modelo época por época e captura a perda
    train_loss = []
    val_loss = []
    
    # Define um número de épocas para o treinamento
    n_epochs = 150 
    
    # Precisamos chamar fit na primeira vez para inicializar os pesos
    # classes_ é necessário para partial_fit ou warm_start em alguns casos, 
    # mas com warm_start=True e fit repetido, o sklearn gerencia.
    
    for epoch in range(n_epochs):
        # Treina por uma época (warm_start mantém o estado anterior)
        final_estimator.fit(X_train_part, y_train_part)
        
        # Armazena a perda de treino
        train_loss.append(final_estimator.loss_)
        
        # Para a perda de validação, calculamos manualmente a Log Loss
        if hasattr(final_estimator, 'predict_proba'):
            y_val_pred_proba = final_estimator.predict_proba(X_val_part)
            from sklearn.metrics import log_loss
            # Tratamento para evitar erro se houver apenas 1 classe no batch (raro aqui)
            if len(np.unique(y_train)) > 1:
                val_loss.append(log_loss(y_val_part, y_val_pred_proba))
            else:
                val_loss.append(0)
        else:
            val_loss.append(0)

    print("Histórico de aprendizado capturado.")
    return train_loss, val_loss
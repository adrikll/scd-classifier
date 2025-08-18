import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def load_and_split_data(path, target_col, drop_cols, test_size, random_state):
    """Carrega os dados, separa features/alvo e divide em treino/teste."""
    print(f"Carregando dados de: {path}")
    df = pd.read_excel(path)
    
    y = df[target_col].values
    X = df.drop(columns=drop_cols).values
    
    print(f"Shape das features (X): {X.shape}")
    print(f"Shape do alvo (y): {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print("Dados divididos em conjuntos de treino e teste.")
    return X_train, X_test, y_train, y_test

def get_sample_weights(y_train):
    """Calcula os pesos das classes para lidar com desbalanceamento."""
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    sample_weights = np.where(y_train == 1, class_weights_dict[1], class_weights_dict[0])
    print("Pesos de amostra calculados para o treino.")
    return sample_weights
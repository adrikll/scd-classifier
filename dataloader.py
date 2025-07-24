import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config

def load_and_prepare_data():
    """
    Carrega os dados de Chagas, exclui 'patient_id', 'Obito_MS', 'TimeObito_MS' das features,
    separa a variável alvo 'Obito_MS_FU_5years', e divide em conjuntos de treino, validação e teste.
    """
    print("Carregando dataset base_chagas.csv...")
    df = pd.read_excel(config.BASE_FILE)

    target_column_name = 'Obito_MS_FU-5 years'
    
    columns_to_exclude = ['ID', 'Name', 'FE', 'Filename', 'Age', 'Obito_MS', 'Time', 'Date Holter', 'Sex', 'Nat',	'Event (FU-5 years)', 'Rassi Score', 'Rassi Points', target_column_name]
    
    y = df[target_column_name].values

    X = df.drop(columns=columns_to_exclude, axis=1).values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=config.VALIDATION_SIZE / (1 - config.VALIDATION_SIZE),
        random_state=config.RANDOM_STATE,
        stratify=y_temp
    )
    
    if X_train.ndim == 1:
        config.NN_INPUT_SHAPE = (1,)
    else:
        config.NN_INPUT_SHAPE = (X_train.shape[1],) 

    scaler = StandardScaler()

    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Número de features: {X_train_scaled.shape[1]}")
    print(f"Tamanho do treino: {len(X_train_scaled)}")
    print(f"Tamanho da validação: {len(X_val_scaled)}")
    print(f"Tamanho do teste: {len(X_test_scaled)}")
    print("Dados carregados, divididos e normalizados com sucesso.")

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)



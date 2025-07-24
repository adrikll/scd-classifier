import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
import config

def load_and_prepare_data():
    """
    Carrega os dados de Chagas, exclui colunas de identificação/clínicas/Rassi/outras das features,
    separa a variável alvo 'Obito_MS_FU-5 years', divide em conjuntos de treino, validação e teste.
    Aplica SMOTE ao conjunto de treino se configurado***.
    """
    print("Carregando dataset base_chagas.csv...")
    df = pd.read_excel(config.BASE_FILE)

    target_column_name = 'Obito_MS_FU-5 years' 
    
    columns_to_exclude = ['ID', 'Name', 'FE', 'Filename', 'Age', 'Obito_MS', 'Time', 'Date Holter', 'Sex', 'Nat', 'Event (FU-5 years)', 'Rassi Score', 'Rassi Points', target_column_name]
    
    y = df[target_column_name].values
    X = df.drop(columns=columns_to_exclude, axis=1).values

    #usa o 'stratify' para manter a proporção das classes.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    #divide o conjunto temporário em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=config.VALIDATION_SIZE / (1 - config.VALIDATION_SIZE),
        random_state=config.RANDOM_STATE,
        stratify=y_temp
    )
    
    if config.APPLY_SMOTE:
        print("Aplicando SMOTE no conjunto de treino...")
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Tamanho do treino ANTES do SMOTE: {len(X_train)}")
        print(f"Distribuição de classes ANTES do SMOTE (treino): {pd.Series(y_train).value_counts()}")
        print(f"Tamanho do treino DEPOIS do SMOTE: {len(X_train_resampled)}")
        print(f"Distribuição de classes DEPOIS do SMOTE (treino): {pd.Series(y_train_resampled).value_counts()}")
        
        X_train = X_train_resampled
        y_train = y_train_resampled

    if X_train.ndim == 1:
        config.NN_INPUT_SHAPE = (1,)
    else:
        config.NN_INPUT_SHAPE = (X_train.shape[1],) 

    scaler = StandardScaler()

    #ajusta apenas no conjunto de treino (pode ser o resampled)
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Número de features: {X_train_scaled.shape[1]}")
    print(f"Tamanho do treino (final): {len(X_train_scaled)}")
    print(f"Tamanho da validação: {len(X_val_scaled)}")
    print(f"Tamanho do teste: {len(X_test_scaled)}")
    print("Dados carregados, divididos e normalizados com sucesso (com SMOTE se aplicado).")

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)
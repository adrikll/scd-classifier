import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer

def clean_categorical_features(df):
    """
    Limpa dados sujos (vírgulas), trata ordinais e aplica One-Hot.
    """
    # 1. CORREÇÃO DE VÍRGULAS (Decimais brasileiros)
    # TSH, AE diam, IMC e outras numéricas que vieram como texto
    numeric_candidates = ['TSH', 'AE diam.', 'IMC', 'FC', 'FC media'] 
    
    for col in numeric_candidates:
        if col in df.columns and df[col].dtype == 'O':
            # Troca vírgula por ponto e converte para float (força NaN se der erro)
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. NYHA (Ordinal) -> 1, 2, 3, 4
    if 'NYHA' in df.columns:
        map_nyha = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, '1': 1, '2': 2, '3': 3, '4': 4}
        # Converte apenas se for texto/mixed
        if df['NYHA'].dtype == 'O':
             df['NYHA'] = df['NYHA'].map(map_nyha)
        df['NYHA'] = pd.to_numeric(df['NYHA'], errors='coerce')

    # 3. Sexo (Binário) -> 0/1
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].replace({'M': 0, 'F': 1, 'Masculino': 0, 'Feminino': 1})
        df['Sex'] = pd.to_numeric(df['Sex'], errors='coerce')

    # 4. One-Hot Encoding (Distúrbios de Condução)
    # Definimos explicitamente quais colunas são nominais
    cols_to_encode = ['Dist Cond InterVent', 'Dist Cond AtrioVent', 'Dist Cond AtrioVent.1']
    
    # Filtra apenas as presentes
    cols_present = [c for c in cols_to_encode if c in df.columns]
    
    if cols_present:
        print(f"Aplicando One-Hot Encoding em: {cols_present}")
        # Converte para string para garantir que seja tratado como categoria
        # '3.0' vira categoria '3.0', '0.0' vira '0.0'
        for c in cols_present:
            df[c] = df[c].astype(str).replace('nan', np.nan)
            
        df = pd.get_dummies(df, columns=cols_present, drop_first=False, dummy_na=False)
        
        # Garante que as novas colunas sejam 0/1 (int)
        new_cols = [c for c in df.columns if any(x in c for x in cols_present)]
        for nc in new_cols:
            df[nc] = df[nc].astype(int)

    return df

def load_and_split_data(path, target_col, drop_cols, test_size, random_state):
    print(f"Carregando dados de: {path}")
    
    if path.endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    
    # 1. Remover colunas indesejadas (Exclusão Segura)
    # Removemos APENAS as que não são o target
    cols_to_drop = [c for c in drop_cols if c in df.columns and c != target_col]
    
    if cols_to_drop:
        print(f"Removendo {len(cols_to_drop)} colunas configuradas em DROP_COLUMNS...")
        df = df.drop(columns=cols_to_drop)
    
    # Verificação de Segurança: Data_MSC ainda está lá?
    bad_cols = ['Data_MSC', 'Tempo', 'Date Holter']
    remaining_bad = [c for c in bad_cols if c in df.columns]
    if remaining_bad:
        print(f"AVISO CRÍTICO: Colunas de vazamento detectadas! Removendo forçadamente: {remaining_bad}")
        df = df.drop(columns=remaining_bad)

    # 2. Limpeza e Tratamento (Vírgulas, One-Hot)
    df = clean_categorical_features(df)
    
    # 3. Separar X e y
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada! Verifique se o nome está correto no CSV.")
        
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])
    
    # 4. Imputação (Mediana)
    # Como limpamos as vírgulas antes, agora isso vai funcionar
    imputer = SimpleImputer(strategy='median')
    
    # Seleciona apenas colunas numéricas para evitar erros se sobrou algum texto perdido
    X_df_numeric = X_df.select_dtypes(include=[np.number])
    if len(X_df_numeric.columns) < len(X_df.columns):
        diff = set(X_df.columns) - set(X_df_numeric.columns)
        print(f"Aviso: As seguintes colunas não-numéricas foram descartadas antes do treino: {diff}")
    
    X = imputer.fit_transform(X_df_numeric)
    feature_names = X_df_numeric.columns.tolist()
    
    print(f"Shape final (X): {X.shape}")
    print(f"Shape final (y): {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_names

def get_sample_weights(y_train):
    if len(np.unique(y_train)) < 2: return np.ones(len(y_train))
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    d = dict(enumerate(class_weights))
    return np.where(y_train == 1, d[1], d[0])
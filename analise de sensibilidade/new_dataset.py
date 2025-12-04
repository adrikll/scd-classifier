import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def preprocess_smart_merge():

    # 1. Carregar Arquivos
    file_features = 'dados/chagas_all_features.xlsx'
    file_clinical = 'dados/Banco Chagas Geral-22-08-2025.xlsx'
    
    df_chagas = pd.read_excel(file_features)
    df_banco_geral = pd.read_excel(file_clinical)
    
    # 2. Normalização Básica (Datas e Nomes)
    # Converter para datetime
    df_chagas['Date Holter'] = pd.to_datetime(df_chagas['Date Holter'], errors='coerce')
    df_banco_geral['Data Holter'] = pd.to_datetime(df_banco_geral['Data Holter'], errors='coerce')
    df_banco_geral['Data Obito MSC'] = pd.to_datetime(df_banco_geral['Data Obito MSC'], errors='coerce')

    # Criar chaves de merge normalizadas (sem hora)
    df_chagas['Merge_Date'] = df_chagas['Date Holter'].dt.normalize()
    df_banco_geral['Merge_Date'] = df_banco_geral['Data Holter'].dt.normalize()
    
    # Padronizar nomes da chave principal (remover espaços extras se houver)
    df_banco_geral.rename(columns={'Nome do Paciente': 'Name'}, inplace=True)
    
    # === PASSO A: Lógica Especial para Data de MSC (Global por Paciente) ===
    print("Extraindo Data de MSC varrendo todo o histórico do paciente...")
    
    # Filtra apenas linhas que têm data de óbito preenchida
    df_obitos_validos = df_banco_geral.dropna(subset=['Data Obito MSC'])
    
    # Cria um dicionário {Nome_Paciente: Data_Obito}. 
    # Se houver duplicatas (mesmo paciente com data repetida), pegamos a primeira (first).
    # Isso assume que a data de morte é um fato único.
    mapa_msc = df_obitos_validos.groupby('Name')['Data Obito MSC'].first()
    
    print(f"Encontradas datas de óbito para {len(mapa_msc)} pacientes únicos no banco geral.")

    # === PASSO B: Cruzamento dos Dados Clínicos (Estrito por Data do Exame) ===
    print("Cruzando dados clínicos respeitando a data do exame...")
    
    cols_clinicas_originais = [
        'IMC ', 'HAS', 'DM2', 'Sincope', 'I R Crônica', 'Coronariopatia', 
        'Ins Cardiaca ', 'TSH', 'FC', 'Dist Cond InterVent ', 'Dist Cond AtrioVent ', 
        'ESV', 'EV', 'Area Elet inativa', 'Dist Cond AtrioVent.1', 
        'Disf Nodulo Sinusal', 'Fibri/Flutter Atrial', 'FC media', 
        'TVS', 'TVMNS.1', 'AE diam.', 'NYHA'
    ]
    
    # Filtrar apenas as colunas que existem
    cols_clinicas_validas = [c for c in cols_clinicas_originais if c in df_banco_geral.columns]
    
    # Renomear para remover espaços
    rename_map = {
        'IMC ': 'IMC', 'Ins Cardiaca ': 'Ins Cardiaca',
        'Dist Cond InterVent ': 'Dist Cond InterVent', 'Dist Cond AtrioVent ': 'Dist Cond AtrioVent'
    }
    
    # Preparar subset do banco geral para merge (apenas colunas clínicas + chaves)
    df_banco_subset = df_banco_geral[['Name', 'Merge_Date'] + cols_clinicas_validas].copy()
    df_banco_subset.rename(columns=rename_map, inplace=True)
    
    # Merge Left (Mantém os pacientes do estudo original)
    df_merged = pd.merge(
        df_chagas, 
        df_banco_subset, 
        on=['Name', 'Merge_Date'], 
        how='left'
    )

    # === PASSO C: Aplicação da Data de MSC e Cálculo de Tempo ===
    print("Consolidando as datas de óbito...")
    
    # 1. Cria a coluna Data_MSC vazia
    df_merged['Data_MSC'] = pd.NaT
    
    # 2. Identifica quem morreu no dataset de estudo (Obito_MS == 1)
    mask_obito_estudo = df_merged['Obito_MS'] == 1
    
    # 3. Mágica: Para esses pacientes, busca a data no mapa global (Passo A), não no merge local
    # O map() usa o Nome para buscar a data no dicionário que criamos
    datas_recuperadas = df_merged.loc[mask_obito_estudo, 'Name'].map(mapa_msc)
    
    # 4. Atribui os valores
    df_merged.loc[mask_obito_estudo, 'Data_MSC'] = datas_recuperadas

    # 5. Cálculo do Tempo (Anos)
    df_merged['Tempo'] = ((df_merged['Data_MSC'] - df_merged['Date Holter']).dt.days / 365.25).round(2)

    # --- RELATÓRIO DE SUCESSO ---
    total_obitos = df_merged['Obito_MS'].sum()
    total_datas = df_merged['Data_MSC'].notnull().sum()
    faltantes = total_obitos - total_datas
    
    print("-" * 40)
    print(f"RELATÓRIO DE EXTRAÇÃO (Lógica Melhorada):")
    print(f"Pacientes marcados como óbito (MS): {total_obitos}")
    print(f"Datas recuperadas com sucesso: {total_datas}")
    print(f"Ainda sem data (verificar grafia nomes): {faltantes}")
    
    if faltantes > 0:
        print("\nPacientes com óbito mas sem data encontrada:")
        print(df_merged[mask_obito_estudo & df_merged['Data_MSC'].isnull()]['Name'].unique())
    print("-" * 40)
    
    # === PASSO D: Reorganização Final ===
    print("Reorganizando colunas...")
    
    # Lista limpa das novas colunas clínicas
    new_clinical_clean = [rename_map.get(c, c) for c in cols_clinicas_validas]
    
    # Definir ordem: Primeiras 15 originais -> MSC/Tempo -> Clínicas -> Resto
    cols_orig = [c for c in df_chagas.columns if c not in ['Merge_Date']]
    
    # Evitar duplicação se alguma coluna já existia
    cols_to_insert = ['Data_MSC', 'Tempo'] + [c for c in new_clinical_clean if c in df_merged.columns]
    
    final_cols = cols_orig[:15] + cols_to_insert + cols_orig[15:]
    
    df_final = df_merged[final_cols]
    
    # Salvar
    output_filename = 'analise de sensibilidade/chagas_all_features_updated.xlsx'
    df_final.to_excel(output_filename, index=False)
    print(f"Arquivo salvo com sucesso: {output_filename}")

if __name__ == "__main__":
    preprocess_smart_merge()
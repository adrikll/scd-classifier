#gera um dataset unificado agora com as datas das MSC para posteriormente definirmos uma coluna Time que não se limita a 5

import pandas as pd
import numpy as np
import re
import os

print("Iniciando script de enriquecimento de dados...")

DIRETORIO_DADOS = 'dados'

ARQUIVO_BASE_XLSX = 'chagas_all_features.xlsx'
ARQUIVO_FONTE_XLSX = 'Banco Geral Chagas-19-02-2024.xlsx'

COLUNA_CHAVE_BASE = 'ID' 
COLUNA_CHAVE_FONTE = 'Prontuario'
COLUNA_OBSERVACOES = 'Observações'

REGEX_DATA = r'MSC em .*\((.*?)\)'

ARQUIVO_SAIDA = 'chagas_features_com_data_msc.xlsx'

caminho_base = os.path.join(DIRETORIO_DADOS, ARQUIVO_BASE_XLSX)
caminho_fonte = os.path.join(DIRETORIO_DADOS, ARQUIVO_FONTE_XLSX)
caminho_saida = os.path.join(DIRETORIO_DADOS, ARQUIVO_SAIDA)

try:
    df_base = pd.read_excel(caminho_base)
    print(f"Arquivo base '{caminho_base}' carregado. Total de pacientes: {len(df_base)}")
except FileNotFoundError:
    print(f"ERRO: Arquivo base '{caminho_base}' não encontrado.")
    exit()
except Exception as e:
    print(f"ERRO ao ler '{caminho_base}': {e}")
    exit()

try:
    df_fonte = pd.read_excel(caminho_fonte)
    print(f"Arquivo fonte '{caminho_fonte}' carregado. Total de registros: {len(df_fonte)}")
except FileNotFoundError:
    print(f"ERRO: Arquivo fonte '{caminho_fonte}' não encontrado.")
    exit()
except Exception as e:
    print(f"ERRO ao ler '{caminho_fonte}': {e}")
    exit()

print(f"\nProcessando '{ARQUIVO_FONTE_XLSX}' para extrair datas de MSC...")

if COLUNA_CHAVE_FONTE not in df_fonte.columns:
    print(f"ERRO: Coluna chave '{COLUNA_CHAVE_FONTE}' não encontrada em '{ARQUIVO_FONTE_XLSX}'.")
    exit()
if COLUNA_OBSERVACOES not in df_fonte.columns:
    print(f"ERRO: Coluna de observações '{COLUNA_OBSERVACOES}' não encontrada.")
    exit()

df_fonte[COLUNA_OBSERVACOES] = df_fonte[COLUNA_OBSERVACOES].astype(str)

df_msc = df_fonte[df_fonte[COLUNA_OBSERVACOES].str.contains("MSC em", case=False, na=False)].copy()
print(f"Encontrados {len(df_msc)} registros contendo 'MSC em'.")

df_msc['Data_MSC_Extraida'] = df_msc[COLUNA_OBSERVACOES].str.extract(REGEX_DATA)

df_msc_datas = df_msc.dropna(subset=['Data_MSC_Extraida'])
print(f"Encontradas {len(df_msc_datas)} datas de MSC extraídas com sucesso.")

mapa_datas_msc = df_msc_datas.drop_duplicates(subset=[COLUNA_CHAVE_FONTE])

mapa_datas_msc = mapa_datas_msc[[COLUNA_CHAVE_FONTE, 'Data_MSC_Extraida']]

print(f"Mapa final de datas criado com {len(mapa_datas_msc)} pacientes únicos com MSC.")

print(f"\nMesclando datas de MSC com '{ARQUIVO_BASE_XLSX}'...")

if COLUNA_CHAVE_BASE not in df_base.columns:
    print(f"ERRO: A coluna chave '{COLUNA_CHAVE_BASE}' não foi encontrada em '{ARQUIVO_BASE_XLSX}'.")
    exit()

df_enriquecido = pd.merge(
    df_base,
    mapa_datas_msc,
    left_on=COLUNA_CHAVE_BASE,
    right_on=COLUNA_CHAVE_FONTE,
    how='left'
)

if COLUNA_CHAVE_FONTE in df_enriquecido.columns:
    df_enriquecido = df_enriquecido.drop(columns=[COLUNA_CHAVE_FONTE])

print("Merge concluído.")
print(f"O dataset final tem {len(df_enriquecido)} linhas (deve ser igual ao original).")

try:
    df_enriquecido.to_excel(caminho_saida, index=False)
    print(f"\nSucesso! Novo dataset salvo como: '{caminho_saida}'")
    
    datas_adicionadas = df_enriquecido['Data_MSC_Extraida'].notna().sum()
    print(f"Total de datas de MSC adicionadas ao arquivo final: {datas_adicionadas}")

except Exception as e:
    print(f"\nERRO ao salvar o arquivo '{caminho_saida}': {e}")
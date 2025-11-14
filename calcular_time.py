import pandas as pd
import numpy as np
import os

print("Iniciando script para cálculo de tempo (follow-up) [VERSÃO CORRIGIDA]...")

# --- 1. CONFIGURAÇÕES ---

DIRETORIO_DADOS = 'dados'
ARQUIVO_ENTRADA = 'chagas_features_com_data_msc.xlsx'
ARQUIVO_SAIDA = 'chagas_final_com_time.xlsx'

# Nomes das colunas
COLUNA_HOLTER = 'Date Holter'
COLUNA_MSC = 'Data_MSC_Extraida'
COLUNA_TIME_ANTIGA = 'Time' # Coluna âncora
COLUNA_SAIDA_TIME = 'Time_Calculado_Anos'
COLUNA_OBITO_MS = 'Obito_MS' 
COLUNA_NOME = 'Name' # Adicionada para o diagnóstico

TEMPO_CENSOR_ANOS = 10.0

# --- 2. CARREGAR O DATASET ---

caminho_entrada = os.path.join(DIRETORIO_DADOS, ARQUIVO_ENTRADA)

try:
    df = pd.read_excel(caminho_entrada)
    print(f"Arquivo '{caminho_entrada}' carregado. ({len(df)} linhas)")
except FileNotFoundError:
    print(f"ERRO: Arquivo de entrada '{caminho_entrada}' não encontrado.")
    exit()
except Exception as e:
    print(f"ERRO ao ler '{caminho_entrada}': {e}"); exit()

# --- 3. PROCESSAMENTO DAS DATAS ---

print("Convertendo colunas de data (com dayfirst=True para ambas)...")

# --- ★★★ ESTA É A LINHA CORRIGIDA ★★★ ---
# Adiciona dayfirst=True para ler formatos DD/MM/AAAA corretamente
df[COLUNA_HOLTER] = pd.to_datetime(df[COLUNA_HOLTER], dayfirst=True, errors='coerce')
# --- ★★★ FIM DA CORREÇÃO ★★★ ---

df[COLUNA_MSC] = pd.to_datetime(df[COLUNA_MSC], dayfirst=True, errors='coerce')

# Verificação de Diagnóstico (Melhorada)
holter_nat = df[df[COLUNA_HOLTER].isna() & (df[COLUNA_OBITO_MS] == 1)]
msc_nat = df[df[COLUNA_MSC].isna() & (df[COLUNA_OBITO_MS] == 1)]

if not holter_nat.empty:
    print(f"AVISO: {len(holter_nat)} pacientes com 'Obito_MS == 1' têm 'Date Holter' inválida ou vazia.")
    print(f"       Nomes: {holter_nat[COLUNA_NOME].tolist()}")
if not msc_nat.empty:
    print(f"AVISO: {len(msc_nat)} pacientes com 'Obito_MS == 1' têm 'Data_MSC_Extraida' inválida ou vazia.")
    print(f"       Nomes: {msc_nat[COLUNA_NOME].tolist()}")

print("Datas convertidas com sucesso.")

# --- 4. CÁLCULO DA DURAÇÃO (TIME) ---

print(f"Calculando a duração em anos... (Censura em {TEMPO_CENSOR_ANOS} anos)")

delta_tempo_dias = (df[COLUNA_MSC] - df[COLUNA_HOLTER]).dt.days
duracao_em_anos = delta_tempo_dias / 365.25

df[COLUNA_SAIDA_TIME] = np.where(
    pd.notna(duracao_em_anos), 
    duracao_em_anos, 
    TEMPO_CENSOR_ANOS
)

# Arredonda a coluna final para duas casas decimais
df[COLUNA_SAIDA_TIME] = df[COLUNA_SAIDA_TIME].round(2)

# --- 5. VERIFICAÇÃO E SUMÁRIO ---

print("\n--- Verificação do Novo Tempo ---")
tempos_negativos = df[df[COLUNA_SAIDA_TIME] < 0]
if not tempos_negativos.empty:
    print(f"AVISO: {len(tempos_negativos)} pacientes têm tempo de follow-up negativo (MSC antes do Holter).")
    print(f"       Nomes: {tempos_negativos[COLUNA_NOME].tolist()}")
else:
    print("Verificação de tempo negativo: OK (Nenhum encontrado).")

# Verificação de Censura (Melhorada)
msc_pacientes = df[df[COLUNA_OBITO_MS] == 1]
msc_sem_tempo = msc_pacientes[msc_pacientes[COLUNA_SAIDA_TIME] == TEMPO_CENSOR_ANOS]

if not msc_sem_tempo.empty:
    print(f"AVISO: {len(msc_sem_tempo)} pacientes têm 'Obito_MS == 1', mas o tempo foi censurado em {TEMPO_CENSOR_ANOS} anos.")
    print("       Isto ocorreu porque a 'Date Holter' ou 'Data_MSC_Extraida' não pôde ser convertida.")
    print(f"       Nomes dos pacientes censurados: {msc_sem_tempo[COLUNA_NOME].tolist()}")
else:
    print("Verificação de 'Obito_MS vs Tempo': OK (Todos os óbitos têm tempo calculado).")

print("\nEstatísticas do tempo calculado (em anos):")
print(df[COLUNA_SAIDA_TIME].describe())

# --- 6. REORDENAR COLUNAS ---

print("\nReordenando colunas...")

all_cols = df.columns.tolist()
anchor_col = COLUNA_TIME_ANTIGA
cols_to_move = [COLUNA_MSC, COLUNA_SAIDA_TIME]

try:
    original_cols = [col for col in all_cols if col not in cols_to_move]
    anchor_index = original_cols.index(anchor_col)
    
    new_col_order = original_cols[:anchor_index + 1] + \
                    cols_to_move + \
                    original_cols[anchor_index + 1:]
    
    df = df[new_col_order]
    
    print(f"Colunas '{', '.join(cols_to_move)}' movidas para depois de '{anchor_col}'.")

except ValueError:
    print(f"AVISO: A coluna âncora '{anchor_col}' não foi encontrada. As colunas serão salvas na ordem padrão.")
except Exception as e:
    print(f"AVISO: Ocorreu um erro ao reordenar as colunas: {e}.")

# --- 7. SALVAR O RESULTADO ---

caminho_saida = os.path.join(DIRETORIO_DADOS, ARQUIVO_SAIDA)
try:
    df.to_excel(caminho_saida, index=False)
    print(f"\nSucesso! Novo dataset salvo como: '{caminho_saida}'")
    print("Este é o arquivo que você deve usar para a próxima etapa (gerar os datasets de limiar).")

except Exception as e:
    print(f"\nERRO ao salvar o arquivo '{caminho_saida}': {e}")
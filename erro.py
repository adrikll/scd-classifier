import pandas as pd
import numpy as np
import os

print("--- Iniciando Script de Verificação de Discrepância (5 Anos) ---")

# --- 1. CONFIGURAÇÕES ---
DIRETORIO_DADOS = 'dados'
ARQUIVO_ENTRADA = 'chagas_final_com_time.xlsx' # O arquivo com todos os dados

COLUNA_ALVO_ORIGINAL = 'Obito_MS_FU-5 years'
COLUNA_EVENTO = 'Obito_MS'
COLUNA_TEMPO_CALCULADO = 'Time_Calculado_Anos'
COLUNA_ALVO_NOVA = 'Target_FU_5_Anos' # Vamos recriar esta
LIMIAR_T = 5

caminho_entrada = os.path.join(DIRETORIO_DADOS, ARQUIVO_ENTRADA)

# --- 2. CARREGAR O DATASET ---
try:
    df = pd.read_excel(caminho_entrada)
    print(f"Arquivo '{caminho_entrada}' carregado. ({len(df)} linhas)")
except FileNotFoundError:
    print(f"ERRO: Arquivo de entrada '{caminho_entrada}' não encontrado.")
    exit()
except Exception as e:
    print(f"ERRO ao ler '{caminho_entrada}': {e}")
    exit()

# --- 3. RECRIAR A LÓGICA DO NOVO ALVO ---
# Classe 1 (Evento ANTES do limiar T)
condicao_positiva_nova = (df[COLUNA_EVENTO] == 1) & (df[COLUNA_TEMPO_CALCULADO] < LIMIAR_T)
df[COLUNA_ALVO_NOVA] = np.where(condicao_positiva_nova, 1, 0)
    
# --- 4. ENCONTRAR OS PACIENTES DISCREPANTES ---
# Pacientes que ERAM classe 1, mas AGORA são classe 0
filtro_discrepancia = (df[COLUNA_ALVO_ORIGINAL] == 1) & (df[COLUNA_ALVO_NOVA] == 0)
pacientes_discrepantes = df[filtro_discrepancia]
    
# --- 5. EXIBIR OS RESULTADOS ---
print(f"\nContagem no Alvo Original ({COLUNA_ALVO_ORIGINAL}):")
print(df[COLUNA_ALVO_ORIGINAL].value_counts())
    
print(f"\nContagem na Nova Lógica ({COLUNA_ALVO_NOVA}):")
print(df[COLUNA_ALVO_NOVA].value_counts())
    
if pacientes_discrepantes.empty:
    print("\nNenhuma discrepância encontrada.")
else:
    print(f"\n--- DISCREPÂNCIA ENCONTRADA ---")
    print(f"Encontrados {len(pacientes_discrepantes)} pacientes que eram '1' e agora são '0'.")
    print("Isto ocorre porque o tempo calculado (preciso) é >= 5.00 anos.")
        
    colunas_para_exibir = ['Name', COLUNA_ALVO_ORIGINAL, COLUNA_ALVO_NOVA, COLUNA_TEMPO_CALCULADO, COLUNA_EVENTO, 'Date Holter', 'Data_MSC_Extraida']
    print("\nPacientes Discrepantes:")
    print(pacientes_discrepantes[colunas_para_exibir])

print("\n--- FIM DO SCRIPT DE VERIFICAÇÃO ---")
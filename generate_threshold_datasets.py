import pandas as pd
import numpy as np
import os

print("Iniciando script para geração de datasets por limiar (Versão 2 - Lógica <= T)...")

# --- 1. CONFIGURAÇÕES ---

DIRETORIO_DADOS = 'dados'
ARQUIVO_ENTRADA = 'chagas_final_com_time.xlsx'

COLUNA_EVENTO = 'Obito_MS'
COLUNA_TEMPO = 'Time_Calculado_Anos' 
LIMIARES_ANOS = [3, 5, 7, 10]


caminho_entrada = os.path.join(DIRETORIO_DADOS, ARQUIVO_ENTRADA)

try:
    df_base = pd.read_excel(caminho_entrada)
    print(f"Arquivo '{caminho_entrada}' carregado. ({len(df_base)} linhas)")
except FileNotFoundError:
    print(f"ERRO: Arquivo de entrada '{caminho_entrada}' não encontrado.")
    exit()
except Exception as e:
    print(f"ERRO ao ler '{caminho_entrada}': {e}"); exit()

# --- 3. CRIAR TODAS AS COLUNAS-ALVO (LÓGICA <= T) ---

print("Criando colunas-alvo para cada limiar...")

df_processado = df_base.copy()
novas_colunas_alvo = []

for T in LIMIARES_ANOS:
    nova_coluna = f'Target_FU_{T}_Anos'
    novas_colunas_alvo.append(nova_coluna)
    
    # --- LÓGICA ALTERADA ---
    # Classe 1: Evento (Obito_MS=1) E tempo MENOR OU IGUAL (<=) ao limiar T
    condicao_positiva = (df_processado[COLUNA_EVENTO] == 1) & (df_processado[COLUNA_TEMPO] <= T)
    
    # Classe 0: Todos os outros
    # (Obito_MS=0) OU (Evento ocorreu DEPOIS (>) do limiar T)
    df_processado[nova_coluna] = np.where(condicao_positiva, 1, 0)
    # --- FIM DA ALTERAÇÃO ---
    
    print(f"\n  - Limiar {T} anos ({nova_coluna}):")
    print(f"    Classe 1 (Evento <= {T} anos): {df_processado[nova_coluna].sum()} pacientes")
    print(f"    Classe 0 (Outros):          {(df_processado[nova_coluna] == 0).sum()} pacientes")

# --- 4. VERIFICAÇÃO ESPECIAL PARA T=5 ---
if 5 in LIMIARES_ANOS:
    print("\n--- Verificação Especial (T=5) ---")
    nova_contagem_5a = df_processado['Target_FU_5_Anos'].sum()
    print(f"Nova contagem (Classe 1, <= 5 anos): {nova_contagem_5a}")
    
    if 'Obito_MS_FU-5 years' in df_processado.columns:
        contagem_original_5a = df_processado['Obito_MS_FU-5 years'].sum()
        print(f"Contagem Original ('Obito_MS_FU-5 years'): {contagem_original_5a}")
        if nova_contagem_5a == contagem_original_5a:
            print("VERIFICAÇÃO: OK. As contagens agora batem (59).")
        else:
            print(f"AVISO: As contagens ainda não batem ({nova_contagem_5a} vs {contagem_original_5a}).")
            print("Isso significa que a coluna 'Obito_MS_FU-5 years' foi criada com uma lógica diferente.")
    else:
        print("Coluna 'Obito_MS_FU-5 years' não encontrada para verificação.")

# --- 5. SALVAR CADA DATASET SEPARADAMENTE ---

print("\nSalvando os 4 datasets...")

for T in LIMIARES_ANOS:
    df_final = df_processado.copy()
    
    alvo_atual = f'Target_FU_{T}_Anos'
    alvos_para_remover = [col for col in novas_colunas_alvo if col != alvo_atual]
    
    df_final = df_final.drop(columns=alvos_para_remover)
    
    arquivo_saida = f'chagas_limiar_{T}_anos.xlsx'
    caminho_saida = os.path.join(DIRETORIO_DADOS, arquivo_saida)
    
    try:
        df_final.to_excel(caminho_saida, index=False)
        print(f"Arquivo salvo com sucesso: {caminho_saida}")
    except Exception as e:
        print(f"ERRO ao salvar o arquivo '{caminho_saida}': {e}")

print("\nProcesso concluído.")
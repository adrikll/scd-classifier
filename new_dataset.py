import pandas as pd
import numpy as np
import re
import os
import unicodedata

print("Iniciando script de enriquecimento de dados (Versão 11 - Diagnóstico Sem Falha)...")

# --- 1. FUNÇÃO DE NORMALIZAÇÃO DE TEXTO/NOME ---
def normalize_text(text):
    try:
        text = str(text).strip().upper()
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                       if unicodedata.category(c) != 'Mn')
        text = text.replace('Y', 'I')
        text_padded = ' ' + text + ' '
        text_padded = re.sub(r'\sD(A|E|O|AS|OS)\s', ' ', text_padded)
        text = text_padded.strip()
        text = text.replace('.', '')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except:
        return None

# --- 2. CONFIGURAÇÕES ---
DIRETORIO_DADOS = 'dados'
ARQUIVO_BASE_XLSX = 'chagas_all_features.xlsx'
ARQUIVO_FONTE_XLSX = 'Banco Chagas Geral-22-08-2025.xlsx'
ARQUIVO_SAIDA = 'chagas_features_com_data_msc.xlsx'
COLUNA_CHAVE_BASE = 'Name' 
COLUNA_CHAVE_FONTE = 'Nome do Paciente'
COLUNA_OBSERVACOES = 'Observações'
REGEX_DATA = r'(?:MSC|MORTE SUBITA)\s*(?:EM)?\s*([0-9/,-]+)'

# --- 3. CARREGAR OS DATASETS ---
caminho_base = os.path.join(DIRETORIO_DADOS, ARQUIVO_BASE_XLSX)
caminho_fonte = os.path.join(DIRETORIO_DADOS, ARQUIVO_FONTE_XLSX)
caminho_saida = os.path.join(DIRETORIO_DADOS, ARQUIVO_SAIDA)
try:
    df_base = pd.read_excel(caminho_base)
    print(f"Arquivo base '{caminho_base}' carregado. ({len(df_base)} linhas)")
except Exception as e:
    print(f"ERRO ao ler '{caminho_base}': {e}"); exit()
try:
    df_fonte = pd.read_excel(caminho_fonte)
    print(f"Arquivo fonte '{caminho_fonte}' carregado. ({len(df_fonte)} linhas)")
except Exception as e:
    print(f"ERRO ao ler '{caminho_fonte}': {e}"); exit()

# --- 4. PROCESSAR O ARQUIVO FONTE (Banco Geral) ---
print(f"\nProcessando '{ARQUIVO_FONTE_XLSX}' para extrair datas de MSC...")
if COLUNA_CHAVE_FONTE not in df_fonte.columns:
    print(f"ERRO: Coluna chave '{COLUNA_CHAVE_FONTE}' não encontrada no arquivo fonte."); exit()
if COLUNA_OBSERVACOES not in df_fonte.columns:
    print(f"ERRO: Coluna '{COLUNA_OBSERVACOES}' não encontrada no arquivo fonte."); exit()

df_fonte['_obs_norm'] = df_fonte[COLUNA_OBSERVACOES].apply(normalize_text)
df_fonte['Data_MSC_Extraida'] = df_fonte['_obs_norm'].str.extract(REGEX_DATA, flags=0)
df_msc_datas = df_fonte.dropna(subset=['Data_MSC_Extraida']).copy()
print(f"[DEBUG] {len(df_msc_datas)} datas de MSC extraídas com sucesso (Regex funcionou).")

df_msc_datas['_key_norm_fonte'] = df_msc_datas[COLUNA_CHAVE_FONTE].apply(normalize_text)
mapa_datas_msc = df_msc_datas.drop_duplicates(subset=['_key_norm_fonte'])
mapa_datas_msc = mapa_datas_msc[['_key_norm_fonte', 'Data_MSC_Extraida']]
print(f"[DEBUG] Mapa final de datas criado com {len(mapa_datas_msc)} pacientes únicos com MSC.")

print(f"\nMesclando datas de MSC com '{ARQUIVO_BASE_XLSX}' usando NOME...")
if COLUNA_CHAVE_BASE not in df_base.columns:
    print(f"ERRO: Coluna chave '{COLUNA_CHAVE_BASE}' não encontrada no arquivo base."); exit()
if 'Obito_MS' not in df_base.columns:
    print(f"ERRO: Coluna 'Obito_MS' não encontrada no arquivo base."); exit()

df_base['_key_norm_base'] = df_base[COLUNA_CHAVE_BASE].apply(normalize_text)
df_enriquecido = pd.merge(
    df_base,
    mapa_datas_msc,
    left_on='_key_norm_base',
    right_on='_key_norm_fonte',
    how='left'
)
df_enriquecido = df_enriquecido.drop(columns=['_key_norm_base', '_key_norm_fonte'])
print("Merge concluído.")

print("\nCorrigindo a coluna 'Obito_MS' com base nas datas de MSC extraídas...")
filtro_correcao = (
    df_enriquecido['Data_MSC_Extraida'].notna() & 
    (df_enriquecido['Obito_MS'] != 1)
)
pacientes_corrigidos = df_enriquecido[filtro_correcao]
if not pacientes_corrigidos.empty:
    print(f"AVISO: {len(pacientes_corrigidos)} pacientes tinham 'Obito_MS != 1' mas uma data de MSC foi encontrada.")
    print("       Corrigindo 'Obito_MS' para 1.")
    df_enriquecido.loc[filtro_correcao, 'Obito_MS'] = 1
else:
    print("Verificação de consistência: OK.")

# --- 7. VERIFICAÇÃO DE FALHAS (Imprime AVISO em vez de falhar) ---
print("\nVerificando se todos os pacientes com 'Obito_MS == 1' tiveram a data encontrada...")
pacientes_com_msc = df_enriquecido[df_enriquecido['Obito_MS'] == 1]
pacientes_sem_data = pacientes_com_msc[pacientes_com_msc['Data_MSC_Extraida'].isna()]

if not pacientes_sem_data.empty:
    nomes_problematicos = pacientes_sem_data[COLUNA_CHAVE_BASE].tolist()
    
    # --- LÓGICA DE AVISO (EM VEZ DE EXCEÇÃO) ---
    print("\n" + "="*30 + " AVISO DE MAPEAMENTO " + "="*30)
    print(f"Os {len(nomes_problematicos)} pacientes abaixo têm 'Obito_MS == 1', mas a data de MSC não foi encontrada.")
    print("Isso significa que o nome no 'Banco Geral' está muito diferente ou a observação tem um formato de data que a Regex não pegou.")
    
    # Diagnóstico de merge (da Versão 7)
    df_fonte_norm_keys = df_fonte[[COLUNA_CHAVE_FONTE, COLUNA_OBSERVACOES, '_obs_norm']].copy()
    df_fonte_norm_keys['_key_norm_fonte'] = df_fonte_norm_keys[COLUNA_CHAVE_FONTE].apply(normalize_text)
    chaves_fonte_sucesso = mapa_datas_msc['_key_norm_fonte'].tolist()
    
    for nome in nomes_problematicos:
        chave_base_problema = normalize_text(nome)
        print(f"\n--- Paciente: '{nome}' (Chave Base: '{chave_base_problema}') ---")
        
        if chave_base_problema in chaves_fonte_sucesso:
            print("  - DIAGNÓSTICO: ERRO GRAVE. A chave foi encontrada no mapa, mas o merge falhou.")
        else:
            print("  - DIAGNÓSTICO: A chave normalizada NÃO ESTAVA no mapa de datas.")
            candidatos_exatos = df_fonte_norm_keys[df_fonte_norm_keys['_key_norm_fonte'] == chave_base_problema]
            
            if candidatos_exatos.empty:
                print(f"  - FALHA: O nome normalizado '{chave_base_problema}' não foi encontrado NENHUMA VEZ no 'Banco Geral'.")
                primeiro_nome = nome.split(' ')[0]
                df_fonte_candidatos = df_fonte[df_fonte[COLUNA_CHAVE_FONTE].str.contains(primeiro_nome, case=False, na=False)]
                if not df_fonte_candidatos.empty:
                     print("  - Candidatos de nome parecido encontrados no 'Banco Geral':")
                     for nome_fonte in df_fonte_candidatos[COLUNA_CHAVE_FONTE].unique():
                         print(f"    - Nome Original: '{nome_fonte}' (Chave Norm: '{normalize_text(nome_fonte)}')")
            else:
                print(f"  - SUCESSO NO MERGE: O nome foi encontrado no 'Banco Geral' ({len(candidatos_exatos)} vez(es)).")
                print("  - FALHA NA REGEX: A extração da data falhou. Verificando as observações:")
                for index, row in candidatos_exatos.iterrows():
                    print(f"    - Linha {index} (do 'Banco Geral'): ")
                    print(f"      - Obs. Original:    '{row[COLUNA_OBSERVACOES]}'")
                    print(f"      - Obs. Normalizada: '{row['_obs_norm']}'")
        print(f"  - (Regex Aplicada: '{REGEX_DATA}')")
    print("="*80 + "\n")
    # --- FIM DA LÓGICA DE AVISO ---
else:
    print("Verificação bem-sucedida! Todos os pacientes com MSC foram mapeados.")

# --- 8. SALVAR O RESULTADO ---
caminho_saida = os.path.join(DIRETORIO_DADOS, ARQUIVO_SAIDA)
try:
    df_enriquecido.to_excel(caminho_saida, index=False)
    print(f"\nSucesso! Novo dataset salvo como: '{caminho_saida}'")
    
    datas_adicionadas = df_enriquecido['Data_MSC_Extraida'].notna().sum()
    print(f"Total de datas de MSC adicionadas ao arquivo final: {datas_adicionadas}")

except Exception as e:
    print(f"\nERRO ao salvar o arquivo '{caminho_saida}': {e}")
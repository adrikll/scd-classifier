import pandas as pd
import numpy as np
import re
import os
import unicodedata

#extrai as datas de MSC dos pacientes e calcula o tempo em anos para fazernos a analise de sensibilidade do modelo
#como algumas datas estavam "incompletas" com mm-yyyy ou apenas yyyy vamos usar apenas o ano para fazer o calculo 
# da coluna Time, que agora tem como valor maximo 10

def normalize_text(text, aggressive=True):
    try:
        text = str(text).strip().upper()
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                       if unicodedata.category(c) != 'Mn')
        if aggressive:
            text = text.replace('Y', 'I')
            text_padded = ' ' + text + ' '
            text_padded = re.sub(r'\sD(A|E|O|AS|OS)\s', ' ', text_padded)
            text = text_padded.strip()
            text = text.replace('.', '')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except:
        return None

def converter_data_flexivel(data_texto):
    if pd.isna(data_texto):
        return pd.NaT
    data_texto_str = str(data_texto).replace('-', '/')
    try:
        return pd.to_datetime(data_texto_str, format='%d/%m/%Y')
    except ValueError:
        try:
            data = pd.to_datetime(data_texto_str, format='%m/%Y')
            return data + pd.DateOffset(days=14) 
        except ValueError:
            try:
                data = pd.to_datetime(data_texto_str, format='%Y')
                return data + pd.DateOffset(months=6) 
            except ValueError:
                return pd.NaT

DIRETORIO_DADOS = 'dados'
ARQUIVO_BASE_XLSX = 'chagas_all_features.xlsx'
ARQUIVO_FONTE_XLSX = 'Banco Chagas Geral-22-08-2025.xlsx'
ARQUIVO_SAIDA = 'chagas_msc_time_calculado.xlsx'

COLUNA_CHAVE_BASE = 'Name' 
COLUNA_CHAVE_FONTE = 'Nome do Paciente'
COLUNA_OBSERVACOES = 'Observações'
COLUNA_OBITO_MS = 'Obito_MS'
REGEX_DATA = r'(?:MSC|MORTE SUBITA)\s*(?:EM)?\s*([0-9/,-]+)'

COLUNA_HOLTER = 'Date Holter'
COLUNA_MSC = 'Data_MSC_Extraida' 
COLUNA_TIME_ANTIGA = 'Time' 
COLUNA_SAIDA_TIME = 'Time_Calculado_Anos' 

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

print(f"\nProcessando '{ARQUIVO_FONTE_XLSX}' para extrair datas de MSC...")
if COLUNA_CHAVE_FONTE not in df_fonte.columns:
    print(f"ERRO: Coluna chave '{COLUNA_CHAVE_FONTE}' não encontrada no arquivo fonte."); exit()
if COLUNA_OBSERVACOES not in df_fonte.columns:
    print(f"ERRO: Coluna '{COLUNA_OBSERVACOES}' não encontrada no arquivo fonte."); exit()

df_fonte['_obs_norm'] = df_fonte[COLUNA_OBSERVACOES].apply(normalize_text)
df_fonte[COLUNA_MSC] = df_fonte['_obs_norm'].str.extract(REGEX_DATA, flags=0)
df_msc_datas = df_fonte.dropna(subset=[COLUNA_MSC]).copy()
print(f"[DEBUG] {len(df_msc_datas)} datas de MSC extraídas com sucesso (Regex funcionou).")

df_msc_datas['_key_norm_fonte'] = df_msc_datas[COLUNA_CHAVE_FONTE].apply(normalize_text)
mapa_datas_msc = df_msc_datas.drop_duplicates(subset=['_key_norm_fonte'])
mapa_datas_msc = mapa_datas_msc[['_key_norm_fonte', COLUNA_MSC]]
print(f"[DEBUG] Mapa final de datas criado com {len(mapa_datas_msc)} pacientes únicos com MSC.")

print(f"\nMesclando datas de MSC com '{ARQUIVO_BASE_XLSX}' usando NOME...")
if COLUNA_CHAVE_BASE not in df_base.columns:
    print(f"ERRO: Coluna chave '{COLUNA_CHAVE_BASE}' não encontrada no arquivo base."); exit()
if COLUNA_OBITO_MS not in df_base.columns:
    print(f"ERRO: Coluna '{COLUNA_OBITO_MS}' não encontrada no arquivo base."); exit()

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
    df_enriquecido[COLUNA_MSC].notna() & 
    (df_enriquecido[COLUNA_OBITO_MS] != 1)
)
pacientes_corrigidos = df_enriquecido[filtro_correcao]
if not pacientes_corrigidos.empty:
    print(f"AVISO: {len(pacientes_corrigidos)} pacientes tinham 'Obito_MS != 1' mas uma data de MSC foi encontrada.")
    print("       Corrigindo 'Obito_MS' para 1.")
    df_enriquecido.loc[filtro_correcao, COLUNA_OBITO_MS] = 1
else:
    print("Verificação de consistência: OK.")

print("Iniciando cálculo de tempo (follow-up)...")

print("Convertendo colunas de data (com lógica de imputação flexível)...")
df_enriquecido[COLUNA_HOLTER] = pd.to_datetime(df_enriquecido[COLUNA_HOLTER], errors='coerce')
df_enriquecido[COLUNA_MSC] = df_enriquecido[COLUNA_MSC].apply(converter_data_flexivel)

print(f"Calculando a duração em anos... (Pacientes com Obito_MS=0 terão valor nulo)")
delta_tempo_dias = (df_enriquecido[COLUNA_MSC] - df_enriquecido[COLUNA_HOLTER]).dt.days
duracao_em_anos = delta_tempo_dias / 365.25

df_enriquecido[COLUNA_SAIDA_TIME] = duracao_em_anos
df_enriquecido[COLUNA_SAIDA_TIME] = df_enriquecido[COLUNA_SAIDA_TIME].round(2)

print("\nVerificando resultados do cálculo de tempo...")
msc_pacientes_final = df_enriquecido[df_enriquecido[COLUNA_OBITO_MS] == 1]

msc_sem_tempo_final = msc_pacientes_final[msc_pacientes_final[COLUNA_SAIDA_TIME].isna()]

if not msc_sem_tempo_final.empty:
    print(f"AVISO FINAL: {len(msc_sem_tempo_final)} pacientes têm 'Obito_MS == 1', mas o tempo é NULO.")
    print("       Isto é esperado para pacientes com 'Date Holter' FALTANTE ou inválida.")
    print(f"       Nomes: {msc_sem_tempo_final[COLUNA_CHAVE_BASE].tolist()}")
else:
    print("Verificação de 'Obito_MS vs Tempo': OK (Todos os óbitos válidos têm tempo calculado).")

print("\nReordenando colunas para melhor visualização...")
all_cols = df_enriquecido.columns.tolist()
cols_to_move = [COLUNA_MSC, COLUNA_SAIDA_TIME]
anchor_col = COLUNA_TIME_ANTIGA

try:
    original_cols = [col for col in all_cols if col not in cols_to_move]
    anchor_index = original_cols.index(anchor_col)
    new_col_order = original_cols[:anchor_index + 1] + \
                    cols_to_move + \
                    original_cols[anchor_index + 1:]
    df_enriquecido = df_enriquecido[new_col_order]
    print(f"Colunas movidas para depois de '{anchor_col}'.")
except ValueError:
    print(f"AVISO: A coluna âncora '{anchor_col}' não foi encontrada. As novas colunas ficarão no final.")

try:
    df_enriquecido.to_excel(caminho_saida, index=False)
    print(f"\n{'-'*20} SUCESSO! {'-'*20}")
    print(f"Pipeline concluído. Novo dataset salvo como: '{caminho_saida}'")
    
    print("\nEstatísticas do tempo calculado (em anos):")
    print(df_enriquecido[COLUNA_SAIDA_TIME].describe())
    
    print("\n--- Verificação Final de Contagem ---")
    nulos = df_enriquecido[COLUNA_SAIDA_TIME].isna().sum()
    calculados = df_enriquecido[COLUNA_SAIDA_TIME].notna().sum()
    print(f"Total de pacientes: {len(df_enriquecido)}")
    print(f"Valores Nulos (Obito_MS=0 ou Holter Faltante): {nulos}")
    print(f"Valores Calculados (Obito_MS=1): {calculados}")
    
    if nulos == 108 and calculados == 68:
        print("VEREDITO: OK! As contagens (108 nulos, 68 calculados) batem exatamente.")
    else:
        print(f"VEREDITO: ATENÇÃO! As contagens (Esperado: 108/68, Real: {nulos}/{calculados}) não bateram.")
        if nulos > 108:
            print(f"  -> Isso significa que {nulos - 108} pacientes com Obito_MS=1 não puderam ter o tempo calculado (provavelmente 'Date Holter' faltante).")


except Exception as e:
    print(f"\nERRO ao salvar o arquivo '{caminho_saida}': {e}")
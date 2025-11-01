import pandas as pd
import numpy as np
import os

# Pega as configurações do seu projeto
from src import config 

def create_time_horizon_datasets():
    """
    Carrega o dataset original e cria 4 novos datasets com diferentes
    horizontes temporais para a definição do alvo.
    """
    print(f"Carregando dataset original de: {config.DATA_PATH}")
    
    output_dir = os.path.dirname(config.DATA_PATH)
    if not output_dir:
        output_dir = '.' 
        
    try:
        df_original = pd.read_excel(config.DATA_PATH)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo Excel: {e}")
        return

    horizons = [1, 2, 3, 4]
    all_new_target_cols = [f'Target_FU_{t}_years' for t in horizons]

    df_processed = df_original.copy()
    for t in horizons:
        new_col_name = f'Target_FU_{t}_years'
        
        cond_positive = (df_processed['Obito_MS'] == 1) & (df_processed['Time'] < t)
        
        df_processed[new_col_name] = np.where(cond_positive, 1, 0)
        
        print(f"Coluna {new_col_name} criada. Contagem de classe 1: {df_processed[new_col_name].sum()}")

    for t in horizons:
        df_final = df_processed.copy()
        
        current_target_col = f'Target_FU_{t}_years'
        
        cols_to_drop = [col for col in all_new_target_cols if col != current_target_col]
        
        if config.TARGET_COLUMN not in cols_to_drop:
             cols_to_drop.append(config.TARGET_COLUMN)
        
        df_final = df_final.drop(columns=cols_to_drop, errors='ignore')

        output_filename = os.path.join(output_dir, f'chagas_{t}_year.xlsx')
        
        try:
            df_final.to_excel(output_filename, index=False)
            print(f"Dataset salvo com sucesso em: {output_filename}")
        except Exception as e:
            print(f"Erro ao salvar o arquivo {output_filename}: {e}")

    print("\nGeração de datasets concluída.")

if __name__ == "__main__":
    create_time_horizon_datasets()
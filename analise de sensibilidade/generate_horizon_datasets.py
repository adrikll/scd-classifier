import pandas as pd
import numpy as np
import os
import sys

def generate_horizon_datasets(input_file='analise de sensibilidade/chagas_all_features_updated.xlsx'):

    df = pd.read_excel(input_file)

    # Garantir que a coluna Tempo seja numérica
    df['Tempo'] = pd.to_numeric(df['Tempo'], errors='coerce')

    # Lista de horizontes desejados
    horizons = [3, 5, 7, 10]

    for anos in horizons:
        print(f"\n{'='*40}")
        print(f"PROCESSANDO HORIZONTE: {anos} ANOS")
        print(f"{'='*40}")

        # 3. Lógica de Construção das Classes
        
        # CLASSE 1 (Evento):
        # Paciente morreu de MS (Obito_MS == 1) E o tempo foi menor ou igual ao horizonte.
        mask_c1 = (df['Obito_MS'] == 1) & (df['Tempo'] < anos)
        
        #classe 0: (Tempo >= anos)
        
        mask_c0 = (df['Obito_MS'] == 0) | (df['Tempo'] >= anos)

        # Filtragem e Atribuição do Target
        df_c1 = df[mask_c1].copy()
        df_c1['Target'] = 1

        df_c0 = df[mask_c0].copy()
        df_c0['Target'] = 0

        # Concatenação
        df_final = pd.concat([df_c0, df_c1])

        # 5. Estatísticas e Salvamento
        total = len(df_final)
        n_c1 = len(df_c1)
        n_c0 = len(df_c0)
        
        output_filename = f'dados/dataset_chagas_{anos}anos.csv'
        df_final.to_csv(output_filename, index=False)

        print(f"Arquivo gerado: {output_filename}")
        print(f"Dimensões: {df_final.shape}")
        print(f" - Classe 0 (Controle > {anos} anos): {n_c0}")
        print(f" - Classe 1 (Evento <= {anos} anos): {n_c1}")
        if total > 0:
            print(f" - Proporção de Eventos: {(n_c1/total*100):.2f}%")


    print(f"\n{'='*40}")
    print("Processo concluído com sucesso.")

if __name__ == "__main__":
    generate_horizon_datasets()
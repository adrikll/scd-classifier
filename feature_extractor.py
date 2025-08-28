import os
import re
import numpy as np
import torch
import pandas as pd
import neurokit2 as nk
from torch.utils.data import Dataset
import json

class ChagasDataset(Dataset):
    def __init__(self, data_dir, patients_df, weights=None, return_ids=True, max_minutes=27):
        self.data_dir       = data_dir
        self.patients_df    = self._filter_patients(patients_df)
        self.return_ids     = return_ids
        self.max_minutes    = max_minutes
        self.sampling_rate  = 480 

        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.txt')
        ])

        sample_signal = np.loadtxt(self.files[0], dtype=np.float32)
        self.feature_names = self._extract_ecg_features(sample_signal).index.tolist()

    def _filter_patients(self, df):
        c1 = df[(df['Obito_MS'] == 1) & (df['Time'] < 5)]
        c0 = df[(df['Obito_MS'] == 0) | (df['Time'] == 5)]
        return pd.concat([c1, c0])

    def _extract_ecg_features(self, signal: np.ndarray) -> pd.Series:
        """
        Extrai features de variabilidade da frequência cardíaca (HRV) a partir de um sinal de ECG,
        segmentando o sinal em janelas consecutivas de 9 minutos. Para cada janela completa, são
        extraídas as features de HRV no domínio do tempo, da frequência e não linear. Após o processamento
        de todas as janelas, a função retorna a média e o desvio padrão das features extraídas para o paciente.

        Parâmetros: 
        
        signal : np.ndarray
            Sinal de ECG em formato unidimensional (1D) e com frequência de amostragem definida
            por `self.sampling_rate`.

        Return:
        
        - pd.Series
            A média e o desvio padrão das features extraídas de todas as janelas de 9 minutos.
        """
        
        window_duration = 9 * 60 
        window_samples = int(window_duration * self.sampling_rate)

        segment = signal
        num_windows = len(segment) // window_samples

        feature_list = []

        for i in range(num_windows):
            start = i * window_samples
            end = start + window_samples
            window = segment[start:end]

            try:
                clean_ecg = nk.ecg_clean(window, sampling_rate=self.sampling_rate, method="pantompkins1985")
                _, info = nk.ecg_peaks(clean_ecg, sampling_rate=self.sampling_rate, method="pantompkins1985")

                try:
                    hrv_time = nk.hrv_time(info['ECG_R_Peaks'], sampling_rate=self.sampling_rate, show=False)
                except Exception as e:
                    print(f"[Janela {i+1}] Erro na extração: HRV_TIME")
                    print(e)
                    hrv_time = pd.DataFrame()

                try:
                    hrv_freq = nk.hrv_frequency(info['ECG_R_Peaks'], sampling_rate=self.sampling_rate, show=False)
                except Exception as e:
                    print(f"[Janela {i+1}] Erro na extração: HRV_FREQUENCY")
                    print(e)
                    hrv_freq = pd.DataFrame()

                try:
                    hrv_nonlinear = nk.hrv_nonlinear(info['ECG_R_Peaks'], sampling_rate=self.sampling_rate, show=False)
                except Exception as e:
                    print(f"[Janela {i+1}] Erro na extração: HRV_NONLINEAR")
                    print(e)
                    hrv_nonlinear = pd.DataFrame()

                if 'HRV_ULF' in hrv_freq.columns:
                    hrv_freq = hrv_freq.drop(columns=['HRV_ULF'])

                features_df = pd.concat([hrv_time, hrv_freq, hrv_nonlinear], axis=1)

                if not features_df.empty:
                    feature_list.append(features_df.iloc[0])

            except Exception as e:
                print(f"[Janela {i+1}] Erro geral: {e}")
                continue

        if not feature_list:
            raise ValueError("Nenhuma feature extraída de nenhuma janela.")

        # Concatena todas as features de todas as janelas em um único DataFrame
        all_features_df = pd.DataFrame(feature_list)
        
        # Calcula a média das features
        aggregated_mean = all_features_df.mean(axis=0)
        
        # Calcula o desvio padrão das features 
        aggregated_std = all_features_df.std(axis=0)
        aggregated_std.index = 'std_' + aggregated_std.index
        
        # Concatena a média e o desvio padrão
        final_features = pd.concat([aggregated_mean, aggregated_std])

        return final_features


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        basename = os.path.basename(path)
        m = re.match(r'sinal(\d+)\.txt', basename)
        if not m:
            raise ValueError(f"Arquivo inválido: {basename}")
        patient_id = int(m.group(1))

        info = self.patients_df[self.patients_df['ID'] == patient_id]
        if info.empty:
            raise ValueError(f"ID {patient_id} não está na tabela de pacientes.")
        obito = info['Obito_MS'].iloc[0]
        tempo = info['Time'].iloc[0]
        event = 1 if (obito == 1 and tempo < 5) else 0

        signal = np.loadtxt(path, dtype=np.float32)
        feats = self._extract_ecg_features(signal)
        x = torch.tensor(feats.values, dtype=torch.float32)
        y = torch.tensor([event], dtype=torch.float32)

        out = [x, y]
        if self.return_ids:
            out.append(patient_id)
        return tuple(out)


def save_features(dataset, output_file):
    """
    Salva as features extraídas do dataset em um arquivo CSV. Cada linha corresponde a um paciente,
    com as médias e o desvio padrão das features extraídas de todas as janelas de 9 minutos do sinal de ECG.

    Percorre o dataset, processa o sinal de ECG de cada paciente e salva as médias e o desvio padrão das
    features extraídas em um arquivo CSV, juntamente com informações do paciente e o rótulo do evento.
    """
    header_written = os.path.exists(output_file)

    for idx in range(len(dataset)):
        try:
            features, event, weight, patient_id = dataset[idx]
            patient_info = dataset.patients_df[dataset.patients_df['ID'] == patient_id]

            if patient_info.empty:
                continue

            obito = patient_info['Obito_MS'].values[0]
            tempo = patient_info['Time'].values[0]

            feature_dict = {name: features[i].item() for i, name in enumerate(dataset.feature_names)}
            feature_dict.update({
                'patient_id': patient_id,
                'Obito_MS_FU-5 years': event.item()
            })

            df_row = pd.DataFrame([feature_dict])

            col_order = ['patient_id', 'Obito_MS_FU-5 years'] + dataset.feature_names
            df_row = df_row[col_order]

            df_row.to_csv(output_file, mode='a', header=not header_written, index=False)
            header_written = True

            print(f"Paciente {patient_id} ok.")

        except Exception as e:
            print(f"Erro ao processar paciente {patient_id} (idx={idx}): {e}")

if __name__ == "__main__":

    df_chagas = pd.read_excel('chagas_idades.xlsx')

    ds = ChagasDataset('base_chagas', df_chagas, return_ids=True)

    save_features(ds, 'mean_std_features_9min.csv') 
import re
import os
import numpy as np
import pandas as pd


def set_label(df, label):
    labels = label.split(';')
    df['label'] = 'normal'
    for label in labels:
        vals = [l.split('=')[1] for l in label.split('&')]
        cond = np.all([df[val[0]] == val for val in vals], axis=0)
        df.loc[cond, 'label'] = ' & '.join(vals)
    return df


dataset_path = "./data/"
folders = ['S', 'L', 'H']

deep_a = False

normal_predict_amount = 0
normal_predict_error = 0
sigs = []
num_files = 0
for folder in folders:
    info = os.path.join(dataset_path, folder, 'injection_info.csv')
    df = pd.read_csv(info)

    print('folder', folder)

    if 'normal_predict_amount' in df.columns:
        normal_predict_amount += df['normal_predict_amount'].sum()
        normal_predict_error += df['normal_predict_error'].sum()

        print('residual', df['normal_predict_error'].sum() / df['normal_predict_amount'].sum() * 100)
        res = df['normal_predict_error'] / df['normal_predict_amount'] * 100
        print('max residual', res.max())

    if 'significance' in df.columns:  # B datasets
        significance = df['significance'].mean()
        sigs.extend(df['significance'].values)
    elif 'anomaly_significance' in df.columns:  # synthetic
        significance = df['anomaly_significance'].mean()
        sigs.extend(df['anomaly_significance'].values)
    else:  # A and D
        files = os.listdir(os.path.join(dataset, folder))

        if folder.startswith('new_dataset_A'):
            layer = int(re.search(r"layers?_(\d)", folder).group(1))
            elements = int(re.search(r"elements?_(\d)", folder).group(1))

            print('layer', layer, 'elements', elements)

            if deep_a:
                if layer <= 3 and elements <= 3:
                    print('lower')
                    continue
            else:
                if layer > 3 or elements > 3:
                    print('upper')
                    continue

        num_files += len(files) - 1

        folder_sigs = []
        normal_predict_amount_file = 0
        normal_predict_error_file = 0
        for file in files:
            if file == 'injection_info.csv' or file == 'truth_prediction.csv':
                continue

            file_path = os.path.join(dataset_path, folder, file)
            df_file = pd.read_csv(file_path)

            n = 4 if dataset_path[-1] == 'A' else 6
            label = df.loc[df['timestamp'] == int(file[:-n]), 'set'].iloc[0]

            df_file = set_label(df_file, label)

            mask = df_file['label'] != 'normal'

            normal_predict_amount_file += df_file.loc[~mask, 'predict'].sum()
            normal_predict_error_file += (df_file.loc[~mask, 'real'] - df_file.loc[~mask, 'predict']).abs().sum()

            abnormal_predict_error = (df_file.loc[mask, 'real'] - df_file.loc[mask, 'predict']).abs().sum()
            sig = abnormal_predict_error / df_file['predict'].sum()
            folder_sigs.append(sig)

        print('residual', normal_predict_error_file / normal_predict_amount_file * 100)
        normal_predict_amount += normal_predict_amount_file
        normal_predict_error += normal_predict_error_file

        significance = np.mean(folder_sigs)
        sigs.extend(folder_sigs)

    print('mean significant', significance, '\n')

residual = normal_predict_error / normal_predict_amount
print('total residual', residual * 100)

print('total significance', np.mean(sigs))

print('num_files (for A & D)', num_files)


import os
import re
import numpy as np
import pandas as pd
from utils.robustspot_data_utils import get_rs_label, read_rs_dataframe


def set_label(df, label):
    """
    Creates a label column for a dataframe following the given label.
    :param df: pandas dataframe.
    :param label: str, the label to use.
    :return pandas dataframe with added label column.
    """
    labels = label.split(';')
    df['label'] = 'normal'
    for label in labels:
        vals = [(l.split('=')[0], l.split('=')[1]) for l in label.split('&')]
        cond = np.all([df[val[0]] == val[1] for val in vals], axis=0)
        df.loc[cond, 'label'] = label
    return df


def analyze_single_B_folder(dataset_path, folder, significance_column='significance'):
    """
    Analyze instances in the B_i datasets.
    :param dataset_path: str, the path to the dataset.
    :param folder: str, the sub-folder to analyze.
    :param significance_column: str, the column with anomaly significance values.
    :return total prediction amount and error for normal leaf elements, significance values, number of considered files.
    """
    print('folder', folder)

    info = os.path.join(dataset_path, folder, 'injection_info.csv')
    df = pd.read_csv(info)

    all_files = os.listdir(os.path.join(dataset_path, folder))
    case_files = [file for file in all_files if file != 'injection_info.csv' and file != 'truth_prediction.csv']
    num_files = len(case_files)
    
    normal_predict_amount = df['normal_predict_amount'].sum()
    normal_predict_error = df['normal_predict_error'].sum()
    significance_values = df[significance_column].values

    print('residual', df['normal_predict_error'].sum() / df['normal_predict_amount'].sum() * 100)
    res = df['normal_predict_error'] / df['normal_predict_amount'] * 100
    print('max residual', res.max())
    print('min residual', res.min())
    print('mean significant', df[significance_column].mean(), '\n')
    return normal_predict_amount, normal_predict_error, significance_values, num_files


def analyze_B_data(dataset_path):
    """
    Analyzes all B_i datasets.
    :param dataset_path: str, the path to the dataset.
    :return total prediction amount and error for normal leaf elements, significance values, number of considered files.
    """
    normal_predict_amount = 0
    normal_predict_error = 0
    significance_values = []
    num_files = 0

    folders = os.listdir(dataset_path)
    for folder in folders:
        npa, npe, sv, nf = analyze_single_B_folder(dataset_path, folder)
        normal_predict_amount += npa
        normal_predict_error += npe
        significance_values.extend(sv)
        num_files += nf

    return normal_predict_amount, normal_predict_error, significance_values, num_files


def analyze_synthetic_data(dataset_path):
    """
    Analyzes the synthetic datasets (S, L, H, or any others created with the generate_dataset.py code.
    :param dataset_path: str, the path to the dataset.
    :return total prediction amount and error for normal leaf elements, significance values, number of considered files.
    """
    return analyze_single_B_folder(dataset_path, "", significance_column='anomaly_significance')


def analyze_A_D_data(dataset_path, deep_a):
    """
    Analyzes the A (both A and A_star) or the D dataset.
    :param dataset_path: str, the path to the dataset.
    :param deep_a: boolean, if true then considers A_star, otherwise A.
    :return total prediction amount and error for normal leaf elements, significance values, number of considered files.
    """
    normal_predict_amount = 0
    normal_predict_error = 0
    significance_values = []
    num_files = 0

    folders = os.listdir(dataset_path)
    for folder in folders:
        print('folder', folder)

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

        df_info = pd.read_csv(os.path.join(dataset_path, folder, 'injection_info.csv'))
        all_files = os.listdir(os.path.join(dataset_path, folder))
        case_files = [file for file in all_files if file != 'injection_info.csv' and file != 'truth_prediction.csv']
        num_files += len(case_files)

        folder_significance_values = []
        folder_normal_predict_amount = 0
        folder_normal_predict_error = 0
        for file in case_files:
            df_file = pd.read_csv(os.path.join(dataset_path, folder, file))

            n = 4 if dataset_path[-1] == 'A' else 6
            label = df_info.loc[df_info['timestamp'] == int(file[:-n]), 'set'].iloc[0]

            df_file = set_label(df_file, label)
            mask = df_file['label'] != 'normal'

            folder_normal_predict_amount += df_file.loc[~mask, 'predict'].sum()
            folder_normal_predict_error += (df_file.loc[~mask, 'real'] - df_file.loc[~mask, 'predict']).abs().sum()

            abnormal_predict_error = (df_file.loc[mask, 'real'] - df_file.loc[mask, 'predict']).abs().sum()
            significance = abnormal_predict_error / df_file['predict'].sum()
            folder_significance_values.append(significance)

        normal_predict_amount += folder_normal_predict_amount
        normal_predict_error += folder_normal_predict_error
        significance_values.extend(folder_significance_values)

        print('mean significant', np.mean(folder_significance_values), '\n')
        print('residual', folder_normal_predict_error / folder_normal_predict_amount * 100)

    return normal_predict_amount, normal_predict_error, significance_values, num_files


def analyze_RS_data(dataset_path):
    """
    Analyzes the RS dataset.
    :param dataset_path: str, the path to the dataset.
    :return total prediction amount and error for normal leaf elements, significance values, number of considered files.
    """
    normal_predict_amount = 0
    normal_predict_error = 0
    significance_values = []

    files = [file for file in os.listdir(dataset_path) if file != 'anomaly.yaml']
    num_files = len(files)

    for file in files:
        label = get_rs_label(dataset_path, file[:-4])
        df_file, attributes, df_a, df_b = read_rs_dataframe(dataset_path, file[:-4])

        df_file = set_label(df_file, label)
        mask = df_file['label'] != 'normal'

        normal_predict_amount += df_file.loc[~mask, 'predict'].sum()
        normal_predict_error += (df_file.loc[~mask, 'real'] - df_file.loc[~mask, 'predict']).abs().sum()
        abnormal_predict_error = (df_file.loc[mask, 'real'] - df_file.loc[mask, 'predict']).abs().sum()

        significance_value = abnormal_predict_error / df_file['predict'].sum()
        significance_values.append(significance_value)

    print('mean significant', np.mean(significance_values), '\n')
    return normal_predict_amount, normal_predict_error, significance_values, num_files


dataset_path = "../data/"
datasets = ['B0', 'A', 'L', 'D', 'RS']  # Example of some datasets
deep_a = False

for dataset in datasets:
    print(f"Running dataset {dataset}")
    dataset_folder = os.path.join(dataset_path, dataset)
    if dataset.startswith('B'):
        normal_predict_vals, normal_predict_error, sig_vals, n = analyze_B_data(dataset_folder)
    elif dataset in ['D', 'A']:
        normal_predict_vals, normal_predict_error, sig_vals, n = analyze_A_D_data(dataset_folder, deep_a=deep_a)
        if dataset == 'D':
            n = n // 2
    elif dataset in ['S', 'L', 'H']:
        normal_predict_vals, normal_predict_error, sig_vals, n = analyze_synthetic_data(dataset_folder)
    elif dataset == 'RS':
        normal_predict_vals, normal_predict_error, sig_vals, n = analyze_RS_data(dataset_folder)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

    residual = normal_predict_error / normal_predict_vals
    print(f"dataset {dataset}")
    print('total residual:', residual * 100)
    print('total significance:', np.mean(sig_vals))
    print('num_files:', n)
    print("-----------------------------------------------------")

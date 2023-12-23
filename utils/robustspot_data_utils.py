import os
import yaml
import pandas as pd


def transform_rs_data(data_folder, anomaly, predict_periods=4):
    # Adapted from RobustSpot's 'get_predict_df' in predict.py
    anomaly_raw_data = pd.read_csv(f'{data_folder}/{anomaly["data"]}.csv')
    col_list = anomaly_raw_data.columns.values.tolist()
    col_list.remove('min')
    col_list.remove('value')
    col_list.remove('cnt')
    anomaly['header'] = col_list

    anomaly_raw_data['value'] = anomaly_raw_data['cnt'] - anomaly_raw_data['value']  # TODO: changed
    anomaly_raw_data['k_real'] = anomaly_raw_data['value'] / anomaly_raw_data['cnt']

    current_time = anomaly['timestamp']
    history_time_list = [current_time - i * 60 for i in range(1, predict_periods + 1)]
    predict_df = anomaly_raw_data[anomaly_raw_data['min'] == current_time]
    predict_df = predict_df.drop(columns=['min'])
    predict_df.rename(columns={'value': 'value_real', 'cnt': 'cnt_real'}, inplace=True)
    predict_df['value_predict'] = 0
    predict_df['cnt_predict'] = 0
    predict_df.reset_index(inplace=True)
    for predict_df_item in predict_df.itertuples():
        history_df_item = anomaly_raw_data[anomaly_raw_data['min'].isin(history_time_list)]
        for header in anomaly['header']:
            history_df_item = history_df_item[history_df_item[header] == getattr(predict_df_item, header)]
        predict_df.loc[predict_df_item.Index, 'value_predict'] = history_df_item['value'].mean()
        predict_df.loc[predict_df_item.Index, 'cnt_predict'] = history_df_item['cnt'].mean()
        predict_df.loc[predict_df_item.Index, 'k_predict'] = history_df_item['k_real'].mean()
    predict_df.fillna(0, inplace=True)
    return predict_df, col_list


def read_rs_dataframe(run_directory, file):
    anomaly = get_rs_anomaly(run_directory, file)
    df, attributes = transform_rs_data(run_directory, anomaly)

    # Keep the same format as the other derived dataset (D)
    df = df.rename(columns={'k_real': 'real', 'k_predict': 'predict', 'value_real': 'real_a',
                            'value_predict': 'predict_a', 'cnt_real': 'real_b', 'cnt_predict': 'predict_b'})
    df[attributes] = df[attributes].astype(str)
    df = df.drop(columns='index')

    df_a = df[attributes + ['real_a', 'predict_a']].copy(deep=True)
    df_a = df_a.rename(columns={'real_a': 'real', 'predict_a': 'predict'})

    df_b = df[attributes + ['real_b', 'predict_b']].copy(deep=True)
    df_b = df_b.rename(columns={'real_b': 'real', 'predict_b': 'predict'})

    return df, attributes, df_a, df_b


def get_rs_anomaly(run_directory, file):
    anomaly_config = os.path.join(run_directory, 'anomaly.yaml')
    anomaly_config = open(anomaly_config, mode='r', encoding='utf-8')
    anomaly_info = yaml.load(anomaly_config.read(), Loader=yaml.FullLoader)
    anomaly = [d for d in anomaly_info if d['data'] == file][0]
    return anomaly


def get_rs_label(run_directory, file):
    labels = get_rs_anomaly(run_directory, file)['cause']
    if not isinstance(labels, list):
        labels = [labels]

    all_labels = []
    for label in labels:
        label = '&'.join([k + '=' + str(v) for k, v in label.items()])
        all_labels.append(label)
    final_label = ';'.join(all_labels)
    return final_label

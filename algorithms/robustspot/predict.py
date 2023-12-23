import pandas as pd
import algorithms.robustspot.config.global_data as g_data


def get_predict_df(anomaly_index):
    anomaly = g_data.anomaly_list[anomaly_index]
    anomaly_raw_data = pd.read_csv(f'data/{anomaly["data"]}.csv')
    col_list = anomaly_raw_data.columns.values.tolist()
    col_list.remove('min')
    col_list.remove('value')
    col_list.remove('cnt')
    anomaly['header'] = col_list
    anomaly_raw_data['k_real'] = anomaly_raw_data['value'] / anomaly_raw_data['cnt']
    current_time = anomaly['timestamp']
    history_time_list = [current_time - i * 60 for i in range(1, g_data.predict_periods + 1)]
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
    return predict_df

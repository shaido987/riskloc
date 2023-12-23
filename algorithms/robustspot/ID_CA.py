import numpy as np
import math


def get_sum_f_v(df):
    sum_f = df['k_predict'].sum()
    sum_v = df['k_real'].sum()
    df_len = df.shape[0]
    return sum_f, sum_v, df_len


def get_influence_degree(dataframe):
    ID = []
    sum_f, sum_v, df_len = get_sum_f_v(dataframe)
    num = dataframe.columns.tolist().__len__()
    df_f = np.array(dataframe['k_predict']).tolist()
    df_v = np.array(dataframe['k_real']).tolist()
    for i in range(0, df_len):
        v = df_v[i]
        f = df_f[i]
        avg_f_S2 = (sum_f - f) / (df_len - 1)
        avg_v_S2 = (sum_v - v) / (df_len - 1)
        ID_row = 1 - (1 / (abs((v - f) / (avg_v_S2 - avg_f_S2)) + 1))
        ID.append(ID_row)
    dataframe.insert(num, 'ID', ID)
    return dataframe


def get_sum_m(df):
    sum_f = df['value_predict'].sum()
    sum_v = df['value_real'].sum()
    df_len = df.shape[0]
    return sum_f, sum_v, df_len


def get_sum_d(df):
    sum_f = df['cnt_predict'].sum()
    sum_v = df['cnt_real'].sum()
    return sum_f, sum_v


def get_contribution_ability(dataframe):
    CA = []
    sum_f_d, sum_v_d = get_sum_d(dataframe)
    sum_f_m, sum_v_m, df_len = get_sum_m(dataframe)
    num = dataframe.columns.tolist().__len__()
    df_f_d = np.array(dataframe['cnt_predict']).tolist()
    df_v_d = np.array(dataframe['cnt_real']).tolist()
    df_f_m = np.array(dataframe['value_predict']).tolist()
    df_v_m = np.array(dataframe['value_real']).tolist()
    for i in range(0, df_len):
        v_m_s1 = df_v_m[i]
        f_m_s1 = df_f_m[i]
        v_d_s1 = df_v_d[i]
        f_d_s1 = df_f_d[i]
        rate1 = sum_f_d / sum_f_m
        rate2 = (v_m_s1 + (sum_f_m - f_m_s1)) / (v_d_s1 + (sum_f_d - f_d_s1))
        CA_row = (rate1 * rate2) - 1
        if math.isnan(CA_row):
            CA_row = 0
        
        # Changed: Roungding like this does not work for very small values (e.g. D dataset)
        # CA_row = math.floor(CA_row * 100) / 100
        CA.append(CA_row)
    dataframe.insert(num, 'CA', CA)
    try:
        dataframe.replace([np.inf, -np.inf], np.nan)
        dataframe.fillna(0)
    except:
        pass
    return dataframe

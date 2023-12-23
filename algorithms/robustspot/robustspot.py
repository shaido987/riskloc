import copy
import numpy as np

import algorithms.robustspot.final as final
import algorithms.robustspot.mining as mining
import algorithms.robustspot.config.global_data as g_data
import algorithms.robustspot.select_expand as select_expand


def cause_mining(dim1, dim2, anomaly_index, contribution_ability_threshold, confidence_threshold, lambda_amplification):
    iter_index = select_expand.select_expand(dim1, dim2, anomaly_index, contribution_ability_threshold, lambda_amplification)
    mining.mining(iter_index, confidence_threshold)


def adapt_dataframe(df):
    # Remove rows with all zeroes
    df = df[(df['real'] > 0) | (df['predict'] > 0)]
    df = df.rename(columns={'real': 'k_real', 'predict': 'k_predict'})
    df[['cnt_real', 'cnt_predict', 'value_real', 'value_predict']] = [1, 1, 1, 1]
    return df


def adapt_derived_dataframe(df):
    # Remove rows with all zeroes
    df = df[(df['real_a'] > 0) | (df['predict_a'] > 0) | (df['real_b'] > 0) | (df['predict_b'] > 0)]

    df = df.rename(columns={'real': 'k_real', 'predict': 'k_predict', 'real_a': 'value_real',
                           'predict_a': 'value_predict', 'real_b': 'cnt_real', 'predict_b': 'cnt_predict'})

    # Fix k to be '1 - success rate'
    df['value_real'] = df['cnt_real'] - df['value_real']
    df['value_predict'] = df['cnt_predict'] - df['value_predict']

    df['k_real'] = df['value_real'] / df['cnt_real']
    df['k_predict'] = df['value_predict'] / df['cnt_predict']
    return df


def robustspot(df, attributes, k, derived, contribution_ability_threshold=0.0, confidence_threshold=0.8,
               lambda_amplification=100):
    anomaly_index = 0
    g_data.anomaly_list = [{'header': attributes, 'data': []}]

    if not derived:
        contribution_ability_threshold = -1  # TODO: CA can not be used for fundamental measures
        g_data.derived_measure = False

    # Adjust the dataframe to the original robustspot format
    if derived:
        g_data.predict_dataframe = adapt_derived_dataframe(df)
    else:
        g_data.predict_dataframe = adapt_dataframe(df)

    cause_mining(0, 0, anomaly_index, contribution_ability_threshold, confidence_threshold, lambda_amplification)
    cause_mining(1, 1, anomaly_index, contribution_ability_threshold, confidence_threshold, lambda_amplification)
    if g_data.mining_root_cause[1]:
        cause_mining(1, 2, anomaly_index, contribution_ability_threshold, confidence_threshold, lambda_amplification)
    else:
        g_data.mining_root_cause[2] = []
    cause_mining(2, 1, anomaly_index, contribution_ability_threshold, confidence_threshold, lambda_amplification)
    if g_data.mining_root_cause[3]:
        cause_mining(2, 2, anomaly_index, contribution_ability_threshold, confidence_threshold, lambda_amplification)
    else:
        g_data.mining_root_cause[4] = []
    cause_mining(3, 1, anomaly_index, contribution_ability_threshold, confidence_threshold, lambda_amplification)
    if g_data.mining_root_cause[5]:
        cause_mining(3, 2, anomaly_index, contribution_ability_threshold, confidence_threshold, lambda_amplification)
    else:
        g_data.mining_root_cause[6] = []

    merge_res = []
    merge_res += final.get_merge_res(
        [g_data.mining_root_cause[0][:1], g_data.mining_root_cause[1][:1], g_data.mining_root_cause[2]])
    merge_res += final.get_merge_res(
        [g_data.mining_root_cause[0][1:2], g_data.mining_root_cause[3][:1], g_data.mining_root_cause[4]])
    merge_res += final.get_merge_res(
        [g_data.mining_root_cause[0][2:3], g_data.mining_root_cause[5][:1], g_data.mining_root_cause[6]])
    merge_res += [[item] for item in g_data.mining_root_cause[0]]

    for index in range(len(merge_res)):
        if len(merge_res[index]) == 2:
            copy_item = [copy.deepcopy(set(merge_res[index][0])), copy.deepcopy(set(merge_res[index][1]))]
            copy_item[0].discard(('p2p', 1))  # TODO
            copy_item[0].discard(('p2p', 0))
            copy_item[1].discard(('p2p', 1))
            copy_item[1].discard(('p2p', 0))
            if copy_item[0] == copy_item[1] and copy_item[0]:
                merge_res[index] = [tuple(copy_item[0])]
    for index in range(len(merge_res)):
        if len(merge_res[index]) > 1:
            final.merge_larger_dimension(merge_res, index)
    merge_res_set_list = [set(item) for item in merge_res]
    merge_res_new = []
    for item in merge_res_set_list:
        if item not in merge_res_new:
            merge_res_new.append(item)
    merge_res = [list(item) for item in merge_res_new]
    support_delta = [
        mining.get_support(item, g_data.before_df_list[0]) - mining.get_support(item, g_data.after_df_list[0])
        for item in merge_res
    ]
    support_delta = np.array(support_delta)
    sorted_index = np.argsort(support_delta)[:k]
    final_root_cause = []
    for index in sorted_index:
        final_root_cause.append(merge_res[index])
    g_data.before_df_list = [None] * 7
    g_data.after_df_list = [None] * 7
    g_data.expand_df_list = [None] * 7
    g_data.mining_root_cause = [None] * 7
    g_data.predict_dataframe = None
    g_data.col_num = len(g_data.anomaly_list[anomaly_index]['header'])
    return final_root_cause

from kneed import KneeLocator
import pandas as pd
from functools import reduce
from operator import and_

import algorithms.robustspot.ID_CA as ID_CA
import algorithms.robustspot.config.global_data as g_data


def drop_last_top(df, last_mining_root_cause):
    def get_all_mask(value):
        def single_mask(v):
            column = v[0]
            return df[column] == v[1]

        mask = reduce(and_, map(single_mask, value))
        return mask
    return df[~get_all_mask(last_mining_root_cause)]


def select_expand(dim1, dim2, anomaly_index, contribution_ability_threshold=0, lambda_amplification=100):
    before_df = None
    after_df = None
    expand_df = None
    index = None
    if dim1 == 0 and dim2 == 0:
        before_df = g_data.predict_dataframe
        g_data.row_num = before_df.shape[0]
        index = 0
    else:
        index = 2 * dim1 + dim2 - 2
        if dim2 == 1:
            before_df = drop_last_top(g_data.before_df_list[0], g_data.mining_root_cause[0][dim1-1])
        else:
            before_df = drop_last_top(g_data.before_df_list[index-1], g_data.mining_root_cause[index-1][0])
    before_df = ID_CA.get_influence_degree(before_df)
    before_df = ID_CA.get_contribution_ability(before_df)

    influence_degree_threshold = 0.5
    influence_degree_list = before_df['ID'].values.tolist()
    if influence_degree_list:
        influence_degree_list.sort()
        influence_degree_list = list(filter(lambda x: x > 0, influence_degree_list))

        def get_cdf():
            cdf_list = [0] * len(influence_degree_list)
            for i in range(len(influence_degree_list)):
                cdf_list[i] = (i + 1) / len(influence_degree_list)
            return cdf_list

        influence_degree_cdf = get_cdf()

        try:
            knee_concave = KneeLocator(influence_degree_list, influence_degree_cdf, S=6,
                                       interp_method='polynomial', curve='concave', direction='increasing').knee
            knee = knee_concave
            if knee is None:
                pass
            else:
                influence_degree_threshold = knee
        except:
            pass

    after_df = before_df[
        (before_df['ID'] > influence_degree_threshold)
        &
        (before_df['CA'] > contribution_ability_threshold)
    ]

    before_df = before_df.drop(columns=['ID', 'CA'])
    attribute_columns = g_data.anomaly_list[anomaly_index]['header']
    expand_list = []
    for _, row in after_df.iterrows():
        if g_data.derived_measure:
            expand_times = int(row['ID'] * row['CA'] * lambda_amplification)
        else:
            # For fundamental measures the CA is 0 or slightly negative and can't be used
            expand_times = int(row['ID'] * lambda_amplification)
        expand_list.extend([dict(row[attribute_columns])] * expand_times)

    expand_df = pd.DataFrame(expand_list)
    g_data.before_df_list[index] = before_df
    g_data.after_df_list[index] = after_df
    g_data.expand_df_list[index] = expand_df
    return index

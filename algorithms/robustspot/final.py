import algorithms.robustspot.mining as mining
import algorithms.robustspot.config.global_data as g_data


def get_merge_res(top5_3iter):
    merge_causes = []
    merge_causes += get_merge_causes_2(top5_3iter, 0, 1)
    merge_causes += get_merge_causes_2(top5_3iter, 0, 2)
    merge_causes += get_merge_causes_2(top5_3iter, 1, 2)
    merge_causes += get_merge_cause_3(top5_3iter)
    return merge_causes


def get_merge_causes_2(top5_3iter, first, second):
    merge_causes = []
    for cause1 in top5_3iter[first]:
        for cause2 in top5_3iter[second]:
            if len(cause1) == len(cause2):
                flag_same_column = True
                flag_diff_value = False
                for cause_index in range(len(cause1)):
                    if cause1[cause_index][0] != cause2[cause_index][0]:
                        flag_same_column = False
                    if cause1[cause_index][1] != cause2[cause_index][1]:
                        flag_diff_value = True
                if flag_same_column and flag_diff_value:
                    merge_causes.append([cause1, cause2])
    return merge_causes


def get_merge_cause_3(top5_3iter):
    merge_causes = []
    for cause1 in top5_3iter[0]:
        for cause2 in top5_3iter[1]:
            for cause3 in top5_3iter[2]:
                if len(cause1) == len(cause2) and len(cause2) == len(cause3):
                    flag_same_column = True
                    flag_diff_value = False
                    for cause_index in range(len(cause1)):
                        if not (cause1[cause_index][0] == cause2[cause_index][0]
                                and cause1[cause_index][0] == cause3[cause_index][0]):
                            flag_same_column = False
                        if cause1[cause_index][1] != cause2[cause_index][1] and \
                                cause1[cause_index][1] != cause3[cause_index][1] and \
                                cause2[cause_index][1] != cause3[cause_index][1]:
                            flag_diff_value = True
                    if flag_same_column and flag_diff_value:
                        merge_causes.append([cause1, cause2, cause3])
    return merge_causes


def merge_larger_dimension(merge_res, index):
    merge_cause = merge_res[index]
    record_dict = dict()
    for merge_cause_item in merge_cause:
        for item in merge_cause_item:
            if item[0] in record_dict.keys():
                if item[1] not in record_dict[item[0]]:
                    record_dict[item[0]].append(item[1])
            else:
                record_dict[item[0]] = [item[1]]
    keep_items = []
    for k, v in record_dict.items():
        if len(v) == 1:
            keep_items.append((k, v[0]))
    if len(keep_items) > 0:
        before_support = mining.get_support(merge_cause, g_data.before_df_list[0])
        after_support = mining.get_support(tuple(keep_items), g_data.before_df_list[0])
        if before_support / after_support >= 0.9:
            merge_res[index] = [tuple(keep_items)]

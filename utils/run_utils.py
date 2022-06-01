import os
import pandas as pd
import numpy as np
from algorithms.hotspot import hotspot
from algorithms.squeeze.squeeze import Squeeze, SqueezeOption
from algorithms.autoroot import autoroot
from algorithms.adtributor import adtributor
from algorithms.riskloc import riskloc
from algorithms.rev_rec_adtributor import rev_rec_adtributor


def run_method(df, dfs, attributes, algorithm, algorithm_args, derived, debug):
    """
    Runs the specified algorithm on a given instance.
    :param df: pandas dataframe, the data to use.
    :param dfs: list of pandas dataframe, input for squeeze when using derived data.
    :param attributes: list, the attributes.
    :param algorithm: str, name of the algorithm.
    :param algorithm_args: dict, algorithm input parameters.
    :param derived: boolean, if using derived data.
    :param debug: boolean, if running in debug mode.
    :return: list, root cause predictions.
    """
    if algorithm == "riskloc":
        root_causes = riskloc(df, attributes, derived=derived, debug=debug, **algorithm_args)
    elif algorithm == 'autoroot':
        root_causes = autoroot(df, attributes, debug=debug, **algorithm_args)
    elif algorithm == "squeeze":
        if not derived:
            model = Squeeze(
                data_list=[df],
                op=lambda x: x,
                option=SqueezeOption(debug=debug, **algorithm_args)
            )
        else:
            divide = lambda x, y: np.divide(x, y, out=np.zeros_like(x), where=y != 0)
            model = Squeeze(
                data_list=dfs,
                op=divide,
                option=SqueezeOption(debug=debug, **algorithm_args)
            )
        model.run()
        root_causes = model.root_cause_string_list
    elif algorithm == "hotspot":
        root_causes = [hotspot(df, attributes, debug=debug, **algorithm_args)]
    elif algorithm == "adtributor":
        root_causes = adtributor(df, attributes, derived=derived, **algorithm_args)
    elif algorithm == "r_adtributor":
        root_causes = rev_rec_adtributor(df, attributes, derived=derived, **algorithm_args)
    else:
        raise ValueError("method", algorithm, "not implemented.")
    return root_causes


def read_dataframe(directory, file, derived):
    """
    Reads a root cause example file.
    :param directory: str, teh directory with files.
    :param file: str, the csv file to use (note: without appending .csv).
    :param derived: boolean, if the dataset is a derived measure.
    :return: pandas dataframe, attributes, if derived: non-merged dataframes (used for squeeze).
    """
    def get_attributes(df):
        return sorted(df.columns.drop(['real', 'predict']).tolist())

    if derived:
        file_a = file + '.a.csv'
        file_b = file + '.b.csv'
        df_a = pd.read_csv(os.path.join(directory, file_a))
        df_b = pd.read_csv(os.path.join(directory, file_b))

        attributes = get_attributes(df_a)

        df = pd.merge(df_a, df_b, on=attributes, suffixes=('_a', '_b'))
        df['real'] = df['real_a'] / df['real_b']
        df['predict'] = df['predict_a'] / df['predict_b']
        df = df.fillna(0.0)  # Fix any nans that appeared after dividing by 0.
    else:
        df = pd.read_csv(os.path.join(directory, file + '.csv'))
        attributes = get_attributes(df)
        df_a = df_b = None

    return df, attributes, df_a, df_b


def get_instances(data_root, directory):
    """
    Obtains all instances (files) to be run within a folder.
    :param: data_root: str, the root directory for the datasets.
    :param directory: str, the directory to be run.
    :return: list with tuples containing all files to be run.
    """
    path = os.path.join(data_root, directory)
    subdirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    if len(subdirs) > 0:
        # if there are any directories then we want to recursively go deeper
        subdir_insts = [get_instances(data_root, os.path.join(directory, subdir)) for subdir in subdirs]
        instances = [inst for subdir_inst in subdir_insts for inst in subdir_inst]
    else:
        # deepest level, get all files
        dir_split = directory.split(os.sep)
        dataset = dir_split[0]
        subdir = os.path.join(*dir_split[1:]) if len(dir_split) > 1 else ''

        instances = []
        for file in os.listdir(path):
            # ignore the file with labels
            if os.path.isfile(os.path.join(path, file)) and file != 'injection_info.csv':
                instance = (dataset, subdir, file.split(".")[0])
                instances.append(instance)

        # the D folder contains 2 files for each timestamp, so only keep unique ones.
        instances = list(set(instances))
    return instances


def result_post_processing(parallel_run_results, algorithm, csv_suffix):
    """
    Postprocess of the results and saving result csv files.
    :param parallel_run_results: algorithm results.
    """
    df = pd.DataFrame(parallel_run_results)
    df.columns = ['Dataset', 'Folder', 'File', 'F1', 'TP', 'FP', 'FN', 'Time']
    df = df.sort_values(['Dataset', 'Folder'])

    df_summary = df.copy()
    A_folder_split = 'layer_' + df_summary['Folder'].str.split('_').str[-1] + \
                     '_elements_' + df_summary['Folder'].str.split('_').str[-3]
    df_summary['Folder'] = np.where(df_summary['Dataset'] == 'A', A_folder_split, df_summary['Folder'])

    df_summary = df_summary.groupby(['Dataset', 'Folder'], as_index=False). \
        agg({'TP': sum, 'FP': sum, 'FN': sum, 'Time': sum})
    df_summary = df_summary.sort_values(['Dataset', 'Folder'])
    df_summary['F1-score'] = 2 * df_summary['TP'] / (2 * df_summary['TP'] + df_summary['FP'] + df_summary['FN'])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_summary)

    df.to_csv(algorithm + '-all' + csv_suffix + '.csv', index=False)
    df_summary.to_csv(algorithm + '-summary' + csv_suffix + '.csv', index=False)
import os
import time
import pandas as pd
from multiprocessing import Pool
from utils.argument_parser import get_input_arguments
from utils.run_utils import get_instances, read_dataframe, run_method, result_post_processing
from utils.evaluation import root_cause_postprocessing, score_root_causes


def run_directory(data_root, run_path, algorithm, algorithm_args, derived, n_threads, csv_suffix, debug):
    """
    Run all files in all subdirectories of run_path.
    """
    parallel_run_results = []

    def parallel_callback(result):
        parallel_run_results.append(result)

    instances = get_instances(data_root, run_path)

    pool = Pool(n_threads)
    for dataset, sub_directory, file in instances:

        if derived is None:
            derived = os.path.basename(dataset) == 'D'

        pool.apply_async(run_instance,
                         args=(data_root, dataset, sub_directory, file, algorithm, algorithm_args, derived, debug),
                         callback=parallel_callback)
    pool.close()
    pool.join()

    result_post_processing(parallel_run_results, algorithm, csv_suffix)


def run_single_file(data_root, run_path, algorithm, algorithm_args, derived):
    """
    Run a single file.
    """
    directory_structure = list(filter(None, run_path.split(os.sep)))
    dataset_name = directory_structure[0] if len(directory_structure) > 1 else ''
    sub_directory = os.path.join(*directory_structure[1:-1]) if len(directory_structure) > 2 else ''
    file = directory_structure[-1].split('.')[0]

    if derived is None:
        derived = dataset_name == 'D'

    run_instance(data_root, dataset_name, sub_directory, file, algorithm, algorithm_args, derived, debug=True)


def run_instance(data_root, dataset_name, sub_directory, file, algorithm, algorithm_args, derived=False, debug=False):
    """
    Runs a single instance (file) and evaluates the result.
    :param data_root: str, the root directory for all datasets.
    :param dataset_name: str, the name of the dataset to run (must be located within data_root).
    :param sub_directory: str, subdirectory of the dataset (can be an empty string or of a depth >= 1).
    :param file: str, the file to run. Should not have any file extension (assumed to be csv).
    :param algorithm: str, the name of the algorithm that should be run.
    :param algorithm_args: dict, any algorithm specific arguments.
    :param derived: boolean, if the dataset is derived.
           In this case, two files `file`.a.csv and `file`.b.csv. must exist.
    :param debug: boolean, if debug mode should be used.
    :return: (str, str, str, float, float, float, float, float), the dataset name, subdirectory and file name
             are all returned for collecting the results when using multiple threads. Moreover, the F1-score,
             true positive count, false positive count, false negative count and the run time are also returned.
    """
    run_directory = os.path.join(data_root, dataset_name, sub_directory)
    print('Running file:', os.path.join(run_directory, file), ', derived:', derived)

    df, attributes, df_a, df_b = read_dataframe(run_directory, file, derived)
    start_time = time.time()
    root_causes = run_method(df, [df_a, df_b], attributes, algorithm, algorithm_args, derived, debug)
    root_cause_predictions = root_cause_postprocessing(root_causes, algorithm)
    run_time = time.time() - start_time

    # Get the label.
    labels = pd.read_csv(os.path.join(run_directory, 'injection_info.csv'))
    label = labels.loc[labels['timestamp'] == int(file), 'set'].iloc[0]

    # Evaluate the root cause.
    TP, FP, FN, true_labels = score_root_causes(root_cause_predictions, label)
    F1 = 2 * TP / (2 * TP + FP + FN)

    print('dataset:', dataset_name, 'sub_directory:', sub_directory, 'file:', file, 'label:', label)
    if debug or FP > 0 or FN > 0:
        print('Run time:', run_time)
        print('TP:', TP, 'FP:', FP, 'FN:', FN)
        print('True labels:     ', true_labels)
        print('Predicted labels:', root_cause_predictions)

    return dataset_name, sub_directory, file, F1, TP, FP, FN, run_time


if __name__ == "__main__":

    # Get the parsed input arguments.
    args, data_root, run_path, algorithm_args, is_single_file = get_input_arguments()

    print('Running', args.algorithm, 'with arguments:', algorithm_args)
    if is_single_file:
        run_single_file(data_root, run_path, args.algorithm, algorithm_args, args.derived)
    else:
        # Add algorithm specific arguments to the given csv suffix.
        argument_list = [k + '-' + str(v).replace('.', '') for k, v in algorithm_args.items()]
        csv_suffix = '-'.join(['', args.output_suffix, *argument_list])
        csv_suffix = csv_suffix if args.output_suffix != '' else csv_suffix[1:]

        run_directory(data_root, run_path, args.algorithm, algorithm_args, args.derived, args.n_threads, csv_suffix,
                      args.debug)

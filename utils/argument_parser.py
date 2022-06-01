import os
import argparse


def get_input_arguments():
    """
    Construct argument parser.
    :return: argparse.ArgumentParser
    """
    # Parent parser
    parser = argparse.ArgumentParser(description='RiskLoc')
    subparsers = parser.add_subparsers(help='algorithm specific help', dest='algorithm', required=True)
    common_arguments = ['algorithm', 'data_root', 'run_path', 'derived', 'n_threads', 'output_suffix', 'debug']

    # Riskloc algorithm specific parameters
    subparser_riskloc = subparsers.add_parser('riskloc', help='riskloc help')
    subparser_riskloc = add_common_arguments(subparser_riskloc)
    subparser_riskloc.add_argument('--risk-threshold', type=float, default=0.5, help='risk threshold')
    subparser_riskloc.add_argument('--pep-threshold', type=float, default=0.02,
                                   help='proportional explanatory power threshold')
    subparser_riskloc.add_argument('--prune-elements', type=str2bool, nargs='?', const=True, default=True,
                                   help='use element pruning (True/False)')

    # AutoRoot algorithm specific parameters
    subparser_autoroot = subparsers.add_parser('autoroot', help='autoroot help')
    subparser_autoroot = add_common_arguments(subparser_autoroot)
    subparser_autoroot.add_argument('--delta-threshold', type=float, default=0.25, help='delta threshold')

    # Squeeze algorithm specific parameters
    subparser_squeeze = subparsers.add_parser('squeeze', help='squeeze help')
    subparser_squeeze = add_common_arguments(subparser_squeeze)
    subparser_squeeze.add_argument('--ps-upper-bound', type=float, default=0.9, help='threshold')
    subparser_squeeze.add_argument('--max-num-elements-single-cluster', type=int, default=12,
                                   help='maximum number of elements returned for a cluster')

    # HotSpot algorithm specific parameters
    subparser_hotspot = subparsers.add_parser('hotspot', help='autoroot help')
    subparser_hotspot = add_common_arguments(subparser_hotspot)
    subparser_hotspot.add_argument('--pt', type=float, default=0.8, help='PT threshold')
    subparser_hotspot.add_argument('--m', type=float, default=200, help='maximum number of MCTS iterations')
    subparser_hotspot.add_argument('--scoring', type=str, default='gps', choices=['gps', 'ps'],
                                   help='scoring method to use')

    # R_adtributor algorithm specific parameters
    subparser_r_adtributor = subparsers.add_parser('r_adtributor', help='r_adtributor help')
    subparser_r_adtributor = add_common_arguments(subparser_r_adtributor)
    subparser_r_adtributor.add_argument('--teep', type=float, default=0.2,
                                        help='per-element explanatory power threshold')
    subparser_r_adtributor.add_argument('--k', type=int, default=3, help='number of returned root cause elements')

    # Adtributor algorithm specific parameters
    subparser_adtributor = subparsers.add_parser('adtributor', help='adtributor help')
    subparser_adtributor = add_common_arguments(subparser_adtributor)
    subparser_adtributor.add_argument('--tep', type=float, default=0.1, help='total explanatory power threshold')
    subparser_adtributor.add_argument('--teep', type=float, default=0.1,
                                      help='per-element explanatory power threshold')
    subparser_adtributor.add_argument('--k', type=int, default=3, help='number of returned root cause elements')

    args = parser.parse_args()
    data_root, run_path, algorithm_args, is_single_file = process_arguments(args, common_arguments)
    return args, data_root, run_path, algorithm_args, is_single_file


def add_common_arguments(parser):
    """
    Adds shared arguments to a parser.
    :param parser: argparse.ArgumentParser, the parser to use.
    :return: parser with added arguments.
    """
    parser.add_argument('--data-root', type=str, default='./data',
                        help='root directory for all datasets (default ./data)')
    parser.add_argument('--run-path', type=str, default='./data',
                        help='''directory or file to be run; 
                                if a directory, any subdirectories will be considered as well;
                                must contain data-path as a prefix
                        ''')
    parser.add_argument('--derived', type=str2bool, nargs='?', const=True, default=None,
                        help='derived dataset (defaults to True for the D dataset and False for others)')
    parser.add_argument('--n-threads', type=int, default=10, help='number of threads to run')
    parser.add_argument('--output-suffix', type=str, default='', help='suffix for output csv file')
    parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False,
                        help='debug mode')
    return parser


def process_arguments(args, common_arguments):
    """
    Preprocessing for the run arguments. Normalizes paths and checks their validity.
    Obtains all algorithm specific arguments and checks whether a single file or a directory should be run.
    :param args: namespace, the run arguments.
    :param common_arguments: list of strings, all argument names that are shared between the algorithms.
    :return: (str, str, dict, boolean), normalized data root directory, normalized run path with root removed,
             algorithm specific arguments, if running a single file.
    """
    data_root = os.path.normpath(args.data_root).lstrip('/')
    run_path = os.path.normpath(args.run_path).lstrip('/')
    if not run_path.startswith(data_root):
        raise argparse.ArgumentTypeError(f'data-root ({args.data_root}) must be prefix of run-path ({args.run_path}).')

    run_path = run_path[len(data_root) + 1:]

    # Get all algorithm specific arguments
    algorithm_args = {k: v for k, v in vars(args).items() if k not in common_arguments}

    is_single_file = run_path.endswith('csv')
    return data_root, run_path, algorithm_args, is_single_file


def limited_number_type(return_type=float, minimum=None, maximum=None):
    """
    Helper function to process input arguments with a minimum and maximum value.
    :param return_type: the type of the return values, typically float or int.
    :param minimum: float or None, the minimum value allowed.
    :param maximum: float or None, the maximum value allowed.
    :return: a function that will convert the input to `return_type` and check it's bounds.
    """
    def range_checker(arg):
        try:
            n = return_type(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(''.join(['Must be a ', str(return_type), ' number.']))
        if minimum is not None and n < minimum:
            raise argparse.ArgumentTypeError(''.join(['Minimum value must be larger than ', str(minimum), '.']))
        if maximum is not None and n > maximum:
            raise argparse.ArgumentTypeError(''.join(['Maximum value must be smaller than ', str(maximum), '.']))
        return n
    return range_checker


def str2bool(argument):
    """
    Helper function to process input arguments of boolean type.
    :param argument: str
    :return: boolean
    """
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


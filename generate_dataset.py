import os
import argparse
import itertools
import numpy as np
import pandas as pd
from functools import reduce
from utils.utils import str2bool, limited_number_type

parser = argparse.ArgumentParser(description='Generate synthetic dataset.')
parser.add_argument('--num', required=True, type=int, help='number of instances to generate')
parser.add_argument('--dims', type=int, nargs='+', default=[10, 12, 10, 8, 5], help='dimension sizes')
parser.add_argument('--seed', type=int, default=123, help='initial random seed')
parser.add_argument('--data-root', type=str, default='./data', help='output directory')
parser.add_argument('--dataset-name', type=str, help='name of the new dataset')
parser.add_argument('--weibull-alpha', type=limited_number_type(float, minimum=0), nargs=2, default=[0.5, 1.0],
                    metavar=('min', 'max'),
                    help='range of Weibull alpha to control the real value distribution of normal leaf elements')
parser.add_argument('--zero-rate', type=limited_number_type(float, 0, 1), nargs=2, default=[0.0, 0.25],
                    metavar=('min', 'max'), help='range of percentage of leaf elements to set to zero')
parser.add_argument('--noise-level', type=limited_number_type(float, 0, 1), nargs=2, default=[0.0, 0.25],
                    metavar=('min', 'max'), help='range of relative forecasting error of the normal leaf elements')
parser.add_argument('--anomaly-severity', type=limited_number_type(float, 0, 1), nargs=2, default=[0.2, 1.0],
                    metavar=('min', 'max'), help='range of severity of the anomalies')
parser.add_argument('--anomaly-deviation', type=limited_number_type(float, 0, 1), nargs=2, default=[0.0, 0.1],
                    metavar=('min', 'max'), help='range of deviation of the anomalies')
parser.add_argument('--num-anomaly', type=limited_number_type(int, minimum=1), nargs=2, default=[1, 3],
                    metavar=('min', 'max'), help='range of number of anomalies')
parser.add_argument('--num-anomaly-elements', type=limited_number_type(int, minimum=1), nargs=2, default=[1, 3],
                    metavar=('min', 'max'), help='range of number of elements within each anomaly')
parser.add_argument('--only-last-layer', type=str2bool, nargs='?', const=True, default=False,
                    help='place all anomalies in the highest layer')
args = parser.parse_args()

# S: 121, L: 122, H: 123
rng = np.random.default_rng(args.seed)

# Basic settings
# S: dimensions = {'a': 10, 'b': 12, 'c': 10, 'd': 8, 'e': 5}
# L: dimensions = {'a': 10, 'b': 24, 'c': 10, 'd': 15}
# 2: dimensions = {'a': 10, 'b': 5, 'c': 250, 'd': 20, 'e': 8, 'f': 12}
dimensions = dict(zip('abcdefghijklmnopqrstuvwxyz', args.dims))

# S and L uses 1000, H dataset uses 100
number_of_files = args.num

# The normal ('real') value distribution
weibull_alpha = [min(args.weibull_alpha), max(args.weibull_alpha)]

# Noise parameters
zero_rate = [min(args.zero_rate), max(args.zero_rate)]
noise_level = [min(args.noise_level), max(args.noise_level)]  # S,H: [0.0,0.25], L: [0.0,0.1]

# Anomaly parameters
anomaly_severity = [min(args.anomaly_severity), max(args.anomaly_severity)]              # S,H: [0.2,1.0], L: [0.5,1.0]
anomaly_deviation = [min(args.anomaly_deviation), max(args.anomaly_deviation)]           # S,H: [0.0,0.1], L: [0.0,0.0]
num_anomaly = [min(args.num_anomaly), max(args.num_anomaly)]                             # S,H: [1,3], L: [1,5]
num_anomaly_elements = [min(args.num_anomaly_elements), max(args.num_anomaly_elements)]  # S,H: [1,3], L: [1,1]
only_last_layer = args.only_last_layer                                                   # S,H: False, L: True

# Set the output path
data_root = args.data_root
if args.dataset_name is None:
    folder = '-'.join(sorted([k + str(v) for k, v in dimensions.items()]))
else:
    folder = args.dataset_name
save_path = os.path.join(data_root, folder)
os.makedirs(save_path, exist_ok=True)


def get_dataset_properties(dimensions):
    sel_zero_rate = rng.uniform(zero_rate[0], zero_rate[1])
    sel_noise_level = rng.uniform(noise_level[0], noise_level[1])

    print('zero_rate', sel_zero_rate)
    print('noise_level', sel_noise_level)

    sel_num_anomalies = rng.integers(num_anomaly[0], num_anomaly[1], endpoint=True)
    print('num_anomalies', sel_num_anomalies)

    anomaly_properties = dict()
    for i in range(sel_num_anomalies):
        if not only_last_layer:
            anomaly_level = rng.integers(1, len(dimensions), endpoint=True)
        else:
            anomaly_level = len(dimensions)

        elements = rng.integers(num_anomaly_elements[0], num_anomaly_elements[1], endpoint=True)

        # Add noise_level to the severity to avoid anomalies too alike to normal data.
        severity = rng.uniform(anomaly_severity[0], anomaly_severity[1]) + sel_noise_level
        deviation = rng.uniform(anomaly_deviation[0], anomaly_deviation[1])
        anomaly_properties[i] = {'level': anomaly_level, 'elements': elements,
                                 'severity': severity, 'deviation': deviation}

    print('anomaly_properties', anomaly_properties)
    return sel_zero_rate, sel_noise_level, anomaly_properties


def get_anomaly_locations(df, dimensions, anomaly_properties):
    def _get_anomaly_locations(anomaly_level, anomaly_elements, current_anomalies, depth=0):
        if len(dimensions) < anomaly_level:
            raise ValueError("anaomly_level should be less or equal to number of dimensions.")

        if np.prod(sorted(list(dimensions.values()), reverse=True)[:anomaly_level]) < anomaly_elements:
            raise ValueError("Impossible to get that many anomaly elements with specified input.")

        anomaly_dims = sorted(rng.choice(list(dimensions.keys()), size=anomaly_level, replace=False))
        lowest_layer = len(anomaly_dims) == len(dimensions)

        # Do not reuse the same anomaly dimension again
        used_dims = [ca['dimensions'] for ca in current_anomalies]
        if anomaly_dims in used_dims and not lowest_layer:
            print('Anomaly dimension used, recursive call.')
            return _get_anomaly_locations(anomaly_level, anomaly_elements, current_anomalies)

        all_selected_dim_elements = []
        for ad in anomaly_dims:
            dim_elements = list(range(1, dimensions[ad] + 1))

            # We do not want any overlap on current anomalies
            # For example, if one anomaly is c=c5 then we do not want an overlapping one at e.g. c=c5&b=b3.
            for ca in current_anomalies:
                if ad in ca['dimensions']:
                    # Overlap
                    idx = ca['dimensions'].index(ad)
                    used_elements = set([int(a[idx][len(ad):]) for a in ca['cuboids']])
                    dim_elements = list(set(dim_elements) - used_elements)

            if len(dim_elements) == 0:
                print('All elements in dimension used, recursive call.')
                return _get_anomaly_locations(anomaly_level, anomaly_elements, current_anomalies, depth=depth+1)

            selected_dim_elements = rng.choice(dim_elements, size=anomaly_elements, replace=True)
            selected_dim_elements = [ad + str(e) for e in selected_dim_elements]
            all_selected_dim_elements.append(selected_dim_elements)

        anomaly_cuboids = list(zip(*all_selected_dim_elements))

        # When in the lowest layer, we do not want any anomalies with real == 0 and predict == 0,
        # since we can't predict these (i.e., they would automatically be a false negative when evaluating).
        if lowest_layer:
            for cuboid in anomaly_cuboids:
                element = list(zip(anomaly_dims, cuboid))
                v = df.loc[reduce(lambda c1, c2: c1 & c2, [df[k] == v for k, v in element]), 'real'].iloc[0]
                if v == 0:
                    print('Tried anomaly value is 0, recursive call.')
                    return _get_anomaly_locations(anomaly_level, anomaly_elements, current_anomalies)

        # Make sure the anomalies are unique
        if len(np.unique(anomaly_cuboids, axis=0)) < anomaly_elements:
            print('Non-unique elements found, recursive call.')
            return _get_anomaly_locations(anomaly_level, anomaly_elements, current_anomalies)

        return anomaly_dims, anomaly_cuboids

    anomalies = []
    for properties in anomaly_properties.values():
        level = properties['level']
        elements = properties['elements']
        anomaly_dims, anomaly_cuboids = _get_anomaly_locations(level, elements, anomalies)
        anomalies.append({'dimensions': anomaly_dims, 'cuboids': anomaly_cuboids})
    return anomalies


def get_anomaly_masks(df, anomalies):
    anomaly_masks = []
    for anomaly in anomalies:
        dims = anomaly['dimensions']

        cuboid_mask = []
        for cuboid in anomaly['cuboids'] :
            tup = zip(dims, cuboid) if len(dims) > 1 else [(dims[0], cuboid[0])]
            masks = [df[d] == c for d, c in tup]
            mask = np.logical_and.reduce(masks)
            cuboid_mask.append(mask)
        cuboid_mask = np.logical_or.reduce(cuboid_mask)

        anomaly_masks.append(cuboid_mask)
    return anomaly_masks


def scale_anomaly(row, properties):
    severity = properties['severity']
    deviation = properties['deviation']

    r = rng.normal(loc=severity, scale=deviation)
    v = max(row * (1 - r), 0.0)
    return v


def generate_labels(anomalies):
    labels = []
    for anomaly in anomalies:
        dims = anomaly['dimensions']
        for element in anomaly['cuboids']:
            ts = zip(dims, element) if len(dims) > 1 else [(dims[0], element[0])]
            label = "&".join(sorted(["=".join(t) for t in ts]))
            labels.append(label)
    labels = ';'.join(labels)
    return labels


def create_metadata(df, anomaly_masks, anomaly_properties, zero_rate, noise_level, direction):
    mask = np.logical_or.reduce(anomaly_masks)
    normal_predict_amount = df.loc[~mask, 'predict'].abs().sum()
    normal_predict_error = (df.loc[~mask, 'real'] - df.loc[~mask, 'predict']).abs().sum()

    abnormal_predict_amount = df.loc[mask, 'predict'].abs().sum()
    abnormal_predict_error = (df.loc[mask, 'real'] - df.loc[mask, 'predict']).abs().sum()
    anomaly_significance = abnormal_predict_error / df['predict'].sum()

    anomaly_severities = ';'.join([str(round(anomaly_properties[i]['severity'], 2)) for i in range(len(anomaly_masks))])
    anomaly_deviations = ';'.join([str(round(anomaly_properties[i]['deviation'], 2)) for i in range(len(anomaly_masks))])
    anomaly_elements = ';'.join([str(round(anomaly_properties[i]['elements'], 2)) for i in range(len(anomaly_masks))])
    metadata = {'total_real_amount': df['real'].sum().round(2),
                 'total_predict_amount': df['predict'].sum().round(2),
                 'normal_predict_amount': round(normal_predict_amount, 2),
                 'normal_predict_error': round(normal_predict_error, 2),
                 'abnormal_predict_amount': round(abnormal_predict_amount, 2),
                 'abnormal_predict_error': round(abnormal_predict_error, 2),
                 'anomaly_significance': round(anomaly_significance, 2),
                 'zero_rate': round(zero_rate, 2),
                 'noise_level': round(noise_level, 2),
                 'elements_per_anomaly': anomaly_elements,
                 'anomaly_severity': anomaly_severities,
                 'anomaly_deviation': anomaly_deviations,
                 'anomaly_direction': direction
                 }
    return metadata


def generate_dataset(dimensions):
    sel_zero_rate, sel_noise_level, anomaly_properties = get_dataset_properties(dimensions)
    dimension_values = [[dimension + str(i) for i in range(1, num + 1)] for dimension, num in dimensions.items()]

    # Add all combinations
    df = pd.DataFrame(list(itertools.product(*dimension_values)), columns=dimensions.keys())

    # Add real values
    alpha = rng.uniform(weibull_alpha[0], weibull_alpha[1])
    df['real'] = rng.weibull(alpha, len(df)) * 100
    print('Added real')

    # Add zero rows
    df['real'] = df['real'] * (rng.uniform(size=len(df)) > sel_zero_rate)
    print('Added zero rows')

    # Apply noise to the predictions
    df['predict'] = df['real'] + df['real'] * rng.normal(loc=0, size=len(df), scale=sel_noise_level)
    print('Added noise')

    # Swap for equal distribution on both sides
    p = df['predict'].copy()
    r = rng.integers(0, 1, size=len(df), endpoint=True)
    df['predict'] = np.where(r == 1, df['real'], df['predict'])
    df['real'] = np.where(r == 0, df['real'], p)
    df.loc[df['predict'] < 0, 'predict'] = 0.0
    del p
    del r
    print('Swap predict/real')

    # Create and add anomalies
    anomalies = get_anomaly_locations(df, dimensions, anomaly_properties)
    print('anomalies', anomalies)

    anomaly_masks = get_anomaly_masks(df, anomalies)

    # We set the direction following the prediction error direction in the normal data.
    # This is to make sure that the error direction in the normal data does not overweigh
    # the error introduced by the anomalies.
    direction = 1 if df['real'].sum() > df['predict'].sum() else 0

    for i, anomaly_mask in enumerate(anomaly_masks):
        properties = anomaly_properties[i]
        if direction == 0:
            df.loc[anomaly_mask, 'real'] = df.loc[anomaly_mask, 'predict']  # reset the noise
            df.loc[anomaly_mask, 'real'] = df.loc[anomaly_mask, 'real'].apply(scale_anomaly, properties=properties)
        else:
            df.loc[anomaly_mask, 'predict'] = df.loc[anomaly_mask, 'real']  # reset the noise
            df.loc[anomaly_mask, 'predict'] = df.loc[anomaly_mask, 'predict'].apply(scale_anomaly, properties=properties)

    labels = generate_labels(anomalies)
    metadata = create_metadata(df, anomaly_masks, anomaly_properties, sel_zero_rate, sel_noise_level, direction)
    return df, labels, metadata


def generate_filename(n, used_names):
    filename = 0
    while filename == 0 or filename in used_names:
        range_start = 10**(n-1)
        range_end = (10**n)-1
        filename = str(rng.integers(range_start, range_end))
    return filename


if __name__ == "__main__":
    used_names = set()
    dataset_info = dict()
    for i in range(number_of_files):
        filename = generate_filename(6, used_names)
        used_names.add(filename)

        print('creating file', filename)
        df, labels, metadata = generate_dataset(dimensions)
        df.to_csv(os.path.join(save_path, filename + '.csv'), index=False)
        dataset_info[filename] = {'set': labels,  **metadata}
        del df
        print('-----------------------')

        # Save inside the loop for intermediate results
        df_info = pd.DataFrame.from_dict(dataset_info, orient='index')
        df_info = df_info.rename_axis('timestamp').reset_index()
        df_info.to_csv(os.path.join(save_path, 'injection_info.csv'), index=False)

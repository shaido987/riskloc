import numpy as np
import pandas as pd
from itertools import combinations
from utils.element_scores import add_deviation_score
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema


def get_unique_elements(df, cuboid):
    return np.vstack(list({tuple(row) for row in df[cuboid].values}))


def get_elements_mask(df, cuboid, elements):
    return np.logical_and.reduce(np.logical_or.reduce([(df[cuboid] == e).values for e in elements], axis=0), axis=1)


def nps(selection, non_selection):
    sel_real, sel_pred = selection['real'], selection['predict']
    non_sel_real, non_sel_pred = non_selection['real'], non_selection['predict']

    with np.errstate(divide='ignore', invalid='ignore'):
        selection_a = np.nan_to_num(sel_pred * (sel_real.sum() / sel_pred.sum()))

    a = np.mean(np.nan_to_num(np.abs(sel_real - selection_a) / sel_real, posinf=0, neginf=0, nan=0))
    b = np.mean(np.nan_to_num(np.abs(sel_real - sel_pred) / sel_real, posinf=0, neginf=0, nan=0))
    c = np.mean(np.nan_to_num(np.abs(non_sel_real - non_sel_pred) / non_sel_real, posinf=0, neginf=0, nan=0))
    return 1 - ((a + c) / (b + c))


def kde_clustering(df):
    values = df['deviation'].values

    if len(np.unique(values)) == 1:
        df['cluster'] = 1
        return df

    kernel = gaussian_kde(values, bw_method='silverman')

    s = np.linspace(-2, 2, 400)
    e = kernel.evaluate(s)
    mi = argrelextrema(e, np.less)[0]

    # All ends in reverse order
    ends = sorted(np.concatenate((s[mi], [np.inf])), reverse=True)
    for i, end in enumerate(ends):
        df.loc[df['deviation'] <= end, 'cluster'] = i
    return df


def is_subset(parent, child):
    return all([any([p.issubset(c) for p in parent]) for c in child])


def remove_crc(cluster_root_causes, elem_to_remove):
    def filter_crc(crc):
        root_cause_set = set([frozenset(elem) for elem in crc['elements']])
        return root_cause_set == elem_to_remove

    return [crc for crc in cluster_root_causes if not filter_crc(crc)]


def remove_same_layer(cluster_root_causes):
    # Merge if exactly the same root cause
    duplicates = []
    for p, c in combinations(enumerate(cluster_root_causes), 2):
        if p[1]['layer'] == c[1]['layer']:
            parent_set = set([frozenset(elems) for elems in p[1]['elements']])
            child_set = set([frozenset(elems) for elems in c[1]['elements']])
            if is_subset(parent_set, child_set):
                duplicates.append(p[0])
    mask = np.full(len(cluster_root_causes), True, dtype=bool)
    mask[duplicates] = False
    cluster_root_causes = np.array(cluster_root_causes)[mask].tolist()
    return cluster_root_causes


def merge_root_causes(cluster_root_causes, max_layer=4):
    cluster_root_causes = remove_same_layer(cluster_root_causes)

    for layer in range(max_layer - 1, 0, -1):
        layer_root_causes = [set([frozenset(elems) for elems in crc['elements']]) for crc in cluster_root_causes if
                             crc['layer'] == layer]
        higher_layer_root_causes = [set([frozenset(elems) for elems in crc['elements']]) for crc in cluster_root_causes
                                    if crc['layer'] > layer]

        for child in higher_layer_root_causes:
            for parent in layer_root_causes:
                if is_subset(parent, child):
                    print('parent', parent, 'child', child)
                    cluster_root_causes = remove_crc(cluster_root_causes, child)
    return cluster_root_causes


def search_cluster(df, df_cluster, attributes, delta_threshold, debug=False):
    z = len(df_cluster)

    best_root_cause = {'avg': -1.0}
    for layer in range(1, len(attributes) + 1):
        if debug: print('Layer:', layer)
        cuboids = [list(c) for c in combinations(attributes, layer)]
        for cuboid in cuboids:
            if debug: print('Cuboid:', cuboid)

            # Way too many to go through. This is probably not what is done.
            # elements = get_unique_elements(df_cluster, cuboid)
            # splits = [t for r in range(1, len(elements) + 1) for t in list(combinations(elements, r))]

            # if last layer, we only run if CF can be above the threshold
            best_candidate = {'NPS': -1.0}
            if layer == len(attributes):
                CF = 1 / len(df_cluster)
                if CF <= delta_threshold:
                    continue

            xs = df_cluster.groupby(cuboid)['real'].count()
            xs = xs.loc[(xs / z) > delta_threshold]
            xs.name = 'x'

            ys = df.groupby(cuboid)['real'].count()
            ys.name = 'y'
            splits = pd.concat([xs, ys], axis=1, join='inner')
            splits['LF'] = splits['x'] / splits['y']
            splits = splits.loc[splits['LF'] > delta_threshold]

            for s, row in splits.iterrows():
                split = [s] if layer == 1 else s
                mask = get_elements_mask(df, cuboid, split)

                selection = df.loc[mask]
                non_selection = df.loc[~mask]
                nps_score = nps(selection, non_selection)
                if nps_score > best_candidate['NPS']:
                    CF = row['x'] / z
                    avg_score = (nps_score + row['LF'] + CF) / 3
                    candidate = {'elements': [split], 'layer': layer, 'cuboid': cuboid,
                                 'LF': row['LF'], 'CF': CF, 'NPS': nps_score, 'avg': avg_score}
                    best_candidate = candidate.copy()

            if 'elements' in best_candidate and best_candidate['avg'] > best_root_cause['avg']:
                best_root_cause = best_candidate.copy()

    if 'elements' not in best_root_cause:
        return None
    return best_root_cause


def autoroot(df, attributes, delta_threshold=0.1, debug=False):
    df = add_deviation_score(df)

    # Filter away the uninteresting elements with a score [-0.2,0.2].
    # (The deviation score here uses a multiple 2.)
    df_relevant = df.loc[df['deviation'].abs() > 0.2].copy()

    df_relevant = kde_clustering(df_relevant)
    clusters = df_relevant['cluster'].unique()
    if debug: print('clusters:', clusters)

    cluster_root_causes = []
    for cluster in clusters:
        if debug: print("Cluster:", cluster)
        df_cluster = df_relevant.loc[df_relevant['cluster'] == cluster].copy()

        root_cause = search_cluster(df, df_cluster, attributes, delta_threshold, debug)
        if root_cause is not None:
            root_cause['cluster'] = cluster
            cluster_root_causes.append(root_cause)

    if debug: print('root causes before merge:', cluster_root_causes)
    cluster_root_causes = merge_root_causes(cluster_root_causes, max_layer=len(attributes))
    return cluster_root_causes

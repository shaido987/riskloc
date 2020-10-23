import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from kneed import KneeLocator
from itertools import combinations


def filter_with_knee_point(df):
    vals = df['deviation'].abs().sort_values(ascending=False).values
    y = np.cumsum(np.log(vals + 1))
    x = list(range(len(y)))

    kneedle = KneeLocator(x, y)
    threshold = vals[kneedle.knee]
    
    df_filtered = df[abs(df['deviation']) >= threshold]
    return df_filtered


def density_clustering(df, bins=10, debug=False):

    # Use static sized bins over the whole range
    bins = np.linspace(-2, 2, bins + 1)
    df['bin'] = pd.cut(df['deviation'], bins)
    # df['bin'] = pd.cut(df['deviation'], bins)
    hist = df['bin'].value_counts().sort_index()

    if debug: print('hist', hist.values)

    bins, hists = hist.index.categories, hist.values
    hists = np.concatenate([[0], hists, [0]])
    bins = np.concatenate([[pd.Interval(bins[0].left - 1, bins[0].left)],
                           bins,
                           [pd.Interval(bins[-1].right, bins[-1].right + 1)]])

    centers = find_peaks(hists)[0]
    boundaries = np.concatenate([[0], find_peaks(-hists)[0], [len(hists)]])

    for center in centers:
        l = [bound for bound in boundaries if bound < center][-1]
        r = [bound for bound in boundaries if bound > center][0]
        cluster_bins = list(map(lambda x: str(x), bins[l:r+1]))

        if debug: print("cluster:", center, ", l:", l, ", r:", r)

        # Each element can only be in a single cluster.
        df.loc[df['bin'].astype(str).isin(cluster_bins), 'cluster'] = center
    return df


def get_unqiue_elements(df, cuboid):
    return np.vstack(list({tuple(row) for row in df[cuboid].values}))


def get_element_mask(df, cuboid, element):
    return np.logical_and.reduce((df[cuboid] == element).values, axis=1)


def GPS(selection, non_selection):
    selection_a = selection['predict'] * (selection['real'].sum() / selection['predict'].sum())
    a = np.mean(np.abs(selection['real'] - selection_a))
    b = np.mean(np.abs(selection['real'] - selection['predict']))
    c = np.nan_to_num(np.mean(np.abs(non_selection['real'] - non_selection['predict'])))
    return 1 - ((a + c) / (b + c))


def search_cluster(df_full, df_cluster_and_normal, df_cluster, attributes, C, threshold, debug=False):
    root_casues = []
    
    stop_next_layer = False
    for layer in range(1, len(attributes) + 1):
        if debug: print('Layer:', layer)
        cuboids = [list(c) for c in combinations(attributes, layer)]
        for cuboid in cuboids:
            if debug: print('Cuboid:', cuboid)
            elems = get_unqiue_elements(df_cluster, cuboid)

            elem_scores = []
            for elem in elems:
                n_ele = np.sum(get_element_mask(df_cluster, cuboid, elem))
                n_descents = np.sum(get_element_mask(df_full, cuboid, elem))
                mask = get_element_mask(df_cluster_and_normal, cuboid, elem)
                score = n_ele / n_descents
                
                if debug: print('element', elem, 'n_ele', n_ele, 'n_descents', n_descents, 'decent score', score)
                
                elem_scores.append((score, elem, n_ele, mask))
                
            elem_scores = sorted(elem_scores, key=lambda t: t[0], reverse=True)
        
            scores, split = [], []
            mask = np.full(len(df_cluster_and_normal), False)
            for i in range(len(elem_scores)):
                mask |= elem_scores[i][3]
                selection = df_cluster_and_normal.loc[mask]
                non_selection = df_cluster_and_normal.loc[~mask]
                score = GPS(selection, non_selection)
                split.append(elem_scores[i][1])
                scores.append((score, split.copy()))
            
            max_score, elements = max(scores, key=lambda t: t[0])
            root_casues.append({
                'elements': elements,
                'layer': layer,
                'cuboid': cuboid,
                'score': max_score,
            })
            
            if debug: print('Root cause:', root_casues[-1])
            
            if max_score > threshold:
                stop_next_layer = True
        
        if stop_next_layer:
            break    
        
    # Sort the root causes
    root_causes = sorted(root_casues, key=lambda rc: rc['score'] * C - len(rc['elements']) * rc['layer'], reverse=True)
    if debug: print('Sorted root causes', root_causes)
    
    # Return the top root cause
    return root_causes[0]


def squeeze(df, attributes, delta_threshold=0.9, debug=False):
    # Compute the deviation scores (f - v / f, while accounting for f=0)
    df['deviation'] = 2 * (df['predict'] - df['real']).divide(df['predict'] + df['real']).fillna(0.0)
    
    # Filter away the uninteresting elements.
    # Keep a copy to get the normal counts later on.
    df_original = df.copy()
    df = filter_with_knee_point(df)
    
    # Cluster by deviation score.
    df = density_clustering(df.copy(), debug=debug)
    clusters = df['cluster'].unique()

    # Get all normal elements
    in_any_cluster = df_original.index.isin(df.index)
    df_not_any_cluster = df_original.loc[~in_any_cluster]

    # Used to tune the trade-off between GPS and succinctness.
    n = np.log(len(clusters) * len(df) / len(df_original))
    attr_values = np.sum([len(df[a].unique()) for a in attributes])
    d = np.log(attr_values)
    C = -(n / d) * attr_values
    if debug: print('C', C)
    
    cluster_root_causes = []
    for cluster in clusters:
        # Ignore the normal cluster
        #if cluster == (bins \\ 2):
        #    continue

        if debug: print("Cluster:", cluster)
        df_cluster = df.loc[df['cluster'] == cluster].copy()
        df_cluster_and_normal = pd.concat([df_not_any_cluster, df_cluster], sort=True)
        root_cause = search_cluster(df_original, df_cluster_and_normal, df_cluster, attributes, C, delta_threshold, debug)

        root_cause['cluster'] = cluster
        cluster_root_causes.append(root_cause)
        
    return cluster_root_causes

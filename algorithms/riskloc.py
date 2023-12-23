import math
import numpy as np
from itertools import combinations
from collections import defaultdict
from utils.element_scores import add_explanatory_power, add_deviation_score


def get_element_mask(df, cuboid, element):
    return np.logical_and.reduce((df[cuboid] == element).values, axis=1)


def add_weight(df, cutoff):
    df['weight'] = (cutoff - df['deviation']).abs()
    df.loc[(df['real'] == 0) & (df['predict'] == 0), 'weight'] = 0
    df['weight'] = np.where(df['partition'] == 1, df['deviation'].abs(), df['weight'])
    df.loc[df['weight'] > 1, 'weight'] = 1  # max weight is 1
    return df


def add_partition(df, cutoff):
    if cutoff == 0:
        # We need to find the correct direction by checking the sign
        anomaly_right = math.copysign(1, cutoff) > 0
    else:
        anomaly_right = cutoff >= 0

    df['partition'] = 0
    if anomaly_right:
        df.loc[df['deviation'] > cutoff, 'partition'] = 1
    else:
        df.loc[df['deviation'] < cutoff, 'partition'] = 1
    return df


def get_cutoff(df, n_remove=5, relative=False):
    devs = df['deviation'].drop_duplicates()

    if relative:
        n_remove = int(math.ceil(len(devs) * n_remove / 100))

    # Ablation test
    # n_remove = 0

    min_val = devs.nsmallest(n_remove + 1).max()
    max_val = devs.nlargest(n_remove + 1).min()
    t = -min(min_val, max_val, key=abs)
    return t


def high_risk(selection):
    n_anomaly = selection.loc[selection['partition'] == 1, 'weight'].sum()
    n_normal = selection.loc[selection['partition'] == 0, 'weight'].sum() + 1
    high_risk_score = n_anomaly / (n_anomaly + n_normal)
    return high_risk_score


def low_risk(selection):
    # We can only do relative comparision when the real value != 0.
    # Predict = 0 will give a = 0, so not relevant for our d / selection['dev'] comparison.
    selection = selection.loc[(selection['real'] != 0) & (selection['predict'] != 0)]

    low_risk_score = 0.0
    if len(selection) > 0:
        a = selection['predict'] * selection['real'].sum() / selection['predict'].sum()
        d = (2 * (a - selection['real']) / (a + selection['real'])).fillna(0.0)

        w1 = d.abs().mean()
        w2 = selection['deviation'].abs().mean()

        if w2 != 0.0:
            low_risk_score = w1 / w2
    return low_risk_score


def element_pruning(df, cuboid, pruned_elements):
    if pruned_elements is None:
        return df, ['ep', 'partition']

    df_c = df.copy()
    keys = [key for key in pruned_elements.keys() if set(cuboid).issuperset(set(key))]
    for key in keys:
        ics = pruned_elements[key]
        # print('ignored cuboids:', ics)
        if len(ics) > 0:
            df_c = df_c.loc[~df_c.set_index(list(key)).index.isin(ics)]
    return df_c, ['ep', 'ep_z', 'partition']


def add_prune_element(eps, adj_ep_threshold, layer, cuboid, pruned_elements, max_layer=1):
    if layer <= max_layer:
        ics = eps.loc[(eps['ep_z'] < adj_ep_threshold) | (eps['partition'] == 0)]
        if len(ics) > 0:
            ics = ics.index.tolist()
            # print('added ignored cuboids:', ics)
            pruned_elements[tuple(cuboid)].extend(ics)
    return pruned_elements


def search_anomaly(df, attributes, pruned_elements, risk_threshold=0.5, adj_ep_threshold=0.0, debug=True):
    for layer in range(1, len(attributes) + 1):
        if debug: print('Layer:', layer)
        cuboids = [list(c) for c in combinations(attributes, layer)]

        best_root_cause = {'ep_score': adj_ep_threshold}
        for cuboid in cuboids:
            if debug: print('Cuboid:', cuboid)

            # Prune irrelevant cuboids if needed
            df_c, sum_cols = element_pruning(df, cuboid, pruned_elements)

            # Compute the cuboid element values
            eps = df_c.groupby(cuboid)[sum_cols].sum()

            # Check for cuboid pruning
            # need to use ep_z here instead of ep to make sure no potential root causes are missed.
            if pruned_elements is not None:
                pruned_elements = add_prune_element(eps, adj_ep_threshold, layer, cuboid, pruned_elements)

            # Filter away elements with too low EP scores.
            # Make sure at least 1 leaf element is in the anomaly partition (1).
            eps = eps.loc[(eps['partition'] > 0) & (eps['ep'] > best_root_cause['ep_score'])]
            eps = eps['ep'].sort_values(ascending=False)

            for e, ep_score in eps.items():
                element = (e,) if layer == 1 else e

                mask = get_element_mask(df_c, cuboid, element)
                selection = df_c[mask]

                high_risk_score = high_risk(selection)
                low_risk_score = low_risk(selection)

                # Ablation test
                # high_risk_score = 1.0

                # Ablation test
                # low_risk_score = 0.0

                risk_score = high_risk_score - low_risk_score

                if debug: print('element', element, 'ep score', ep_score, 'high', high_risk_score,
                                'low', low_risk_score, 'risk', risk_score)

                if risk_score >= risk_threshold:
                    if debug: print('New best score')

                    best_root_cause = {
                        'elements': [element],
                        'high risk score': high_risk_score,
                        'low risk score': low_risk_score,
                        'risk score': risk_score,
                        'ep_score': ep_score,
                        'layer': layer,
                        'cuboid': cuboid,
                    }

                    # We know the eps scores are in order (i.e., this is the one with highest ep in this cuboid)
                    # so we continue with the next cuboid in this layer.
                    break

        # If an element has been found
        if 'elements' in best_root_cause:
            return best_root_cause, pruned_elements
    return None, pruned_elements


def riskloc(df, attributes, risk_threshold=0.5, pep_threshold=0.02, n_remove=5, remove_relative=False, derived=False, 
            prune_elements=True, debug=False):
    df = add_explanatory_power(df, derived)
    df = add_deviation_score(df)

    cutoff = get_cutoff(df, n_remove, relative=remove_relative)
    if debug: print('cutoff:', cutoff)

    # Ablation test
    # cutoff = -0.2 if cutoff < 0 else 0.2

    df = add_partition(df, cutoff)
    df = add_weight(df, cutoff)

    # Ablation test
    # df['weight'] = 1

    # Negate all EP values if the sum if negative for the abnormal part.
    anomaly_ep_sum = df.loc[df['partition'] == 1, 'ep'].sum()
    if anomaly_ep_sum < 0:
        df['ep'] = -df['ep']
        anomaly_ep_sum *= -1

    # Get the adjusted EP threshold
    adj_ep_threshold = anomaly_ep_sum * pep_threshold

    # Compute the ep_{>0} values (for element pruning)
    df['ep_z'] = np.where(df['ep'] > 0, df['ep'], 0)

    root_causes = []
    pruned_elements = defaultdict(list) if prune_elements else None
    while True:
        anomaly_ep_sum = df.loc[df['partition'] == 1, 'ep'].sum()
        if debug: print('ep sum:', anomaly_ep_sum, 'ths', adj_ep_threshold)
        if anomaly_ep_sum < adj_ep_threshold:
            break

        root_cause, pruned_elements = search_anomaly(df, attributes, pruned_elements, risk_threshold, adj_ep_threshold,
                                                     debug)
        if root_cause is None:
            break

        if debug: print('Found root cause', root_cause)

        root_causes.append(root_cause)
        mask = get_element_mask(df, root_cause['cuboid'], root_cause['elements'][0])
        df = df.loc[~mask]

    return root_causes

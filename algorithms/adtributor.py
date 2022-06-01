import numpy as np
import pandas as pd
from utils.element_scores import add_explanatory_power, add_surpise


def merge_dimensions(df, dimensions, derived):
    elements = pd.DataFrame(columns=list(set(df.columns) - set(dimensions)), dtype=float)
    for d in dimensions:
        dim = df.groupby(d).sum().reset_index()
        dim['element'] = dim[d]
        dim['dimension'] = d
        dim = dim.drop(columns=d)
        elements = pd.concat([elements, dim], axis=0, sort=False)

    if derived:
        elements['predict'] = elements['predict_a'] / elements['predict_b']
        elements['real'] = elements['real_a'] / elements['real_b']

    elements = elements.reset_index(drop=True)
    return elements


def adtributor(df, dimensions, teep=0.1, tep=0.1, k=3, derived=False):
    elements = merge_dimensions(df, dimensions, derived)
    elements = add_explanatory_power(elements, derived)
    elements = add_surpise(elements, derived, merged_divide=len(dimensions))

    candidate_set = []
    for d in dimensions:
        dim_elems = elements.loc[elements['dimension'] == d].set_index('element')
        dim_elems = dim_elems.sort_values('surprise', ascending=False)
        cumulative_ep = dim_elems.loc[dim_elems['ep'] > teep, 'ep'].cumsum()
        if np.any(cumulative_ep > tep):
            idx = (cumulative_ep > tep).idxmax()
            candidate = {'elements': cumulative_ep[:idx].index.values.tolist(),
                         'explanatory_power': cumulative_ep[idx],
                         'surprise': dim_elems.loc[:idx, 'surprise'].sum(),
                         'dimension': d}
            candidate_set.append(candidate)

    # Sort by surprise and return the top k
    candidate_set = sorted(candidate_set, key=lambda t: t['surprise'], reverse=True)[:k]
    return candidate_set

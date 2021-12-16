import numpy as np
import pandas as pd


def adtributor(df, dimensions, Teep=0.1, Tep=0.67, k=3):
    # For each element, compute the explanatory power.
    F = df['predict'].sum()
    A = df['real'].sum()
    df['ep'] = (df['real'] - df['predict']) / (A - F)

    # For each element, compute the surprise.
    # Ignore any divide by zero.
    with np.errstate(divide='ignore'):
        p = df['predict'] / F
        q = df['real'] / A
        p_term = np.nan_to_num(p * np.log(2 * p / (p + q)))
        q_term = np.nan_to_num(q * np.log(2 * q / (p + q)))
        df['surprise'] = 0.5 * (p_term + q_term)
        
    candidate_set = []
    for d in dimensions:
        elements = df.groupby(d).sum()
        elements = elements.sort_values('surprise', ascending=False)
        cumulative_ep = elements.loc[elements['ep'] > Teep, 'ep'].cumsum()
        if np.any(cumulative_ep > Tep):
            idx = (cumulative_ep > Tep).idxmax()
            candidate = {'elements': cumulative_ep[:idx].index.values.tolist(),
                         'explanatory_power': cumulative_ep[idx],
                         'surprise': elements.loc[:idx, 'surprise'].sum(),
                         'dimension': d}
            candidate_set.append(candidate)

    # Sort by surprise and return the top k
    candidate_set = sorted(candidate_set, key=lambda t: t['surprise'], reverse=True)[:k]
    return candidate_set


def adtributor_new(df, dimensions, Teep=0.1, Tep=0.67, k=3):
    elements = pd.DataFrame(columns=['real', 'predict', 'dimension', 'element'])
    for d in dimensions:
        dim = df.groupby(d).sum().reset_index()
        dim['element'] = dim[d]
        dim['dimension'] = d
        dim = dim.drop(columns=d)
        elements = pd.concat([elements, dim], axis=0, sort=False)

    # For each element, compute the explanatory power.
    F = df['predict'].sum()
    A = df['real'].sum()
    elements['ep'] = (elements['real'] - elements['predict']) / (A - F)

    # For each element, compute the surprise.
    # Ignore any divide by zero.
    with np.errstate(divide='ignore'):
        p = elements['predict'] / F
        q = elements['real'] / A
        p_term = np.nan_to_num(p * np.log(2 * p / (p + q)))
        q_term = np.nan_to_num(q * np.log(2 * q / (p + q)))
        df['surprise'] = 0.5 * (p_term + q_term)

    candidate_set = []
    for d in dimensions:
        dim_elems = elements.loc[elements['dimension'] == d].set_index('element')
        dim_elems = dim_elems.sort_values('surprise', ascending=False)
        cumulative_ep = dim_elems.loc[dim_elems['ep'] > Teep, 'ep'].cumsum()
        if np.any(cumulative_ep > Tep):
            idx = (cumulative_ep > Tep).idxmax()
            candidate = {'elements': cumulative_ep[:idx].index.values.tolist(),
                         'explanatory_power': cumulative_ep[idx],
                         'surprise': dim_elems.loc[:idx, 'surprise'].sum(),
                         'dimension': d}
            candidate_set.append(candidate)

    # Sort by surprise and return the top k
    candidate_set = sorted(candidate_set, key=lambda t: t['surprise'], reverse=True)[:k]
    return candidate_set

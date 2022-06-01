import numpy as np
from utils.element_scores import add_explanatory_power, add_surpise
from algorithms.adtributor import merge_dimensions


def remove_duplicates(explanatory_set):
    seen = dict()
    for cs in explanatory_set:
        key = ''.join(np.array(cs['elements']).flatten())
        if key not in seen:
            seen[key] = cs
    return list(seen.values())


def rev_adtributor(df, dimensions, teep=0.1, k=3, derived=False):
    elements = merge_dimensions(df, dimensions, derived)
    elements = add_explanatory_power(elements, derived)
    elements = add_surpise(elements, derived, merged_divide=len(dimensions))

    explainatory_set = []
    for d in dimensions:
        partition_set = elements.loc[elements['dimension'] == d].set_index('element')
        candidate_set = partition_set.loc[partition_set['ep'] > teep]

        if 0 < len(candidate_set) < len(partition_set):
            candidate = {'elements': candidate_set.index.values.tolist(),
                         'explanatory_power': candidate_set['ep'].sum(),
                         'surprise': candidate_set['surprise'].sum(),
                         'dimension': d}
            explainatory_set.append(candidate)

    # Sort by surprise and return the top k
    explainatory_set = sorted(explainatory_set, key=lambda t: t['surprise'], reverse=True)[:k]
    return explainatory_set


def rev_rec_adtributor(df, dimensions, teep=0.1, k=3, derived=False):
    explanatory_set = rev_adtributor(df, dimensions, teep, k, derived)

    new_explanatory_set = []
    for candidate_set in explanatory_set:
        candidate_set['elements'] = [[e] for e in candidate_set['elements']]
        candidate_set['cuboid'] = [candidate_set['dimension']]

        # Only search dimensions 'after' the current dimension to avoid repeating (ordered differently) elements.
        remaining_dims = list(set(dimensions) - set(candidate_set['dimension']))

        # Keep the old candidate set if no deeper explainatory set is found within a subcube.
        new_candidate_set = []
        if len(remaining_dims) > 0:
            for candidate in candidate_set['elements']:
                df_candidate = df[df[candidate_set['dimension']] == candidate[0]].copy()
                c_explanatory_set = rev_rec_adtributor(df_candidate, remaining_dims, teep, k, derived)

                if len(c_explanatory_set) == 0:
                    # if one of the candidates do not have any explainatory_set then
                    # we use the candidate set of this layer
                    new_candidate_set = []
                    break

                for es in c_explanatory_set:
                    es['elements'] = [sorted(e + candidate) for e in es['elements']]
                    es['explanatory_power'] = es['explanatory_power'] * candidate_set['explanatory_power']
                    es['cuboid'] = sorted(candidate_set['cuboid'] + es['cuboid'])
                new_candidate_set.extend(c_explanatory_set)

        if len(new_candidate_set) > 0:
            new_explanatory_set.extend(new_candidate_set)
        else:
            new_explanatory_set.append(candidate_set)

    # Remove any duplicates
    new_explanatory_set = remove_duplicates(new_explanatory_set)
    return new_explanatory_set

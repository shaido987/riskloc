import numpy as np


def score_root_causes(root_cause_predictions, label):
    """
    Evaluates an instances given the prediction and label.
    :param root_cause_predictions: list of strings, the root cause predictions.
    :param label: str, the ground-truth label.
    :return: (float, float, float, list of strings), true positive count, false positive count, false negative count,
             and the ground-truth labels as a list.
    """
    true_labels = label.split(';')
    true_labels = ['&'.join(sorted(tl.split('&'))) for tl in true_labels]
    true_labels = np.unique(true_labels)

    # Return early if pred_labels is empty to avoid FutureWarning
    if len(root_cause_predictions) == 0:
        return 0, 0, len(true_labels), true_labels

    TP, FN = 0, 0
    for true_label in true_labels:
        if true_label in root_cause_predictions:
            TP += 1
        else:
            FN += 1

    FP = max(len(root_cause_predictions) - TP, 0)
    return TP, FP, FN, true_labels


def root_cause_postprocessing(root_causes, algorithm):
    """
    Postprocessing for the returned root causes for unified evaluation as the return from
    adtributor and squeeze are slightly different.
    :param root_causes: list of string, all predicted root causes.
    :param algorithm: str, the algorithm used.
    :return: list of strings, the final root cause predictions in a unified format.
    """
    # To make Adtibutor match other algorithms' outputs.
    if algorithm == 'adtributor':
        for rc in root_causes:
            rc['elements'] = [[e] for e in rc['elements']]
            rc['cuboid'] = [rc['dimension']]

    # To get strings (added here for uniformity since squeeze returns strings).
    root_cause_predictions = []
    if algorithm == 'robustspot':
        for rc in root_causes:
            root_cause_predictions.extend(['&'.join([e[0] + '=' + str(e[1]) for e in c]) for c in rc])
    elif algorithm == 'squeeze':
        root_cause_predictions = root_causes
    else:
        for rc in root_causes:
            elems = np.array([d + '=' for d in rc['cuboid']], dtype=object) + np.array(rc['elements'], dtype=object)
            root_cause_predictions.extend(['&'.join(e) for e in elems])

    # Get unique elements and sort each element in order
    root_cause_predictions = np.unique(root_cause_predictions)
    root_cause_predictions = np.array(['&'.join(sorted(rc.split('&'))) for rc in root_cause_predictions])
    return root_cause_predictions

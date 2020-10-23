import os
import pandas as pd
import numpy as np
from squeeze import squeeze
from adtributor import adtributor, adtributor_new


def run_method(data, attributes, method, debug):
    if method == "squeeze":
        root_causes = squeeze(data, attributes, debug=debug)
    elif method == "adtributor":
        root_causes = adtributor(data, attributes)

        # Postprocessing for evaluation
        for rc in root_causes:
            rc['elements'] = [[e] for e in rc['elements']]
            rc['cuboid'] = [rc['dimension']]
    elif method == "adtributor_new":
        root_causes = adtributor_new(data, attributes)

        # Postprocessing for evaluation
        for rc in root_causes:
            rc['elements'] = [[e] for e in rc['elements']]
            rc['cuboid'] = [rc['dimension']]
    else:
        raise ValueError("method", method, "not implemented.")
    return root_causes



def evaluate(root_casues, label):
    true_labels = label.split(';')

    pred_labels = []
    for rc in root_casues:
        elems = np.array([d + '=' for d in rc['cuboid']], dtype=np.object) + np.array(rc['elements'], dtype=np.object)
        print('elems', elems)
        pred_labels.extend(['&'.join(e) for e in elems])

    # If multiple clusters have the same predictions
    pred_labels = np.unique(pred_labels)

    TP, FN = 0, 0
    for true_label in true_labels:
        if true_label in pred_labels:
            TP += 1
        else:
            FN += 1

    FP = len(pred_labels) - TP
    return TP, FP, FN


def evaluate_folder(main_folder, attributes, method, run_folder=0, debug=False):
    folders = os.listdir(main_folder)
    for folder in folders[run_folder:run_folder+1]:
        print('Folder', folder)

        label_file = os.path.join(main_folder, folder, 'injection_info.csv')
        labels = pd.read_csv(label_file)

        files = os.listdir(os.path.join(main_folder, folder))
        total_TP, total_FP, total_FN = 0, 0, 0
        for i, file in enumerate(files[:-1]):  # Skip the label file
            print('file number', i, ', file', file)
            timestamp = int(os.path.splitext(file)[0])
            label = labels.loc[labels['timestamp'] == timestamp, 'set'].iloc[0]
            data = pd.read_csv(os.path.join(main_folder, folder, file))
            
            cluster_root_causes = run_method(data, attributes, method, debug)

            TP, FP, FN = evaluate(cluster_root_causes, label, debug)
            if FP > 0 or FN > 0:
                print('TP:', TP, 'FP:', FP, 'FN:', FN)
            total_TP += TP
            total_FP += FP
            total_FN += FN

        F1 = 2 * total_TP / (2 * total_TP + total_FP + total_FN)

        print('total_TP:', total_TP)
        print('total_FP:', total_FP)
        print('total_FN:', total_FN)
        print('F1-score:', F1)
        

if __name__ == "__main__":
    main_folder = './encrypted_B7'  # B0
    attributes = ['a', 'b', 'c', 'd']
    method = "squeeze"
    
    evaluate_folder(main_folder, attributes, method, run_folder=0, debug=False)

import math
import random
import numpy as np
from itertools import combinations
from copy import deepcopy


class Node:
    def __init__(self):
        self.parent = None
        self.state = []
        self.children = []
        self.fully_expanded = False
        self.Q = 0
        self.N = 0

    def __str__(self):
        return f"node state: {self.state}, Q: {self.Q}, N: {self.N}, fully expanded: {self.fully_expanded}"


def ripple(v_sum, f_sum, f_leaves):
    return f_leaves - (f_sum - v_sum) * (f_leaves / f_sum) if f_sum != 0 else 0


def distance(v1, v2):
    return np.sqrt(np.sum(np.power(v1 - v2, 2)))


def ps(df, v, f, selections):
    a = np.copy(f)
    for selection in selections:
        v_sum = df.loc[selection, 'real'].sum()
        f_sum = df.loc[selection, 'predict'].sum()
        a[selection] = ripple(v_sum, f_sum, df.loc[selection, 'predict'])

    score = max(1 - distance(v, a) / distance(v, f), 0)
    return score


def gps(v, f, selections):
    a, b = [], []
    for selection in selections:
        selection_v = v[selection]
        selection_f = f[selection]
        with np.errstate(divide='ignore', invalid='ignore'):
            selection_a = f[selection] * (selection_v.sum() / selection_f.sum())
            selection_a = np.nan_to_num(selection_a)
        a.extend(np.abs(selection_v - selection_a))
        b.extend(np.abs(selection_v - selection_f))

    selection = np.logical_or.reduce(selections)
    non_selection_v = v[~selection]
    non_selection_f = f[~selection]

    a = np.mean(a)
    b = np.mean(b)
    c = np.nan_to_num(np.mean(np.abs(non_selection_v - non_selection_f)))
    score = 1 - ((a + c) / (b + c))
    return score


def get_unqiue_elements(df, cuboid):
    return {tuple(row) for row in df[cuboid].values}


def get_element_mask(df, cuboid, combination):
    return [np.logical_and.reduce([df[d] == e for d, e in zip(cuboid, c)]) for c in combination]


def ucb(node, C=math.sqrt(2.0)):
    best_child = None
    max_score = -1
    for child in node.children:
        if child.N > 0 and not child.fully_expanded:
            left = child.Q
            right = C * math.sqrt(math.log(node.N) / child.N)
            score = left + right
            if score > max_score:
                best_child = child
                max_score = score
    return best_child


def init_children(node, elements):
    children = [e for e in elements if e not in set(node.state)]
    for c in children:
        child = Node()
        child.state = node.state + [c]
        child.parent = node
        node.children.append(child)


def get_initial_scores(df, elements, cuboid, v, f, scoring):
    element_scores = dict()
    for leaf in elements:
        selections = get_element_mask(df, cuboid, [leaf])
        if scoring == 'ps':
            element_scores[leaf] = ps(df.copy(), v, f, selections)
        else:
            element_scores[leaf] = gps(v, f, selections)
    return element_scores


def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)


def selection(node, elements):
    while len(node.state) < len(elements):
        if len(node.children) == 0:  # First time to search this node.
            init_children(node, elements)
            return node

        q_max = 0
        all_visit = True
        for child in node.children:
            q_max = max(q_max, child.Q)
            if child.N == 0:  # Not all children have been visited.
                all_visit = False

        if not all_visit and random.random() > q_max:
            return node  # Expand current node

        child_node = ucb(node)  # Select the best path got go deeper into the tree.
        if child_node is None:  # If all children are already fully expanded.
            if all_visit:
                node.fully_expanded = True
                if node.parent is None:
                    return node  # The tree is fully exanded.
                node = node.parent  # Continue again with parent node.
            else:
                return node  # Expand current node.
        else:
            node = child_node

    node.fully_expanded = True
    return node


def expand(node, element_scores):
    best_child = None
    max_score = -1
    for child in node.children:
        if child.N == 0:
            score = element_scores[child.state[-1]]
            if score > max_score:
                max_score = score
                best_child = child
    return best_child


def evaluate(df, selected_node, cuboid, v, f, scoring):
    selections = get_element_mask(df, cuboid, selected_node.state)
    if scoring == 'ps':
        score = ps(df.copy(), v, f, selections)
    else:
        score = gps(v, f, selections)
    return score


def backup(node, new_q):
    while node is not None:
        node.N += 1
        node.Q = max(node.Q, new_q)
        node = node.parent


def MCTS(df, elements, cuboid, v, f, pt, m, scoring):
    root = Node()
    max_q = -1
    best_selection = Node()

    element_scores = get_initial_scores(df, elements, cuboid, v, f, scoring)
    for i in range(m):
        node = selection(root, elements)
        if not node.fully_expanded:
            node = expand(node, element_scores)

        if root.fully_expanded:
            break

        new_q = evaluate(df, node, cuboid, v, f, scoring)
        backup(node, new_q)

        if new_q > max_q:
            max_q = root.Q
            best_selection = deepcopy(node)
        elif (new_q == max_q) and not sublist(node.state, best_selection.state) and len(node.state) < len(
                best_selection.state):
            max_q = root.Q
            best_selection = deepcopy(node)

        if max_q >= pt:
            break

    return best_selection.state, max_q


def hierarchical_pruning(elements, layer, cuboid, candidate_set):
    previous_layer_candidates = [candidate for candidate in candidate_set if candidate['layer'] == layer - 1]
    parent_selections = [cand['elements'] for cand in previous_layer_candidates if set(cand['cuboid']) < set(cuboid)]

    for parent_selection in parent_selections:
        elements = [e for e in elements if np.any([set(pe) < set(e) for pe in parent_selection])]
    return elements


def get_best_candidate(candidate_set):
    # Sort by score, layer, number of elements
    sorted_cands = sorted(candidate_set, key=lambda c: (c['score'], -c['layer'], -len(c['elements'])), reverse=True)
    return sorted_cands[0]


def hotspot(df, dimensions, pt=0.67, m=200, scoring='gps', debug=False):
    assert scoring in ['ps', 'gps'], "Supported scoring is 'ps' and 'gps'."

    # Hierarcical pruning does not seem to work well when using gps scoring
    use_pruning = scoring != 'gps'

    v = df['real'].values
    f = df['predict'].values

    candidate_set = []
    for layer in range(1, len(dimensions) + 1):
        if debug:
            print('Layer:', layer)

        cuboids = [list(c) for c in combinations(dimensions, layer)]
        for cuboid in cuboids:
            if debug:
                print('Cuboid:', cuboid)

            elements = get_unqiue_elements(df, cuboid)
            # if debug: print('Elements:', elements)

            if use_pruning and layer > 1:
                elements = hierarchical_pruning(elements, layer, cuboid, candidate_set)
                # if debug: print('Filtered elements:', elements)

            selected_set, score = MCTS(df, elements, cuboid, v, f, pt, m, scoring)
            if debug:
                print('Best subset:', selected_set, 'score', score)

            candidate = {
                'layer': layer,
                'cuboid': cuboid,
                'score': score,
                'elements': np.array(selected_set)
            }

            if candidate['score'] >= pt:
                return candidate

            candidate_set.append(candidate)

    return get_best_candidate(candidate_set)

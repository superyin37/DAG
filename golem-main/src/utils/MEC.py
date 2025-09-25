import numpy as np

def get_skeleton(G):
    return ((G + G.T) > 0).astype(int)

def find_v_structures(G):
    p = G.shape[0]
    v_structures = set()
    for i in range(p):
        for j in range(p):
            for k in range(p):
                if i == j or j == k or i == k:
                    continue
                # i → j ← k and i, k not connected
                if G[i, j] == 1 and G[k, j] == 1 and G[i, k] == 0 and G[k, i] == 0:
                    v = tuple(sorted([i, k])) + (j,)
                    v_structures.add(v)
    return v_structures

def is_in_markov_equiv_class(G_true, G_est):

    if not np.array_equal(get_skeleton(G_est), get_skeleton(G_true)):
        return False

    v1 = find_v_structures(G_est)
    v2 = find_v_structures(G_true)
    return v1 == v2

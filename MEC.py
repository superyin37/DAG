import numpy as np


def to_binary_adj(G):
    """
    Convert a weighted adjacency matrix to a binary adjacency matrix.
    Any positive entry is treated as an edge.
    """
    return (G > 0).astype(int)


def get_skeleton(G):
    """
    Return the skeleton (undirected structure) of a directed graph.
    Works for weighted adjacency matrices.
    """
    G_bin = to_binary_adj(G)
    return ((G_bin + G_bin.T) > 0).astype(int)


def find_v_structures(G):
    """
    Find all v-structures (colliders) in a directed graph.

    A v-structure is:
        i -> j <- k
    where i and k are not connected.
    """
    G_bin = to_binary_adj(G)
    p = G_bin.shape[0]
    v_structures = set()

    for i in range(p):
        for j in range(p):
            for k in range(p):
                if i == j or j == k or i == k:
                    continue

                # i -> j <- k
                if (
                    G_bin[i, j] == 1 and
                    G_bin[k, j] == 1 and
                    G_bin[i, k] == 0 and
                    G_bin[k, i] == 0
                ):
                    v = tuple(sorted([i, k])) + (j,)
                    v_structures.add(v)

    return v_structures


def is_in_markov_equiv_class(G_true, G_est):
    """
    Check whether two DAGs belong to the same Markov equivalence class.

    Two DAGs are Markov equivalent iff:
        1. They have the same skeleton
        2. They have the same v-structures
    """

    if not np.array_equal(get_skeleton(G_est), get_skeleton(G_true)):
        return False

    v1 = find_v_structures(G_est)
    v2 = find_v_structures(G_true)

    return v1 == v2

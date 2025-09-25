import numpy as np

def generate_scm_data(scm_id, n_samples = 1000, seed = None):
    """
    Generate data for one of six SCMs, returning raw NumPy arrays A, B, C.
    
    scm_id:
        1: A, B, C independent
        2: A -> B, C independent
        3: A -> B <- C
        4: A -> B -> C
        5: A -> B -> C and A -> C
        6: A -> B <- C and A -> C
    n_samples: number of data points
    seed: random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    def scm1():
        # A, B, C ~ N(0, 1) independent
        G = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        CPDAG = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        A = np.random.randn(n_samples)
        B = np.random.randn(n_samples)
        C = np.random.randn(n_samples)
        return A, B, C, G, CPDAG

    def scm2():
        # A -> B, C independent
        G = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        CPDAG = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 0]])
        A = np.random.randn(n_samples)
        C = np.random.randn(n_samples)
        B = A + np.random.randn(n_samples)  # B = A + noise
        return A, B, C, G, CPDAG

    def scm3():
        # A -> B <- C
        G = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
        CPDAG = np.array([[0, -1, 0], [1, 0, 1], [0, -1, 0]])
        A = np.random.randn(n_samples)
        C = np.random.randn(n_samples)
        B = A + C + np.random.randn(n_samples)
        return A, B, C, G, CPDAG

    def scm4():
        # A -> B -> C
        G = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        CPDAG = np.array([[0, -1, 0], [-1, 0, -1], [0, -1, 0]])
        A = np.random.randn(n_samples)
        B = A + np.random.randn(n_samples)
        C = B + np.random.randn(n_samples)
        return A, B, C, G, CPDAG

    def scm5():
        # A -> B -> C and A -> C
        G = np.array([[0, 1, 2], [0, 0, 3], [0, 0, 0]])
        CPDAG = np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]])
        A = np.random.randn(n_samples)
        B = A + np.random.randn(n_samples)
        C = 2*A + 3*B + np.random.randn(n_samples)
        return A, B, C, G, CPDAG

    def scm6():
        # A -> B <- C and A -> C
        G = np.array([[0, 1, 1], [0, 0, 0], [0, 1, 0]])
        CPDAG = np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]])
        A = np.random.randn(n_samples)
        C = A + np.random.randn(n_samples)
        B = 2*A + 3*C + np.random.randn(n_samples)
        return A, B, C, G, CPDAG

    funcs = {1: scm1, 2: scm2, 3: scm3, 4: scm4, 5: scm5, 6: scm6}
    if scm_id not in funcs:
        raise ValueError(f"Invalid scm_id {scm_id}, must be 1–6.")

    return funcs[scm_id]()
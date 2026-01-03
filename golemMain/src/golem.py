import os
import numpy as np

from .models import GolemModel
from .trainers import GolemTrainer
from .data_loader import SyntheticDataset
from .data_loader import SCM_data

from utils import MEC


# For logging of tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def golem(X, lambda_1, lambda_2, equal_variances=True,
          num_iter=1e+5, learning_rate=1e-3, seed=1,
          checkpoint_iter=None, output_dir=None, B_init=None):
    """Solve the unconstrained optimization problem of GOLEM, which involves
        GolemModel and GolemTrainer.

    Args:
        X (numpy.ndarray): [n, d] data matrix.
        lambda_1 (float): Coefficient of L1 penalty.
        lambda_2 (float): Coefficient of DAG penalty.
        equal_variances (bool): Whether to assume equal noise variances
            for likelibood objective. Default: True.
        num_iter (int): Number of iterations for training.
        learning_rate (float): Learning rate of Adam optimizer. Default: 1e-3.
        seed (int): Random seed. Default: 1.
        checkpoint_iter (int): Number of iterations between each checkpoint.
            Set to None to disable. Default: None.
        output_dir (str): Output directory to save training outputs.
        B_init (numpy.ndarray or None): [d, d] weighted matrix for initialization.
            Set to None to disable. Default: None.

    Returns:
        numpy.ndarray: [d, d] estimated weighted matrix.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """
    # Center the data
    X = X - X.mean(axis=0, keepdims=True)

    # Set up model
    n, d = X.shape
    model = GolemModel(n, d, lambda_1, lambda_2, equal_variances, seed, B_init)

    # Training
    trainer = GolemTrainer(learning_rate)
    B_est = trainer.train(model, X, num_iter, checkpoint_iter, output_dir)

    return B_est    # Not thresholded yet

def weight_to_adjacency(W, threshold=0.05):
    """
    Convert a weight matrix to an adjacency matrix.
    
    Parameters:
        W (np.ndarray): Weight matrix (square matrix).
        threshold (float): Values with absolute weight <= threshold are treated as 0.
    
    Returns:
        np.ndarray: Binary adjacency matrix of the same shape.
    """
    if not isinstance(W, np.ndarray):
        raise TypeError("Input W must be a numpy array.")
    if W.shape[0] != W.shape[1]:
        raise ValueError("Input W must be a square matrix.")
    
    G = (np.abs(W) > threshold).astype(int)
    return G


if __name__ == '__main__':
    # Minimal code to run GOLEM.
    import logging

    from data_loader import SyntheticDataset
    from data_loader.synthetic_dataset import dataset_based_on_B
    from utils.train import postprocess
    from utils.utils import count_accuracy, set_seed

    # Setup for logging
    # Required for printing histories if checkpointing is activated
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s - %(name)s - %(message)s'
    )

    # Reproducibility
    set_seed(1)

    # Load dataset
    n, d = 1000, 4
    graph_type, degree = 'ER', 0.5    # ER2 graph
    B_scale = 1.0
    noise_type = 'gaussian_ev'
    #dataset = SyntheticDataset(n, d, graph_type, degree,
    #                           noise_type, B_scale, seed=1)

    times = 20
    for i in range(1, 6):
        true_count = [0] * 6
        for seed in range(times):
            X, Y, Z, G_true, CPDAG = SCM_data.generate_scm_data(i,10000, seed = seed)
            data = np.array([X, Y, Z]).T
            #print(data.T@ data / 10000)
            W_est = golem(data, lambda_1=2e-2, lambda_2=5.0,
                  equal_variances=True, num_iter=1e+4)
            G_est = weight_to_adjacency(W_est, 0.05)
            if MEC.is_in_markov_equiv_class(G_true, G_est): true_count[i-1] += 1
        print(f"SCM {i} : {true_count[i-1]/times}")


'''
X, Y, Z, G_true, CPDAG = SCM_data.generate_scm_data(4,10000, seed = 1)
data = np.array([X, Y, Z]).T
W_est = golem(data, lambda_1=2e-2, lambda_2=5.0,
        equal_variances=True, num_iter=1e+4)
G_est = weight_to_adjacency(W_est, 0.05)
print(G_est)
print(G_true)
'''
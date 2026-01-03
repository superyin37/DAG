import numpy as np
import optuna
from sklearn.model_selection import KFold
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(r"C:\Users\super\DAG")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print(os.getcwd())

# =============================
# Import your algorithm
# =============================
from coordinate_descent.coordinate0 import dag_coordinate_descent_l0


# =============================
# Likelihood (evaluation only)
# =============================

def neg_loglik(A, S):
    """
    Negative log-likelihood up to a constant:
        -2 log det(A) + tr(A^T S A)
    """
    return -2.0 * np.log(np.linalg.det(A)) + np.trace(A.T @ S @ A)


# =============================
# Cross-validation (no Optuna)
# =============================

def cv_neg_loglik(
    X,
    lambda_l0,
    threshold,
    T,
    algo_seed,
    n_splits
):
    """
    K-fold CV negative log-likelihood for a fixed hyperparameter setting.
    """
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=algo_seed
    )

    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train = X[train_idx]
        X_val   = X[val_idx]

        # second moments
        S_train = X_train.T @ X_train / X_train.shape[0]
        S_val   = X_val.T   @ X_val   / X_val.shape[0]

        # train (penalized)
        A_hat, _, _ = dag_coordinate_descent_l0(
            S_train,
            T=T,
            lambda_l0=lambda_l0,
            threshold=threshold,
            seed=algo_seed
        )

        # evaluate (unpenalized)
        scores.append(neg_loglik(A_hat, S_val))

    return float(np.mean(scores))


# =============================
# Main tuning function 
# =============================

def tune_dag_l0_cv(
    X,
    *,
    T=1000,
    n_splits=5,
    seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    n_trials=50,
    lambda_range=(1e-2, 1.0),
    threshold_range=(0.01, 0.2),
    optuna_seed=0
):
    """
    Tune lambda_l0 and threshold using CV likelihood + Optuna.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Raw data matrix.
    T : int
        Fixed number of coordinate descent iterations.
    n_splits : int
        Number of CV folds.
    seeds : iterable of int
        Random seeds for algorithm randomness.
    n_trials : int
        Optuna trial budget.
    lambda_range : tuple
        (min, max) range for lambda_l0.
    threshold_range : tuple
        (min, max) range for threshold.
    optuna_seed : int
        Random seed for Optuna sampler.

    Returns
    -------
    best_params : dict
    best_score : float
    study : optuna.study.Study
    """

    def objective(trial):
        lambda_l0 = trial.suggest_float(
            "lambda_l0",
            lambda_range[0],
            lambda_range[1],
            log=True
        )

        threshold = trial.suggest_float(
            "threshold",
            threshold_range[0],
            threshold_range[1]
        )

        seed_scores = []

        for seed in seeds:
            score = cv_neg_loglik(
                X=X,
                lambda_l0=lambda_l0,
                threshold=threshold,
                T=T,
                algo_seed=seed,
                n_splits=n_splits
            )
            seed_scores.append(score)

        return float(np.mean(seed_scores))

    sampler = optuna.samplers.TPESampler(seed=optuna_seed)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    return study.best_params, study.best_value, study


# =============================
# Example usage (entry point)
# =============================

if __name__ == "__main__":
    # load data (example)
    X = np.load("data.npy")   # shape (n, d)

    best_params, best_score, study = tune_dag_l0_cv(
        X
    )

    print("\n===== Tuning Result =====")
    print("Best CV NLL:", best_score)
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

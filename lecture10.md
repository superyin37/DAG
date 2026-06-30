
# Rethinking Matrix Representation in Sparse DAG Learning for Large-scale Causal Discovery

My research focuses on large-scale causal graph discovery under a linear Gaussian structural equation model. In this setting, a causal structure is usually represented as a weighted adjacency matrix, and the learning problem is formulated as an optimization problem over this matrix. For example, the objective can be written as


$$
f(A)=-2\log\det(A)+\mathrm{tr}(A^\top S A)+\lambda_0\|A\|_0
$$

where $S$ is the sample covariance matrix, $A$ is the optimization variable, and the $L_0$ term encourages sparsity. In addition, a DAG constraint is imposed so that the learned graph has no directed cycles.

The lecture emphasizes that representation is not only a technical choice, but also a central factor that determines the efficiency, robustness, and theoretical behavior of machine learning. Tensor Representation Learning is introduced as a way to represent data, parameters, and functions in a more structured form, thereby revealing intrinsic structure and improving learning efficiency. Inspired by this viewpoint, I reconsider whether the standard matrix representation used in sparse DAG learning is truly the most economical and statistically natural representation for large-scale causal discovery.

## Hidden Assumptions Behind the Current Representation

First, the current data representation assumes that the essential information can be captured by the sample covariance matrix

$$
S = \frac{1}{n}X^\top X.
$$

This is natural under a linear Gaussian model, because second-order moments are sufficient for likelihood-based estimation. However, this also hides a strong assumption: causal mechanisms are expected to be linear, Gaussian, and well described by covariance information. If the true mechanisms are nonlinear, non-Gaussian, heteroscedastic, or affected by latent variables, then this representation may lose important causal information.

Second, the current parameter representation uses a dense $d \times d$ matrix even though the true causal graph is assumed to be sparse. The number of possible directed edges is $O(d^2)$, while the number of true edges may be much smaller. This creates a mismatch between the modeling assumption and the representation. Sparsity is introduced later through the $L_0$ penalty, but the representation itself still asks the algorithm to search over a dense candidate edge space. For large-scale causal discovery, this can be computationally expensive.

Third, the weighted adjacency matrix forces the algorithm to output a single DAG. However, in observational causal discovery, a unique DAG is often not identifiable. Several DAGs may belong to the same Markov equivalence class and imply the same conditional independence relations. Therefore, a single-DAG matrix representation is convenient for optimization, but it may be stronger than what observational data can statistically justify. A CPDAG or Markov equivalence class representation may be more faithful to the identifiability structure of the problem.

Fourth, acyclicity is not built into the matrix representation itself. A general matrix can represent a directed graph with cycles, so the DAG property has to be imposed as an additional constraint. In my coordinate descent approach, each candidate update must be checked to ensure that it does not create a directed cycle. This is workable, but it also means that the representation itself does not naturally encode the DAG structure.

## Toward a More Economical Representation

The current matrix representation is useful because it is compact, differentiable, and compatible with likelihood-based optimization. However, it is not necessarily economical for large-scale problems. The lecture gives an instructive analogy: low-rank tensor parameterization can change the relevant complexity from the total number of parameters to the effective number of parameters. A similar idea applies to sparse DAG learning. Instead of treating all $O(d^2)$ possible edges equally, a better representation should make the effective sparse structure explicit.

One possible alternative is to represent the graph by a support set and active weights:

$$
E = \{(i,j): A_{ij} \neq 0\},
\quad
\theta_E = \{A_{ij} : (i,j) \in E\}.
$$

This representation directly reflects the sparse graph assumption. The effective number of parameters becomes $O(|E|)$ rather than $O(d^2)$. The drawback is that the optimization problem becomes more explicitly mixed discrete-continuous, because adding or deleting an edge changes the support set.

Another practical alternative is to introduce a candidate edge mask

$$
M \in \{0,1\}^{d \times d}
$$

and restrict the search to

$$
A = M \odot A.
$$

The mask can be constructed by preliminary screening methods such as correlation, partial correlation, graphical lasso, or moral graph estimation. This keeps the matrix optimization framework while reducing the effective candidate edge space. The main risk is that screening may remove true edges, so the mask must be designed conservatively.

A third possibility is to represent the graph through a topological order and a triangular parameterization. If the order were known, acyclicity would be guaranteed by construction, and no cycle check would be required. However, the topological order is usually unknown, so this representation introduces another difficult combinatorial search problem.

Finally, for multi-environment or time-varying causal discovery, tensor representation may become useful. For example, covariance matrices or causal graphs from multiple environments could be represented as

$$
\mathcal{S} \in \mathbb{R}^{d \times d \times m},
\quad
\mathcal{B} \in \mathbb{R}^{d \times d \times m},
$$

where the third mode represents environments or time periods. This could help model shared causal structure and environment-specific variation. Although this is beyond my current single-environment setting, it is a natural extension inspired by the lecture’s view that tensor representations are useful when matrix representations cannot express multi-dimensional structure.

## Conclusion

The standard weighted adjacency matrix representation is powerful because it connects causal discovery with continuous optimization. It makes likelihood-based objectives, sparsity penalties, and coordinate-wise updates easy to formulate. However, it also hides several assumptions: linear Gaussian structure, covariance sufficiency, dense candidate edge space, single-DAG identifiability, and acyclicity as an external constraint.

For large-scale causal discovery, a more economical representation should explicitly exploit sparsity, candidate edge restrictions, topological structure, or Markov equivalence. In this sense, improving the representation of sparse DAGs may be as important as improving the optimization algorithm itself. The main lesson from the lecture is that representation determines what structure a learning algorithm can exploit; for my research, this means that the design of the graph representation is central to scalability, statistical reliability, and future extensions.

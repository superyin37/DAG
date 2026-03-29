\section{Greedy Coordinate Descent}
\begin{algorithm}[H]
\caption{Greedy-Grad Coordinate Descent for the $A$-formulation}
\begin{algorithmic}[1]
\Require Initial matrix $A^{(0)}$, sample covariance matrix $S$, number of iterations $T$
\For{$t=0,1,\dots,T-1$}
    \State Compute the gradient
    \[
    G^{(t)} \gets \nabla_A f(A^{(t)}) = -2(A^{(t)})^{-T} + 2SA^{(t)}.
    \]
    \State Select
    \[
    (i_t,j_t)\in
    \arg\max_{i<j}
    \max\left\{
    |G^{(t)}_{ij}|,\,
    |G^{(t)}_{ji}|
    \right\},
    \]
    subject to feasibility under the DAG constraint.
    \State Set $A^{(t+\frac12)} \gets A^{(t)}$, and
    \[
    A^{(t+\frac12)}_{i_tj_t}\gets 0,\qquad
    A^{(t+\frac12)}_{j_ti_t}\gets 0.
    \]
    \State Initialize $\Delta_{i_tj_t}\gets -\infty$, $\Delta_{j_ti_t}\gets -\infty$
    \If{$(A^{(t+\frac12)}+\mathbb{R}E_{i_tj_t})\cap \mathrm{DAG}_k \neq \varnothing$}
        \State Compute
        \[
        \delta_{i_tj_t}^*
        \gets
        \arg\min_{\delta} f\!\left(A^{(t+\frac12)}+\delta E_{i_tj_t}\right)
        \]
        \State
        \[
        \Delta_{i_tj_t}
        \gets
        f(A^{(t+\frac12)})
        -
        f\!\left(A^{(t+\frac12)}+\delta_{i_tj_t}^* E_{i_tj_t}\right)
        \]
    \EndIf
    \If{$(A^{(t+\frac12)}+\mathbb{R}E_{j_ti_t})\cap \mathrm{DAG}_k \neq \varnothing$}
        \State Compute
        \[
        \delta_{j_ti_t}^*
        \gets
        \arg\min_{\delta} f\!\left(A^{(t+\frac12)}+\delta E_{j_ti_t}\right)
        \]
        \State
        \[
        \Delta_{j_ti_t}
        \gets
        f(A^{(t+\frac12)})
        -
        f\!\left(A^{(t+\frac12)}+\delta_{j_ti_t}^* E_{j_ti_t}\right)
        \]
    \EndIf
    \If{$\Delta_{i_tj_t} > \Delta_{j_ti_t}$}
        \State
        \[
        A^{(t+1)}
        \gets
        A^{(t+\frac12)}+\delta_{i_tj_t}^*E_{i_tj_t}
        \]
    \Else
        \State
        \[
        A^{(t+1)}
        \gets
        A^{(t+\frac12)}+\delta_{j_ti_t}^*E_{j_ti_t}
        \]
    \EndIf
\EndFor
\State \Return $A^{(T)}$
\end{algorithmic}
\end{algorithm}
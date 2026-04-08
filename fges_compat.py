import os
from pathlib import Path

import jpype
import numpy as np
import pandas as pd


_JDK_CANDIDATE = r"C:\Program Files\Microsoft\jdk-21.0.10.7-hotspot"
_JAR_CANDIDATES = [
    Path.home() / "AppData" / "Roaming" / "Python" / "Python310" / "site-packages" / "pytetrad" / "resources" / "tetrad-current.jar",
]


def _ensure_java_home():
    if os.environ.get("JAVA_HOME"):
        return
    if os.path.isdir(_JDK_CANDIDATE):
        os.environ["JAVA_HOME"] = _JDK_CANDIDATE


def _find_tetrad_jar() -> str:
    for candidate in _JAR_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("tetrad-current.jar not found in the expected py-tetrad location")


def _ensure_jvm():
    _ensure_java_home()
    if jpype.isJVMStarted():
        return
    jpype.startJVM(jpype.getDefaultJVMPath(), classpath=[_find_tetrad_jar()])


class TetradSearch:
    """
    Small compatibility wrapper for the FGES subset used in this repo.

    This bypasses jpype.imports and py-tetrad's package import hook, which is
    broken in the current Windows/JDK environment, while keeping the notebook
    call sites unchanged.
    """

    def __init__(self, df: pd.DataFrame):
        _ensure_jvm()
        self._ArrayList = jpype.JClass("java.util.ArrayList")
        self._ContinuousVariable = jpype.JClass("edu.cmu.tetrad.data.ContinuousVariable")
        self._DoubleDataBox = jpype.JClass("edu.cmu.tetrad.data.DoubleDataBox")
        self._BoxDataSet = jpype.JClass("edu.cmu.tetrad.data.BoxDataSet")
        self._Knowledge = jpype.JClass("edu.cmu.tetrad.data.Knowledge")
        self._Parameters = jpype.JClass("edu.cmu.tetrad.util.Parameters")
        self._Params = jpype.JClass("edu.cmu.tetrad.util.Params")
        self._SemBicScore = jpype.JClass("edu.cmu.tetrad.algcomparison.score.SemBicScore")
        self._Fges = jpype.JClass("edu.cmu.tetrad.algcomparison.algorithm.oracle.cpdag.Fges")

        self.data = self._pandas_to_tetrad(df)
        self.SCORE = None
        self.java = None
        self.knowledge = self._Knowledge()
        self.params = self._Parameters()
        self.bootstrap_graphs = None

    def _pandas_to_tetrad(self, df: pd.DataFrame):
        df = df.astype("float64", copy=False)
        values = df.to_numpy(dtype=np.float64, copy=False)
        n, p = values.shape

        variables = self._ArrayList()
        for col in df.columns:
            variables.add(self._ContinuousVariable(str(col)))

        databox = self._DoubleDataBox(n, p)
        for row in range(n):
            for col in range(p):
                databox.set(row, col, float(values[row, col]))

        return self._BoxDataSet(databox, variables)

    def set_verbose(self, verbose):
        self.params.set(self._Params.VERBOSE, bool(verbose))

    def use_sem_bic(self, penalty_discount=2, structurePrior=0, sem_bic_rule=1, singularity_lambda=0.0):
        self.params.set(self._Params.PENALTY_DISCOUNT, float(penalty_discount))
        self.params.set(self._Params.SEM_BIC_STRUCTURE_PRIOR, float(structurePrior))
        self.params.set(self._Params.SEM_BIC_RULE, int(sem_bic_rule))
        self.params.set(self._Params.SINGULARITY_LAMBDA, float(singularity_lambda))
        self.SCORE = self._SemBicScore()

    def run_fges(self, symmetric_first_step=False, max_degree=-1, parallelized=False, faithfulness_assumed=False):
        if self.SCORE is None:
            raise RuntimeError("use_sem_bic() must be called before run_fges()")

        alg = self._Fges(self.SCORE)
        alg.setKnowledge(self.knowledge)

        self.params.set(self._Params.SYMMETRIC_FIRST_STEP, bool(symmetric_first_step))
        self.params.set(self._Params.MAX_DEGREE, int(max_degree))
        self.params.set(self._Params.PARALLELIZED, bool(parallelized))
        self.params.set(self._Params.FAITHFULNESS_ASSUMED, bool(faithfulness_assumed))

        self.java = alg.search(self.data, self.params)
        self.bootstrap_graphs = alg.getBootstrapGraphs()

    def get_graph_to_matrix(self, java=None, nullEpt=0, circleEpt=1, arrowEpt=2, tailEpt=3):
        graph = self.java if java is None else java
        if graph is None:
            raise RuntimeError("run_fges() must be called before get_graph_to_matrix()")

        endpoint_map = {
            "NULL": nullEpt,
            "CIRCLE": circleEpt,
            "ARROW": arrowEpt,
            "TAIL": tailEpt,
        }

        nodes = graph.getNodes()
        p = graph.getNumNodes()
        matrix = np.zeros((p, p), dtype=int)

        for edge in graph.getEdges():
            i = nodes.indexOf(edge.getNode1())
            j = nodes.indexOf(edge.getNode2())
            matrix[j, i] = endpoint_map[edge.getEndpoint1().name()]
            matrix[i, j] = endpoint_map[edge.getEndpoint2().name()]

        columns = [str(nodes.get(i)) for i in range(p)]
        return pd.DataFrame(matrix, columns=columns, index=columns)

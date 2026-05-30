import numpy as np
import pandas as pd
from sklearn.tree import _tree

from .arf import arf
from .nodes import (
    GaussianLeaf,
    Leaf,
    MultinomialLeaf,
    ProdNode,
    SumNode,
    fit_gaussian,
    fit_multinomial,
)
from .pc import PC


def infer_ncat_from_arf(model):
    """
    Infer GeF-style ncat metadata from a fitted ARF object.

    Continuous variables get ncat=1. ARF categorical variables get the number
    of category levels seen during ARF preprocessing.
    """
    _check_fitted_arf(model)
    ncat = []
    for col in model.orig_colnames:
        if bool(model.factor_cols[col]):
            ncat.append(len(model.levels[col]))
        else:
            ncat.append(1)
    return np.asarray(ncat, dtype=np.int64)


def arf_to_gef(model, ncat=None, minstd=1, smoothing=1e-6):
    """
    Compile a fitted ARF into a GeF-style probabilistic circuit.

    The ARF forest supplies the density-oriented partition structure. Each
    terminal node gets the original GeF leaf model: a fully factorized product
    of univariate Gaussian leaves for continuous variables and Multinomial
    leaves for categorical variables.

    Parameters
    ----------
    model : gefs.arf.arf
        Fitted ARF instance.
    ncat : np.ndarray, optional
        Number of categories per variable. Use 1 for continuous variables.
        If omitted, this is inferred from ARF's pandas categorical metadata.
    minstd : float
        Minimum standard deviation for Gaussian leaves.
    smoothing : float
        Additive smoothing for categorical leaves.

    Returns
    -------
    pc : gefs.pc.PC
        Probabilistic circuit over the ARF input variables.
    """
    _check_fitted_arf(model)
    data = model.x_real.to_numpy(dtype=np.float64)
    if ncat is None:
        ncat = infer_ncat_from_arf(model)
    else:
        ncat = np.asarray(ncat, dtype=np.int64)
    _validate_ncat(ncat, data.shape[1])

    scope = np.arange(data.shape[1], dtype=np.int64)
    pc = PC(ncat)
    pc.root = SumNode(scope=scope, n=1)
    for tree in model.clf.estimators_:
        tree_root = _tree_to_pc_root(
            tree,
            data,
            ncat,
            minstd=minstd,
            smoothing=smoothing,
        )
        pc.root.add_child(tree_root)
    pc.is_ensemble = True
    return pc


class ARFGeF:
    """
    Convenience wrapper: train ARF, then compile it into a GeF-style PC.

    Input data should be a pandas DataFrame containing the full joint vector
    to model, e.g. columns [X_1, ..., X_d, Y] when the target is included.
    """

    def __init__(
        self,
        num_trees=30,
        delta=0,
        max_iters=10,
        early_stop=True,
        verbose=True,
        min_node_size=5,
        ncat=None,
        minstd=1,
        smoothing=1e-6,
        **arf_kwargs
    ):
        self.num_trees = num_trees
        self.delta = delta
        self.max_iters = max_iters
        self.early_stop = early_stop
        self.verbose = verbose
        self.min_node_size = min_node_size
        self.ncat = ncat
        self.minstd = minstd
        self.smoothing = smoothing
        self.arf_kwargs = arf_kwargs
        self.arf_model = None
        self.pc = None

    def fit(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "ARFGeF.fit expects a pandas DataFrame containing the joint variables."
            )
        self.arf_model = arf(
            data,
            num_trees=self.num_trees,
            delta=self.delta,
            max_iters=self.max_iters,
            early_stop=self.early_stop,
            verbose=self.verbose,
            min_node_size=self.min_node_size,
            **self.arf_kwargs
        )
        self.pc = arf_to_gef(
            self.arf_model,
            ncat=self.ncat,
            minstd=self.minstd,
            smoothing=self.smoothing,
        )
        return self

    def log_likelihood(self, data):
        return self.pc.log_likelihood(_to_arf_numpy(self.arf_model, data))

    def likelihood(self, data):
        return self.pc.likelihood(_to_arf_numpy(self.arf_model, data))

    def classify(self, X, classcol=None, return_prob=False):
        return self.pc.classify(
            _to_arf_numpy(self.arf_model, X),
            classcol=classcol,
            return_prob=return_prob,
        )

    def classify_avg(self, X, classcol=None, return_prob=False, naive=False):
        return self.pc.classify_avg(
            _to_arf_numpy(self.arf_model, X),
            classcol=classcol,
            return_prob=return_prob,
            naive=naive,
        )

    def sample(self, n_samples=1):
        return self.pc.sample(n_samples)

    def sample_conditional(self, evidence):
        return self.pc.sample_conditional(_to_arf_numpy(self.arf_model, evidence))


def _tree_to_pc_root(tree, data, ncat, minstd, smoothing):
    scope = np.arange(data.shape[1], dtype=np.int64)
    lp = np.sum(np.where(ncat == 1, 0, ncat)) * smoothing
    upper = ncat.astype(np.float64)
    upper[upper == 1] = np.Inf
    lower = np.ones(data.shape[1], dtype=np.float64) * -np.Inf
    tree_ = tree.tree_

    def recurse(parent, node_id, rows, upper_bounds, lower_bounds):
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            split_var = int(tree_.feature[node_id])
            split_value = np.array([tree_.threshold[node_id]], dtype=np.float64)

            sumnode = SumNode(scope=scope, n=rows.shape[0] + lp)
            if parent is not None:
                parent.add_child(sumnode)

            left_mask = rows[:, split_var] <= split_value[0]
            left_rows = rows[left_mask, :]
            left_upper = upper_bounds.copy()
            left_lower = lower_bounds.copy()
            left_upper[split_var] = min(split_value[0], left_upper[split_var])
            left_prod = ProdNode(scope=scope, n=left_rows.shape[0] + lp)
            sumnode.add_child(left_prod)
            left_prod.add_child(
                Leaf(
                    scope=np.array([split_var], dtype=np.int64),
                    n=left_rows.shape[0] + lp,
                    value=split_value,
                    comparison=3,
                )
            )
            recurse(
                left_prod,
                tree_.children_left[node_id],
                left_rows.copy(),
                left_upper,
                left_lower,
            )

            right_rows = rows[~left_mask, :]
            right_upper = upper_bounds.copy()
            right_lower = lower_bounds.copy()
            right_lower[split_var] = max(split_value[0], right_lower[split_var])
            right_prod = ProdNode(scope=scope, n=right_rows.shape[0] + lp)
            sumnode.add_child(right_prod)
            right_prod.add_child(
                Leaf(
                    scope=np.array([split_var], dtype=np.int64),
                    n=right_rows.shape[0] + lp,
                    value=split_value,
                    comparison=4,
                )
            )
            recurse(
                right_prod,
                tree_.children_right[node_id],
                right_rows.copy(),
                right_upper,
                right_lower,
            )
            return sumnode

        leaf_parent = parent
        if leaf_parent is None:
            leaf_parent = ProdNode(scope=scope, n=rows.shape[0] + lp)
        _add_factorized_leaf_model(
            leaf_parent,
            rows,
            ncat,
            upper_bounds,
            lower_bounds,
            minstd=minstd,
            smoothing=smoothing,
            lp=lp,
        )
        return leaf_parent

    root = recurse(None, 0, data, upper, lower)
    if root is None:
        root = ProdNode(scope=scope, n=data.shape[0] + lp)
        _add_factorized_leaf_model(
            root,
            data,
            ncat,
            upper,
            lower,
            minstd=minstd,
            smoothing=smoothing,
            lp=lp,
        )
    return root


def _add_factorized_leaf_model(
    parent,
    rows,
    ncat,
    upper,
    lower,
    minstd,
    smoothing,
    lp,
):
    if parent is None:
        raise ValueError("Cannot attach a leaf model without a parent node.")
    for var in range(rows.shape[1]):
        scope = np.array([var], dtype=np.int64)
        if ncat[var] > 1:
            leaf = MultinomialLeaf(scope=scope, n=rows.shape[0] + lp)
            parent.add_child(leaf)
            fit_multinomial(leaf, rows, int(ncat[var]), smoothing)
        else:
            leaf = GaussianLeaf(scope=scope, n=rows.shape[0] + lp)
            parent.add_child(leaf)
            fit_gaussian(leaf, rows, upper[var], lower[var], minstd)


def _check_fitted_arf(model):
    if not hasattr(model, "clf") or not hasattr(model, "x_real"):
        raise TypeError("Expected a fitted gefs.arf.arf instance.")


def _validate_ncat(ncat, n_features):
    if ncat.ndim != 1:
        raise ValueError("ncat must be a one-dimensional array.")
    if ncat.shape[0] != n_features:
        raise ValueError(
            "ncat length must match the ARF input dimension: "
            f"got {ncat.shape[0]} for {n_features} variables."
        )
    if np.any(ncat < 1):
        raise ValueError("Every ncat entry must be >= 1.")


def _to_arf_numpy(model, data):
    if model is None:
        raise RuntimeError("ARFGeF must be fitted before this method is called.")
    if isinstance(data, pd.DataFrame):
        if not hasattr(model, "factor_cols"):
            return data.to_numpy(dtype=np.float64)
        encoded = data.copy()
        for col in encoded.columns:
            if col in model.factor_cols.index and bool(model.factor_cols[col]):
                codes = pd.Categorical(
                    encoded[col],
                    categories=model.levels[col],
                ).codes.astype(np.float64)
                codes[codes < 0] = np.nan
                encoded[col] = codes
        return encoded.to_numpy(dtype=np.float64)
    if isinstance(data, pd.Series):
        return data.to_numpy(dtype=np.float64)
    return np.asarray(data, dtype=np.float64)

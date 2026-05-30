import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.special import logsumexp
from tqdm import tqdm

from .pc import PC
from .nodes import SumNode, ProdNode, eval_root
from .utils import bincount


def get_tree_leaf_ids(tree, data):
    """
    Returns the leaf node id for each sample in `data` according to the
    hard splits in `tree` (a jitclass Tree object).

    Parameters
    ----------
    tree : Tree
        A fitted Decision Tree (gefs.trees.Tree).
    data : np.ndarray, shape (n_samples, n_features)
        Input features (without class variable).

    Returns
    -------
    leaf_ids : np.ndarray, shape (n_samples,), dtype int64
        The id of the leaf each sample falls into.
    """
    n_samples = data.shape[0]
    leaf_ids = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        node = tree.root
        while not node.isleaf:
            s = node.split
            if s.type == 'num':
                if data[i, s.var] <= s.threshold[0]:
                    node = node.left_child
                else:
                    node = node.right_child
            else:
                # categorical split: check if value is in threshold set
                if data[i, s.var] in s.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
        leaf_ids[i] = node.id
    return leaf_ids


def compute_mi_matrix(leaf_ids):
    """
    Compute pairwise mutual information between trees based on leaf
    assignments.

    Parameters
    ----------
    leaf_ids : np.ndarray, shape (n_samples, n_trees)
        Leaf id assigned to each sample by each tree.

    Returns
    -------
    mi : np.ndarray, shape (n_trees, n_trees)
        Symmetric mutual information matrix.
    """
    n_samples, n_trees = leaf_ids.shape
    mi = np.zeros((n_trees, n_trees), dtype=np.float64)
    for i in range(n_trees):
        vals_i, counts_i = np.unique(leaf_ids[:, i], return_counts=True)
        p_i = counts_i / n_samples
        for j in range(i + 1, n_trees):
            vals_j, counts_j = np.unique(leaf_ids[:, j], return_counts=True)
            p_j = counts_j / n_samples
            # joint counts
            joint = {}
            for s in range(n_samples):
                key = (leaf_ids[s, i], leaf_ids[s, j])
                joint[key] = joint.get(key, 0) + 1
            p_ij = np.array(list(joint.values()), dtype=np.float64) / n_samples
            # MI = sum p_ij log(p_ij / (p_i * p_j))
            mi_val = 0.0
            for idx_ij, (key, count) in enumerate(joint.items()):
                pij = p_ij[idx_ij]
                pi = p_i[vals_i == key[0]][0]
                pj = p_j[vals_j == key[1]][0]
                if pij > 0 and pi > 0 and pj > 0:
                    mi_val += pij * np.log(pij / (pi * pj))
            mi[i, j] = mi_val
            mi[j, i] = mi_val
    return mi


def chow_liu_max_spanning_tree(mi_matrix):
    """
    Compute the Chow-Liu tree (maximum spanning tree) over the MI matrix.

    Parameters
    ----------
    mi_matrix : np.ndarray, shape (n, n)
        Symmetric mutual information matrix.

    Returns
    -------
    edges : list of tuples (i, j, weight)
        Edges in the maximum spanning tree.
    """
    n = mi_matrix.shape[0]
    # SciPy's minimum_spanning_tree expects a cost matrix.
    # We negate MI to turn max-spanning into min-spanning.
    cost = -mi_matrix
    np.fill_diagonal(cost, 0.0)
    csr = csr_matrix(cost)
    mst = minimum_spanning_tree(csr)
    # Use nonzero() to reliably retrieve edges (sparse matrix may drop zeros)
    rows, cols = mst.nonzero()
    edges = []
    for idx in range(len(rows)):
        i, j = int(rows[idx]), int(cols[idx])
        if i < j:
            w = mi_matrix[i, j]
            edges.append((i, j, w))
    return edges


def cluster_tree_groups(edges, n_groups, n_nodes):
    """
    Cluster the Chow-Liu tree into `n_groups` connected components by
    iteratively removing the weakest edges.

    Parameters
    ----------
    edges : list of (i, j, weight)
        Edges of the Chow-Liu tree.
    n_groups : int
        Desired number of groups.
    n_nodes : int
        Number of nodes (trees) in the graph.

    Returns
    -------
    groups : list of lists
        Each inner list contains the tree indices belonging to that group.
    """
    if n_groups <= 1 or len(edges) == 0:
        return [list(range(n_nodes))]

    # Sort edges by weight ascending; remove the weakest first.
    sorted_edges = sorted(edges, key=lambda x: x[2])
    n_remove = min(n_groups - 1, len(sorted_edges))
    kept_edges = sorted_edges[n_remove:]

    if len(kept_edges) == 0:
        # Fallback: every node its own group if no edges remain
        return [[i] for i in range(n_nodes)]

    # Build adjacency matrix for kept edges
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i, j, w in kept_edges:
        adj[i, j] = w
        adj[j, i] = w

    csr = csr_matrix(adj)
    n_comp, labels = connected_components(csr, directed=False)
    groups = [[] for _ in range(n_comp)]
    for node_id, lbl in enumerate(labels):
        groups[lbl].append(node_id)
    return groups


def learn_hierarchical_weights(log_probs, groups, n_iter=10):
    """
    Learn maximum-likelihood weights for a two-level hierarchical mixture:
        p(x) = sum_k w_k * sum_{t in group_k} w_{k,t} * p_t(x)

    Parameters
    ----------
    log_probs : np.ndarray, shape (n_trees, n_samples)
        Log-likelihood of each training sample under each base tree PC.
    groups : list of lists
        Grouping of tree indices.
    n_iter : int
        Number of EM iterations.

    Returns
    -------
    group_weights : np.ndarray, shape (n_groups,)
        Mixture weights at the root level.
    tree_weights : list of np.ndarray
        Mixture weights inside each group.
    """
    n_trees, n_samples = log_probs.shape
    K = len(groups)

    # initialise uniformly
    group_weights = np.ones(K) / K
    tree_weights = [np.ones(len(g)) / len(g) for g in groups]

    for _ in range(n_iter):
        # ---- E-step ----
        # responsibilities[k, t, n]  (zero for t not in group k)
        resp = np.zeros((K, n_trees, n_samples), dtype=np.float64)
        # pre-compute group log-likelihoods
        group_ll = np.zeros((K, n_samples), dtype=np.float64)
        for k, g in enumerate(groups):
            if len(g) == 0:
                group_ll[k, :] = -np.inf
                continue
            g_logp = log_probs[g, :]          # shape (|g|, N)
            g_w = tree_weights[k][:, None]    # shape (|g|, 1)
            # logsumexp over trees in group
            group_ll[k, :] = logsumexp(g_logp + np.log(g_w), axis=0)

        # root log-likelihood
        root_ll = logsumexp(group_ll + np.log(group_weights[:, None]), axis=0)  # shape (N,)

        for k, g in enumerate(groups):
            if len(g) == 0:
                continue
            for idx_t, t in enumerate(g):
                resp[k, t, :] = np.exp(
                    log_probs[t, :]
                    + np.log(tree_weights[k][idx_t])
                    + np.log(group_weights[k])
                    - root_ll
                )

        # ---- M-step ----
        for k, g in enumerate(groups):
            if len(g) == 0:
                tree_weights[k] = np.array([])
                continue
            group_resp = resp[k, g, :]          # shape (|g|, N)
            tree_w = group_resp.sum(axis=1)     # shape (|g|,)
            total = tree_w.sum()
            if total > 0:
                tree_weights[k] = tree_w / total
            else:
                tree_weights[k] = np.ones(len(g)) / len(g)

            group_weights[k] = group_resp.sum() / n_samples

        # normalise group weights
        s = group_weights.sum()
        if s > 0:
            group_weights /= s
        else:
            group_weights = np.ones(K) / K

    return group_weights, tree_weights


def build_deep_gef(rf, data, ncat, n_groups=5, n_em_iter=10,
                   learnspn=np.inf, max_height=1000000, thr=0.01,
                   minstd=1, smoothing=1e-6, tcr=False, rho=0.5,
                   copula_reg=1e-6, min_samples_copula=5, chow_liu=False):
    """
    Build a Deep GeF (D-GeF): a hierarchical Probabilistic Circuit on top of
    a Random Forest ensemble.

    Parameters
    ----------
    rf : RandomForest
        A fitted RandomForest object (gefs.trees.RandomForest).
    data : np.ndarray, shape (n_samples, n_features + 1)
        Training data *including* the class variable in the last column.
    ncat : np.ndarray
        Number of categories per variable.
    n_groups : int
        Number of tree groups in the Meta-PC (K).
    n_em_iter : int
        EM iterations for weight learning.
    **kwargs : passed to `rf.topc()` for base GeDT learning.

    Returns
    -------
    pc : PC
        A Probabilistic Circuit representing the D-GeF.
    """
    print("[D-GeF] Step 1/5: building base GeF ...")
    base_gef = rf.topc(learnspn=learnspn, max_height=max_height, thr=thr,
                       minstd=minstd, smoothing=smoothing, tcr=tcr, rho=rho,
                       copula_reg=copula_reg, min_samples_copula=min_samples_copula,
                       chow_liu=chow_liu)
    T = rf.n_estimators
    scope = base_gef.root.scope.copy()

    print("[D-GeF] Step 2/5: extracting hard leaf assignments ...")
    X_train = data[:, :-1]
    leaf_ids = np.zeros((data.shape[0], T), dtype=np.int64)
    for t in range(T):
        leaf_ids[:, t] = get_tree_leaf_ids(rf.estimators[t], X_train)

    print("[D-GeF] Step 3/5: computing MI and Chow-Liu tree ...")
    mi_mat = compute_mi_matrix(leaf_ids)
    edges = chow_liu_max_spanning_tree(mi_mat)
    groups = cluster_tree_groups(edges, n_groups, T)
    print(f"[D-GeF]   -> formed {len(groups)} groups: {[len(g) for g in groups]}")

    print("[D-GeF] Step 4/5: evaluating base trees on training data ...")
    # log_probs[t, n] = log p_t(data[n])
    log_probs = np.zeros((T, data.shape[0]), dtype=np.float64)
    for t in tqdm(range(T), desc="Tree LL"):
        log_probs[t, :] = eval_root(base_gef.root.children[t], data)

    print("[D-GeF] Step 5/5: learning hierarchical weights (EM) ...")
    group_weights, tree_weights = learn_hierarchical_weights(
        log_probs, groups, n_iter=n_em_iter)

    # ---- Assemble Meta-PC ----
    pc = PC(ncat)
    pc.root = SumNode(scope=scope, n=data.shape[0])
    pc.is_ensemble = True

    for k, g in enumerate(groups):
        if len(g) == 0:
            continue
        # Each group is a Sum node (mixture of its trees)
        group_sum = SumNode(scope=scope, n=data.shape[0])
        for idx_t, t in enumerate(g):
            group_sum.add_child(base_gef.root.children[t])
        # Override the uniform weights learned by add_child/reweight
        group_sum.w = tree_weights[k].astype(np.float64)
        group_sum.logw = np.log(tree_weights[k]).astype(np.float64)
        pc.root.add_child(group_sum)

    # Override root weights
    pc.root.w = group_weights.astype(np.float64)
    pc.root.logw = np.log(group_weights).astype(np.float64)

    print("[D-GeF] Done.")
    return pc

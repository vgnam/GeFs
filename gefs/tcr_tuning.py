import numpy as np

from .nodes import eval_tcr_class


_EPS = 1e-12


def sanitize_probabilities(prob, n_classes):
    prob = np.asarray(prob, dtype=np.float64)
    prob = np.nan_to_num(prob, nan=0., posinf=0., neginf=0.)
    prob = np.maximum(prob, 0.)
    if prob.ndim != 2 or prob.shape[1] != n_classes:
        raise ValueError("prob must have shape (n_samples, n_classes)")

    prob_sum = prob.sum(axis=1, keepdims=True)
    low_density = prob_sum[:, 0] <= 0.
    if np.any(low_density):
        prob[low_density, :] = 1. / n_classes
        prob_sum = prob.sum(axis=1, keepdims=True)
    return prob / prob_sum


def random_forest_predict_proba(rf, X):
    n_classes = int(rf.ncat[-1])
    probas = np.zeros((X.shape[0], len(rf.estimators), n_classes),
                      dtype=np.float64)
    for i, estimator in enumerate(rf.estimators):
        probas[:, i, :] = estimator.predict_proba(X)
    return sanitize_probabilities(np.mean(probas, axis=1), n_classes)


def classification_kl_loss(y, p_teacher, p_model, beta):
    y = np.asarray(y, dtype=np.int64)
    p_teacher = np.clip(np.asarray(p_teacher, dtype=np.float64), _EPS, 1.)
    p_model = np.clip(np.asarray(p_model, dtype=np.float64), _EPS, 1.)
    ce = -np.log(p_model[np.arange(y.shape[0]), y])
    kl = np.sum(p_teacher * (np.log(p_teacher) - np.log(p_model)), axis=1)
    return float(np.sum(ce + beta * kl))


def tcr_rho_leaves(pc):
    leaves = []
    queue = [pc.root]
    while queue:
        node = queue.pop(0)
        if node.type == 'R':
            leaves.append(node)
        elif node.type not in ['L', 'G', 'M', 'U', 'R']:
            queue.extend(node.children)
    return leaves


def get_tcr_rhos(pc):
    return np.asarray([leaf.rho for leaf in tcr_rho_leaves(pc)],
                      dtype=np.float64)


def set_tcr_rhos(pc, values):
    leaves = tcr_rho_leaves(pc)
    values = np.asarray(values, dtype=np.float64)
    if len(leaves) != values.shape[0]:
        raise ValueError("rho vector length does not match TCR leaves")
    for leaf, value in zip(leaves, values):
        leaf.rho = float(np.clip(value, 0., 1.))


def tcr_validation_loss(pc, X, y, p_rf, beta=1., gamma=0.):
    n_classes = int(pc.ncat[-1])
    _, p_model = pc.classify_avg(X, return_prob=True)
    p_model = sanitize_probabilities(p_model, n_classes)
    data_loss = classification_kl_loss(y, p_rf, p_model, beta)
    rho_penalty = float(np.sum([leaf.rho ** 2 for leaf in tcr_rho_leaves(pc)]))
    return {
        "loss": data_loss + gamma * rho_penalty,
        "data_loss": data_loss,
        "rho_penalty": rho_penalty,
    }


def _split_go_left(split, x):
    value = x[split.var]
    if not np.isnan(value):
        if split.type == 'num':
            return bool(value <= split.threshold[0])
        return bool(np.any(split.threshold == value))

    for var, threshold, go_left in zip(split.surr_var, split.surr_thr,
                                       split.surr_go_left):
        value = x[var]
        if not np.isnan(value):
            if value <= threshold:
                return bool(go_left)
            return not bool(go_left)
    return bool(split.surr_blind)


def _route_leaf_id(tree_node, x):
    node = tree_node
    while not node.isleaf:
        if _split_go_left(node.split, x):
            node = node.left_child
        else:
            node = node.right_child
    return int(node.id)


def _tree_leaf_ids(tree, X):
    leaf_ids = np.empty(X.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        leaf_ids[i] = _route_leaf_id(tree.root, X[i, :])
    return leaf_ids


def _child_sum_node(pc_node):
    if pc_node.type == 'S':
        return pc_node
    for child in pc_node.children:
        if child.type == 'S':
            return child
    raise ValueError("Cannot find split sum node in converted PC")


def _direct_tcr_leaf(pc_node):
    if pc_node.type == 'R':
        return pc_node
    for child in pc_node.children:
        if child.type == 'R':
            return child
    raise ValueError("Cannot find TCR leaf in converted PC")


def _map_tree_to_tcr_leaves(tree_node, pc_node, out):
    if tree_node.isleaf:
        out[int(tree_node.id)] = _direct_tcr_leaf(pc_node)
        return

    split_sum = _child_sum_node(pc_node)
    children = split_sum.children
    if len(children) != 2:
        raise ValueError("Expected binary split in converted PC")
    _map_tree_to_tcr_leaves(tree_node.left_child, children[0], out)
    _map_tree_to_tcr_leaves(tree_node.right_child, children[1], out)


def _tree_pc_roots(tcr_pc):
    if not tcr_pc.is_ensemble:
        return [tcr_pc.root]
    return tcr_pc.root.children


def _leaf_predict_proba(leaf, X, n_classes):
    log_joints = eval_tcr_class(leaf, X, n_classes, False)
    max_values = np.max(log_joints, axis=1, keepdims=True)
    max_values = np.where(np.isfinite(max_values), max_values, 0.)
    probs = np.exp(log_joints - max_values)
    return sanitize_probabilities(probs, n_classes)


def _leaf_loss_for_rho(leaf, rho, X, y, p_rf, beta, gamma, tree_weight,
                       n_classes):
    old_rho = leaf.rho
    leaf.rho = float(rho)
    try:
        p_leaf = _leaf_predict_proba(leaf, X, n_classes)
    finally:
        leaf.rho = old_rho
    data_loss = classification_kl_loss(y, p_rf, p_leaf, beta)
    return tree_weight * data_loss + gamma * float(rho) ** 2


def tune_tcr_rhos(rf, tcr_pc, X_val, y_val, p_rf_val, beta=1., gamma=0.01,
                  rho_grid_size=21):
    """Tune one rho_v per TCR leaf on a validation set.

    The local objective is the validation CE + beta KL term for validation
    examples routed to the corresponding RF leaf, plus gamma * rho_v^2. A
    1 / n_trees factor keeps the sum over tree-local objectives on the same
    scale as the ensemble validation loss.
    """
    if rho_grid_size < 2:
        raise ValueError("rho_grid_size must be at least 2")

    n_classes = int(tcr_pc.ncat[-1])
    candidates = np.linspace(0., 1., int(rho_grid_size), dtype=np.float64)
    tree_roots = _tree_pc_roots(tcr_pc)
    if len(tree_roots) != len(rf.estimators):
        raise ValueError("RF estimators and TCR PC roots do not match")

    tuned = 0
    untouched = 0
    values = []
    tree_weight = 1. / max(len(rf.estimators), 1)

    for tree, pc_root in zip(rf.estimators, tree_roots):
        leaf_map = {}
        _map_tree_to_tcr_leaves(tree.root, pc_root, leaf_map)
        val_leaf_ids = _tree_leaf_ids(tree, X_val)

        for tree_leaf_id, leaf in leaf_map.items():
            indices = np.where(val_leaf_ids == tree_leaf_id)[0]
            if indices.size == 0:
                leaf.rho = 0.
                untouched += 1
                values.append(leaf.rho)
                continue

            X_leaf = X_val[indices, :]
            y_leaf = y_val[indices]
            p_rf_leaf = p_rf_val[indices, :]
            losses = [
                _leaf_loss_for_rho(
                    leaf, rho, X_leaf, y_leaf, p_rf_leaf, beta, gamma,
                    tree_weight, n_classes,
                )
                for rho in candidates
            ]
            best_idx = int(np.argmin(losses))
            leaf.rho = float(candidates[best_idx])
            tuned += 1
            values.append(leaf.rho)

    values = np.asarray(values, dtype=np.float64)
    return {
        "n_leaves": int(values.size),
        "n_tuned": int(tuned),
        "n_untouched": int(untouched),
        "rho_mean": float(np.mean(values)) if values.size else 0.,
        "rho_min": float(np.min(values)) if values.size else 0.,
        "rho_max": float(np.max(values)) if values.size else 0.,
    }

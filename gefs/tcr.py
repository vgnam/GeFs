import numpy as np
from scipy.stats import norm

from .nodes import ResidualGaussianCopulaLeaf


_EPS = 1e-12


def _observed(values):
    return values[~np.isnan(values)]


def _safe_mean_std(values, minstd, fallback_mean=0., fallback_std=None):
    obs = _observed(values)
    if obs.size == 0:
        if fallback_std is None:
            fallback_std = minstd
        return float(fallback_mean), float(max(minstd, fallback_std))
    mean = float(np.mean(obs))
    if obs.size <= 1:
        std = minstd if fallback_std is None else fallback_std
    else:
        std = float(np.std(obs))
    return mean, float(max(minstd, std))


def _categorical_logp(values, n_categories, smoothing):
    obs = _observed(values)
    counts = np.zeros(n_categories, dtype=np.float64)
    if obs.size > 0:
        cats = obs.astype(np.int64)
        cats = cats[(cats >= 0) & (cats < n_categories)]
        if cats.size > 0:
            counts += np.bincount(cats, minlength=n_categories)
    counts += smoothing
    total = np.sum(counts)
    if total <= 0.:
        counts += 1.
        total = np.sum(counts)
    return np.log(counts / total)


def _truncnorm_cdf(x, mean, std, lower, upper):
    lower_cdf = norm.cdf((lower - mean) / std)
    upper_cdf = norm.cdf((upper - mean) / std)
    denom = upper_cdf - lower_cdf
    if denom <= 0.:
        return np.full(x.shape[0], 0.5, dtype=np.float64)
    u = (norm.cdf((x - mean) / std) - lower_cdf) / denom
    return np.clip(u, _EPS, 1. - _EPS)


def _nearest_regular_correlation(corr, reg):
    corr = np.nan_to_num(corr, nan=0., posinf=0., neginf=0.)
    corr = (corr + corr.T) / 2.
    np.fill_diagonal(corr, 1.)
    corr = (1. - reg) * corr + reg * np.eye(corr.shape[0])

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, reg)
    corr = (eigvecs * eigvals) @ eigvecs.T
    diag = np.sqrt(np.maximum(np.diag(corr), reg))
    corr = corr / np.outer(diag, diag)
    corr = (corr + corr.T) / 2.
    np.fill_diagonal(corr, 1.)
    return corr


def _fit_gaussian_copula_corr(data, copula_scope, means, stds, lower, upper,
                              reg, min_samples):
    d = len(copula_scope)
    corr = np.eye(d, dtype=np.float64)
    if d <= 1 or data.shape[0] == 0:
        return corr

    cont = data[:, copula_scope]
    complete = np.isfinite(cont).all(axis=1)
    cont = cont[complete]
    if cont.shape[0] < max(min_samples, d + 1):
        return corr

    z = np.empty(cont.shape, dtype=np.float64)
    for pos, var in enumerate(copula_scope):
        u = _truncnorm_cdf(cont[:, pos], means[var], stds[var],
                           lower[var], upper[var])
        z[:, pos] = norm.ppf(u)

    corr = np.corrcoef(z, rowvar=False)
    if corr.ndim == 0:
        return np.eye(d, dtype=np.float64)
    return _nearest_regular_correlation(corr, reg)


def make_residual_gaussian_copula_leaf(scope, data_leaf, ncat, upper, lower,
                                       rho=0.5, minstd=1.,
                                       smoothing=1e-6, copula_reg=1e-6,
                                       min_samples_copula=5):
    """Fit a TCR leaf with Gaussian copula class-conditional branch.

    The compatible branch is the original fully factorized GeF leaf. The
    expressive branch keeps categorical features independent and models
    dependence among continuous features with a Gaussian copula per class.
    """
    rho = float(np.clip(rho, 0., 1.))
    ncat = np.asarray(ncat, dtype=np.int64)
    scope = np.asarray(scope, dtype=np.int64)
    n_features = len(ncat) - 1
    n_classes = int(ncat[-1])
    max_cat = 1
    if n_features > 0:
        max_cat = int(max(np.max(ncat[:n_features]), 1))

    data_leaf = np.asarray(data_leaf, dtype=np.float64)
    y = data_leaf[:, n_features] if data_leaf.shape[0] else np.empty(0)
    valid_y = y[~np.isnan(y)].astype(np.int64)
    valid_y = valid_y[(valid_y >= 0) & (valid_y < n_classes)]
    class_counts = np.bincount(valid_y, minlength=n_classes).astype(np.float64)
    class_counts += smoothing
    if np.sum(class_counts) <= 0.:
        class_counts += 1.
    class_logp = np.log(class_counts / np.sum(class_counts))

    logp0_cat = np.zeros((n_features, max_cat), dtype=np.float64)
    logp1_cat = np.zeros((n_classes, n_features, max_cat), dtype=np.float64)
    mean0 = np.zeros(n_features, dtype=np.float64)
    std0 = np.ones(n_features, dtype=np.float64)
    mean1 = np.zeros((n_classes, n_features), dtype=np.float64)
    std1 = np.ones((n_classes, n_features), dtype=np.float64)

    feature_upper = np.asarray(upper[:n_features], dtype=np.float64)
    feature_lower = np.asarray(lower[:n_features], dtype=np.float64)

    for var in range(n_features):
        if ncat[var] > 1:
            k = int(ncat[var])
            logp0_cat[var, :k] = _categorical_logp(data_leaf[:, var], k,
                                                   smoothing)
        else:
            mean0[var], std0[var] = _safe_mean_std(data_leaf[:, var], minstd)

    for c in range(n_classes):
        class_data = data_leaf[y == c] if data_leaf.shape[0] else data_leaf
        for var in range(n_features):
            if ncat[var] > 1:
                k = int(ncat[var])
                logp1_cat[c, var, :k] = _categorical_logp(class_data[:, var],
                                                          k, smoothing)
            else:
                mean1[c, var], std1[c, var] = _safe_mean_std(
                    class_data[:, var],
                    minstd,
                    fallback_mean=mean0[var],
                    fallback_std=std0[var],
                )

    copula_scope = np.where(ncat[:n_features] == 1)[0].astype(np.int64)
    d = len(copula_scope)
    corr1 = np.zeros((n_classes, d, d), dtype=np.float64)
    for c in range(n_classes):
        class_data = data_leaf[y == c] if data_leaf.shape[0] else data_leaf
        corr1[c] = _fit_gaussian_copula_corr(
            class_data,
            copula_scope,
            mean1[c],
            std1[c],
            feature_lower,
            feature_upper,
            copula_reg,
            min_samples_copula,
        )

    leaf = ResidualGaussianCopulaLeaf(scope, data_leaf.shape[0])
    leaf.rho = rho
    leaf.logcounts = np.log(class_counts)
    leaf.tcr_ncat = ncat
    leaf.tcr_copula_scope = copula_scope
    leaf.tcr_class_logp = class_logp
    leaf.tcr_logp0_cat = logp0_cat
    leaf.tcr_logp1_cat = logp1_cat
    leaf.tcr_mean0 = mean0
    leaf.tcr_std0 = std0
    leaf.tcr_mean1 = mean1
    leaf.tcr_std1 = std1
    leaf.tcr_corr1 = corr1
    leaf.tcr_lower = feature_lower
    leaf.tcr_upper = feature_upper
    return leaf

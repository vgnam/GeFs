import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from gefs import RandomForest


def factorize_columns(df, columns):
    df = df.copy()
    for col in columns:
        codes, _ = pd.factorize(df[col], sort=True)
        df[col] = codes
    return df


def is_continuous(values):
    observed = values[~np.isnan(values)]
    if observed.size == 0:
        return False
    rules = [
        np.min(observed) < 0,
        np.any(observed != np.round(observed)),
        len(np.unique(observed)) > min(30, observed.size / 3),
    ]
    return any(rules)


def learncats(data, classcol=-1, continuous_ids=None):
    if continuous_ids is None:
        continuous_ids = []
    classcol = data.shape[1] - 1 if classcol < 0 else classcol
    ncat = np.ones(data.shape[1], dtype=np.int64)
    data = data.copy()
    for col in range(data.shape[1]):
        if col != classcol and (col in continuous_ids or is_continuous(data[:, col])):
            continue
        values = data[:, col]
        observed = values[~np.isnan(values)]
        if observed.size == 0:
            ncat[col] = 1
        else:
            values = values.astype(np.int64)
            ncat[col] = int(np.max(values)) + 1
    return ncat


def preprocess_known_dataset(name, data_dir):
    if name == "german":
        df = pd.read_csv(data_dir / "german.csv", sep=" ", header=None)
        cat_cols = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20]
        cont_cols = [1, 4, 7, 10, 12, 15, 17]
        df = factorize_columns(df, cat_cols)
        data = df.values.astype(float)
        return data, learncats(data, classcol=-1, continuous_ids=cont_cols)

    if name == "cmc":
        df = pd.read_csv(data_dir / "cmc.csv")
        cat_cols = [
            "Wifes_education",
            "Husbands_education",
            "Wifes_religion",
            "Wifes_now_working%3F",
            "Husbands_occupation",
            "Standard-of-living_index",
            "Media_exposure",
            "Contraceptive_method_used",
        ]
        cont_cols = ["Wifes_age", "Number_of_children_ever_born"]
        df = factorize_columns(df, cat_cols)
        data = df.values.astype(float)
        continuous_ids = [df.columns.get_loc(col) for col in cont_cols]
        return data, learncats(data, classcol=-1, continuous_ids=continuous_ids)

    if name == "bank":
        df = pd.read_csv(data_dir / "bank-additional-full.csv", sep=";")
        cat_cols = [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
            "y",
        ]
        cont_cols = [
            "age",
            "duration",
            "campaign",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
        ]
        df = factorize_columns(df, cat_cols)
        df["pdays"] = np.where(df["pdays"] == 999, 0, 1)
        data = df.values.astype(float)
        continuous_ids = [df.columns.get_loc(col) for col in cont_cols]
        return data, learncats(data, classcol=-1, continuous_ids=continuous_ids)

    raise ValueError(f"Unknown built-in dataset: {name}")


def load_dataset(name, data_dir):
    if name in {"german", "cmc", "bank"}:
        return preprocess_known_dataset(name, data_dir)

    path = Path(name)
    if not path.exists():
        path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset '{name}'")

    df = pd.read_csv(path)
    object_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    if object_cols:
        df = factorize_columns(df, object_cols)
    target = df.columns[-1]
    if sorted(pd.unique(df[target].dropna())) != list(range(df[target].nunique())):
        df = factorize_columns(df, [target])
    data = df.values.astype(float)
    return data, learncats(data, classcol=-1)


def standardize_from_train(train, test, ncat):
    train = train.copy()
    test = test.copy()
    for col in range(train.shape[1] - 1):
        if ncat[col] != 1:
            continue
        mean = np.nanmean(train[:, col])
        std = np.nanstd(train[:, col])
        if std <= 0 or np.isnan(std):
            continue
        train[:, col] = np.clip((train[:, col] - mean) / std, -6, 6)
        test[:, col] = np.clip((test[:, col] - mean) / std, -6, 6)
    return train, test


def maybe_subsample(data, max_rows, seed):
    if max_rows is None or max_rows >= data.shape[0]:
        return data
    rng = np.random.default_rng(seed)
    y = data[:, -1].astype(np.int64)
    ids = []
    per_class = max(1, max_rows // len(np.unique(y)))
    for cls in np.unique(y):
        cls_ids = np.where(y == cls)[0]
        take = min(per_class, cls_ids.size)
        ids.extend(rng.choice(cls_ids, size=take, replace=False))
    if len(ids) < max_rows:
        rest = np.setdiff1d(np.arange(data.shape[0]), np.array(ids), assume_unique=False)
        extra = rng.choice(rest, size=min(max_rows - len(ids), rest.size), replace=False)
        ids.extend(extra)
    return data[np.array(ids)]


def add_missing_values(X, missing_rate, seed):
    X = X.copy()
    if missing_rate <= 0:
        return X
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) < missing_rate
    X[mask] = np.nan
    return X


def evaluate_method(name, pc, X, y, n_classes):
    start = time.perf_counter()
    pred, prob = pc.classify_avg(X, return_prob=True)
    elapsed = time.perf_counter() - start
    return {
        "method": name,
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, prob, labels=np.arange(n_classes)),
        "inference_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare GeF and TCR-GeF on the same train/test split.")
    parser.add_argument("--dataset", default="cmc", help="Built-in name: cmc, german, bank; or path to a CSV.")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--minstd", type=float, default=0.1)
    parser.add_argument("--smoothing", type=float, default=1e-6)
    parser.add_argument("--copula-reg", type=float, default=1e-6)
    parser.add_argument("--min-samples-copula", type=int, default=5)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional stratified subsample for quick runs.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data, ncat = load_dataset(args.dataset, root / "data")
    data = maybe_subsample(data, args.max_rows, args.seed)

    y = data[:, -1].astype(np.int64)
    train, test = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    train, test = standardize_from_train(train, test, ncat)
    X_train, y_train = train[:, :-1], train[:, -1].astype(np.int64)
    X_test, y_test = test[:, :-1], test[:, -1].astype(np.int64)
    X_eval = add_missing_values(X_test, args.missing_rate, args.seed + 1)

    print(f"Dataset: {args.dataset}")
    print(f"Train/test: {X_train.shape[0]}/{X_test.shape[0]}")
    print(f"Classes: {int(ncat[-1])}")
    print(f"Missing rate on shared eval X: {args.missing_rate:.2f}")

    start = time.perf_counter()
    rf = RandomForest(
        ncat=ncat,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        max_depth=args.max_depth,
        random_state=args.seed,
    )
    rf.fit(X_train, y_train)
    print(f"RF training: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    gef = rf.topc(minstd=args.minstd, smoothing=args.smoothing)
    print(f"GeF conversion: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    tcr = rf.topc(
        tcr=True,
        rho=args.rho,
        minstd=args.minstd,
        smoothing=args.smoothing,
        copula_reg=args.copula_reg,
        min_samples_copula=args.min_samples_copula,
    )
    print(f"TCR-GeF conversion: {time.perf_counter() - start:.2f}s")

    results = [
        evaluate_method("GeF", gef, X_eval, y_test, int(ncat[-1])),
        evaluate_method("TCR-GeF", tcr, X_eval, y_test, int(ncat[-1])),
    ]
    print()
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    gef.delete()
    tcr.delete()
    rf.delete()
    gc.collect()


if __name__ == "__main__":
    main()

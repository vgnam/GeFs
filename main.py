import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from gefs import RandomForest
from gefs.tcr_tuning import (
    get_tcr_rhos,
    random_forest_predict_proba,
    sanitize_probabilities,
    set_tcr_rhos,
    tcr_validation_loss,
    tune_tcr_rhos,
)


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
    if name in {"wine", "winequality"}:
        red = pd.read_csv(data_dir / "winequality_red.csv")
        white = pd.read_csv(data_dir / "winequality_white.csv")
        df = pd.concat([red, white], ignore_index=True)
        df["quality"] = np.where(df["quality"] <= 6, 0, 1)
        data = df.values.astype(float)
        return data, learncats(data, classcol=-1)

    if name in {"wine-red", "wine_red"}:
        df = pd.read_csv(data_dir / "winequality_red.csv")
        df["quality"] = np.where(df["quality"] <= 6, 0, 1)
        data = df.values.astype(float)
        return data, learncats(data, classcol=-1)

    if name in {"wine-white", "wine_white"}:
        df = pd.read_csv(data_dir / "winequality_white.csv")
        df["quality"] = np.where(df["quality"] <= 6, 0, 1)
        data = df.values.astype(float)
        return data, learncats(data, classcol=-1)

    if name in {"breast", "wdbc"}:
        df = pd.read_csv(data_dir / "wdbc.csv")
        df = factorize_columns(df, ["Class"])
        data = df.values.astype(float)
        ncat = np.ones(data.shape[1], dtype=np.int64)
        ncat[-1] = df["Class"].nunique()
        return data, ncat

    if name == "vehicle":
        df = pd.read_csv(data_dir / "vehicle.csv")
        df = factorize_columns(df, ["Class"])
        data = df.values.astype(float)
        ncat = np.ones(data.shape[1], dtype=np.int64)
        ncat[-1] = df["Class"].nunique()
        return data, ncat

    if name == "segment":
        df = pd.read_csv(data_dir / "segment.csv")
        df = df.drop(columns=["region.centroid.col", "region.pixel.count"])
        cat_cols = ["short.line.density.5", "short.line.density.2", "class"]
        cont_cols = [
            "region.centroid.row",
            "vedge.mean",
            "vegde.sd",
            "hedge.mean",
            "hedge.sd",
            "intensity.mean",
            "rawred.mean",
            "rawblue.mean",
            "rawgreen.mean",
            "exred.mean",
            "exblue.mean",
            "exgreen.mean",
            "value.mean",
            "saturation.mean",
            "hue.mean",
        ]
        df = factorize_columns(df, cat_cols)
        data = df.values.astype(float)
        continuous_ids = [df.columns.get_loc(col) for col in cont_cols]
        return data, learncats(data, classcol=-1, continuous_ids=continuous_ids)

    if name == "vowel":
        df = pd.read_csv(data_dir / "vowel.csv")
        df = factorize_columns(df, ["Speaker_Number", "Sex", "Class"])
        data = df.values.astype(float)
        return data, learncats(data, classcol=-1)

    if name in {"mice", "miceprotein"}:
        df = pd.read_csv(data_dir / "miceprotein.csv")
        df = df.replace("?", np.nan)
        df = df.drop(columns=["MouseID", "Genotype", "Treatment", "Behavior"])
        df = factorize_columns(df, ["class"])
        data = df.values.astype(float)
        return data, learncats(data, classcol=-1)

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
    builtin = {
        "german",
        "cmc",
        "bank",
        "wine",
        "winequality",
        "wine-red",
        "wine_red",
        "wine-white",
        "wine_white",
        "breast",
        "wdbc",
        "vehicle",
        "segment",
        "vowel",
        "mice",
        "miceprotein",
    }
    if name in builtin:
        return preprocess_known_dataset(name, data_dir)

    path = Path(name)
    if not path.exists():
        path = data_dir / name
    if not path.exists() and path.suffix == "":
        path = path.with_suffix(".csv")
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


def standardize_from_train(train, test, ncat, *extra):
    train = train.copy()
    others = [test.copy()]
    others.extend(arr.copy() for arr in extra)
    for col in range(train.shape[1] - 1):
        if ncat[col] != 1:
            continue
        mean = np.nanmean(train[:, col])
        std = np.nanstd(train[:, col])
        if std <= 0 or np.isnan(std):
            continue
        train[:, col] = np.clip((train[:, col] - mean) / std, -6, 6)
        for arr in others:
            arr[:, col] = np.clip((arr[:, col] - mean) / std, -6, 6)
    return (train, *others)


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


def macro_auroc(y, prob, n_classes):
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return np.nan

    try:
        if n_classes == 2:
            return float(roc_auc_score(y, prob[:, 1]))
        return float(
            roc_auc_score(
                y,
                prob,
                labels=np.arange(n_classes),
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return np.nan


def expected_calibration_error(y, prob, n_bins=15):
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    y = np.asarray(y, dtype=np.int64)
    confidence = np.max(prob, axis=1)
    pred = np.argmax(prob, axis=1)
    correct = pred == y
    ece = 0.
    bins = np.linspace(0., 1., n_bins + 1)

    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (confidence >= bins[i]) & (confidence <= bins[i + 1])
        else:
            in_bin = (confidence >= bins[i]) & (confidence < bins[i + 1])
        if not np.any(in_bin):
            continue
        bin_weight = np.mean(in_bin)
        bin_accuracy = np.mean(correct[in_bin])
        bin_confidence = np.mean(confidence[in_bin])
        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def evaluate_method(name, pc, X, y, n_classes, ece_bins=15, lspn=False):
    start = time.perf_counter()
    if lspn:
        pred, prob = pc.classify_avg_lspn(X, return_prob=True)
    else:
        pred, prob = pc.classify_avg(X, return_prob=True)
    elapsed = time.perf_counter() - start
    prob = sanitize_probabilities(prob, n_classes)
    pred = np.asarray(pred, dtype=np.int64)
    labels = np.arange(n_classes)
    return {
        "method": name,
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, prob, labels=labels)),
        "auroc_macro_ovr": macro_auroc(y, prob, n_classes),
        "ece": expected_calibration_error(y, prob, ece_bins),
        "f1_macro": float(
            f1_score(y, pred, labels=labels, average="macro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(
                y,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "inference_sec": elapsed,
    }


def _safe_filename(value):
    value = str(value)
    safe = []
    for char in value:
        if char.isalnum() or char in "-_.":
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "run"


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_run_results(results_df, args, root, run_info):
    output_dir = Path(args.results_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dataset = _safe_filename(Path(str(args.dataset)).stem)
    stem = f"{timestamp}_{dataset}_seed{args.seed}"
    csv_path = output_dir / f"{stem}_results.csv"
    json_path = output_dir / f"{stem}_run.json"

    saved = results_df.copy()
    saved.insert(0, "dataset", args.dataset)
    saved.insert(1, "seed", args.seed)
    saved.insert(2, "missing_rate", args.missing_rate)
    saved.to_csv(csv_path, index=False)

    metadata = {
        "args": vars(args),
        "run": run_info,
        "results": saved.to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, default=_json_default)

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Compare GeF, GeF-LearnSPN, and TCR-GeF on the same train/test split.")
    parser.add_argument(
        "--dataset",
        default="cmc",
        help="Built-in name: cmc, german, bank, vehicle, breast, segment, vowel, wine, mice; or path to a CSV.",
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--minstd", type=float, default=0.1)
    parser.add_argument("--smoothing", type=float, default=1e-6)
    parser.add_argument("--learnspn-min-samples", type=int, default=30, help="Minimum leaf samples required to fit LearnSPN.")
    parser.add_argument("--learnspn-max-height", type=int, default=1000000, help="Maximum depth of LearnSPN structures at leaves.")
    parser.add_argument("--learnspn-thr", type=float, default=0.01, help="Independence-test threshold for LearnSPN product splits.")
    parser.add_argument("--skip-learnspn", action="store_true", help="Compare only GeF and TCR-GeF.")
    parser.add_argument("--copula-reg", type=float, default=1e-6)
    parser.add_argument("--min-samples-copula", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1.0, help="Weight of KL(p_RF || p_rho) on validation.")
    parser.add_argument("--gamma", type=float, default=0.01, help="Weight of sum_v rho_v^2 regularization.")
    parser.add_argument("--rho-grid-size", type=int, default=21, help="Number of [0, 1] grid points for validation rho tuning.")
    parser.add_argument("--no-tune-rho", action="store_true", help="Disable validation tuning of per-leaf TCR rho values.")
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--ece-bins", type=int, default=15, help="Number of bins for expected calibration error.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional stratified subsample for quick runs.")
    parser.add_argument("--results-dir", default="results", help="Directory where run results are saved.")
    parser.add_argument("--no-save-results", action="store_true", help="Do not save CSV/JSON result files.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data, ncat = load_dataset(args.dataset, root / "data")
    data = maybe_subsample(data, args.max_rows, args.seed)

    y = data[:, -1].astype(np.int64)
    train_full, test = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    use_validation = args.val_size > 0 and not args.no_tune_rho
    if use_validation:
        train, val = train_test_split(
            train_full,
            test_size=args.val_size,
            random_state=args.seed + 17,
            stratify=train_full[:, -1].astype(np.int64),
        )
    else:
        train, val = train_full, None

    if val is None:
        train, test = standardize_from_train(train, test, ncat)
    else:
        train, val, test = standardize_from_train(train, val, ncat, test)

    X_train, y_train = train[:, :-1], train[:, -1].astype(np.int64)
    if val is not None:
        X_val, y_val = val[:, :-1], val[:, -1].astype(np.int64)
    else:
        X_val, y_val = None, None
    X_test, y_test = test[:, :-1], test[:, -1].astype(np.int64)
    X_eval = add_missing_values(X_test, args.missing_rate, args.seed + 1)
    n_classes = int(ncat[-1])

    print(f"Dataset: {args.dataset}")
    if X_val is None:
        print(f"Train/test: {X_train.shape[0]}/{X_test.shape[0]}")
    else:
        print(f"Train/validation/test: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
    print(f"Classes: {n_classes}")
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
    rf_training_sec = time.perf_counter() - start
    print(f"RF training: {rf_training_sec:.2f}s")

    start = time.perf_counter()
    gef = rf.topc(minstd=args.minstd, smoothing=args.smoothing)
    gef_conversion_sec = time.perf_counter() - start
    print(f"GeF conversion: {gef_conversion_sec:.2f}s")

    gef_lspn = None
    gef_lspn_conversion_sec = None
    if not args.skip_learnspn:
        start = time.perf_counter()
        gef_lspn = rf.topc(
            learnspn=args.learnspn_min_samples,
            max_height=args.learnspn_max_height,
            thr=args.learnspn_thr,
            minstd=args.minstd,
            smoothing=args.smoothing,
        )
        gef_lspn_conversion_sec = time.perf_counter() - start
        print(f"GeF-LearnSPN conversion: {gef_lspn_conversion_sec:.2f}s")

    start = time.perf_counter()
    tcr = rf.topc(
        tcr=True,
        rho=args.rho,
        minstd=args.minstd,
        smoothing=args.smoothing,
        copula_reg=args.copula_reg,
        min_samples_copula=args.min_samples_copula,
    )
    tcr_conversion_sec = time.perf_counter() - start
    print(f"TCR-GeF conversion: {tcr_conversion_sec:.2f}s")

    tuning_summary = None
    if X_val is not None and not args.no_tune_rho:
        p_rf_val = random_forest_predict_proba(rf, X_val)
        initial_rhos = get_tcr_rhos(tcr)
        before = tcr_validation_loss(
            tcr,
            X_val,
            y_val,
            p_rf_val,
            beta=args.beta,
            gamma=args.gamma,
        )
        start = time.perf_counter()
        tuning = tune_tcr_rhos(
            rf,
            tcr,
            X_val,
            y_val,
            p_rf_val,
            beta=args.beta,
            gamma=args.gamma,
            rho_grid_size=args.rho_grid_size,
        )
        elapsed = time.perf_counter() - start
        after = tcr_validation_loss(
            tcr,
            X_val,
            y_val,
            p_rf_val,
            beta=args.beta,
            gamma=args.gamma,
        )
        kept_tuned_rhos = True
        if after["loss"] > before["loss"]:
            set_tcr_rhos(tcr, initial_rhos)
            after = tcr_validation_loss(
                tcr,
                X_val,
                y_val,
                p_rf_val,
                beta=args.beta,
                gamma=args.gamma,
            )
            kept_tuned_rhos = False
        print(f"TCR rho tuning: {elapsed:.2f}s")
        print(
            "Validation loss: "
            f"{before['loss']:.6f} -> {after['loss']:.6f} "
            f"(data {after['data_loss']:.6f}, rho^2 {after['rho_penalty']:.6f})"
        )
        print(
            "Rho leaves: "
            f"{tuning['n_tuned']}/{tuning['n_leaves']} tuned, "
            f"mean={tuning['rho_mean']:.4f}, "
            f"range=[{tuning['rho_min']:.4f}, {tuning['rho_max']:.4f}]"
        )
        if not kept_tuned_rhos:
            print("Tuned rho values were reverted because validation loss increased.")
        tuning_summary = {
            "before": before,
            "after": after,
            "kept_tuned_rhos": kept_tuned_rhos,
            "tuning": tuning,
            "tuning_sec": elapsed,
        }

    results = [evaluate_method("GeF", gef, X_eval, y_test, n_classes, args.ece_bins)]
    if gef_lspn is not None:
        results.append(
            evaluate_method(
                "GeF-LearnSPN",
                gef_lspn,
                X_eval,
                y_test,
                n_classes,
                args.ece_bins,
                lspn=True,
            )
        )
    results.append(
        evaluate_method("TCR-GeF", tcr, X_eval, y_test, n_classes, args.ece_bins)
    )
    results_df = pd.DataFrame(results)
    print()
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if not args.no_save_results:
        run_info = {
            "n_train": X_train.shape[0],
            "n_validation": 0 if X_val is None else X_val.shape[0],
            "n_test": X_test.shape[0],
            "n_features": X_train.shape[1],
            "n_classes": n_classes,
            "rf_training_sec": rf_training_sec,
            "gef_conversion_sec": gef_conversion_sec,
            "gef_learnspn_conversion_sec": gef_lspn_conversion_sec,
            "tcr_conversion_sec": tcr_conversion_sec,
            "tuning": tuning_summary,
        }
        csv_path, json_path = save_run_results(results_df, args, root, run_info)
        print()
        print(f"Saved results: {csv_path}")
        print(f"Saved run metadata: {json_path}")

    gef.delete()
    if gef_lspn is not None:
        gef_lspn.delete()
    tcr.delete()
    rf.delete()
    gc.collect()


if __name__ == "__main__":
    main()

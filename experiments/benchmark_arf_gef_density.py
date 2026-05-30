import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gefs import ARFGeF, RandomForest
from main import load_dataset, maybe_subsample, standardize_from_train


def make_joint_dataframe(data, ncat):
    columns = [f"v{i}" for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=columns)
    for j, col in enumerate(columns):
        if ncat[j] > 1:
            values = df[col].astype(np.int64)
            df[col] = pd.Categorical(values, categories=np.arange(int(ncat[j])))
    return df


def mean_ll(pc, data):
    return float(np.mean(pc.log_likelihood(data)))


def run_one(dataset, seed, args, root):
    started = time.perf_counter()
    data, ncat = load_dataset(dataset, root / "data")
    max_rows = None if args.max_rows <= 0 else args.max_rows
    data = maybe_subsample(data, max_rows, seed)

    y = data[:, -1].astype(np.int64)
    train, test = train_test_split(
        data,
        test_size=args.test_size,
        random_state=seed,
        stratify=y,
    )
    train, test = standardize_from_train(train, test, ncat)
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int64)

    result_base = {
        "dataset": dataset,
        "seed": seed,
        "n_train": train.shape[0],
        "n_test": test.shape[0],
        "n_vars": train.shape[1],
        "n_classes": int(ncat[-1]),
        "n_estimators": args.n_estimators,
        "max_rows": max_rows if max_rows is not None else 0,
    }

    rf = None
    gef = None
    gef_started = time.perf_counter()
    rf = RandomForest(
        ncat=ncat,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        max_depth=args.max_depth,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    gef = rf.topc(minstd=args.minstd, smoothing=args.smoothing)
    gef_sec = time.perf_counter() - gef_started

    gef_train_ll = mean_ll(gef, train)
    gef_test_ll = mean_ll(gef, test)

    train_df = make_joint_dataframe(train, ncat)
    test_df = make_joint_dataframe(test, ncat)
    arf_started = time.perf_counter()
    arf_gef = ARFGeF(
        num_trees=args.n_estimators,
        max_iters=args.arf_max_iters,
        delta=args.arf_delta,
        early_stop=not args.no_arf_early_stop,
        verbose=False,
        min_node_size=args.min_samples_leaf,
        ncat=ncat,
        minstd=args.minstd,
        smoothing=args.smoothing,
        random_state=seed,
    )
    arf_gef.fit(train_df)
    arf_sec = time.perf_counter() - arf_started

    arf_train_ll = float(np.mean(arf_gef.log_likelihood(train_df)))
    arf_test_ll = float(np.mean(arf_gef.log_likelihood(test_df)))
    total_sec = time.perf_counter() - started

    row = {
        **result_base,
        "status": "ok",
        "gef_train_mean_log_xy": gef_train_ll,
        "gef_test_mean_log_xy": gef_test_ll,
        "arf_gef_train_mean_log_xy": arf_train_ll,
        "arf_gef_test_mean_log_xy": arf_test_ll,
        "test_ll_delta_arf_minus_gef": arf_test_ll - gef_test_ll,
        "gef_sec": gef_sec,
        "arf_gef_sec": arf_sec,
        "total_sec": total_sec,
        "arf_acc_trace": ";".join(f"{x:.6f}" for x in arf_gef.arf_model.acc),
        "error": "",
    }

    if gef is not None:
        gef.delete()
    if rf is not None:
        rf.delete()
    gc.collect()
    return row


def save_results(rows, args, root):
    out_dir = Path(args.results_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{timestamp}_arf_gef_density.csv"
    json_path = out_dir / f"{timestamp}_arf_gef_density_args.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2, sort_keys=True)
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare GeF and ARF-GeF on density estimation via mean log p(x,y)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wdbc", "vehicle", "wine-red"],
        help="Built-in dataset names or CSV paths. Default keeps the run moderate.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-rows", type=int, default=1000, help="0 disables subsampling.")
    parser.add_argument("--n-estimators", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=1000000)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--minstd", type=float, default=0.1)
    parser.add_argument("--smoothing", type=float, default=1e-6)
    parser.add_argument("--arf-max-iters", type=int, default=2)
    parser.add_argument("--arf-delta", type=float, default=0.0)
    parser.add_argument("--no-arf-early-stop", action="store_true")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"\n=== {dataset} seed={seed} ===")
            try:
                row = run_one(dataset, seed, args, root)
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "status": "error",
                    "error": repr(exc),
                }
            rows.append(row)
            print(pd.DataFrame([row]).to_string(index=False))

    results = pd.DataFrame(rows)
    print("\n=== Summary ===")
    cols = [
        "dataset",
        "seed",
        "status",
        "gef_test_mean_log_xy",
        "arf_gef_test_mean_log_xy",
        "test_ll_delta_arf_minus_gef",
        "gef_sec",
        "arf_gef_sec",
        "error",
    ]
    cols = [c for c in cols if c in results.columns]
    print(results[cols].to_string(index=False))

    if not args.no_save:
        csv_path, json_path = save_results(rows, args, root)
        print(f"\nSaved results: {csv_path}")
        print(f"Saved args: {json_path}")


if __name__ == "__main__":
    main()

"""Train a random forest to read 8x8 handwritten digits."""

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def split_data(seed=42):
    data = load_digits()
    return train_test_split(
        data.data, data.target, test_size=0.25,
        random_state=seed, stratify=data.target,
    )


def train(output_dir, seed=42, data=None):
    x_train, x_test, y_train, y_test = split_data(seed) if data is None else data
    model = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=1, random_state=seed, n_jobs=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    mistakes = np.flatnonzero(predictions != y_test)
    report = {
        "seed": seed,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "baseline_accuracy": float(baseline.score(x_test, y_test)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "mistakes": [
            {"test_index": int(i), "actual": int(y_test[i]),
             "predicted": int(predictions[i])}
            for i in mistakes[:10]
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "forest.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return model, report


def read_pixels(path):
    """Accept an 8x8 CSV using the dataset's grayscale range, 0 through 16."""
    pixels = np.loadtxt(path, delimiter=",")
    if pixels.shape != (8, 8):
        raise ValueError("Input must be an 8x8 CSV.")
    if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 16).any():
        raise ValueError("Pixel values must be finite numbers between 0 and 16.")
    return pixels.reshape(1, 64)


def predict(model, pixels):
    probabilities = model.predict_proba(pixels)[0]
    index = int(probabilities.argmax())
    return {
        "digit": int(model.classes_[index]),
        "vote_share": round(float(probabilities[index]), 4),
    }


def train_database(engine, output_dir, seed=42):
    from database import load_data, save_run

    data, test_rows, dataset_hash = load_data(engine)
    run_id = str(uuid4())
    output = Path(output_dir) / 'runs' / run_id
    model, metrics = train(output, seed, data=data)
    model_hash = hashlib.sha256((output / 'forest.joblib').read_bytes()).hexdigest()
    model_file = f'runs/{run_id}/forest.joblib'
    save_run(engine, run_id, metrics, dataset_hash, model_hash, model_file, test_rows,
             model.predict(data[1]), model.predict_proba(data[1]).max(axis=1))
    metrics.update(run_id=run_id, dataset_sha256=dataset_hash, model_sha256=model_hash)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--output", default="artifacts")
    fit.add_argument("--seed", type=int, default=42)
    infer = commands.add_parser("predict")
    infer.add_argument("pixels", help="Path to an 8x8 CSV")
    infer.add_argument("--model", default="artifacts/forest.joblib")
    for name in ('db-seed', 'db-train', 'db-report', 'pipeline'):
        command = commands.add_parser(name)
        command.add_argument('--output', default='artifacts')
        command.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == "train":
        _, metrics = train(args.output, args.seed)
        print(json.dumps({k: v for k, v in metrics.items()
                          if k not in {"confusion_matrix", "mistakes"}}, indent=2))
    elif args.command == 'predict':
        # Only load model files you trust; joblib uses pickle internally.
        model = joblib.load(args.model)
        print(json.dumps(predict(model, read_pixels(args.pixels)), indent=2))
    else:
        from database import connect, seed_database, summary

        engine = connect()
        try:
            if args.command in {'db-seed', 'pipeline'}:
                seed_database(engine, args.seed)
            if args.command in {'db-train', 'pipeline'}:
                train_database(engine, args.output, args.seed)
            report = summary(engine)
            if args.command == 'pipeline':
                path = Path(args.output) / 'database_demo.json'
                path.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()

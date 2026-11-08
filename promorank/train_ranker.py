"""Train an editor-calibrated ranker after human labels are collected.

Input CSV must contain the five component score columns and an `editor_score` column.
`editor_score` can be 1-5 or 0-100. This script intentionally refuses to train on
unlabeled screenplay text or on the baseline's own scores as pseudo-ground-truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

FEATURES = [
    "hook_score",
    "cliffhanger_score",
    "standalone_score",
    "emotion_score",
    "duration_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a promo ranker from human editor ratings.")
    parser.add_argument("--labels", required=True, help="CSV exported from PromoRank with an added editor_score column")
    parser.add_argument("--output", default="models/editor_ranker.joblib")
    parser.add_argument("--min-samples", type=int, default=20)
    args = parser.parse_args()

    try:
        import importlib
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor  # type: ignore[reportMissingModuleSource]
        from sklearn.metrics import mean_absolute_error, r2_score  # type: ignore[reportMissingModuleSource]
        from sklearn.model_selection import train_test_split  # type: ignore[reportMissingModuleSource]
        joblib = importlib.import_module("joblib")
    except ImportError as exc:
        raise SystemExit(f"Missing training dependency ({exc.name}). Run: pip install -r requirements-training.txt")

    data = pd.read_csv(args.labels)
    required = FEATURES + ["editor_score"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    data = data[required].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < args.min_samples:
        raise SystemExit(
            f"Need at least {args.min_samples} human-rated candidates; found {len(data)}. "
            "Do not use the baseline score as a substitute for editor labels."
        )

    target = data["editor_score"].astype(float)
    if target.min() >= 1 and target.max() <= 5:
        target = target * 20.0
    if target.min() < 0 or target.max() > 100:
        raise SystemExit("editor_score must be on a 1–5 or 0–100 scale.")

    X = data[FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(f"Human-labeled rows: {len(data)}")
    print(f"Holdout MAE: {mean_absolute_error(y_test, prediction):.2f} / 100")
    if len(y_test) > 1:
        print(f"Holdout R²: {r2_score(y_test, prediction):.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "target_scale": "0-100"}, output)
    print(f"Saved editor-calibrated ranker: {output}")


if __name__ == "__main__":
    main()

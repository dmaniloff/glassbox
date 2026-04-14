"""Train a hallucination detection probe on glassbox-extracted features.

Takes wide-format Parquet output from ``glassbox-extract --parquet`` and
trains a logistic regression classifier on Laplacian eigenvalue features,
following the methodology of LapEigvals (Binkowski et al., EMNLP 2025).

Usage:
    glassbox-train features.parquet -o model.joblib
    glassbox-train features.parquet --pca 512 --test-size 0.3 --layer 15
"""

from __future__ import annotations

import re
import sys

import click


def _find_feature_columns(columns: list[str], signal: str, layer: int | None) -> list[str]:
    """Select feature columns from Parquet matching signal and optional layer filter.

    Matches columns like ``laplacian_lap_eigval_3_L2_H0``.
    """
    pattern = re.compile(
        rf"^{re.escape(signal)}_\w+eigval_\d+_L(\d+)_H\d+$"
    )
    result = []
    for col in columns:
        m = pattern.match(col)
        if m is None:
            continue
        if layer is not None and int(m.group(1)) != layer:
            continue
        result.append(col)
    return sorted(result)


def _train_probe(
    parquet_path: str,
    output: str,
    signal: str,
    pca: int,
    test_size: float,
    layer: int | None,
    seed: int,
) -> dict:
    """Core training logic, separated from CLI for testability.

    Returns the saved model dict.
    """
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(parquet_path)

    # Use "full" phase only (prompt + response attention)
    if "phase" in df.columns:
        df = df[df["phase"] == "full"].reset_index(drop=True)

    if "label" not in df.columns:
        raise click.UsageError("Parquet file must have a 'label' column.")

    feat_cols = _find_feature_columns(list(df.columns), signal, layer)
    if not feat_cols:
        raise click.UsageError(
            f"No feature columns found for signal={signal!r}"
            + (f", layer={layer}" if layer is not None else "")
            + f". Available columns sample: {list(df.columns[:10])}"
        )

    click.echo(f"Features: {len(feat_cols)} columns")
    click.echo(f"  signal={signal}, layer={'all' if layer is None else layer}")

    X = df[feat_cols].values.astype(np.float64)
    y = df["label"].values.astype(np.int64)

    # Drop rows with NaN (cuSOLVER failures)
    valid_mask = ~np.isnan(X).any(axis=1)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        click.echo(f"  dropped {n_dropped} rows with NaN features")
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < 10:
        raise click.UsageError(f"Too few valid samples ({len(X)}) after NaN removal.")

    n_pos = int(y.sum())
    click.echo(f"Samples: {len(X)} ({n_pos} positive / {len(X) - n_pos} negative)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Optional PCA
    pca_transformer = None
    n_features_pca = X_train.shape[1]
    if pca > 0 and X_train.shape[1] > pca:
        effective_dim = min(pca, X_train.shape[0], X_train.shape[1])
        pca_transformer = PCA(n_components=effective_dim, random_state=seed)
        X_train = pca_transformer.fit_transform(X_train)
        X_test = pca_transformer.transform(X_test)
        n_features_pca = effective_dim
        click.echo(f"PCA: {len(feat_cols)} -> {n_features_pca} components")

    # Train logistic regression (matching LapEigvals paper)
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    lr.fit(X_train, y_train)

    # Evaluate
    train_proba = lr.predict_proba(X_train)[:, 1]
    test_proba = lr.predict_proba(X_test)[:, 1]
    train_auroc = float(roc_auc_score(y_train, train_proba))
    test_auroc = float(roc_auc_score(y_test, test_proba))

    click.echo(f"AUROC: train={train_auroc:.4f}  test={test_auroc:.4f}")

    model_dict = {
        "model": lr,
        "pca": pca_transformer,
        "feature_columns": feat_cols,
        "signal": signal,
        "threshold": 0.5,
        "train_auroc": train_auroc,
        "test_auroc": test_auroc,
        "metadata": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features_raw": len(feat_cols),
            "n_features_pca": n_features_pca,
            "pca_dim": pca if pca_transformer is not None else 0,
            "seed": seed,
            "layer": layer,
            "class_distribution": {
                "train_pos": int(y_train.sum()),
                "train_neg": int(len(y_train) - y_train.sum()),
                "test_pos": int(y_test.sum()),
                "test_neg": int(len(y_test) - y_test.sum()),
            },
        },
    }

    joblib.dump(model_dict, output)
    click.echo(f"Model saved to {output}")
    return model_dict


@click.command()
@click.argument("parquet_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    default="model.joblib",
    show_default=True,
    help="Output model path.",
)
@click.option(
    "--signal",
    default="laplacian",
    show_default=True,
    help="Signal type to train on.",
)
@click.option(
    "--pca",
    type=int,
    default=512,
    show_default=True,
    help="PCA components before LR (0 to disable).",
)
@click.option(
    "--test-size",
    type=float,
    default=0.3,
    show_default=True,
    help="Fraction of data for test split.",
)
@click.option(
    "--layer",
    type=int,
    default=None,
    help="Train on a single layer only (default: all layers).",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed.",
)
def main(
    parquet_path: str,
    output: str,
    signal: str,
    pca: int,
    test_size: float,
    layer: int | None,
    seed: int,
) -> None:
    """Train a hallucination detection probe on extracted features."""
    try:
        _train_probe(parquet_path, output, signal, pca, test_size, layer, seed)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""End-to-end helpers: build (and cache) train/test feature tables."""
from pathlib import Path

import pandas as pd

from .data import ID, ROOT, TARGET, load_split
from .features import build_features, fit_type_stats

CACHE_DIR = ROOT / "models" / "feature_cache"


def build_feature_tables(use_cache: bool = True, verbose: bool = True):
    """Return (train_features_with_target, test_features, temporal_reports)."""
    tr_path, te_path = CACHE_DIR / "train_features.parquet", CACHE_DIR / "test_features.parquet"
    if use_cache and tr_path.exists() and te_path.exists():
        return pd.read_parquet(tr_path), pd.read_parquet(te_path), None

    train_sig, train_tx, rep_tr = load_split("train")
    test_sig, test_tx, rep_te = load_split("test")
    type_stats = fit_type_stats(train_tx)
    if verbose:
        print("temporal check train:", rep_tr)
        print("temporal check test: ", rep_te)
    train = build_features(train_tx, train_sig, type_stats)
    test = build_features(test_tx, test_sig, type_stats)
    train = train.merge(train_sig[[ID, TARGET]], on=ID, how="left", validate="one_to_one")

    assert len(train) == len(train_sig) and train[ID].is_unique
    assert len(test) == len(test_sig) and test[ID].is_unique
    assert list(train.columns.drop(TARGET)) == list(test.columns)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(tr_path)
    test.to_parquet(te_path)
    return train, test, {"train": rep_tr, "test": rep_te}

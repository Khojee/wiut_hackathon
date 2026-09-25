"""Light Optuna search for LightGBM, scored on a fold split (seed 7) different from the
evaluation split (seed 42) so the reported OOF AUC is not tuned on its own folds."""
import json
import warnings

import numpy as np
import optuna

from .data import ROOT, TARGET
from .features import feature_columns
from .model import cv_lgb, make_folds
from .pipeline import build_feature_tables

TUNE_SEED = 7
N_TRIALS = 30
OUT = ROOT / "models" / "lgb_best_params.json"


def objective_factory(X, y, folds):
    def objective(trial):
        params = dict(
            learning_rate=0.01,
            n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
            num_leaves=trial.suggest_int("num_leaves", 4, 31),
            min_child_samples=trial.suggest_int("min_child_samples", 30, 300, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 0.6),
            subsample=trial.suggest_float("subsample", 0.5, 0.95),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 50, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
        )
        res = cv_lgb(X, y, X.iloc[:1], folds, params=params, verbose=False, early_stopping_rounds=None)
        return res.oof_auc
    return objective


def run(n_trials=N_TRIALS):
    warnings.filterwarnings("ignore")
    train, _, _ = build_feature_tables()
    cols = feature_columns(train)
    X, y = train[cols], train[TARGET].to_numpy()
    folds = make_folds(y, seed=TUNE_SEED)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
    study.enqueue_trial(dict(n_estimators=700, num_leaves=7, min_child_samples=100, colsample_bytree=0.3,
                             subsample=0.8, reg_lambda=10, reg_alpha=0.01, min_split_gain=0.0))
    study.optimize(objective_factory(X, y, folds), n_trials=n_trials,
                   callbacks=[lambda s, t: print(f"trial {t.number:2d} AUC={t.value:.4f} best={s.best_value:.4f}")])
    best = {"learning_rate": 0.01, **study.best_params}
    OUT.write_text(json.dumps(best, indent=2))
    print("best params:", best, "tuning-split AUC:", round(study.best_value, 4))
    return best


if __name__ == "__main__":
    run()

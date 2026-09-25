"""Cross-validated model training: logistic baseline, LightGBM, XGBoost, CatBoost."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_FOLDS = 5


@dataclass
class CVResult:
    name: str
    oof: np.ndarray
    test: np.ndarray
    fold_auc: list
    importance: pd.Series = None
    best_iters: list = field(default_factory=list)

    @property
    def oof_auc(self) -> float:
        return None if self._y is None else roc_auc_score(self._y, self.oof)

    _y: np.ndarray = None


def make_folds(y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED):
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _run_cv(name, fit_predict, X, y, X_test, folds, verbose=True) -> CVResult:
    oof = np.zeros(len(y))
    test = np.zeros(len(X_test))
    fold_auc, iters, imps = [], [], []
    for i, (tr, va) in enumerate(folds):
        p_va, p_te, imp, it = fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], X_test)
        oof[va] = p_va
        test += p_te / len(folds)
        fold_auc.append(roc_auc_score(y[va], p_va))
        iters.append(it)
        if imp is not None:
            imps.append(imp)
        if verbose:
            print(f"  [{name}] fold {i}: AUC={fold_auc[-1]:.4f}" + (f" best_iter={it}" if it else ""))
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False) if imps else None
    res = CVResult(name, oof, test, fold_auc, imp, iters)
    res._y = y
    if verbose:
        print(f"  [{name}] OOF AUC={res.oof_auc:.4f}  folds mean={np.mean(fold_auc):.4f} +/- {np.std(fold_auc):.4f}")
    return res


def cv_logreg(X, y, X_test, folds, C=0.1, verbose=True) -> CVResult:
    def fp(Xtr, ytr, Xva, yva, Xte):
        m = make_logreg(C).fit(Xtr, ytr)
        coef = pd.Series(np.abs(m[-1].coef_[0][: Xtr.shape[1]]), index=Xtr.columns)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], coef, None
    return _run_cv("logreg", fp, X, y, X_test, folds, verbose)


LGB_PARAMS = dict(
    objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
    n_estimators=5000, verbose=-1,
)


def cv_lgb(X, y, X_test, folds, params=None, seed=SEED, verbose=True, name="lgb",
           early_stopping_rounds=300) -> CVResult:
    """LightGBM CV. early_stopping_rounds=None trains exactly n_estimators trees per fold,
    which keeps the OOF score honest (the validation fold never picks the iteration)."""
    import lightgbm as lgb
    p = {**LGB_PARAMS, **(params or {}), "random_state": seed}

    def fp(Xtr, ytr, Xva, yva, Xte):
        m = lgb.LGBMClassifier(**p)
        if early_stopping_rounds:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
            it = m.best_iteration_
        else:
            m.fit(Xtr, ytr)
            it = None
        imp = pd.Series(m.booster_.feature_importance("gain"), index=Xtr.columns)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], imp, it
    return _run_cv(name, fp, X, y, X_test, folds, verbose)


def fit_lgb_full(X, y, params, seeds=(SEED,)):
    """Refit LightGBM on all training rows, one model per seed (fixed n_estimators)."""
    import lightgbm as lgb
    return [lgb.LGBMClassifier(**{**LGB_PARAMS, **params, "random_state": s}).fit(X, y) for s in seeds]


def make_logreg(C=0.01):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))


XGB_PARAMS = dict(
    learning_rate=0.02, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
    reg_lambda=5.0, n_estimators=5000, tree_method="hist", eval_metric="auc",
    early_stopping_rounds=300,
)


def cv_xgb(X, y, X_test, folds, params=None, seed=SEED, verbose=True, name="xgb",
           early_stopping=True) -> CVResult:
    import xgboost as xgb
    p = {**XGB_PARAMS, **(params or {}), "random_state": seed}
    if not early_stopping:
        p.pop("early_stopping_rounds", None)

    def fp(Xtr, ytr, Xva, yva, Xte):
        m = xgb.XGBClassifier(**p)
        if early_stopping:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            it = m.best_iteration
        else:
            m.fit(Xtr, ytr, verbose=False)
            it = None
        imp = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(Xtr.columns).fillna(0)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], imp, it
    return _run_cv(name, fp, X, y, X_test, folds, verbose)


CAT_PARAMS = dict(
    learning_rate=0.03, depth=5, l2_leaf_reg=5.0, iterations=5000, eval_metric="AUC",
    od_type="Iter", od_wait=300, verbose=False, thread_count=-1, allow_writing_files=False,
)


def cv_cat(X, y, X_test, folds, params=None, seed=SEED, verbose=True, name="cat",
           early_stopping=True) -> CVResult:
    from catboost import CatBoostClassifier
    p = {**CAT_PARAMS, **(params or {}), "random_seed": seed}
    if not early_stopping:
        p.pop("od_type", None)
        p.pop("od_wait", None)

    def fp(Xtr, ytr, Xva, yva, Xte):
        m = CatBoostClassifier(**p)
        if early_stopping:
            m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
            it = m.get_best_iteration()
        else:
            m.fit(Xtr, ytr)
            it = None
        imp = pd.Series(m.get_feature_importance(), index=Xtr.columns)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], imp, it
    return _run_cv(name, fp, X, y, X_test, folds, verbose)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend_logits(preds, weights) -> np.ndarray:
    """Weighted sum of already-standardised score vectors."""
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    return sum(wi * p for wi, p in zip(w, preds))


def blend_pairs(pairs, weights):
    """Blend (oof, test) prediction pairs in logit space, z-scoring each model with its OOF stats."""
    oofs, tests = [], []
    for oof, test in pairs:
        lo, lt = logit(oof), logit(test)
        mu, sd = lo.mean(), lo.std()
        oofs.append((lo - mu) / sd)
        tests.append((lt - mu) / sd)
    return blend_logits(oofs, weights), blend_logits(tests, weights)


def fit_platt(score_oof, y):
    """Platt scaling fit on out-of-fold scores -> calibrated probabilities (monotone, AUC-preserving)."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(score_oof).reshape(-1, 1), y)
    return lambda s: lr.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


def rank_average(preds, weights=None) -> np.ndarray:
    """Weighted average of per-model rank-normalised predictions (ROC-AUC only depends on ranks)."""
    preds = [pd.Series(p).rank(pct=True).to_numpy() for p in preds]
    w = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=float)
    return np.average(np.vstack(preds), axis=0, weights=w)

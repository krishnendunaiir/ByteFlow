"""Machine learning tools. All preprocessing lives INSIDE sklearn Pipelines, so it is
fitted on training folds only. This is how data leakage is prevented."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from tools.data_inspection import (
    BOOLEAN, CATEGORICAL, DATETIME, IDENTIFIER, NUMERIC, TEXT, get_numeric_columns,
    infer_column_kind, require_columns, safe_float,
)
from utils.security import UserFacingError, sanitize_text_for_llm

MAX_CLASSES = 30
_PRIMARY = {"classification": "balanced_accuracy", "regression": "r2"}
_SCORING = {
    "classification": ["accuracy", "balanced_accuracy", "f1_macro"],
    "regression": ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
}
_METRIC_NAMES = {"neg_root_mean_squared_error": "rmse", "neg_mean_absolute_error": "mae"}


def infer_task(df: pd.DataFrame, target: str) -> str:
    """Decide classification vs regression from the target column."""
    require_columns(df, [target])
    s = df[target].dropna()
    if s.empty:
        raise UserFacingError("The target column has no values.")
    if s.nunique() <= 1:
        raise UserFacingError("The target column has only one distinct value, so there is nothing to predict.")
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        return "classification"
    n_unique = s.nunique()
    integer_valued = bool((s % 1 == 0).all())
    if integer_valued and (n_unique <= 10 or (n_unique <= 20 and n_unique / len(s) < 0.05)):
        return "classification"
    return "regression"


def select_features(df: pd.DataFrame, target: str, task: str) -> dict[str, Any]:
    """Choose usable feature columns and exclude risky ones (identifiers, leaks, junk)."""
    require_columns(df, [target])
    features, excluded, leaks = [], [], []
    y = df[target]
    y_codes = pd.Series(pd.factorize(y.astype(str))[0], index=y.index) if task == "classification" else \
        pd.to_numeric(y, errors="coerce")
    n_classes = y.nunique()
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        kind = infer_column_kind(s)
        if kind in (IDENTIFIER, TEXT, DATETIME):
            excluded.append({"column": str(c), "reason": f"{kind} column (not a usable feature)"})
            continue
        if s.isna().mean() > 0.5:
            excluded.append({"column": str(c), "reason": "more than 50% missing"})
            continue
        if s.nunique(dropna=True) <= 1:
            excluded.append({"column": str(c), "reason": "constant column"})
            continue
        leak = None
        if kind in (NUMERIC, BOOLEAN) and (task == "regression" or n_classes == 2):
            x = pd.to_numeric(s.astype(float) if kind == BOOLEAN else s, errors="coerce")
            pair = pd.DataFrame({"x": x, "y": y_codes}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) > 10 and pair["x"].nunique() > 1 and pair["y"].nunique() > 1:
                r = abs(pair["x"].corr(pair["y"]))
                if r >= 0.98:
                    leak = f"correlation with target is {r:.3f}"
        elif kind == CATEGORICAL and task == "classification":
            levels = s.nunique()
            if levels <= max(2 * n_classes, 10):
                tab = pd.crosstab(s.astype(str), y.astype(str))
                purity = tab.max(axis=1).sum() / tab.to_numpy().sum()
                if purity >= 0.99 and levels <= 2 * n_classes:
                    leak = f"categories map almost perfectly to the target (purity {purity:.3f})"
        if leak:
            excluded.append({"column": str(c), "reason": f"possible target leakage: {leak}"})
            leaks.append(f"'{sanitize_text_for_llm(c, 40)}' excluded: {leak}.")
            continue
        features.append(c)
    return {"features": features, "excluded": excluded, "leakage_warnings": leaks}


def _prepare_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    for c in features:
        kind = infer_column_kind(X[c])
        if kind == BOOLEAN:
            X[c] = X[c].astype(float)
        elif kind == NUMERIC:
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def build_pipeline(X: pd.DataFrame, model: Any) -> Pipeline:
    """Preprocessing + model in ONE pipeline: imputers/scalers/encoders are fitted only on
    the data the pipeline is trained on (a CV training fold), never on validation/test rows."""
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                               ("scale", StandardScaler())]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=15, sparse_output=False)),
        ]), cat))
    return Pipeline([("preprocess", ColumnTransformer(transformers, remainder="drop")), ("model", model)])


def _candidates(task: str, seed: int) -> dict[str, Any]:
    if task == "classification":
        return {
            "baseline_dummy": DummyClassifier(
                strategy="most_frequent"
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=seed,
                class_weight="balanced",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=150,
                random_state=seed,
                n_jobs=1,
                class_weight="balanced",
            ),
        }

    return {
        "baseline_dummy": DummyRegressor(strategy="mean"),
        "ridge_regression": Ridge(
            alpha=1.0,
            random_state=seed,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=150,
            random_state=seed,
            n_jobs=1,
        ),
    }


def _summarise_cv(raw: dict[str, Any], primary: str) -> dict[str, Any]:
    metrics = {}
    for key, vals in raw.items():
        if key.startswith("test_"):
            name, v = key[5:], np.asarray(vals, dtype=float)
            if name.startswith("neg_"):
                v, name = -v, _METRIC_NAMES.get(name, name[4:])
            metrics[name] = {"mean": safe_float(v.mean(), 4), "std": safe_float(v.std(), 4)}
    primary_vals = np.asarray(raw[f"test_{primary}"], dtype=float)
    return {"primary_metric": primary, "primary_cv_mean": safe_float(primary_vals.mean(), 4),
            "primary_cv_std": safe_float(primary_vals.std(), 4),
            "primary_train_mean": safe_float(np.asarray(raw[f"train_{primary}"], dtype=float).mean(), 4),
            "metrics": metrics}


def run_supervised_ml(df: pd.DataFrame, target: str, *, task: str | None = None, seed: int = 42,
                      max_rows: int = 50_000, test_size: float = 0.2, cv_folds: int = 5) -> dict[str, Any]:
    """Train and honestly evaluate baseline + candidate models for classification/regression."""
    require_columns(df, [target])
    notes = ["Preprocessing (imputation, scaling, encoding) is inside the model pipeline and fitted on training folds only.",
             "The data was split into train and test BEFORE any fitting; the test set was used once, for the final model."]
    warns: list[str] = []
    data = df[df[target].notna()].copy()
    if len(data) < 30:
        raise UserFacingError("At least 30 rows with a known target value are needed for machine learning.")
    task = task or infer_task(data, target)
    if task not in _PRIMARY:
        raise UserFacingError("task must be 'classification' or 'regression'.")
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=seed)
        warns.append(f"Dataset was randomly sampled to {max_rows:,} rows for speed.")
    if task == "classification":
        y = data[target].astype(str)
        counts = y.value_counts()
        if len(counts) > MAX_CLASSES:
            raise UserFacingError(f"The target has more than {MAX_CLASSES} classes; classification is not sensible.")
        rare = counts[counts < 2].index
        if len(rare):
            warns.append(f"{len(rare)} class(es) with a single example were removed.")
            data, y = data[~y.isin(rare)], y[~y.isin(rare)]
        if y.nunique() < 2:
            raise UserFacingError("Fewer than two classes remain after removing rare classes.")
        counts = y.value_counts()
        if counts.max() / counts.min() > 3:
            warns.append("Classes are imbalanced: accuracy can be misleading, so balanced accuracy is the main metric.")
    else:
        y = pd.to_numeric(data[target], errors="coerce")
        keep = y.notna() & np.isfinite(y)
        data, y = data[keep], y[keep]
        if len(data) < 30:
            raise UserFacingError("Too few numeric target values for regression.")
    sel = select_features(data, target, task)
    if not sel["features"]:
        raise UserFacingError("No usable feature columns remain after exclusions.")
    X = _prepare_X(data, sel["features"])

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed,
                                                   stratify=y if task == "classification" else None)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        warns.append("Stratified splitting was not possible; a plain random split was used.")
    if len(y_te) < 30:
        warns.append(f"The test set has only {len(y_te)} rows, so test metrics are noisy.")
    min_count = int(y_tr.value_counts().min()) if task == "classification" else 0
    if task == "classification" and min_count >= 2:
        cv = StratifiedKFold(n_splits=min(cv_folds, min_count), shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=min(cv_folds, len(y_tr)), shuffle=True, random_state=seed)

    primary = _PRIMARY[task]
    cv_results: dict[str, Any] = {}
    for name, model in _candidates(task, seed).items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = cross_validate(build_pipeline(X_tr, model), X_tr, y_tr, cv=cv, scoring=_SCORING[task],
                                     return_train_score=True, n_jobs=1, error_score="raise")
            cv_results[name] = _summarise_cv(raw, primary)
        except Exception as exc:  # one failing model must not kill the run
            warns.append(f"Model '{name}' could not be evaluated ({type(exc).__name__}).")
    contenders = {k: v for k, v in cv_results.items() if k != "baseline_dummy"}
    if not contenders:
        raise UserFacingError("None of the candidate models could be trained on this data.")
    best_name = max(contenders, key=lambda k: contenders[k]["primary_cv_mean"])
    best_cv = cv_results[best_name]
    base_cv = cv_results.get("baseline_dummy")
    # must beat the baseline by more than 0.02 AND by more than the fold-to-fold noise
    beats = base_cv is not None and (best_cv["primary_cv_mean"] - base_cv["primary_cv_mean"]
                                     > max(0.02, best_cv["primary_cv_std"] or 0))
    if base_cv is not None and not beats:
        warns.append("The best model does not meaningfully beat the baseline: the features may carry little signal.")
    if best_cv["primary_train_mean"] - best_cv["primary_cv_mean"] > 0.15:
        warns.append("Training score is much higher than cross-validated score: possible overfitting.")

    models = _candidates(task, seed)
    final = build_pipeline(X_tr, models[best_name]).fit(X_tr, y_tr)
    pred = final.predict(X_te)
    if task == "classification":
        test = {"accuracy": safe_float(accuracy_score(y_te, pred), 4),
                "balanced_accuracy": safe_float(balanced_accuracy_score(y_te, pred), 4),
                "f1_macro": safe_float(f1_score(y_te, pred, average="macro", zero_division=0), 4),
                "precision_macro": safe_float(precision_score(y_te, pred, average="macro", zero_division=0), 4),
                "recall_macro": safe_float(recall_score(y_te, pred, average="macro", zero_division=0), 4)}
        labels = [str(c) for c in final.classes_]
        if len(labels) == 2 and y_te.nunique() == 2:
            proba = final.predict_proba(X_te)[:, 1]
            test["roc_auc"] = safe_float(roc_auc_score(y_te == labels[1], proba), 4)
        confusion = {"labels": labels, "matrix": confusion_matrix(y_te, pred, labels=labels).tolist()}
    else:
        test = {"r2": safe_float(r2_score(y_te, pred), 4),
                "rmse": safe_float(np.sqrt(mean_squared_error(y_te, pred)), 4),
                "mae": safe_float(mean_absolute_error(y_te, pred), 4),
                "target_std": safe_float(float(np.std(y_te)), 4)}
        confusion = None
    base_final = build_pipeline(X_tr, _candidates(task, seed)["baseline_dummy"]).fit(X_tr, y_tr)
    bp = base_final.predict(X_te)
    base_test = balanced_accuracy_score(y_te, bp) if task == "classification" else r2_score(y_te, bp)
    if test[primary] is not None and best_cv["primary_cv_mean"] - test[primary] > 0.1:
        warns.append("The test score is noticeably lower than the cross-validated score.")

    sub = np.random.default_rng(seed).choice(len(X_te), size=min(len(X_te), 1000), replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(final, X_te.iloc[sub], y_te.iloc[sub], n_repeats=3, random_state=seed,
                                      scoring=primary, n_jobs=1)
    order = np.argsort(-perm.importances_mean)[:10]
    importances = [{"feature": str(X_te.columns[i]), "importance_mean": safe_float(perm.importances_mean[i], 5),
                    "importance_std": safe_float(perm.importances_std[i], 5)} for i in order]
    return {
        "task": task, "target": str(target), "n_rows_used": int(len(X)), "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)), "cv_folds": int(cv.get_n_splits()), "primary_metric": primary,
        "features_used": [str(c) for c in sel["features"]], "features_excluded": sel["excluded"],
        "leakage_warnings": sel["leakage_warnings"], "cv_results": cv_results, "best_model": best_name,
        "beats_baseline": bool(beats), "test_metrics": test, "baseline_test_primary": safe_float(base_test, 4),
        "confusion": confusion, "permutation_importance": importances,
        "warnings": warns, "notes": notes,
    }


def feature_importance_mutual_info(df: pd.DataFrame, target: str, task: str | None = None, seed: int = 42,
                                   top: int = 15) -> dict[str, Any]:
    """Exploratory ranking of features by mutual information with the target."""
    require_columns(df, [target])
    data = df[df[target].notna()]
    if len(data) < 30:
        raise UserFacingError("At least 30 rows with a known target are needed.")
    task = task or infer_task(data, target)
    if len(data) > 20_000:
        data = data.sample(20_000, random_state=seed)
    sel = select_features(data, target, task)
    if not sel["features"]:
        raise UserFacingError("No usable feature columns for this target.")
    X = _prepare_X(data, sel["features"])
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        parts.append(pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X[num]), columns=num, index=X.index))
    if cat:
        filled = X[cat].fillna("missing").astype(str)
        parts.append(pd.DataFrame(OrdinalEncoder().fit_transform(filled), columns=cat, index=X.index))
    M = pd.concat(parts, axis=1)
    mask = [c in cat for c in M.columns]
    if task == "classification":
        mi = mutual_info_classif(M, pd.factorize(data[target].astype(str))[0], discrete_features=mask, random_state=seed)
    else:
        mi = mutual_info_regression(M, pd.to_numeric(data[target], errors="coerce").fillna(0), discrete_features=mask,
                                    random_state=seed)
    ranking = sorted(zip(M.columns, mi), key=lambda t: -t[1])[:top]
    return {"task": task, "target": str(target), "method": "mutual information",
            "ranking": [{"feature": str(c), "score": safe_float(v, 5)} for c, v in ranking],
            "features_excluded": sel["excluded"], "leakage_warnings": sel["leakage_warnings"],
            "notes": ["Exploratory ranking on all rows; it is NOT a model evaluation. Mutual information "
                      "captures non-linear association but not causation."]}


def run_clustering(df: pd.DataFrame, features: list[str] | None = None, max_k: int = 8, seed: int = 42) -> dict[str, Any]:
    """K-means on scaled numeric columns; k chosen by silhouette score."""
    cols = features if features is not None else get_numeric_columns(df)
    require_columns(df, cols)
    cols = [c for c in cols if pd.to_numeric(df[c], errors="coerce").nunique() > 1][:30]
    if len(cols) < 2 or len(df) < 20:
        raise UserFacingError("Clustering needs at least 2 numeric columns and 20 rows.")
    raw = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan) for c in cols})
    raw = raw.dropna(how="all")
    filled = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(raw), columns=cols, index=raw.index)
    Xs = StandardScaler().fit_transform(filled)
    scores = {}
    for k in range(2, max(3, min(max_k, len(filled) // 10) + 1)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xs)
        if len(set(labels)) > 1:
            scores[k] = float(silhouette_score(Xs, labels, sample_size=min(5000, len(Xs)), random_state=seed))
    if not scores:
        raise UserFacingError("Clustering could not find a valid structure.")
    best_k = max(scores, key=scores.get)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit_predict(Xs)
    prof = filled.assign(cluster=labels).groupby("cluster")
    return {
        "features": [str(c) for c in cols], "best_k": int(best_k),
        "silhouette_by_k": {int(k): safe_float(v, 4) for k, v in scores.items()},
        "best_silhouette": safe_float(scores[best_k], 4),
        "cluster_sizes": {int(k): int(v) for k, v in prof.size().items()},
        "cluster_means": {int(k): {str(c): safe_float(v, 3) for c, v in row.items()} for k, row in prof.mean().iterrows()},
        "notes": ["Silhouette below about 0.25 suggests weak or no real cluster structure."]
                 + (["Cluster structure looks weak."] if scores[best_k] < 0.25 else []),
    }
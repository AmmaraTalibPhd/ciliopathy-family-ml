"""Sensitivity of the results to related disease entries.

Run after code/ciliopathy_family_pipeline.py (it uses the tuned models and the feature matrix in results/).
Analyses, all with the tuned hyperparameters on the training set (312 records):
  * repeated stratified 4-fold cross-validation (10 repeats) - reproduces the main analysis
  * the same without the family-level records
  * the same with single OMIM phenotype entries only
  * grouped folds (StratifiedGroupKFold, 10 repetitions): records sharing an OMIM entry, or a causal
    gene within the same disease family, are always placed in the same fold
  * models retrained without the family-level records and evaluated on the test set
  * HPO frequency filter fitted on the training records only (test set) and re-fitted within each
    training fold (repeated cross-validation), instead of on all records
  * the 13 disease families classified as primary or motile ciliopathies in the CiliaMiner database
    (column "classification"), without the 7 families classified there as secondary diseases: repeated
    cross-validation on their training records, models retrained on their training records and evaluated
    on their test records, and the test accuracy of the main models for each class
Writes results/sensitivity_results.json.

Run from the repository root:  python code/sensitivity_analysis.py
"""
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ciliopathy_family_pipeline as pipe  # noqa: E402  (helpers and settings of the main analysis)

RES = Path(pipe.OUTPUT_DIR)
X = pd.read_csv(RES / "feature_matrix.csv")
y_labels = pd.read_csv(RES / "target_labels.csv")["target"].values

# rebuild the list of the 391 records used for modelling, in the same order as feature_matrix.csv
raw = pd.read_csv(pipe.INPUT_FILE)
raw[pipe.TARGET_COL] = raw[pipe.TARGET_COL].astype(str).str.strip()
counts = raw[pipe.TARGET_COL].value_counts()
raw = raw[raw[pipe.TARGET_COL].map(counts) >= pipe.TARGET_MIN_SAMPLES]
quality = json.loads((RES / "data_quality_report.json").read_text())
removed = {r["disease_id"] for r in quality["removed_no_hpo"] + quality["removed_same_family_duplicates"]}
rec = raw[~raw["disease_id"].isin(removed)].reset_index(drop=True)
assert len(rec) == len(X) and list(rec[pipe.TARGET_COL]) == list(y_labels), "records do not match the feature matrix"


def mim(v):
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "") or "no mim" in s.lower() else s.replace(".0", "")


rec["mim"] = rec["omim_phenotype_number"].apply(mim)
rec["genes"] = rec["associated_genes"].fillna("").astype(str).apply(
    lambda s: [g.strip() for g in re.split(r"[;,]", s) if g.strip()])
family_level = rec["match_strategy"].eq("family_aggregate_from_exact_omim").values
single_omim = rec["match_strategy"].eq("exact_omim").values

le = LabelEncoder()
y = le.fit_transform(y_labels)
n_classes = len(le.classes_)
idx = np.arange(len(X))
train_idx, test_idx = train_test_split(idx, test_size=pipe.TEST_SIZE, random_state=pipe.RANDOM_STATE, stratify=y)

models = [("Random Forest", joblib.load(RES / "random_forest_model.joblib"), False),
          ("XGBoost", joblib.load(RES / "xgboost_model.joblib"), True)]


def cv_accuracy(rows, splits):
    Xs, ys = X.iloc[rows].reset_index(drop=True), y[rows]
    jobs = [(name, est, w, a, b) for a, b in splits for name, est, w in models]
    out = Parallel(n_jobs=-1)(delayed(pipe._fit_score_fold)(est, Xs.iloc[a], ys[a], Xs.iloc[b], ys[b], n_classes, w)
                              for name, est, w, a, b in jobs)
    acc = {}
    for (name, *_), o in zip(jobs, out):
        acc.setdefault(name, []).append(o["accuracy"])
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n_folds": len(v)} for k, v in acc.items()}


def repeated(rows):
    rcv = RepeatedStratifiedKFold(n_splits=4, n_repeats=pipe.CV_REPEATS, random_state=pipe.RANDOM_STATE)
    return cv_accuracy(rows, list(rcv.split(X.iloc[rows], y[rows])))


results = {
    "main_analysis_repeated_cv": repeated(train_idx),
    "without_family_level_records": repeated(train_idx[~family_level[train_idx]]),
    "single_omim_entries_only": repeated(train_idx[single_omim[train_idx]]),
}

# grouped folds: union of records sharing an OMIM entry, or a gene within the same family
parent = list(range(len(rec)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


first = {}
for i, r in rec.iterrows():
    keys = ([("mim", r["mim"])] if r["mim"] else []) + [("gene_family", g, r[pipe.TARGET_COL]) for g in r["genes"]]
    for k in keys:
        if k in first:
            parent[find(i)] = find(first[k])
        else:
            first[k] = i
groups = np.array([find(i) for i in range(len(rec))])[train_idx]
splits = []
for rep in range(pipe.CV_REPEATS):
    sgk = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=pipe.RANDOM_STATE + rep)
    splits += list(sgk.split(X.iloc[train_idx], y[train_idx], groups=groups))
results["grouped_folds"] = cv_accuracy(train_idx, splits)

# test set: models retrained without family-level records
test_c = test_idx[~family_level[test_idx]]
train_c = train_idx[~family_level[train_idx]]
results["test_set_without_family_level_records"] = {}
for name, est, weighted in models:
    m = clone(est)
    fit_params = {"classifier__sample_weight": compute_sample_weight("balanced", y[train_c])} if weighted else {}
    m.fit(X.iloc[train_c], y[train_c], **fit_params)
    results["test_set_without_family_level_records"][name] = {
        "n_test": int(len(test_c)),
        "correct_full_model": int((est.predict(X.iloc[test_c]) == y[test_c]).sum()),
        "correct_retrained_without_family_level": int((m.predict(X.iloc[test_c]) == y[test_c]).sum()),
    }

# HPO frequency filter fitted on training data only (test set) and inside every training fold (CV)
class MinCountSelector(BaseEstimator, TransformerMixin):
    """Keep HPO columns present in at least `min_count` rows of the fitting data; keep all other columns."""

    def __init__(self, min_count=pipe.MIN_HPO_FREQUENCY):
        self.min_count = min_count

    def fit(self, X_, y_=None):
        X_ = pd.DataFrame(X_)
        hpo = [c for c in X_.columns if str(c).startswith("HPO__")]
        self.columns_ = [c for c in hpo if X_[c].sum() >= self.min_count] + \
                        [c for c in X_.columns if not str(c).startswith("HPO__")]
        return self

    def transform(self, X_):
        return pd.DataFrame(X_)[self.columns_]


hpo_lists = rec[pipe.HPO_COL].apply(pipe.parse_hpo_list)
mlb = MultiLabelBinarizer()
H = pd.DataFrame(mlb.fit_transform(hpo_lists), columns=[f"HPO__{t}" for t in mlb.classes_])
XU = pd.concat([H, X[[c for c in X.columns if not c.startswith("HPO__")]].reset_index(drop=True)], axis=1).astype(float)
filtered_models = [(name, Pipeline([("select", MinCountSelector())] + list(clone(est).steps)), w) for name, est, w in models]

test_res = {}
for name, m, weighted in filtered_models:
    fit_params = {"classifier__sample_weight": compute_sample_weight("balanced", y[train_idx])} if weighted else {}
    m.fit(XU.iloc[train_idx], y[train_idx], **fit_params)
    pred = m.predict(XU.iloc[test_idx])
    prob = m.predict_proba(XU.iloc[test_idx])
    test_res[name] = {"n_hpo_features": int(sum(c.startswith("HPO__") for c in m.named_steps["select"].columns_)),
                      "n_correct": int((pred == y[test_idx]).sum()), "n_test": int(len(test_idx)),
                      "accuracy": float(accuracy_score(y[test_idx], pred)),
                      "f1_weighted": float(f1_score(y[test_idx], pred, average="weighted", zero_division=0)),
                      "top_3_accuracy": float(top_k_accuracy_score(y[test_idx], prob, k=3, labels=np.arange(n_classes)))}
rcv = RepeatedStratifiedKFold(n_splits=4, n_repeats=pipe.CV_REPEATS, random_state=pipe.RANDOM_STATE)
fold_splits = list(rcv.split(XU.iloc[train_idx], y[train_idx]))
Xt, yt = XU.iloc[train_idx].reset_index(drop=True), y[train_idx]
jobs = [(name, m, w, a, b) for a, b in fold_splits for name, m, w in filtered_models]
outs = Parallel(n_jobs=-1)(delayed(pipe._fit_score_fold)(m, Xt.iloc[a], yt[a], Xt.iloc[b], yt[b], n_classes, w)
                           for name, m, w, a, b in jobs)
cv_res = {}
for (name, *_), o in zip(jobs, outs):
    cv_res.setdefault(name, []).append(o["accuracy"])
results["hpo_filter_fitted_on_training_data"] = {
    "test_set": test_res,
    "repeated_cv_accuracy": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n_folds": len(v)}
                             for k, v in cv_res.items()},
}

# primary and motile ciliopathy families only (CiliaMiner classes); the other families are secondary diseases
CORE_CLASSES = ("Primary", "Motile")
ciliopathy_class = rec["classification"].astype(str).str.strip().values
core = np.isin(ciliopathy_class, CORE_CLASSES)


def family_subset(mask):
    """Repeated CV on the training records, and retrained models on the test records, of the selected families."""
    tr, te = train_idx[mask[train_idx]], test_idx[mask[test_idx]]
    le_s = LabelEncoder().fit(y_labels[tr])
    k = len(le_s.classes_)
    ys = np.full(len(y_labels), -1)
    ys[mask] = le_s.transform(y_labels[mask])
    sub_models = [(name, clone(est).set_params(classifier__num_class=k) if w else clone(est), w) for name, est, w in models]
    rcv_s = RepeatedStratifiedKFold(n_splits=4, n_repeats=pipe.CV_REPEATS, random_state=pipe.RANDOM_STATE)
    Xs, yt = X.iloc[tr].reset_index(drop=True), ys[tr]
    jobs = [(name, m, w, a, b) for a, b in rcv_s.split(Xs, yt) for name, m, w in sub_models]
    outs = Parallel(n_jobs=-1)(delayed(pipe._fit_score_fold)(m, Xs.iloc[a], yt[a], Xs.iloc[b], yt[b], k, w)
                               for name, m, w, a, b in jobs)
    cv = {}
    for (name, *_), o in zip(jobs, outs):
        for metric, v in o.items():
            cv.setdefault(name, {}).setdefault(metric, []).append(v)
    res = {"families": list(le_s.classes_), "n_records": int(mask.sum()), "n_train": int(len(tr)), "n_test": int(len(te)),
           "repeated_cv": {name: {m_: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n_folds": len(v)}
                                  for m_, v in d.items()} for name, d in cv.items()},
           "test_set_retrained": {}}
    for name, m, weighted in sub_models:
        fit_params = {"classifier__sample_weight": compute_sample_weight("balanced", ys[tr])} if weighted else {}
        m.fit(X.iloc[tr], ys[tr], **fit_params)
        pred, prob = m.predict(X.iloc[te]), m.predict_proba(X.iloc[te])
        res["test_set_retrained"][name] = {
            "n_correct": int((pred == ys[te]).sum()), "n_test": int(len(te)),
            "accuracy": float(accuracy_score(ys[te], pred)),
            "f1_weighted": float(f1_score(ys[te], pred, average="weighted", zero_division=0)),
            "top_3_accuracy": float(top_k_accuracy_score(ys[te], prob, k=3, labels=np.arange(k)))}
    return res


results["primary_and_motile_ciliopathy_families"] = family_subset(core)
# test records of each CiliaMiner class, classified by the main models (all 20 families)
by_class = {}
for name, est, _ in models:
    correct = est.predict(X.iloc[test_idx]) == y[test_idx]
    by_class[name] = {c: {"n_correct": int(correct[ciliopathy_class[test_idx] == c].sum()),
                          "n_test": int((ciliopathy_class[test_idx] == c).sum())}
                      for c in ("Primary", "Motile", "Secondary")}
results["main_models_test_set_by_ciliopathy_class"] = by_class

(RES / "sensitivity_results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

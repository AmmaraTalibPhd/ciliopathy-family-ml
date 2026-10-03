"""Permutation importance on the held-out test set (feature-level and grouped by organ system).

Run after code/ciliopathy_family_pipeline.py (uses the tuned models and feature matrix in results/).
  * feature level: mean decrease in accuracy over 20 random permutations of each feature that varies in the test set
  * grouped: 100 permutations per group; each organ-system indicator is permuted together with the HPO terms of the
    corresponding HPO category (data/hpo_feature_categories.csv, derived from the HPO release of 2026-09-01), and the
    11 inheritance indicators are permuted together
Writes results/permutation_importance_grouped.csv and results/permutation_importance_features.csv
(the feature-level part takes about 15-20 minutes on 2 CPU cores).

Run from the repository root:  python code/permutation_importance.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ciliopathy_family_pipeline as pipe  # noqa: E402

RES = Path(pipe.OUTPUT_DIR)
X = pd.read_csv(RES / "feature_matrix.csv")
y = LabelEncoder().fit_transform(pd.read_csv(RES / "target_labels.csv")["target"].values)
_, test_idx = train_test_split(np.arange(len(y)), test_size=pipe.TEST_SIZE, random_state=pipe.RANDOM_STATE, stratify=y)
X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y[test_idx]

models = {"Random Forest": joblib.load(RES / "random_forest_model.joblib"),
          "XGBoost": joblib.load(RES / "xgboost_model.joblib")}
for m in models.values():
    m.named_steps["classifier"].set_params(n_jobs=1)  # parallelism over permutations instead

cats = pd.read_csv("data/hpo_feature_categories.csv").fillna("")
SYSTEMS = ["abdomen", "eye", "ear", "cardiovascular", "endocrine", "growth", "genitourinary", "immune_system",
           "head_neck", "limbs", "nervous_system", "metabolism", "respiratory", "skeletal", "skin"]
groups = {s: [f"SYS__{s}"] + [f"HPO__{h}" for h, g in zip(cats["hpo_id"], cats["organ_system_groups"])
                               if s in g.split(";")] for s in SYSTEMS}
groups["inheritance"] = [c for c in X.columns if c.startswith("INH_")]
grouped_hpo = {c for s in SYSTEMS for c in groups[s] if c.startswith("HPO__")}
groups["other HPO terms"] = [c for c in X.columns if c.startswith("HPO__") and c not in grouped_hpo]


def accuracy_drop(model, cols, rep, base):
    rng = np.random.RandomState(1000 + rep)
    Xp = X_test.copy()
    Xp[cols] = X_test[cols].values[rng.permutation(len(Xp))]
    return base - (model.predict(Xp) == y_test).mean()


names = dict(pd.read_csv(pipe.HPO_NAMES_FILE).values) if Path(pipe.HPO_NAMES_FILE).exists() else {}
grouped_rows, feature_rows = [], []
for name, model in models.items():
    base = (model.predict(X_test) == y_test).mean()
    for gname, cols in groups.items():
        drops = Parallel(n_jobs=-1)(delayed(accuracy_drop)(model, cols, r, base) for r in range(100))
        grouped_rows.append({"model": name, "group": gname, "n_features": len(cols),
                             "mean_accuracy_decrease": float(np.mean(drops)), "sd": float(np.std(drops, ddof=1))})
    varying = [c for c in X.columns if X_test[c].nunique() > 1]
    out = Parallel(n_jobs=-1)(delayed(lambda c: [accuracy_drop(model, [c], r, base) for r in range(20)])(c)
                              for c in varying)
    for c, drops in zip(varying, out):
        feature_rows.append({"model": name, "feature": c, "label": pipe.pretty_feature(c, names),
                             "mean_accuracy_decrease": float(np.mean(drops)), "sd": float(np.std(drops, ddof=1))})

order = dict(ascending=[True, False])
pd.DataFrame(grouped_rows).sort_values(["model", "mean_accuracy_decrease"], **order).to_csv(
    RES / "permutation_importance_grouped.csv", index=False)
pd.DataFrame(feature_rows).sort_values(["model", "mean_accuracy_decrease"], **order).to_csv(
    RES / "permutation_importance_features.csv", index=False)
print(pd.DataFrame(grouped_rows).sort_values(["model", "mean_accuracy_decrease"], **order).to_string(index=False))

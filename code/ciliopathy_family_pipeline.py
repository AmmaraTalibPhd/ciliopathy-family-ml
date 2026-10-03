"""
Phenotype-driven classification of ciliopathy disease families
==============================================================
Analysis code for "Phenotype-driven machine learning classification of ciliopathy disease
families using curated clinical and Human Phenotype Ontology features".

Input : data/ciliopathy_curated_entries.csv (451 curated entries; one row per disease entry)
Output: results/ (metrics, statistics, cross-validation scores, figures)

Steps
-----
1. Features: HPO term IDs (terms present in at least two entries), 11 binary inheritance
   indicators and 15 binary organ-system indicators; target = disease family.
2. Families with fewer than five entries are removed; entries without any retained HPO
   term and exact duplicates within the same family are then removed
   (details in results/data_quality_report.json).
3. Stratified 80:20 train/test split (seed 42). Random Forest and XGBoost are tuned by grid
   search with stratified 4-fold cross-validation on the training set (weighted F1).
4. Evaluation: held-out test set (accuracy and Top-3 accuracy with 95% Wilson CIs, weighted
   precision/recall/F1, confusion matrices, exact McNemar test); repeated stratified 4-fold
   cross-validation (10 repeats, 40 folds; mean +/- SD, Nadeau-Bengio corrected 95% CIs,
   corrected repeated k-fold t-test); nested cross-validation.
5. Interpretation: impurity-based (Random Forest) and gain-based (XGBoost) feature importance,
   SHAP (mean |SHAP value|), PCA, t-SNE and an HPO-by-family heatmap.

Run from the repository root:  python code/ciliopathy_family_pipeline.py
Set the environment variable CILIO_QUICK_TEST=1 for a fast check with small grids.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # file-only backend: avoids Tk "main thread is not in main loop" crash on Windows/PyCharm
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest, t as student_t  # extra tests
from joblib import Parallel, delayed  # parallel fold loop

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RepeatedStratifiedKFold,  # repeated CV for fold-level variability
)
from sklearn.base import clone  # never refit the final models inside CV loops
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    top_k_accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

# =========================================================
# USER SETTINGS
# =========================================================
INPUT_FILE = "data/ciliopathy_curated_entries.csv"
OUTPUT_DIR = "results"

TARGET_MIN_SAMPLES = 5
MIN_HPO_FREQUENCY = 2
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5          # NOTE: automatically reduced to the size of the smallest training class (4 here)
TOP_K = 3
SAVE_FEATURE_TABLES = True

# new settings
CV_REPEATS = 10        # repeated stratified k-fold -> 4 folds x 10 repeats = 40 paired scores
RUN_NESTED_CV = True   # unbiased estimate: hyperparameters re-tuned inside every outer fold
FIG_DPI = 600          # publication-quality figures
CONFUSION_TOP_N = 20   # show all 20 families in the confusion matrices
HPO_NAMES_FILE = "data/hpo_term_names.csv"   # optional (columns: hpo_id,name) -> readable plot labels
QUICK_TEST = os.environ.get("CILIO_QUICK_TEST") == "1"  # small grids, only for testing the code
DROP_RECORDS_WITHOUT_HPO = True   # records with no phenotype data cannot be classified from phenotype
DEDUPLICATE_SAME_FAMILY = True    # keep one copy of identical records from the same family
XGB_BALANCED_WEIGHTS = True       # balanced sample weights for XGBoost (as class_weight="balanced" in RF)

# =========================================================
# MANUAL COLUMN MAPPING FOR YOUR FILE
# =========================================================
TARGET_COL = "disease_family"
INHERITANCE_COL = "inheritance_model"
HPO_COL = "hpo_ids_all"

SYSTEM_COLS = [
    "abdomen",
    "eye",
    "ear",
    "cardiovascular",
    "endocrine",
    "growth",
    "genitourinary",
    "immune_system",
    "head_neck",
    "limbs",
    "nervous_system",
    "metabolism",
    "respiratory",
    "skeletal",
    "skin",
]

# =========================================================
# OPTIONAL IMPORTS
# =========================================================
HAS_XGBOOST = False
HAS_SHAP = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# =========================================================
# HELPERS
# =========================================================
def make_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def safe_read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError("Unsupported format. Use .xlsx, .xls, .csv, or .tsv")


def normalize_col_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[_\-]+", " ", col)
    col = re.sub(r"\s+", " ", col)
    return col


def split_terms(text: str) -> List[str]:
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []

    text = text.replace("|", ";")
    text = text.replace(",", ";")
    text = re.sub(r"\s*;\s*", ";", text)

    items = [x.strip() for x in text.split(";") if x.strip()]
    return items


def parse_hpo_list(value) -> List[str]:
    if pd.isna(value):
        return []
    items = split_terms(value)
    parsed = []
    for item in items:
        item = item.strip()
        if item:
            parsed.append(item)
    return parsed


def system_value_to_binary(value) -> int:
    """numeric-aware: 0 / 0.0 / False -> 0 ; any positive number or True -> 1.
    (Before, '0.0' and 'False' were treated as present, making every organ system = 1.)"""
    if pd.isna(value):
        return 0
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(float(value) > 0)
    s = str(value).strip().lower()
    try:
        return int(float(s) > 0)
    except ValueError:
        pass
    if s in {"", "0", "no", "none", "nan", "absent", "negative", "not reported", "false", "f", "n"}:
        return 0
    return 1


# canonical inheritance tokens
_INH_SYNONYMS = {
    "AUTOSOMAL RECESSIVE": "AR", "AUTOSOMAL DOMINANT": "AD",
    "X LINKED RECESSIVE": "XLR", "X LINKED DOMINANT": "XLD", "X LINKED": "XL",
    "ANTICIPATION PATERNAL BIAS": "ANTICIPATION",
    "INCOMPLETE PENETRANCE": "INCOMPLETE_PENETRANCE",
    "AGE RELATED ONSET": "AGE_RELATED_ONSET",
}
_INH_UNKNOWN = {"NEEDS CURATION", "GENETICALLY HETEROGENEOUS", "UNKNOWN", "NAN", "NONE", ""}


def parse_inheritance(value) -> List[str]:
    """'AD; INCOMPLETE_PENETRANCE' -> ['AD', 'INCOMPLETE_PENETRANCE']; 'Autosomal recessive' -> ['AR'].
    Order and spelling no longer create separate categories. Records without a known
    mode get the flag 'UNKNOWN'."""
    if pd.isna(value):
        return ["UNKNOWN"]
    tokens = []
    for t in split_terms(value):
        key = re.sub(r"[_\-]+", " ", t.strip().upper())
        key = re.sub(r"\s+", " ", key)
        if key in _INH_UNKNOWN:
            continue
        tokens.append(_INH_SYNONYMS.get(key, key.replace(" ", "_")))
    modes = {"AR", "AD", "XLR", "XLD", "XL", "DIGENIC", "MITOCHONDRIAL", "YL"}
    if not any(t in modes for t in tokens):
        tokens.append("UNKNOWN")
    return sorted(set(tokens))


def reduce_rare_classes(df: pd.DataFrame, target_col: str, min_samples: int) -> pd.DataFrame:
    counts = df[target_col].value_counts()
    keep = counts[counts >= min_samples].index
    return df[df[target_col].isin(keep)].copy()


def save_json(data: dict, filepath: Path) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_feature_importance(model: Pipeline, feature_names: List[str], top_n: int = 30) -> pd.DataFrame:
    clf = model.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        return imp_df.head(top_n)
    return pd.DataFrame(columns=["feature", "importance"])


def plot_pca(
    X: np.ndarray, y: np.ndarray, filepath: Path,
    label_encoder: LabelEncoder,
) -> None:
    if X.shape[0] < 3:
        return
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 8))
    class_ids = np.unique(y)
    # A categorical palette avoids implying a numeric disease-family scale.
    if len(class_ids) <= 20:
        colours = plt.get_cmap("tab20")(np.arange(len(class_ids)))
    else:
        colours = plt.get_cmap("hsv")(
            np.linspace(0, 1, len(class_ids), endpoint=False)
        )
    for class_id, colour in zip(class_ids, colours):
        mask = y == class_id
        family = label_encoder.inverse_transform([class_id])[0]
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            color=colour, alpha=0.8, s=35,
            label=f"{family} (n={int(mask.sum())})",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA of Ciliopathy Features")
    ax.legend(
        title="Disease family", bbox_to_anchor=(1.02, 1),
        loc="upper left", fontsize=8, title_fontsize=10,
        frameon=True, ncol=max(1, (len(class_ids) + 24) // 25),
    )
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(
    X: np.ndarray, y: np.ndarray, filepath: Path,
    label_encoder: LabelEncoder,
) -> None:
    if X.shape[0] < 5:
        return
    perplexity = min(30, max(2, X.shape[0] // 3))
    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca"
    )
    X_tsne = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 8))
    class_ids = np.unique(y)
    # A categorical palette avoids implying a numeric disease-family scale.
    if len(class_ids) <= 20:
        colours = plt.get_cmap("tab20")(np.arange(len(class_ids)))
    else:
        colours = plt.get_cmap("hsv")(
            np.linspace(0, 1, len(class_ids), endpoint=False)
        )
    for class_id, colour in zip(class_ids, colours):
        mask = y == class_id
        family = label_encoder.inverse_transform([class_id])[0]
        ax.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            color=colour, alpha=0.8, s=35,
            label=f"{family} (n={int(mask.sum())})",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of Ciliopathy Features")
    ax.legend(
        title="Disease family", bbox_to_anchor=(1.02, 1),
        loc="upper left", fontsize=8, title_fontsize=10,
        frameon=True, ncol=max(1, (len(class_ids) + 24) // 25),
    )
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Readable feature labels, statistics helpers, fixed plots
# ----------------------------------------------------------------------
def load_hpo_names(path: str) -> Dict[str, str]:
    """Optional HPO id -> term name lookup (CSV with columns hpo_id,name)."""
    p = Path(path)
    if not p.exists():
        return {}
    tab = pd.read_csv(p)
    return dict(zip(tab["hpo_id"].astype(str), tab["name"].astype(str)))


def pretty_feature(name: str, hpo_names: Dict[str, str], max_len: int = 45) -> str:
    """HPO__HP:0000510 -> 'Rod-cone dystrophy (HP:0000510)'; INH_AR -> 'Inheritance: AR'."""
    if name.startswith("HPO__"):
        hid = name[len("HPO__"):]
        label = hpo_names.get(hid)
        if label:
            if len(label) > max_len:
                label = label[: max_len - 1] + "…"
            return f"{label} ({hid})"
        return hid
    if name.startswith("INH_"):
        return "Inheritance: " + name[len("INH_"):]
    if name.startswith("SYS__"):
        return "Organ system: " + name[len("SYS__"):].replace("_", " ")
    return name


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def corrected_resampled_ci(scores: np.ndarray, n_train: int, n_test: int, alpha: float = 0.05) -> tuple:
    """Nadeau & Bengio (2003) corrected CI for the mean of repeated-CV scores.
    Plain SD/sqrt(J) is too narrow because repeated-CV folds overlap."""
    J = len(scores)
    var = np.var(scores, ddof=1)
    se = np.sqrt((1.0 / J + n_test / n_train) * var)
    tcrit = student_t.ppf(1 - alpha / 2, df=J - 1)
    m = float(np.mean(scores))
    return (m - tcrit * se, m + tcrit * se)


def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> tuple:
    """Corrected repeated k-fold t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004)."""
    J = len(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0:
        return (np.inf if np.mean(diffs) != 0 else 0.0, 0.0 if np.mean(diffs) != 0 else 1.0)
    tstat = np.mean(diffs) / np.sqrt((1.0 / J + n_test / n_train) * var)
    p = 2 * student_t.sf(abs(tstat), df=J - 1)
    return (float(tstat), float(p))


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on the same test cases (model A vs model B)."""
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    n01 = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    if n01 + n10 == 0:
        p = 1.0
    else:
        p = float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
    return {"a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "p_value": p}


def _fit_score_fold(est, X_tr, y_tr, X_te, y_te, n_classes: int, balanced_weights: bool = False) -> dict:
    """Fit a fresh clone on one fold and return its scores (used by the CV loops)."""
    m = clone(est)
    try:
        m.set_params(classifier__n_jobs=1)   # parallelise over folds instead (deterministic)
    except ValueError:
        pass
    if balanced_weights:
        m.fit(X_tr, y_tr, classifier__sample_weight=compute_sample_weight("balanced", y_tr))
    else:
        m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    out = {
        "accuracy": accuracy_score(y_te, pred),
        "f1_weighted": precision_recall_fscore_support(y_te, pred, average="weighted", zero_division=0)[2],
    }
    if hasattr(m.named_steps["classifier"], "predict_proba"):
        prob = m.predict_proba(X_te)
        out["top3_accuracy"] = top_k_accuracy_score(y_te, prob, k=min(TOP_K, n_classes), labels=np.arange(n_classes))
    return out


def plot_feature_importance(df_imp: pd.DataFrame, title: str, filepath: Path,
                            hpo_names: Optional[Dict[str, str]] = None,
                            xlabel: str = "Mean decrease in impurity") -> None:
    if df_imp.empty:
        return
    df_plot = df_imp.sort_values("importance", ascending=True).copy()
    df_plot["label"] = [pretty_feature(f, hpo_names or {}) for f in df_plot["feature"]]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(df_plot["label"], df_plot["importance"], color="#2b6cb0")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Feature", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_for_top_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    filepath: Path,
    top_n_classes: int = 20,
    title: str = "Confusion matrix (test set)",
) -> None:
    true_labels = label_encoder.inverse_transform(y_true)
    counts = pd.Series(true_labels).value_counts().head(top_n_classes)
    selected_classes = counts.index.tolist()
    selected_ids = label_encoder.transform(selected_classes)

    mask = np.isin(y_true, selected_ids)
    if mask.sum() == 0:
        return

    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=selected_ids)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=selected_classes)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=True, cmap="Blues", values_format="d")
    for txt in disp.text_.ravel():
        txt.set_fontsize(11)
    ax.set_xlabel("Predicted family", fontsize=13)
    ax.set_ylabel("True family", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(f"{title} – {len(selected_classes)} families", fontsize=14)
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def create_hpo_target_heatmap(
    df_binary: pd.DataFrame,
    target_col: str,
    feature_prefix: str,
    filepath: Path,
    top_hpo_n: int = 20,
    top_target_n: int = 20,
    hpo_names: Optional[Dict[str, str]] = None,
) -> None:
    hpo_cols = [c for c in df_binary.columns if c.startswith(feature_prefix)]
    if not hpo_cols:
        return

    top_targets = df_binary[target_col].value_counts().head(top_target_n).index.tolist()
    sub = df_binary[df_binary[target_col].isin(top_targets)].copy()

    hpo_freq = sub[hpo_cols].sum().sort_values(ascending=False).head(top_hpo_n)
    top_hpo_cols = hpo_freq.index.tolist()
    if not top_hpo_cols:
        return

    # value = proportion of the records of each family that are annotated with the term (0-1, no further normalization);
    # the terms are the top_hpo_n terms most frequent across all records
    heat = sub.groupby(target_col)[top_hpo_cols].mean().T          # rows: HPO terms (by overall frequency), columns: families
    labels = [pretty_feature(c, hpo_names or {}, max_len=200) for c in heat.index]
    heat.set_axis(labels, axis=0).to_csv(str(filepath).replace(".png", "_values.csv"), float_format="%.4f")

    # drawn at about the printed size (one page width), so that the labels stay readable
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=90, fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(heat.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Disease family", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Proportion of the family's records annotated with the term", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("HPO term frequency by disease family (20 most frequent HPO terms)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(filepath, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def run_shap_analysis(
    model: Pipeline,
    X_train_df: pd.DataFrame,
    filepath_prefix: Path,
    label_encoder: LabelEncoder,
    hpo_names: Optional[Dict[str, str]] = None,
    max_samples: int = 200,
    top_n: int = 20,
    model_name: str = "",
) -> None:
    """Multiclass-safe SHAP.
    Recent SHAP versions return an array of shape (samples, features, classes) for
    multiclass models; passing it straight to summary_plot makes SHAP treat it as
    *interaction* values (this produced the old 'SHAP interaction value' figures).
    Here the class dimension is handled explicitly."""
    if not HAS_SHAP:
        print("SHAP not installed; skipping SHAP analysis.")
        return
    try:
        clf = model.named_steps["classifier"]
        X_small = X_train_df.copy()
        if len(X_small) > max_samples:
            X_small = X_small.sample(max_samples, random_state=RANDOM_STATE)
        X_in = pd.DataFrame(model[:-1].transform(X_small), columns=X_small.columns, index=X_small.index)

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_in)
        if isinstance(sv, list):                      # older SHAP: list of (samples, features)
            sv = np.stack(sv, axis=-1)
        sv = np.asarray(sv)
        if sv.ndim == 2:
            sv = sv[:, :, None]
        # sv: samples x features x classes
        labels = [pretty_feature(c, hpo_names or {}) for c in X_in.columns]

        mean_abs = np.abs(sv).mean(axis=(0, 2))
        order = np.argsort(mean_abs)[::-1]
        pd.DataFrame({"feature": X_in.columns[order], "label": np.array(labels)[order],
                      "mean_abs_shap": mean_abs[order]}).to_csv(
            str(filepath_prefix) + "_mean_abs_shap.csv", index=False)
        # family-specific importance: mean |SHAP| over records, one column per disease family
        by_family = pd.DataFrame(np.abs(sv).mean(axis=0), index=X_in.columns,
                                 columns=list(label_encoder.classes_)[: sv.shape[2]])
        by_family.insert(0, "mean_over_families", mean_abs)
        by_family.sort_values("mean_over_families", ascending=False).to_csv(
            str(filepath_prefix) + "_mean_abs_by_family.csv")

        # Figure A: global importance = mean |SHAP| over samples and classes
        top = order[:top_n][::-1]
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.barh(np.array(labels)[top], mean_abs[top], color="#2b6cb0")
        ax.set_xlabel("Mean |SHAP value| (averaged over samples and disease families)", fontsize=12)
        ax.set_title(f"{model_name + ' – ' if model_name else ''}SHAP feature importance (top {top_n} features)", fontsize=14)
        ax.tick_params(axis="both", labelsize=10)
        fig.tight_layout()
        fig.savefig(str(filepath_prefix) + "_summary.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)

        # Figure B: same features, contribution split by disease family (stacked bars)
        class_names = list(label_encoder.classes_)[: sv.shape[2]]
        plt.figure()
        shap.summary_plot([sv[:, :, k] for k in range(sv.shape[2])], X_in,
                          feature_names=labels, class_names=class_names,
                          plot_type="bar", max_display=top_n, show=False)
        fig = plt.gcf()
        fig.set_size_inches(14, 10)
        ax = fig.axes[0]
        leg = ax.get_legend()
        if leg is not None:  # move the 20-family legend outside the bars
            handles, labs = ax.get_legend_handles_labels()
            leg.remove()
            ax.legend(handles, labs, title="Disease family", bbox_to_anchor=(1.02, 1),
                      loc="upper left", fontsize=9, title_fontsize=10, frameon=False)
        ax.set_xlabel("Mean |SHAP value| (stacked by disease family)", fontsize=12)
        fig.tight_layout()
        fig.savefig(str(filepath_prefix) + "_by_family.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        print(f"SHAP failed: {e}")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    out_dir = make_output_dir(OUTPUT_DIR)

    df = safe_read_table(INPUT_FILE)
    if df.empty:
        raise ValueError("Input file is empty.")

    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nALL COLUMNS IN FILE:")
    for c in df.columns:
        print(repr(c))

    # Validate required columns
    required = [TARGET_COL, INHERITANCE_COL, HPO_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    system_cols_found = [c for c in SYSTEM_COLS if c in df.columns]
    hpo_names = load_hpo_names(HPO_NAMES_FILE)  # readable plot labels (optional)
    print(f"HPO term names loaded: {len(hpo_names)}" if hpo_names else "HPO names file not found; plots will show HPO IDs.")

    print("\nUsing manual columns:")
    print(f"  Target column       : {TARGET_COL}")
    print(f"  Inheritance column  : {INHERITANCE_COL}")
    print(f"  HPO column          : {HPO_COL}")
    print(f"  System columns      : {system_cols_found}")

    # Clean target
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] != "")].copy()

    # Parse HPO
    df["_hpo_list"] = df[HPO_COL].apply(parse_hpo_list)

    # Encode systems
    for col in system_cols_found:
        safe_name = normalize_col_name(col).replace(" ", "_")
        df[f"SYS__{safe_name}"] = df[col].apply(system_value_to_binary)

    # Remove rare target classes
    before_rows = len(df)
    df = reduce_rare_classes(df, TARGET_COL, min_samples=TARGET_MIN_SAMPLES)
    after_rows = len(df)

    print(f"\nRows before rare-class filtering: {before_rows}")
    print(f"Rows after rare-class filtering : {after_rows}")

    if len(df) < 20:
        raise ValueError("Too few rows remain after filtering. Reduce TARGET_MIN_SAMPLES or add more data.")

    # HPO matrix
    mlb = MultiLabelBinarizer()
    hpo_matrix = mlb.fit_transform(df["_hpo_list"])
    hpo_features = [f"HPO__{term}" for term in mlb.classes_]
    df_hpo = pd.DataFrame(hpo_matrix, columns=hpo_features, index=df.index)

    if not df_hpo.empty:
        hpo_freq = df_hpo.sum(axis=0)
        keep_hpo = hpo_freq[hpo_freq >= MIN_HPO_FREQUENCY].index.tolist()
        df_hpo = df_hpo[keep_hpo].copy()
        print(f"HPO features kept after rare-feature filtering: {len(keep_hpo)}")
    else:
        print("No HPO features detected.")

    # Inheritance: canonical multi-hot flags instead of one dummy per raw string
    df["_inh_list"] = df[INHERITANCE_COL].apply(parse_inheritance)
    mlb_inh = MultiLabelBinarizer()
    inh_matrix = mlb_inh.fit_transform(df["_inh_list"])
    df_inherit = pd.DataFrame(inh_matrix, columns=[f"INH_{t}" for t in mlb_inh.classes_], index=df.index)
    print(f"Inheritance features ({df_inherit.shape[1]}): " +
          ", ".join(f"{c[4:]}={int(df_inherit[c].sum())}" for c in df_inherit.columns))

    # System matrix
    sys_feature_cols = [c for c in df.columns if c.startswith("SYS__")]
    df_sys = df[sys_feature_cols].copy() if sys_feature_cols else pd.DataFrame(index=df.index)

    # Final feature matrix
    X = pd.concat([df_hpo, df_inherit, df_sys], axis=1)
    X = X.fillna(0).astype(float)

    if X.shape[1] == 0:
        raise ValueError("No usable features found.")

    # organ-system sanity check
    for c in [c for c in X.columns if c.startswith("SYS__")]:
        frac = X[c].mean()
        if frac in (0.0, 1.0) or frac > 0.98 or frac < 0.02:
            print(f"WARNING: organ-system feature {c} is (almost) constant: present in {frac:.1%} of records")

    # DATA-QUALITY STEP
    id_cols = [c for c in ["disease_id", "omim_phenotype_number", "phenotype_label"] if c in df.columns]
    def _describe(idx_list):
        return [{**{c: (None if pd.isna(df.loc[i, c]) else str(df.loc[i, c])) for c in id_cols},
                 "disease_family": str(df.loc[i, TARGET_COL])} for i in idx_list]
    quality = {"records_before": int(len(X))}
    hpo_cols_now = [c for c in X.columns if c.startswith("HPO__")]
    if DROP_RECORDS_WITHOUT_HPO:
        no_hpo = X.index[X[hpo_cols_now].sum(axis=1) == 0].tolist()
        quality["removed_no_hpo"] = _describe(no_hpo)
        X = X.drop(index=no_hpo); df = df.drop(index=no_hpo)
        print(f"Removed {len(no_hpo)} record(s) without any HPO term")
    sig = X.astype(np.uint8).apply(lambda r: r.values.tobytes(), axis=1)
    if DEDUPLICATE_SAME_FAMILY:
        dup_mask = pd.DataFrame({"sig": sig, "fam": df[TARGET_COL].values}, index=X.index).duplicated(keep="first")
        dup_idx = X.index[dup_mask].tolist()
        quality["removed_same_family_duplicates"] = _describe(dup_idx)
        X = X.drop(index=dup_idx); df = df.drop(index=dup_idx); sig = sig.drop(index=dup_idx)
        print(f"Removed {len(dup_idx)} exact duplicate record(s) of the same family")
    groups = pd.DataFrame({"sig": sig, "fam": df[TARGET_COL].values}, index=X.index).groupby("sig")
    cross = [list(g.index) for _, g in groups if g["fam"].nunique() > 1]
    quality["identical_profiles_different_families"] = [_describe(g) for g in cross]
    print(f"Identical feature profiles shared by different families (kept): {len(cross)} group(s)")
    df = reduce_rare_classes(df, TARGET_COL, min_samples=TARGET_MIN_SAMPLES)  # re-check class sizes
    X = X.loc[df.index]
    quality["records_after"] = int(len(X))
    quality["class_counts_after"] = df[TARGET_COL].value_counts().to_dict()
    save_json(quality, out_dir / "data_quality_report.json")
    print(f"Records used for modelling: {len(X)}")

    y_labels = df[TARGET_COL].astype(str).values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    feature_names = X.columns.tolist()

    print(f"\nFinal feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Number of disease-family classes: {len(label_encoder.classes_)}")

    print(f"\nDisease-family classes (n={len(label_encoder.classes_)}):")
    for i, cls in enumerate(label_encoder.classes_, 1):
        print(f"{i:02d}. {cls}")

    if SAVE_FEATURE_TABLES:
        X.to_csv(out_dir / "feature_matrix.csv", index=False)
        pd.DataFrame({"target": y_labels}).to_csv(out_dir / "target_labels.csv", index=False)

    # Train/test split
    n_classes = len(np.unique(y))
    min_test_size = n_classes / len(X)
    adjusted_test_size = max(TEST_SIZE, min_test_size + 0.02)

    if adjusted_test_size >= 0.5:
        raise ValueError(
            f"Too many classes ({n_classes}) for dataset size ({len(X)}). "
            f"Increase data, reduce classes, or increase TARGET_MIN_SAMPLES."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=adjusted_test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nNumber of classes: {n_classes}")
    print(f"Adjusted test size: {adjusted_test_size:.3f}")
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size : {X_test.shape[0]}")

    min_class_count_train = np.min(np.bincount(y_train))
    effective_cv = min(CV_FOLDS, max(2, min_class_count_train))
    cv = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=RANDOM_STATE)
    # report why the number of folds may be lower than CV_FOLDS
    print(f"Cross-validation folds used: {effective_cv} "
          f"(smallest training class has {min_class_count_train} records; requested {CV_FOLDS})")

    # =====================================================
    # MODEL 1: RANDOM FOREST
    # =====================================================
    rf_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("classifier", RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    rf_param_grid = {
        "classifier__n_estimators": [200, 500],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__max_features": ["sqrt", "log2"],
    }
    if QUICK_TEST:  # tiny grid only for code testing
        rf_param_grid = {"classifier__n_estimators": [50], "classifier__max_depth": [None, 10]}

    rf_grid = GridSearchCV(
        rf_pipeline,
        rf_param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    print("\nTraining Random Forest...")
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print("Best RF params:")
    print(rf_grid.best_params_)

    # =====================================================
    # MODEL 2: XGBOOST OR FALLBACK
    # =====================================================
    if HAS_XGBOOST:
        model2_name = "XGBoost"
        model2_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("classifier", XGBClassifier(
                random_state=RANDOM_STATE,
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=len(label_encoder.classes_),
                n_jobs=1  # one thread per model; GridSearch/CV parallelise over fits (reproducible)
            ))
        ])

        model2_param_grid = {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        }
        if QUICK_TEST:
            model2_param_grid = {"classifier__n_estimators": [30], "classifier__max_depth": [3]}

    else:
        model2_name = "HistGradientBoosting"
        model2_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("classifier", HistGradientBoostingClassifier(
                random_state=RANDOM_STATE,
                max_iter=300
            ))
        ])

        model2_param_grid = {
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__max_depth": [None, 5, 10],
            "classifier__max_leaf_nodes": [15, 31],
            "classifier__min_samples_leaf": [5, 10],
        }

    model2_grid = GridSearchCV(
        model2_pipeline,
        model2_param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    print(f"\nTraining {model2_name}...")
    m2_fit_params = {}
    if XGB_BALANCED_WEIGHTS:  # balanced sample weights (indexed per CV split by GridSearchCV)
        m2_fit_params = {"classifier__sample_weight": compute_sample_weight("balanced", y_train)}
    model2_grid.fit(X_train, y_train, **m2_fit_params)
    best_model2 = model2_grid.best_estimator_
    print(f"Best {model2_name} params:")
    print(model2_grid.best_params_)

    # =====================================================
    # EVALUATION
    # =====================================================
    def evaluate_model(model: Pipeline, model_name: str) -> Dict:
        print(f"\nEvaluating {model_name}...")

        y_pred = model.predict(X_test)

        if hasattr(model.named_steps["classifier"], "predict_proba"):
            y_prob = model.predict_proba(X_test)
            tk = top_k_accuracy_score(
                y_test,
                y_prob,
                k=min(TOP_K, len(label_encoder.classes_)),
                labels=np.arange(len(label_encoder.classes_))
            )
        else:
            y_prob = None
            tk = np.nan

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
            output_dict=True
        )

        n_test = len(y_test)
        k_acc = int(np.sum(y_pred == y_test))
        acc_ci = wilson_ci(k_acc, n_test)                      # 95% CI
        tk_ci = (np.nan, np.nan) if np.isnan(tk) else wilson_ci(int(round(tk * n_test)), n_test)
        results = {
            "model": model_name,
            "accuracy": float(acc),
            "accuracy_95ci": acc_ci,
            "n_correct": k_acc,
            "n_test": n_test,
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "f1_weighted": float(f1),
            "top_3_accuracy": None if np.isnan(tk) else float(tk),
            "top_3_accuracy_95ci": tk_ci,
        }

        save_json(results, out_dir / f"{model_name.lower().replace(' ', '_')}_metrics.json")
        pd.DataFrame(report).transpose().to_csv(
            out_dir / f"{model_name.lower().replace(' ', '_')}_classification_report.csv"
        )

        plot_confusion_for_top_classes(
            y_test,
            y_pred,
            label_encoder,
            out_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png",
            top_n_classes=CONFUSION_TOP_N,
            title=f"{model_name} confusion matrix (test set)",
        )

        return results, y_pred

    rf_results, rf_test_pred = evaluate_model(best_rf, "Random Forest")
    model2_results, m2_test_pred = evaluate_model(best_model2, model2_name)
    mcnemar = mcnemar_exact(y_test, rf_test_pred, m2_test_pred)  # paired test-set comparison
    joblib.dump(best_rf, out_dir / "random_forest_model.joblib")
    joblib.dump(best_model2, out_dir / "xgboost_model.joblib")

    # =====================================================
    # REPEATED STRATIFIED CROSS-VALIDATION (fixed tuned hyperparameters)
    # - every fold fits a fresh clone -> best_rf / best_model2 stay the final models
    #   (the old loop refitted them on the last fold, so feature importance and SHAP
    #   were computed on a fold model, not on the model trained on the full training set)
    # - all CV metrics (fold scores, means, Wilcoxon) come from the SAME fold predictions
    # =====================================================
    n_classes_all = len(label_encoder.classes_)
    n_repeats = 2 if QUICK_TEST else CV_REPEATS
    rcv = RepeatedStratifiedKFold(n_splits=effective_cv, n_repeats=n_repeats, random_state=RANDOM_STATE)
    splits = list(rcv.split(X_train, y_train))
    models = [("Random Forest", best_rf), (model2_name, best_model2)]
    print(f"\nRepeated stratified CV: {effective_cv} folds x {n_repeats} repeats = {len(splits)} folds per model")
    jobs = [(fi, name, est, tr, te) for fi, (tr, te) in enumerate(splits) for name, est in models]
    weighted = {"Random Forest": False, model2_name: bool(XGB_BALANCED_WEIGHTS and HAS_XGBOOST)}
    fold_out = Parallel(n_jobs=-1)(
        delayed(_fit_score_fold)(est, X_train.iloc[tr], y_train[tr], X_train.iloc[te], y_train[te],
                                 n_classes_all, weighted[name])
        for fi, name, est, tr, te in jobs
    )
    fold_rows = [{"fold": fi, "repeat": fi // effective_cv + 1, "model": name, **res}
                 for (fi, name, est, tr, te), res in zip(jobs, fold_out)]
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "cv_fold_scores.csv", index=False)

    n_tr_fold = int(np.mean([len(tr) for tr, te in splits]))
    n_te_fold = int(np.mean([len(te) for tr, te in splits]))
    cv_summary = {}
    for name, _ in models:
        d = fold_df[fold_df["model"] == name]
        entry = {}
        for metric in ["accuracy", "f1_weighted", "top3_accuracy"]:
            if metric in d:
                s = d[metric].values
                entry[metric] = {
                    "mean": float(np.mean(s)), "sd": float(np.std(s, ddof=1)),
                    "min": float(np.min(s)), "max": float(np.max(s)),
                    "corrected_95ci": corrected_resampled_ci(s, n_tr_fold, n_te_fold),
                }
        cv_summary[name] = entry

    rf_acc_f = fold_df[fold_df["model"] == "Random Forest"].sort_values("fold")["accuracy"].values
    m2_acc_f = fold_df[fold_df["model"] == model2_name].sort_values("fold")["accuracy"].values
    diffs = rf_acc_f - m2_acc_f
    t_stat, p_corr = corrected_resampled_ttest(diffs, n_tr_fold, n_te_fold)
    try:
        w_stat, p_wil = wilcoxon(rf_acc_f, m2_acc_f)
    except ValueError:
        w_stat, p_wil = np.nan, np.nan
    comparison = {
        "rf_better_folds": int(np.sum(diffs > 0)), "ties": int(np.sum(diffs == 0)),
        "m2_better_folds": int(np.sum(diffs < 0)), "n_folds": int(len(diffs)),
        "mean_accuracy_difference": float(np.mean(diffs)),
        "mean_accuracy_difference_corrected_95ci": corrected_resampled_ci(diffs, n_tr_fold, n_te_fold),
        "corrected_resampled_ttest": {"t": t_stat, "p_value": p_corr},
        "wilcoxon_signed_rank": {"statistic": float(w_stat), "p_value": float(p_wil),
                                 "note": "repeated-CV folds overlap; Wilcoxon p is optimistic - report the corrected t-test as primary"},
        "mcnemar_test_set": mcnemar,
    }

    print("\nCross-validation (mean ± SD over folds, corrected 95% CI):")
    for name in cv_summary:
        a_ = cv_summary[name]["accuracy"]; f_ = cv_summary[name]["f1_weighted"]
        print(f"  {name:22s} accuracy {a_['mean']:.4f} ± {a_['sd']:.4f}  CI {a_['corrected_95ci'][0]:.3f}–{a_['corrected_95ci'][1]:.3f} | "
              f"F1 {f_['mean']:.4f} ± {f_['sd']:.4f}")
    print(f"  RF better in {comparison['rf_better_folds']}/{len(diffs)} folds (ties {comparison['ties']}); "
          f"corrected t-test p = {p_corr:.4f}; Wilcoxon p = {p_wil:.4f}")
    print(f"  Test-set McNemar: RF-only correct {mcnemar['a_correct_b_wrong']}, "
          f"{model2_name}-only correct {mcnemar['a_wrong_b_correct']}, p = {mcnemar['p_value']:.4f}")

    # =====================================================
    # NESTED CROSS-VALIDATION (optional, unbiased performance estimate)
    # hyperparameters are re-tuned inside every outer training fold, so the outer
    # score is not inflated by tuning on the same folds
    # =====================================================
    nested_summary = {}
    if RUN_NESTED_CV:
        outer = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=RANDOM_STATE + 1)
        print("\nRunning nested cross-validation (this can take a while)...")
        for name, pipe, grid in [("Random Forest", rf_pipeline, rf_param_grid),
                                 (model2_name, model2_pipeline, model2_param_grid)]:
            scores = []
            for tr, te in outer.split(X_train, y_train):
                inner_k = int(min(effective_cv, max(2, np.min(np.bincount(y_train[tr], minlength=n_classes_all)))))
                inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=RANDOM_STATE)
                g = GridSearchCV(clone(pipe), grid, scoring="f1_weighted", cv=inner, n_jobs=-1)
                fp = {}
                if name != "Random Forest" and XGB_BALANCED_WEIGHTS and HAS_XGBOOST:
                    fp = {"classifier__sample_weight": compute_sample_weight("balanced", y_train[tr])}
                g.fit(X_train.iloc[tr], y_train[tr], **fp)
                scores.append(accuracy_score(y_train[te], g.predict(X_train.iloc[te])))
            scores = np.array(scores)
            nested_summary[name] = {"fold_accuracy": scores.tolist(), "mean": float(scores.mean()),
                                    "sd": float(scores.std(ddof=1))}
            print(f"  {name:22s} nested accuracy {scores.mean():.4f} ± {scores.std(ddof=1):.4f}  folds {np.round(scores, 4)}")

    save_json({"cross_validation": cv_summary, "model_comparison": comparison,
               "nested_cv": nested_summary,
               "test_set": {"Random Forest": rf_results, model2_name: model2_results}},
              out_dir / "statistics_summary.json")
    save_json(cv_summary, out_dir / "cross_validation_summary.json")

    # Feature importance
    rf_imp = get_feature_importance(best_rf, feature_names, top_n=30)
    rf_imp["label"] = [pretty_feature(f, hpo_names) for f in rf_imp["feature"]]
    rf_imp.to_csv(out_dir / "random_forest_top_features.csv", index=False)
    plot_feature_importance(
        rf_imp,
        "Random Forest – top 30 features",
        out_dir / "random_forest_feature_importance.png",
        hpo_names,
    )

    model2_imp = get_feature_importance(best_model2, feature_names, top_n=30)
    if not model2_imp.empty:
        model2_imp["label"] = [pretty_feature(f, hpo_names) for f in model2_imp["feature"]]
        model2_imp.to_csv(out_dir / f"{model2_name.lower().replace(' ', '_')}_top_features.csv", index=False)
        plot_feature_importance(
            model2_imp,
            f"{model2_name} – top 30 features",
            out_dir / f"{model2_name.lower().replace(' ', '_')}_feature_importance.png",
            hpo_names,
            xlabel="Feature importance (normalized gain)" if HAS_XGBOOST else "Feature importance",
        )

    # PCA and t-SNE
    plot_pca(X.values, y, out_dir / "pca_plot.png", label_encoder)
    plot_tsne(X.values, y, out_dir / "tsne_plot.png", label_encoder)

    # Heatmap
    plot_df = X.copy()
    plot_df["TARGET_LABEL"] = y_labels
    create_hpo_target_heatmap(
        plot_df,
        target_col="TARGET_LABEL",
        feature_prefix="HPO__",
        filepath=out_dir / "hpo_family_heatmap.png",
        top_hpo_n=20,
        top_target_n=20,
        hpo_names=hpo_names,
    )

    # SHAP
    if HAS_SHAP:
        print("\nRunning SHAP for Random Forest...")
        run_shap_analysis(best_rf, X_train, out_dir / "random_forest_shap", label_encoder, hpo_names,
                          model_name="Random Forest")
        if HAS_XGBOOST:
            print("Running SHAP for XGBoost...")
            run_shap_analysis(best_model2, X_train, out_dir / "xgboost_shap", label_encoder, hpo_names,
                              model_name=model2_name)

    # Save overall summary
    overall_summary = {
        "dataset_rows_used": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_target_classes": int(len(label_encoder.classes_)),
        "hpo_features": int(len([c for c in X.columns if c.startswith("HPO__")])),
        "inheritance_features": int(len([c for c in X.columns if c.startswith("INH_")])),
        "system_features": int(len([c for c in X.columns if c.startswith("SYS__")])),
        "random_forest_test_metrics": rf_results,
        "second_model_test_metrics": model2_results,
        "cross_validation": cv_summary,
        "model_comparison": comparison,
        "nested_cv": nested_summary,
        "cv_folds_used": int(effective_cv),
        "cv_repeats": int(n_repeats),
        "best_random_forest_params": rf_grid.best_params_,
        "best_second_model_params": model2_grid.best_params_,
        "target_classes": label_encoder.classes_.tolist(),
    }
    save_json(overall_summary, out_dir / "overall_summary.json")

    # Comparison table
    def _row(name, res):  # table now includes CIs and fold variability
        c = cv_summary[name]
        row = {
            "Model": name,
            "Accuracy": res["accuracy"],
            "Accuracy 95% CI": f"{res['accuracy_95ci'][0]:.3f}–{res['accuracy_95ci'][1]:.3f}",
            "Precision (weighted)": res["precision_weighted"],
            "Recall (weighted)": res["recall_weighted"],
            "F1 (weighted)": res["f1_weighted"],
            "Top-3 Accuracy": res["top_3_accuracy"],
            "Top-3 95% CI": f"{res['top_3_accuracy_95ci'][0]:.3f}–{res['top_3_accuracy_95ci'][1]:.3f}",
            "CV Accuracy (mean ± SD)": f"{c['accuracy']['mean']:.4f} ± {c['accuracy']['sd']:.4f}",
            "CV Accuracy corrected 95% CI": f"{c['accuracy']['corrected_95ci'][0]:.3f}–{c['accuracy']['corrected_95ci'][1]:.3f}",
            "CV F1 (mean ± SD)": f"{c['f1_weighted']['mean']:.4f} ± {c['f1_weighted']['sd']:.4f}",
        }
        if name in nested_summary:
            row["Nested CV Accuracy (mean ± SD)"] = f"{nested_summary[name]['mean']:.4f} ± {nested_summary[name]['sd']:.4f}"
        return row

    metrics_table = pd.DataFrame([_row("Random Forest", rf_results), _row(model2_name, model2_results)])
    metrics_table.to_csv(out_dir / "model_comparison_table.csv", index=False)

    print("\n==============================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("==============================")
    print(f"Results saved in: {out_dir.resolve()}")
    print("\nModel comparison:")
    print(metrics_table.T.to_string(header=False))

    print("\nTop Random Forest features:")
    print(rf_imp.head(10).to_string(index=False))



if __name__ == "__main__":
    main()
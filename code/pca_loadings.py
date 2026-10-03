"""Loadings of the first two principal components shown in Figure 8 (PCA of the feature matrix).

Run after code/ciliopathy_family_pipeline.py (it uses results/feature_matrix.csv), from the repository root:
    python code/pca_loadings.py
Writes results/pca_loadings.csv (loading of every feature on PC1 and PC2) and results/pca_summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ciliopathy_family_pipeline as pipe  # noqa: E402  (settings and label helper of the main analysis)

RES = Path(pipe.OUTPUT_DIR)
X = pd.read_csv(RES / "feature_matrix.csv")
hpo_names = pipe.load_hpo_names(pipe.HPO_NAMES_FILE)

# same PCA as plot_pca() in the pipeline: 2 components of the binary feature matrix, no scaling
pca = PCA(n_components=2, random_state=pipe.RANDOM_STATE)
scores = pca.fit_transform(X.values)

loadings = pd.DataFrame({
    "feature": X.columns,
    "label": [pipe.pretty_feature(f, hpo_names, max_len=200) for f in X.columns],
    "PC1_loading": pca.components_[0],
    "PC2_loading": pca.components_[1],
})
loadings.sort_values("PC1_loading", ascending=False).to_csv(RES / "pca_loadings.csv", index=False)

hpo = [c for c in X.columns if c.startswith("HPO__")]
sys_cols = [c for c in X.columns if c.startswith("SYS__")]
n_hpo = X[hpo].sum(axis=1).values
summary = {
    "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
    "organ_system_indicators_with_positive_PC1_loading": int((loadings.set_index("feature").loc[sys_cols, "PC1_loading"] > 0).sum()),
    "n_organ_system_indicators": len(sys_cols),
    "share_of_HPO_features_with_positive_PC1_loading": float((loadings.set_index("feature").loc[hpo, "PC1_loading"] > 0).mean()),
    "pearson_r_PC1_vs_number_of_HPO_terms": float(np.corrcoef(scores[:, 0], n_hpo)[0, 1]),
    "pearson_r_PC2_vs_number_of_HPO_terms": float(np.corrcoef(scores[:, 1], n_hpo)[0, 1]),
}
for pc in ("PC1", "PC2"):
    s = loadings.sort_values(f"{pc}_loading")
    summary[f"{pc}_most_negative"] = [[r.label, round(float(r[f"{pc}_loading"]), 4)] for _, r in s.head(10).iterrows()]
    summary[f"{pc}_most_positive"] = [[r.label, round(float(r[f"{pc}_loading"]), 4)] for _, r in s.tail(10)[::-1].iterrows()]
(RES / "pca_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))

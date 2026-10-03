"""Figure 10 - HPO term frequency by disease family (heatmap).

Uses results/feature_matrix.csv and results/target_labels.csv written by the pipeline and draws the figure with the
pipeline's create_hpo_target_heatmap(), so it can be redrawn without re-running the analysis. Each cell is the
proportion of the records of a family that are annotated with the HPO term (0 to 1, no further normalization); the
20 terms are those most frequent across all 391 records. The values are written to
results/hpo_family_heatmap_values.csv.

Run from the repository root:  python code/make_figure10.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ciliopathy_family_pipeline as pipe  # noqa: E402

R = Path(pipe.OUTPUT_DIR)
X = pd.read_csv(R / "feature_matrix.csv")
df = X.copy()
df["TARGET_LABEL"] = pd.read_csv(R / "target_labels.csv")["target"].values
pipe.create_hpo_target_heatmap(df, target_col="TARGET_LABEL", feature_prefix="HPO__",
                               filepath=R / "hpo_family_heatmap.png", top_hpo_n=20, top_target_n=20,
                               hpo_names=pipe.load_hpo_names(pipe.HPO_NAMES_FILE))
print("written:", R / "hpo_family_heatmap.png", "and", R / "hpo_family_heatmap_values.csv")

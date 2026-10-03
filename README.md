# Phenotype-driven classification of ciliopathy disease families

Code and data for the article **"Phenotype-driven machine learning classification of ciliopathy disease families using curated clinical and Human Phenotype Ontology features"** by A. Talib, M. Farman, S. Asir, K. S. Nisar, M. Tabish and F. Khurshid (Journal of the Nigerian Society of Physical Sciences, under review).

The analysis classifies 391 curated disease-level records into 20 ciliopathy disease families from Human Phenotype Ontology (HPO) terms, inheritance and organ-system involvement, using Random Forest and XGBoost.

**Observational unit.** Each record is one disease entry, not a patient. For 374 of the 391 records this is a single OMIM phenotype entry, i.e. one genetic subtype of a disease family with one causal gene (for example, Joubert syndrome 3, caused by *AHI1*); the other records are 10 Orphanet disorder entries and 7 family-level records. The 391 records comprise 366 distinct OMIM phenotype entries (8 entries are curated under two families), and the 374 OMIM entries involve 306 distinct causal genes.

## Repository contents

| Path | Content |
|---|---|
| `data/ciliopathy_curated_entries.csv` | The 451 curated entries used as input: record ID, OMIM/Orphanet identifiers, disease family, inheritance, HPO term IDs and labels, and 15 organ-system indicators. |
| `data/Supplementary_Table_S2.xlsx` / `.csv` | Supplementary Table S2 of the article: every entry with its source identifiers and whether it was used for modelling (and if not, why). |
| `data/hpo_term_names.csv` | HPO term labels (HPO release 2026-09-01), used only for readable figure labels. |
| `data/hpo_feature_categories.csv` | HPO category (organ-system group) of each HPO feature, derived from the HPO release of 2026-09-01; used for grouped permutation importance. |
| `code/ciliopathy_family_pipeline.py` | The complete analysis (data steps, model tuning, evaluation, statistics, interpretation, figures). |
| `code/make_figure3.py` | Figure 3 of the article (model comparison), drawn from `results/statistics_summary.json`. |
| `code/permutation_importance.py` | Permutation importance on the held-out test set, for single features and for organ-system groups (organ-system indicator plus the HPO terms of the matching HPO category, from `data/hpo_feature_categories.csv`); writes `results/permutation_importance_*.csv`. |
| `code/make_figure10.py` | Figure 10 of the article (HPO term frequency by disease family), redrawn from `results/feature_matrix.csv`; writes `results/hpo_family_heatmap.png` and the plotted values (proportion of the records of each family annotated with each term) to `results/hpo_family_heatmap_values.csv`. |
| `code/pca_loadings.py` | Loadings of the first two principal components shown in Figure 8; writes `results/pca_loadings.csv` and `results/pca_summary.json`. |
| `code/sensitivity_analysis.py` | Sensitivity analyses: related entries (without family-level records, single OMIM entries only, grouped folds), the HPO frequency filter fitted on training data only, and the 13 primary and motile ciliopathy families only (CiliaMiner classes); writes `results/sensitivity_results.json`. |
| `results/` | Outputs of the run reported in the article (metrics, statistics, cross-validation fold scores, feature rankings and figures). |

## Data sources and curation

The dataset was assembled in May 2025:

1. Names of ciliopathies and ciliopathy-related disorders reported in the literature were searched in OMIM (https://www.omim.org; accessed May 2025), and the OMIM phenotype entry (MIM number) of each disorder was recorded.
2. For each disorder, the related OMIM phenotype entries (for example, its numbered genetic subtypes) were retrieved, and each entry was assigned to the disease family under whose name it was found. An entry found under two disease names was recorded under both families.
3. For each entry, the mode of inheritance, the organ systems involved and the clinical features were taken from its OMIM Clinical Synopsis; the clinical features are the HPO terms that OMIM links to each Clinical Synopsis feature. Fifteen disorders without a matching OMIM Clinical Synopsis were matched by name to Orphanet (https://www.orpha.net), and 14 family-level records combine the annotations of several entries of the same disease. This gave 451 entries in 59 disease families.
4. Families with fewer than five entries were removed (39 families, 51 entries; 400 entries remain). HPO terms present in only one of these 400 entries are not used as features. Entries left without any HPO term (2) and exact duplicates within the same family (7) were then removed, leaving **391 records in 20 families** for modelling.

**Disease families (classification target).** A family label is the clinical diagnosis shared by the genetically distinct entries assigned to it, so the labels range from named multisystem syndromes (for example, Joubert syndrome) to disorders defined by one organ system (for example, retinitis pigmentosa). The 59 families correspond to the disorders listed in the CiliaMiner ciliopathy database (Turan et al., *Database* 2023, baad047; https://kaplanlab.shinyapps.io/ciliaminer/), and the column `classification` gives the CiliaMiner class of each family: `Primary` (primary ciliopathy), `Motile` (motile ciliopathy) or `Secondary` (secondary disease: a disorder that shares clinical features with ciliopathies but whose disease proteins do not localize to cilia or cilia-related compartments, or the reverse). Of the 20 families used for modelling, 11 are primary ciliopathies (169 records), 2 motile ciliopathies (51 records) and 7 secondary diseases (171 records). The class is not a model feature; it is used only in the sensitivity analysis.

The text of OMIM Clinical Synopses is **not** included in this repository, in accordance with the OMIM terms of use; it can be viewed on the OMIM website using the identifiers in the data files. HPO term identifiers refer to the Human Phenotype Ontology (https://hpo.jax.org).

## Reproducing the analysis

Python 3.13 was used. From the repository root:

```bash
pip install -r requirements.txt
python code/ciliopathy_family_pipeline.py   # writes everything to results/ (about 30-40 min on 2 CPU cores)
python code/make_figure3.py                 # Figure 3
python code/make_figure10.py                # Figure 10 (heatmap) and its values
python code/pca_loadings.py                 # PCA loadings (Figure 8), a few seconds
python code/sensitivity_analysis.py         # sensitivity analyses (a few minutes)
python code/permutation_importance.py       # permutation importance (about 15-20 min)
```

With the package versions in `requirements.txt` the run reproduces the numbers in `results/`. For a quick functional check with small hyperparameter grids, set `CILIO_QUICK_TEST=1` before running the pipeline (results will differ from the article).

Main settings (top of the pipeline script): stratified 80:20 train/test split with seed 42; grid search with stratified 4-fold cross-validation (weighted F1); repeated stratified 4-fold cross-validation with 10 repeats (40 folds); nested cross-validation; families with at least 5 entries; HPO terms present in at least 2 entries.

## Main results (held-out test set, 79 records)

| Model | Accuracy (95% CI) | Weighted F1 | Top-3 accuracy (95% CI) | Repeated CV accuracy (mean ± SD) |
|---|---|---|---|---|
| Random Forest | 0.873 (0.782–0.930) | 0.856 | 0.987 (0.932–0.998) | 0.879 ± 0.032 |
| XGBoost | 0.797 (0.696–0.871) | 0.787 | 0.949 (0.877–0.980) | 0.839 ± 0.042 |

## License

The code is released under the MIT License (see `LICENSE`). The data files contain identifiers and annotations derived from OMIM, Orphanet and the Human Phenotype Ontology and remain subject to the terms of use of these resources; the text of OMIM Clinical Synopses is not included.

## Citation

If you use this code or data, please cite the article above (full reference to be added on publication), together with OMIM, Orphanet, the Human Phenotype Ontology and CiliaMiner.

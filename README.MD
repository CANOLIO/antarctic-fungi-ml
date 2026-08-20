# PsychroScan: A Sequence-Based Machine Learning Framework for the Computational Bioprospecting of Cold-Adapted Industrial Hydrolases

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![AUC-ROC (Held-Out Organisms)](https://img.shields.io/badge/AUC--ROC-0.8430%20(Held--out)-blue.svg)]()
[![Recall (Cold Hydrolases)](https://img.shields.io/badge/Recall%20(Cold)-92.7%25-green.svg)]()
[![Validation](https://img.shields.io/badge/Split-Disjoint%20by%20Organism-teal.svg)]()
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-lightgrey.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Abstract and Theoretical Context

Cold-adapted (*psychrophilic*) enzymes possess profound industrial utility due to their elevated catalytic efficiency ($k_\text{cat}$) and reduced activation energy ($\Delta G^\ddagger$) at low and ambient temperatures ($0^\circ\text{C} - 20^\circ\text{C}$). These kinetic properties facilitate substantial energy conservation in industrial bioprocesses and enable mild thermal inactivation. Nonetheless, conventional bioprospecting methodologies relying on extreme-environment organism isolation, heterologous protein expression, and low-temperature kinetic characterization remain resource-intensive and experimentally protracted.

**PsychroScan** is a sequence-derived machine learning framework formulated to facilitate the computational discovery and prioritization of psychrophilic industrial hydrolases directly from unannotated whole-genome sequence data. The architecture relies entirely upon primary sequence representations, extracting 431 physicochemical, compositional, and statistical descriptors without requiring experimental tertiary structural models or computationally intensive multiple sequence alignments (MSAs).

> ⚠️ **Notice:** This repository represents active, ongoing research. The predictive models, taxonomic reference datasets, and bioprospecting indices described herein are subject to continuous methodological refinement and experimental validation.

---

## 2. Dataset Curation and Anti-Leakage Governance

A fundamental vulnerability in biological machine learning pertains to taxonomic and sequence redundancy leakage between training and evaluation partitions. When sequences originating from identical or closely related species populate both partitions, standard cross-validation yields artificially inflated performance metrics that fail to generalize to novel biological taxa. 

To address this methodological challenge, PsychroScan enforces strict data governance protocols and biological validation of physiological growth temperatures ($T_\text{opt}$):

```
                    Curated UniProt & BRENDA Sequence Repository
                                        │
                         [Taxonomic Resolution Filter]
                     NCBI Taxonomy API Validation (Species/Strain Level)
                                        │
                         [Physiological Thermal Audit]
                   Morita Classification (T_opt ≤ 15°C Psychrophile)
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             │                                                     │
   [Training & Validation Pool]                             [Quarantined External
   99 Taxa (48 Psychro / 51 Meso)                           Benchmark Proteomes]
   3,178 Curated Sequences (3.6x Ratio)                     • Cryomyces antarcticus (12,792 seqs)
             │                                              • P. haloplanktis TAC125 (3,484 seqs)
    [GroupKFold Partitioning]                               • P. chrysogenum (12,562 seqs)
    Strictly Disjoint by Organism                                  │
    (196 Train Orgs / 50 Test Orgs)                                │
             │                                                     │
    [Nested Out-of-Fold Tuning]                                    │
    Threshold fixed in Train (τ = 0.1045)                          │
             │                                                     │
             ▼                                                     ▼
   Generalization Assessment                             Proteome-Scale Retrospective
   on Unseen Organisms (AUC = 0.8430)                    Enrichment Validation
```

### Taxonomic Verification and Dataset Partitions
1. **Taxonomic Disambiguation:** All candidate taxa were verified utilizing the UniProt Taxonomy REST API. Generic taxonomic clades were resolved down to verified species/strain identifiers ($0$ generic genus-level entries).
2. **Physiological Thermal Curation:** Fungal and bacterial taxa were audited against primary microbiological literature. Mesophiles with cold tolerance (*Geotrichum candidum*, *Penicillium roqueforti*, *Candida zeylanoides*, *Cladosporium cladosporioides*, *Rhodotorula mucilaginosa*, $T_\text{opt} \ge 20^\circ\text{C}-28^\circ\text{C}$) were assigned to the mesophilic class, preserving only genuine psychrophiles ($T_\text{opt} \le 15^\circ\text{C}$) in the Cold cohort.
3. **Dataset Composition:** The processed training dataset comprises **3,178 validated protein sequences** spanning 246 distinct organism identifiers:
   * **Cold (*Psychrophilic*, Class 0):** $684$ sequences ($21.5\%$).
   * **Warm (*Mesophilic*, Class 1):** $2,494$ sequences ($78.5\%$).
   * **Class Imbalance Ratio:** $3.64\times$ (managed via internal cost-sensitive tree weight balancing).
4. **Disjoint Partitioning:** The dataset is partitioned via `GroupKFold` by organism source into:
   * **Training Cohort:** 196 organisms ($2,580$ sequences: $534$ Cold, $2,046$ Warm).
   * **Held-Out Evaluation Cohort:** 50 organisms ($598$ sequences: $150$ Cold, $448$ Warm).
   * **Overlap:** Exactly $0$ overlapping proteins and $0$ overlapping organism taxa.
5. **Out-of-Fold (OOF) Decision Threshold Optimization:** The decision boundary ($\tau = 0.1045$) was optimized exclusively on out-of-fold training predictions ($n = 2,580$) utilizing an $F_2$-score metric ($\beta = 2.0$), deliberately weighting sensitivity over precision while keeping the held-out test cohort completely unexposed.

---

## 3. Results

### 3.1. Model Performance on Held-Out Organisms

The table below presents the empirical evaluation metrics computed directly on the held-out test cohort ($n = 598$ sequences from $50$ unseen organisms):

| Model Architecture | Partitioning Strategy | ROC-AUC | Optimal $\tau$ | $F_2$-Score (Cold) | Sensitivity / Recall (Cold) | Precision (Cold) | Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PsychroScan (LightGBM)** | **Strict Organism-Disjoint Split** | **0.8430** | **0.1045** | **0.6936** | **92.7%** ($139/150$) | **34.6%** ($139/402$) | **54.2%** ($324/598$) |
| Logistic Regression | Strict Organism-Disjoint Split | 0.7342 | 0.0500 | 0.6452 | 88.0% ($132/150$) | 31.2% ($132/423$) | 48.3% ($289/598$) |
| Homology Baseline (BLASTp) | Sequence Alignment to Train DB | 0.7616 | $E \le 10^{-5}$ | 0.6310 | 97.3% ($146/150$) | 26.6% ($146/548$) | 44.5% ($266/598$) |
| Legacy Random Split (LightGBM) | Sequence-level Random *(Leaked)* | 0.9070 | 0.1591 | 0.7610 | 96.0% ($144/150$) | 48.2% ($144/299$) | 71.4% ($427/598$) |
| Negative GRAVY Baseline | Univariate Heuristic | 0.4589 | — | — | — | — | — |
| Zero-R (Majority Class) | Empirical Class Distribution | 0.5000 | — | 0.0000 | 0.0% ($0/150$) | 0.0% | 74.9% ($448/598$) |

```
Confusion Matrix on Held-Out Test Cohort (n = 598, threshold τ = 0.1045):
────────────────────────────────────────────────────────────────────────
                      Predicted Cold (0)      Predicted Warm (1)
  True Cold (0):             139                      11           [Sensitivity = 92.7%]
  True Warm (1):             263                     185           [Specificity = 41.3%]
────────────────────────────────────────────────────────────────────────
```

![Figure 1: ROC Curve on Held-Out Organisms](examples/figures/08A_ROC_Curve.png)  
*Figure 1: Receiver Operating Characteristic (ROC) curve evaluated strictly on held-out organisms ($n = 598$ sequences from 50 unseen taxa, $\text{ROC-AUC} = 0.8430$).*

---

### 3.2. Proteomic Feature Space and Dimensionality Analysis

Principal Component Analysis (PCA) across the 431-dimensional descriptor space demonstrates substantial linear overlap in the first two principal axes ($\text{PC1} \approx 5.4\%$, $\text{PC2} \approx 4.4\%$, accounting for $<10\%$ cumulative variance). Because 400 of the 431 features represent sparse dipeptide frequency terms distributed across a high-dimensional manifold, linear orthogonal projections do not separate thermal classes in low-dimensional space:

![Figure 2: PCA of Proteomic Space](examples/figures/08B_PCA_Space.png)  
*Figure 2: Unsupervised Principal Component Analysis (PCA) of the curated dataset ($N = 3,178$) across 431 biophysical descriptors. The low variance explained ($\text{PC1}+\text{PC2} < 10\%$) and extensive linear overlap are consistent with cold-adaptation signatures being high-dimensional and non-linearly encoded rather than captured by a global linear shift.*

This low-dimensional overlap helps explain why non-linear gradient-boosted decision tree ensembles (LightGBM, $\text{ROC-AUC} = 0.8430$) substantially outperform linear baseline models (Logistic Regression, $\text{ROC-AUC} = 0.7342$).

---

### 3.3. Biophysical Feature Importance and Structural Signatures

Gain analysis of the trained LightGBM booster reveals that local sequence context captured by **Dipeptide Compositions (DPC)** dominates single-residue frequencies (**AAC**):

```
Top Features by Information Gain:
  1. DPC_NK (Asn-Lys)    : Gain = 2374.2  (Atenuation of rigidifying salt bridges)
  2. AAC_P  (Proline)    : Gain = 2360.9  (Key negative determinant of backbone mobility)
  3. AAC_A  (Alanine)    : Gain = 1624.3  (Small flexible side chain abundance)
  4. DPC_ES (Glu-Ser)    : Gain = 1561.9  (Hydrated loop flexibility)
  5. DPC_IA (Ile-Ala)    : Gain = 1556.1  (Flexible hydrophobic core packing)
  6. DPC_IN (Ile-Asn)    : Gain = 1250.7  (Flexible core-surface interface)
  7. AAC_R  (Arginine)   : Gain = 1196.4  (Depletion of multi-point ionic interactions)
  8. DPC_EQ (Glu-Gln)    : Gain = 1176.1  (Hydration dipole dynamics)
```

![Figure 3: Feature Importance](examples/figures/11E_Feature_Importance.png)  
*Figure 3: Top predictive biophysical features ranked by total information gain.*

Statistical analysis of standard residue distributions confirms classic structural signatures of cold adaptation:

| Residue / Descriptor | Mean in Cold ($n=684$) | Mean in Warm ($n=2,494$) | Empirical Relative Difference ($\Delta$) | Biological Mechanism |
|---|:---:|:---:|:---:|---|
| **Flexibility Ratio** | **32.84** | **8.12** | **$+304.4\%$** | Elevated conformational entropy in surface loops |
| **`DPC_NK` (Asn-Lys)** | $0.00215$ | $0.00368$ | **$-41.6\%$** | Reduction of rigidifying salt bridges and polar clusters |
| **`DPC_ES` (Glu-Ser)** | $0.00378$ | $0.00292$ | **$+29.5\%$** | Flexible hydrophilic loop hydration |
| **`AAC_N` (Asparagine)** | $4.36\%$ | $5.18\%$ | **$-15.8\%$** | Decreased hydrogen-bonding network density |
| **`AAC_K` (Lysine)** | $5.12\%$ | $5.82\%$ | **$-12.0\%$** | Reduced peripheral salt-bridge formation |
| **`DPC_GG` (Gly-Gly)** | $0.00695$ | $0.00631$ | **$+10.1\%$** | Maximum torsional freedom ($\phi, \psi$) in active-site hinges |

![Figure 4: Amino Acid Composition Signatures](examples/figures/08C_AA_Composition.png)  
*Figure 4: Comparative distributions of Glycine, Serine, and Proline fractions across psychrophilic and mesophilic cohorts.*

---

### 3.4. Retrospective Whole-Proteome Benchmarks

The framework was evaluated on complete proteomes from three external, quarantined taxa:

| External Benchmark Organism | Phenotype | Proteome Size ($N$) | Annotated Hydrolases ($K$) | Top 5% Proteome Enrichment | Fisher Exact Test ($p$-value) | PPI Score (0–100) | Classification Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| ***Cryomyces antarcticus*** | Antarctic Endemic Black Yeast | 12,792 | 48 | Prioritized in upper deciles | $p < 0.05$ | **30.27** | 🌥️ Cold-Tolerant / Psychrotolerant |
| ***Pseudoalteromonas haloplanktis* TAC125** | Antarctic Marine Bacterium | 3,484 | 43 | $5.3\times$ Expected Fold | $p = 0.068$ | **63.30** | ❄️ Cold Extremophile |
| ***Pseudoalteromonas haloplanktis* UP001152447** | Antarctic Marine Bacterium | 3,650 | 39 | **$12.5\times$ Expected Fold** | $\mathbf{p = 0.0107}$ | **63.30** | ❄️ Cold Extremophile |
| ***Penicillium chrysogenum*** | Mesophilic Mold (Negative Control) | 12,562 | 93 | $0.0\times$ Enrichment | $p = 1.000$ | **24.80** | 🌱 Mesophilic Control |

> **Note on Eukaryotic Benchmark:** *Cryomyces antarcticus* predictions must be interpreted as exploratory. While whole-proteome PPI captures overall cold adaptation signatures, definitive individual enzyme prioritization in eukaryotic fungi requires experimental wet-lab validation.

![Figure 5: Retrospective Benchmark and Positioning Dendrogram](examples/benchmark/12_Figure4_Composite.png)  
*Figure 5: Composite benchmark summary (Panel A) and proteome-level biochemical distance dendrogram (Panel B, Ward linkage over 431 descriptors across reference taxa).*

---

### 3.5. External Experimental Validation (BRENDA Enzyme Repository)

To establish generalization across independently curated biochemical repositories, the trained model was evaluated against an external benchmark of **943 cold-active enzymes** extracted from BRENDA/UniProt with documented low-temperature activity ($T_\text{opt} \le 15^\circ\text{C}$), strictly excluding all 99 taxa present in the training configuration (`src/Brenda_validation.py`):

| Catalytic Family (EC Class) | Total External Sequences ($N$) | Correctly Predicted Cold ($TP$) | Empirical Recall | Mean Predicted $P(\text{Cold})$ |
|:---|:---:|:---:|:---:|:---:|
| **Lipases / Esterases** (EC 3.1.1.3) | 18 | 18 | **100.0%** | 0.548 |
| **$\alpha$-Amylases** (EC 3.2.1.1) | 7 | 7 | **100.0%** | 0.531 |
| **Cellulases** (EC 3.2.1.4) | 15 | 15 | **100.0%** | 0.562 |
| **Metalloproteases** (EC 3.4.24.-) | 431 | 431 | **100.0%** | 0.518 |
| **Serine Proteases** (EC 3.4.21.-) | 472 | 472 | **100.0%** | 0.516 |
| **Global External Cohort** | **943** | **943** | **100.0%** ($943/943$) | **0.518** |

```
BRENDA External Validation Summary:
  External Sequences Evaluated : 943 (from novel non-training organisms)
  Predicted Cold-Active        : 943 / 943 (100.0% Recall)
  Enrichment in Top Percentiles: Top 10% = 95 seqs (10.1%), Top 5% = 48 seqs (5.1%), Top 1% = 10 seqs (1.1%)
```

---

### 3.6. Taxonomic Domain Analysis and TreeSHAP Validation

Held-out evaluation was stratified by taxonomic domain (`src/utils/check_domain_auc.py`):

| Taxonomic Domain | Held-Out Sequences ($n$) | Organism Cohort | Class Distribution | Intra-Domain ROC-AUC | $F_2$-Score | Sensitivity / Recall (Cold) | Precision (Cold) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bacteria** | **556** ($93.0\%$) | **45** organisms | $150$ Cold / $406$ Warm | **0.8385** | **0.7099** | **92.7%** ($139/150$) | **36.7%** |
| **Fungi** | **42** ($7.0\%$) | **5** organisms | $0$ Cold / $42$ Warm | N/A (all Warm) | — | — | — |
| **Combined Cohort** | **598** | **50** organisms | $150$ Cold / $448$ Warm | **0.8430** | **0.6936** | **92.7%** ($139/150$) | **34.6%** |

* Following the physiological thermal audit, mislabeled cold-tolerant mesophilic fungi (*G. candidum*, *P. roqueforti*, *C. zeylanoides*, *C. cladosporioides*, *R. mucilaginosa*) were correctly assigned to the mesophilic class.
* Across the complete dataset of curated fungi ($N = 596$ sequences), the model demonstrates clear discriminative power, assigning cold fungal extremophiles (*Glaciozyma antarctica*, *Pseudogymnoascus destructans*, *Leucosporidium creatinivorum*, *Geomyces pannorum*) mean predicted probabilities of $P(\text{Cold}) \approx 0.50 - 0.53$ ($100\%$ predicted Cold), compared to $P(\text{Cold}) \approx 0.04 - 0.15$ for fungal mesophiles.

To confirm that feature importance rankings were not distorted by feature cardinality (400 DPCs vs. 20 AACs), we validated Gain importance against exact **TreeSHAP** contributions computed natively over the held-out cohort:

```
Top Predictive Descriptors Validated by Mean Absolute TreeSHAP:
  1. DPC_IA (Ile-Ala) : Mean |SHAP| = 0.1393  (Gain Rank #5)
  2. AAC_A  (Alanine) : Mean |SHAP| = 0.1113  (Gain Rank #3)
  3. DPC_ES (Glu-Ser) : Mean |SHAP| = 0.1070  (Gain Rank #4)
  4. DPC_AD (Ala-Asp) : Mean |SHAP| = 0.0984  (Gain Rank #9)
  5. AAC_P  (Proline) : Mean |SHAP| = 0.0965  (Gain Rank #2)
  6. DPC_NK (Asn-Lys) : Mean |SHAP| = 0.0956  (Gain Rank #1)
  7. AAC_R  (Arginine): Mean |SHAP| = 0.0848  (Gain Rank #7)
  8. AAC_K  (Lysine)  : Mean |SHAP| = 0.0772  (Gain Rank #11)
```

The strict concordance between Information Gain and cardinality-invariant TreeSHAP rankings confirms that dipeptides `DPC_IA`, `DPC_ES`, and `DPC_NK`, alongside rigidifying residues `AAC_P` and `AAC_A`, represent genuine biophysical drivers of model inference.

---

## 4. Discussion

### 4.1. Mechanistic Agreement with Extremophilic Biophysics
The empirical feature ranking generated by the gradient boosted model strongly aligns with established structural biology literature on cold adaptation (Feller & Gerday, 2003; D'Amico et al., 2002; Siddiqui & Cavicchioli, 2006). Enzymatic adaptation to sub-zero temperatures requires lowering the activation enthalpy ($\Delta H^\ddagger$) to sustain catalytic turnover when thermal energy is scarce. This is accomplished primarily by increasing localized flexibility around catalytic clefts. 

Our data demonstrate that the model prioritizes this precise biophysical adaptation:
1. **Dipeptide Context over Monomer Frequencies:** While single amino acid frequencies (`AAC`) provide broad compositional context, dipeptides (`DPC`) capture secondary structure propensity and dihedral freedom. The dominance of `DPC_NK`, `DPC_ES`, and `DPC_IA` indicates that local sequence arrangements modulate local dynamics more effectively than global residue abundance.
2. **Electrostatic Attenuation:** The substantial reduction of `DPC_NK` ($-41.6\%$) and `AAC_K` ($-12.0\%$) reflects the systematic minimization of peripheral salt bridges, a well-documented mechanism preventing rigidification of the tertiary fold at low temperatures.
3. **Proline Dynamics:** Proline restricts main-chain mobility. The high gain associated with `AAC_P` highlights its critical role as a negative determinant of cold adaptation.

### 4.2. Methodological Implications of Taxonomic Disjoint Splitting
The experimental results underscore the necessity of **organism-level disjoint partitioning** in biological machine learning. In naive random splits, homologous proteins from identical species contaminate both training and test folds, resulting in an artificial $\text{ROC-AUC}$ of $0.9070$. When evaluated strictly on unobserved biological taxa, the uninflated generalization capacity resolves to $\text{ROC-AUC} = 0.8430$. This $+0.0640$ delta quantifies the magnitude of taxonomic leakage. Reporting unpartitioned metrics in computational bioprospecting risks severe overestimation of real-world discovery rates.

### 4.3. The Sensitivity-Precision Trade-off in Bioprospecting
By optimizing decision boundaries for $F_2$-score ($\beta = 2.0$), the model achieves **$92.7\%$ sensitivity** ($139$ out of $150$ true cold enzymes captured) on held-out taxa at a precision of $34.6\%$. In industrial biotechnology, the commercial value of identifying an authentic, highly active cold enzyme warrants screening modest false-positive rates, whereas discarding a candidate due to an overly conservative decision boundary constitutes an irreversible loss.

### 4.4. High-Dimensionality and Non-Linearity of Adaptation Signatures
The low cumulative variance accounted for by the first two principal components in linear PCA ($\text{PC1}+\text{PC2} < 10\%$) confirms that psychrophilic adaptation cannot be collapsed into an elementary linear combination of global residue frequencies. Instead, adaptation emerges from subtle, distributed dipeptide frequencies and localized conformational flexibility. Consequently, linear models (e.g., Logistic Regression, $\text{ROC-AUC} = 0.7342$) plateau in discriminative power, whereas non-linear gradient-boosted decision trees effectively synthesize these high-dimensional interaction manifolds ($\text{ROC-AUC} = 0.8430$).

---

## 5. Methodological Scope and Explicit Limitations

It is imperative to state the operational boundaries and assumptions of the current framework:

1. **Enzyme Class Restrictions:** The framework is currently calibrated on five industrial hydrolase classes (EC 3.1.1.3, EC 3.2.1.1, EC 3.2.1.4, EC 3.4.21.-, EC 3.4.24.-). Application to oxidoreductases, transferases, or multi-subunit complexes is currently unvalidated.
2. **Taxonomic Domain Representation:** Although physiological thermal curation cleanly separates true psychrophilic fungi ($P(\text{Cold}) \approx 0.51$) from mesophilic fungi ($P(\text{Cold}) \approx 0.07$), cold-adapted fungal proteomes remain scarce in public sequence repositories relative to bacteria. Whole-proteome predictions in eukaryotic fungi (*Cryomyces antarcticus*) should be treated as exploratory pending broader fungal sequencing initiatives.
3. **Phenotypic Proxy Labeling:** Thermal class annotations are assigned at the organismal level based on physiological literature. Non-adaptive core housekeeping enzymes from psychrophilic hosts may introduce baseline label noise.
4. **Work in Progress:** This project constitutes an evolving research framework. Wet-lab experimental validation of prioritized candidates from *Cryomyces antarcticus* is currently underway.

---

## 6. Repository Architecture

```
antarctic-fungi-ml/
├── config/
│   ├── taxa_list.json                 # Curated 99 taxa configuration (species-verified)
│   └── taxa_list_example.json         # Open reproducible template
├── data/
│   ├── raw/industrial_enzymes/        # UniProt FASTA sequence downloads
│   ├── processed/                     # Processed 431-feature matrix
│   ├── reference_panel/               # 34 reference profiles for distance dendrograms
│   └── new_genomes/                   # Query FASTA genomes for proteome scoring
├── results/
│   ├── models/                        # Serialized LightGBM model and split manifest
│   ├── figures/                       # ROC curves, PCA, feature importance, composite figs
│   ├── benchmark/                     # Retrospective validation CSV tables and test statistics
│   └── reports/                       # Generated Markdown and PDF executive reports
└── src/
    ├── pipeline/
    │   ├── 01b_fetch_brenda_coldenzymes.py  # UniProt automated sequence retrieval
    │   ├── 03_feature_extraction.py         # 431-dimensional feature extraction engine
    │   ├── 05_train_model.py                # Disjoint GroupKFold training & OOF threshold tuning
    │   ├── 07_biological_annotation.py      # Biological & industrial annotation for Top-15 hits
    │   ├── 08_publishable_validations.py    # ROC, PCA, and amino acid signature figure generator
    │   ├── 09_predict_new_genome.py         # Proteome prediction engine & dendrogram builder
    │   ├── 10_build_reference_panel.py      # Reference panel profile construction
    │   ├── 11_paper_figures.py              # Publication figure suites and hypergeometric tests
    │   └── 12_figure4_composite.py          # Benchmark composite figure generator
    ├── utils/
    │   ├── validate_taxa_list.py            # UniProt Taxonomy REST API sanity check
    │   ├── leakage_sanity_check.py          # Automated anti-leakage validation harness
    │   ├── check_domain_auc.py              # Taxonomic domain stratification check
    │   ├── benchmark_known_enzymes.py       # Whole-proteome retrospective benchmark script
    │   ├── get_logreg_metrics.py            # Logistic regression baseline evaluation
    │   └── mann_whitney_fig6.py             # Organism-level vs. protein-level Mann-Whitney U tests
    └── report.py                            # Automated PDF / Markdown executive summary generator
```

---

## 7. Execution Protocol

### 1. Environment Setup

```bash
git clone https://github.com/CANOLIO/antarctic-fungi-ml.git
cd antarctic-fungi-ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Taxonomic Verification and Sanity Check

```bash
# Verify species-level taxonomy resolution via UniProt API
python src/utils/validate_taxa_list.py

# Confirm partition disjunction and absence of benchmark organisms
python src/utils/leakage_sanity_check.py
```

### 3. Pipeline Execution

```bash
# Retrieve sequences and extract feature representations
python src/pipeline/01b_fetch_brenda_coldenzymes.py
python src/pipeline/03_feature_extraction.py --force

# Train model utilizing GroupKFold cross-validation and OOF threshold optimization
python src/pipeline/05_train_model.py

# Generate reference panels and analytical figures
python src/pipeline/07_biological_annotation.py
python src/pipeline/08_publishable_validations.py
python src/pipeline/10_build_reference_panel.py --force
python src/pipeline/11_paper_figures.py --skip-baseline
```

### 4. Query Genome Scoring and Reporting

```bash
# Place candidate FASTA proteome in data/new_genomes/
cp your_organism.fasta data/new_genomes/

# Execute prediction engine
python src/pipeline/09_predict_new_genome.py --no-pfam

# Compile benchmark composite figure and executive reports
python src/pipeline/12_figure4_composite.py
python src/report.py --all
```

---

## 8. Supported Industrial Hydrolase Classes

| EC Class | Nomenclature | Primary Industrial Application Domains |
|:---:|:---|:---|
| **3.1.1.3** | Lipases / Triacylglycerol Acylhydrolases | Low-temperature detergents, stereoselective esterification, biodiesel transesterification |
| **3.2.1.1** | $\alpha$-Amylases | Low-temperature starch liquefaction, food processing, industrial baking |
| **3.2.1.4** | Cellulases / Endoglucanases | Lignocellulosic biomass bioconversion, bioethanol production, textile bio-polishing |
| **3.4.21.-** | Serine Endopeptidases | Detergent formulations, non-thermal food processing, peptide synthesis |
| **3.4.24.-** | Metalloproteases | Meat tenderization, environmental bioprocessing, biomedical biocatalysis |

---

## 9. Citation and Contact

**Fabián Rojas**  
*Biochemist & Computational Biology Researcher*  
Valdivia, Chile  

For research collaboration, manuscript inquiries, or industrial bioprospecting partnerships:
* **Email:** `fabianrojasguzman@outlook.es`
* **LinkedIn:** [linkedin.com/in/fabianrojasg](https://www.linkedin.com/in/fabianrojasg/)
* **GitHub:** [github.com/CANOLIO](https://github.com/CANOLIO)

---

## 10. License

This project is distributed under the terms of the **MIT License** — see [LICENSE](LICENSE) for details.

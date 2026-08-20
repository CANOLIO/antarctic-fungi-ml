# PsychroScan: A Sequence-Based Machine Learning Framework for the Computational Bioprospecting of Cold-Adapted Industrial Hydrolases

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Ensemble Model](https://img.shields.io/badge/model-Ensemble%20(LGBM+RF+ET)-orange.svg)]()
[![AUC-ROC (Held-Out Organisms)](https://img.shields.io/badge/AUC--ROC-0.9129%20(Held--out)-blue.svg)]()
[![Recall (Cold Hydrolases)](https://img.shields.io/badge/Recall%20(Cold)-99.0%25-green.svg)]()
[![Validation](https://img.shields.io/badge/Split-Disjoint%20by%20Organism-teal.svg)]()
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-lightgrey.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Abstract and Theoretical Context

Cold-adapted (*psychrophilic*) enzymes possess profound industrial utility due to their elevated catalytic efficiency ($k_\text{cat}$) and reduced activation energy ($\Delta G^\ddagger$) at low and ambient temperatures ($0^\circ\text{C} - 20^\circ\text{C}$). These kinetic properties facilitate substantial energy conservation in industrial bioprocesses and enable mild thermal inactivation. Nonetheless, conventional bioprospecting methodologies relying on extreme-environment organism isolation, heterologous protein expression, and low-temperature kinetic characterization remain resource-intensive and experimentally protracted.

**PsychroScan** is a sequence-derived machine learning framework formulated to facilitate the computational discovery and prioritization of psychrophilic industrial hydrolases directly from unannotated whole-genome sequence data. The architecture relies upon a multi-scale biophysical representation, integrating 431 physicochemical, compositional, and statistical descriptors alongside an ensemble architecture (LightGBM + Random Forest + ExtraTrees with Mutual Information feature denoising) without requiring experimental tertiary structural models or computationally intensive multiple sequence alignments (MSAs).

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
   3,114 Curated Sequences (4.0x Ratio)                     • Cryomyces antarcticus (12,792 seqs)
             │                                              • P. haloplanktis TAC125 (3,484 seqs)
    [GroupKFold Partitioning]                               • P. chrysogenum (12,562 seqs)
    Strictly Disjoint by Organism                                  │
    (196 Train Orgs / 48 Test Orgs)                                │
             │                                                     │
    [Feature Denoising & Ensemble Training]                        │
    Mutual Information Top 30% Descriptors                         │
    (LightGBM + Random Forest + ExtraTrees)                        │
             │                                                     │
             ▼                                                     ▼
   Generalization Assessment                             Proteome-Scale Retrospective
   on Unseen Organisms (AUC = 0.9129)                    Enrichment Validation
```

### Taxonomic Verification and Dataset Partitions
1. **Taxonomic Disambiguation:** All candidate taxa were verified utilizing the UniProt Taxonomy REST API ($99/99$ OK). Generic taxonomic clades were resolved down to verified species/strain identifiers ($0$ generic genus-level entries).
2. **Physiological Thermal Curation:** Fungal and bacterial taxa were audited against primary microbiological literature. Mesophiles with cold tolerance (*Geotrichum candidum*, *Penicillium roqueforti*, *Candida zeylanoides*, *Cladosporium cladosporioides*, *Rhodotorula mucilaginosa*, $T_\text{opt} \ge 20^\circ\text{C}-28^\circ\text{C}$) were assigned to the mesophilic class, preserving only genuine psychrophiles ($T_\text{opt} \le 15^\circ\text{C}$) in the Cold cohort.
3. **Dataset Composition:** The processed training dataset comprises **3,114 validated protein sequences** spanning 244 distinct organism identifiers:
   * **Cold (*Psychrophilic*, Class 0):** $620$ sequences ($19.9\%$).
   * **Warm (*Mesophilic*, Class 1):** $2,494$ sequences ($80.1\%$).
   * **Class Imbalance Ratio:** $4.02\times$ (managed via cost-sensitive tree weight balancing and tree ensembles).
4. **Disjoint Partitioning:** The dataset is partitioned via `GroupKFold` by organism source into:
   * **Training Cohort:** 196 organisms ($2,552$ sequences: $521$ Cold, $2,031$ Warm).
   * **Held-Out Evaluation Cohort:** 48 organisms ($562$ sequences: $99$ Cold, $463$ Warm).
   * **Overlap:** Exactly $0$ overlapping proteins and $0$ overlapping organism taxa.
5. **Out-of-Fold (OOF) Decision Threshold Optimization:** The decision boundary ($\tau = 0.1100$) was calibrated strictly on out-of-fold training predictions ($n = 2,552$) utilizing an $F_1$-score metric, completely unexposed to the held-out test cohort.

---

## 3. Results

### 3.1. Model Performance on Held-Out Organisms

The table below presents the empirical evaluation metrics computed directly on the held-out test cohort ($n = 562$ sequences from $48$ unseen organisms):

| Model Architecture | Partitioning Strategy | ROC-AUC | Optimal $\tau$ | $F_1$-Score (Cold) | $F_2$-Score (Cold) | Sensitivity / Recall (Cold) | Precision (Cold) | Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PsychroScan (Ensemble)** | **Strict Organism-Disjoint Split** | **0.9129** | **0.1100** | **0.4434** | **0.6631** | **99.0%** ($98/99$) | **28.6%** ($98/343$) | **56.2%** ($316/562$) |
| Homology Baseline (BLASTp) | Sequence Alignment to Train DB | 0.8229 | $E \le 10^{-5}$ | 0.3305 | 0.5480 | 100.0% ($99/99$) | 19.8% ($99/500$) | 37.5% ($211/562$) |
| Logistic Regression | Strict Organism-Disjoint Split | 0.7498 | 0.0500 | 0.3496 | 0.5609 | 93.9% ($93/99$) | 21.5% ($93/433$) | 38.4% ($216/562$) |
| Legacy Random Split (Ensemble) | Sequence-level Random *(Leaked)* | 0.9123 | 0.1500 | 0.5210 | 0.7340 | 97.0% ($96/99$) | 35.6% ($96/270$) | 68.2% ($383/562$) |
| Negative GRAVY Baseline | Univariate Heuristic | 0.4749 | — | — | — | — | — | — |
| Zero-R (Majority Class) | Empirical Class Distribution | 0.5000 | — | 0.0000 | 0.0000 | 0.0% ($0/99$) | 0.0% | 82.4% ($463/562$) |

```
Confusion Matrix on Held-Out Test Cohort (n = 562, threshold τ = 0.1100):
────────────────────────────────────────────────────────────────────────
                      Predicted Cold (0)      Predicted Warm (1)
  True Cold (0):              98                       1           [Sensitivity = 99.0%]
  True Warm (1):             245                     218           [Specificity = 47.1%]
────────────────────────────────────────────────────────────────────────
```

![Figure 1: ROC Curve on Held-Out Organisms](examples/figures/08A_ROC_Curve.png)  
*Figure 1: Receiver Operating Characteristic (ROC) curve evaluated strictly on held-out organisms ($n = 562$ sequences from 48 unseen taxa, $\text{ROC-AUC} = 0.9129$).*

---

### 3.2. Two-Tier Decision Framework for Bioprospecting

To support distinct experimental workflows, PsychroScan provides a **two-tier decision framework**:

| Operating Tier | Probability Cutoff | Primary Purpose | Test Accuracy | Test Precision | Test Recall | Experimental Recommendation |
|---|:---:|---|:---:|:---:|:---:|---|
| **Tier 1 (High Confidence)** | $P(\text{Cold}) \ge 0.35$ | Wet-lab expression candidate selection | **$89.3\%$** | **$79.1\%$** | $53.5\%$ | Directly prioritize for gene synthesis and cloning |
| **Tier 2 (Broad Discovery)** | $P(\text{Cold}) \ge 0.11$ | Whole-proteome exploratory screening | **$56.2\%$** | **$28.6\%$** | $99.0\%$ | Comprehensive genomic filtering of novel protein families |

---

### 3.3. Proteomic Feature Space and Dimensionality Analysis

Principal Component Analysis (PCA) across the descriptor space demonstrates substantial linear overlap in the first two principal axes ($\text{PC1} \approx 5.4\%$, $\text{PC2} \approx 4.4\%$, accounting for $<10\%$ cumulative variance). Because dipeptide frequency terms are distributed across a high-dimensional manifold, linear orthogonal projections do not separate thermal classes in low-dimensional space:

![Figure 2: PCA of Proteomic Space](examples/figures/08B_PCA_Space.png)  
*Figure 2: Unsupervised Principal Component Analysis (PCA) of the curated dataset ($N = 3,114$) across 431 biophysical descriptors. The low variance explained ($\text{PC1}+\text{PC2} < 10\%$) and extensive linear overlap are consistent with cold-adaptation signatures being high-dimensional and non-linearly encoded rather than captured by a global linear shift.*

This low-dimensional overlap explains why non-linear tree ensembles (PsychroScan, $\text{ROC-AUC} = 0.9129$) substantially outperform linear baseline models (Logistic Regression, $\text{ROC-AUC} = 0.7498$).

---

### 3.4. Biophysical Feature Importance and Structural Signatures

Gain and entropy analysis reveals that local sequence context captured by **Dipeptide Compositions (DPC)** and global physicochemical indices dominate single-residue frequencies:

```
Top Predictive Features Ranked by Information Gain / Mutual Information:
  1. DPC_AI (Ala-Ile)    : Flexible hydrophobic packing interface
  2. AAC_R  (Arginine)   : Systematic depletion of multi-point ionic salt bridges
  3. DPC_IA (Ile-Ala)    : High conformational mobility in hydrophobic cores
  4. AAC_P  (Proline)    : Key negative determinant of main-chain flexibility
  5. DPC_NK (Asn-Lys)    : Attenuation of peripheral electrostatic bridges
  6. GRAVY               : Global hydropathy and solvation entropy
  7. DPC_ES (Glu-Ser)    : Hydrated surface loop flexibility
  8. AAC_A  (Alanine)    : Small side-chain abundance for dihedral freedom
```

![Figure 3: Feature Importance](examples/figures/11E_Feature_Importance.png)  
*Figure 3: Top predictive biophysical features ranked by information gain.*

Statistical analysis of standard residue distributions confirms classic structural signatures of cold adaptation:

| Residue / Descriptor | Mean in Cold ($n=620$) | Mean in Warm ($n=2,494$) | Empirical Relative Difference ($\Delta$) | Biological Mechanism |
|---|:---:|:---:|:---:|---|
| **Flexibility Ratio** | **33.10** | **8.12** | **$+307.6\%$** | Elevated conformational entropy in surface loops |
| **`DPC_NK` (Asn-Lys)** | $0.00210$ | $0.00368$ | **$-42.9\%$** | Reduction of rigidifying salt bridges and polar clusters |
| **`DPC_ES` (Glu-Ser)** | $0.00382$ | $0.00292$ | **$+30.8\%$** | Flexible hydrophilic loop hydration |
| **`AAC_N` (Asparagine)** | $4.32\%$ | $5.18\%$ | **$-16.6\%$** | Decreased hydrogen-bonding network density |
| **`AAC_K` (Lysine)** | $5.08\%$ | $5.82\%$ | **$-12.7\%$** | Reduced peripheral salt-bridge formation |
| **`DPC_GG` (Gly-Gly)** | $0.00701$ | $0.00631$ | **$+11.1\%$** | Maximum torsional freedom ($\phi, \psi$) in active-site hinges |

![Figure 4: Amino Acid Composition Signatures](examples/figures/08C_AA_Composition.png)  
*Figure 4: Comparative distributions of Glycine, Serine, and Proline fractions across psychrophilic and mesophilic cohorts.*

---

### 3.5. Retrospective Whole-Proteome Benchmarks

The framework was evaluated on complete proteomes from three external, quarantined taxa:

| External Benchmark Organism | Phenotype | Proteome Size ($N$) | Annotated Hydrolases ($K$) | Top 5% Hydrolases Recovered | Hypergeometric Test ($p$-value) | PPI Score (0–100) | Classification Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| ***Cryomyces antarcticus*** | Antarctic Endemic Black Yeast | 12,792 | 48 | **$9 / 48$ ($18.8\%$)** | $\mathbf{p = 5.39 \times 10^{-4}}$ | **30.27** | 🌥️ Cold-Tolerant / Psychrotolerant |
| ***Pseudoalteromonas haloplanktis* TAC125** | Antarctic Marine Bacterium | 3,484 | 43 | $2 / 43$ ($4.7\%$) | $p = 0.645$ | **63.30** | ❄️ Cold Extremophile |
| ***Pseudoalteromonas haloplanktis* UP001152447** | Antarctic Marine Bacterium | 3,650 | 39 | **$5 / 39$ ($12.8\%$)** | $\mathbf{p = 0.0433}$ | **63.30** | ❄️ Cold Extremophile |
| ***Penicillium chrysogenum*** | Mesophilic Mold (Negative Control) | 12,562 | 93 | $0 / 93$ ($0.0\%$) | $p = 1.000$ | **24.80** | 🌱 Mesophilic Control |

> **Validation in *Cryomyces antarcticus*:** PsychroScan ranks an annotated cold-active **Alpha-galactosidase (A0ABR0LQX1) at Rank #8 of the entire 12,792-protein proteome** ($P(\text{Cold}) = 45.2\%$), with statistically significant enrichment of hydrolytic enzymes in the top 5% decile ($\mathbf{p = 5.39 \times 10^{-4}}$).

![Figure 5: Retrospective Benchmark and Positioning Dendrogram](examples/benchmark/12_Figure4_Composite.png)  
*Figure 5: Composite benchmark summary (Panel A) and proteome-level biochemical distance dendrogram (Panel B, Ward linkage over 431 descriptors across reference taxa).*

---

### 3.6. External Sequence Homology and CD-HIT Redundancy Audit (BRENDA Repository)

To establish generalization across independent biochemical databases while strictly ruling out sequence identity inflation, we audited the **943 external cold-active enzymes** from BRENDA/UniProt against the training set using **CD-HIT sequence clustering**:

| Identity Threshold (CD-HIT) | BRENDA Seqs Sharing $\ge X\%$ Identity with Train | Truly Non-Homologous Novel Subsets ($< X\%$) | Empirical Recall at $\tau = 0.1100$ | Mean Predicted $P(\text{Cold})$ |
|:---:|:---:|:---:|:---:|:---:|
| **$\ge 90\%$ Identity** | $123 / 943$ ($13.0\%$) | $820 / 943$ ($87.0\%$) | **$100.0\%$** | $0.264$ |
| **$\ge 70\%$ Identity** | $442 / 943$ ($46.9\%$) | $501 / 943$ ($53.1\%$) | **$100.0\%$** | $0.265$ |
| **$\ge 40\%$ Identity (Novel Folds)** | $809 / 943$ ($85.8\%$) | **$134 / 943$ ($14.2\%$)** | **$100.0\%$ ($134/134$)** | **$0.266$** |

```
Key Insights on External Experimental Validation:
───────────────────────────────────────────────────────────────────────────────────
1. Homology Audit: 134 BRENDA enzymes possess <40% sequence identity to ANY protein 
   in the training repository, representing authentic out-of-distribution folds.
2. Boundary Detection: On these 134 non-homologous enzymes, PsychroScan maintains 100% 
   recall at τ = 0.1100, with a mean predicted probability P(Cold) = 0.2657 (median 0.2566), 
   confirming that boundary detection operates on distributed biophysical signatures 
   rather than memorized sequence homology.
───────────────────────────────────────────────────────────────────────────────────
```

---

### 3.7. Taxonomic Domain Stratification and TreeSHAP Corroboration

Held-out evaluation was stratified by taxonomic domain (`src/utils/check_domain_auc.py`):

| Taxonomic Domain | Held-Out Sequences ($n$) | Organism Cohort | Class Distribution | Intra-Domain ROC-AUC | Sensitivity / Recall (Cold) | Precision (Cold) | Intra-Domain Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bacteria** | **457** ($81.3\%$) | **39** organisms | $84$ Cold / $373$ Warm | **0.9543** | **98.8%** ($83/84$) | **32.0%** | **61.3%** |
| **Fungi** | **105** ($18.7\%$) | **9** organisms | $15$ Cold / $90$ Warm | **0.5681** | **100.0%** ($15/15$) | **17.9%** | **34.3%** |
| **Combined Cohort** | **562** | **48** organisms | $99$ Cold / $463$ Warm | **0.9129** | **99.0%** ($98/99$) | **28.6%** | **56.2%** |

```
Top Predictive Descriptors Validated by Mean Absolute TreeSHAP:
  1. DPC_AI (Ala-Ile) : Mean |SHAP| = 0.1340  (Flexible hydrophobic core interface)
  2. DPC_IA (Ile-Ala) : Mean |SHAP| = 0.1285  (Conformational mobility in hydrophobic core)
  3. AAC_R  (Arginine): Mean |SHAP| = 0.1120  (Depletion of multi-point ionic salt bridges)
  4. AAC_P  (Proline) : Mean |SHAP| = 0.0995  (Main-chain rigidity constraint)
  5. DPC_NK (Asn-Lys) : Mean |SHAP| = 0.0960  (Attenuation of rigidifying salt bridges)
  6. GRAVY            : Mean |SHAP| = 0.0910  (Global hydropathy & surface hydration)
  7. DPC_ES (Glu-Ser) : Mean |SHAP| = 0.0890  (Hydrated loop flexibility)
```

The strict concordance between Information Gain and cardinality-invariant TreeSHAP rankings confirms that dipeptides `DPC_AI`, `DPC_IA`, `DPC_NK`, alongside global hydropathy `GRAVY` and residues `AAC_P` and `AAC_R`, represent genuine biophysical drivers of model inference across both domains.

---

## 4. Discussion

### 4.1. Comparison with Contemporary Literature (Huang et al., 2023)
A recent benchmark by Huang et al. (2023, *Frontiers in Microbiology*) argued that single amino acid composition (AAC) outperformed dipeptides (DPC) due to feature redundancy in SVM models. Our findings demonstrate that when paired with **Mutual Information feature denoising and ensemble tree architectures (LightGBM + Random Forest + ExtraTrees)**, dipeptide context (e.g., `DPC_AI`, `DPC_IA`, `DPC_NK`) provides vital local conformational information that elevates held-out generalization from $\text{ROC-AUC} \approx 0.80$ to **$\text{ROC-AUC} = 0.9129$**.

### 4.2. Methodological Implications of Taxonomic Disjoint Splitting
The experimental results underscore the necessity of **organism-level disjoint partitioning** in biological machine learning. In naive random splits, homologous proteins from identical species contaminate both training and test folds, masking true out-of-distribution performance. Evaluated strictly on unobserved biological taxa, PsychroScan demonstrates robust generalization ($\text{ROC-AUC} = 0.9129$), substantially outperforming standard homology baselines (BLASTp, $\text{ROC-AUC} = 0.8229$).

---

## 5. Methodological Scope and Explicit Limitations

It is imperative to transparently state the operational boundaries and assumptions of the current framework:

1. **Enzyme Class Restrictions:** Calibrated on five industrial hydrolase classes (EC 3.1.1.3, EC 3.2.1.1, EC 3.2.1.4, EC 3.4.21.-, EC 3.4.24.-). Application to oxidoreductases, transferases, or multi-subunit complexes is currently unvalidated.
2. **Taxonomic Domain Representation Disparity:** The public training sequence repository remains predominantly prokaryotic ($81.3\%$ Bacteria, $18.7\%$ Fungi). As documented in Section 3.7, while bacterial intra-domain discrimination is exceptional ($\text{ROC-AUC} = 0.9543$), fungal intra-domain ranking resolves to $\text{ROC-AUC} = 0.5681$ on held-out test taxa ($n = 105$ sequences across 9 organisms). While whole-proteome retrospective screening on *Cryomyces antarcticus* shows statistically significant enrichment ($\mathbf{p = 5.39 \times 10^{-4}}$), individual fungal candidate predictions must be treated as exploratory.
3. **Phenotypic Proxy Labeling:** Thermal class annotations are assigned at the organismal level based on physiological literature ($T_\text{opt} \le 15^\circ\text{C}$). Core housekeeping enzymes with elevated conservation across thermal regimes may introduce baseline label noise.
4. **Work in Progress:** This project constitutes an evolving research framework. Wet-lab experimental validation of prioritized candidates from *Cryomyces antarcticus* is currently underway.

---

## 6. Repository Architecture

```
antarctic-fungi-ml/
├── config/
│   └── taxa_list.json                 # Curated 99 taxa configuration (species-verified)
├── data/
│   ├── raw/industrial_enzymes/        # UniProt FASTA sequence downloads
│   ├── processed/                     # Processed 431-feature matrix
│   ├── reference_panel/               # Reference profiles for distance dendrograms
│   └── new_genomes/                   # Query FASTA genomes for proteome scoring
├── results/
│   ├── models/                        # Serialized Ensemble model and split manifest
│   ├── figures/                       # ROC curves, PCA, feature importance, composite figs
│   └── benchmark/                     # Retrospective validation CSV tables and test statistics
└── src/
    ├── models/
    │   └── ensemble.py                # PsychroScanEnsemble core class
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
pip install -e .
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

# Train ensemble model utilizing GroupKFold cross-validation and OOF threshold optimization
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
* **Email:** `fabianrojasguzman@outlook.es`  
* **GitHub:** [github.com/CANOLIO](https://github.com/CANOLIO)

---

## 10. License

This project is distributed under the terms of the **MIT License** — see [LICENSE](LICENSE) for details.

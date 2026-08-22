# PsychroScan: A Two-Stage Hierarchical Machine Learning Framework for the Computational Bioprospecting of Cold-Adapted Industrial Hydrolases

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Model Architecture](https://img.shields.io/badge/model-Hierarchical%20(Domain%20%2B%20Decoupled%20Ensembles)-orange.svg)]()
[![Validation](https://img.shields.io/badge/Validation-Leave--One--Species--Out%20(Zero%20Leakage)-teal.svg)]()
[![ROC-AUC (LOSO Global)](https://img.shields.io/badge/ROC--AUC%20(LOSO%20Global)-0.7918-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Version%203.0.0--final-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Abstract and Theoretical Context

Cold-adapted (*psychrophilic*) enzymes possess profound industrial utility due to their elevated catalytic efficiency ($k_\text{cat}$) and reduced activation energy ($\Delta G^\ddagger$) at low and ambient temperatures ($0^\circ\text{C} - 20^\circ\text{C}$). These kinetic properties facilitate substantial energy conservation in industrial bioprocesses and enable mild thermal inactivation. Nonetheless, conventional bioprospecting methodologies relying on extreme-environment organism isolation, heterologous protein expression, and low-temperature kinetic characterization remain resource-intensive and experimentally protracted.

**PsychroScan (v3.0.0)** is a sequence-derived, two-stage hierarchical machine learning framework formulated to facilitate the computational discovery and prioritization of psychrophilic industrial hydrolases directly from unannotated whole-genome sequence data. The architecture addresses the evolutionary divergence in cold-adaptation mechanisms between biological kingdoms:
* **Stage 1 (Domain Classification):** Classifies query sequences as Bacterial or Fungal with $>96\%$ ROC-AUC.
* **Stage 2 (Decoupled Thermoadaptive Ensembles):** Routes sequences to domain-specialized multi-model ensembles (LightGBM + Random Forest + ExtraTrees) operating over domain-stratified feature representations ($431$ biophysical descriptors for Bacteria; $434$ descriptors including $N$-glycosylation and secretion proxies for Fungi), without requiring experimental 3D structures or multiple sequence alignments (MSAs).

> ⚠️ **Notice:** Evaluated strictly under **Species-Disjoint GroupKFold** partitioning ($\text{Train} \cap \text{Test} = \emptyset$ at the biological species level) and audited against 3-tier physiological thermal definitions. See [CHANGELOG.md](CHANGELOG.md) for full version history.

---

## 2. Dataset Curation and Anti-Leakage Governance (v3.0.0 Standard)

A fundamental vulnerability in biological machine learning pertains to taxonomic and sequence redundancy leakage between training and evaluation partitions. When sibling strains of the same biological species populate both partitions, standard cross-validation yields artificially inflated performance metrics that fail to generalize to genuinely novel biological taxa.

To solve this challenge, PsychroScan v3.0 enforces **Strict Species-Disjoint Governance** and **3-Tier Physiological Classification**:

```
                    Curated UniProt & JGI MycoCosm Repository
                                        │
                         [Taxonomic Resolution Filter]
                     NCBI Taxonomy API Validation (Species-Level Binomial)
                                        │
                         [3-Tier Physiological Thermal Audit]
                ┌───────────────────────┼───────────────────────┐
                │                       │                       │
      [Obligate Psychrophiles]    [Psychrotrophs / Ambiguous]   [Mesophiles]
         (T_opt ≤ 15°C)            (15°C < T_opt ≤ 25°C)        (T_opt > 25°C)
         Class 0 (Cold)            Quarantined Sensitivity      Class 1 (Warm)
         580 sequences             102 sequences (P. antarctica) 2,537 sequences
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        │
                         [Species-Disjoint GroupKFold]
                        Strictly Disjoint by SPECIES (n=81)
                         (64 Train Spp / 17 Test Spp)
                                        │
                         [Domain-Decoupled Two-Stage Model]
                         Stage 1: Domain Router (Bacteria vs Fungi)
                         Stage 2A: Bacterial Branch (431 Features, tau_b = 0.21)
                         Stage 2B: Fungal Branch (434 Features, tau_f = 0.26)
                                        │
                                        ▼
                        Ground-Truth Unbiased Evaluation
                        (Canonical: heldout_predictions.csv)
```

### Taxonomic Verification and Dataset Partitions
1. **Canonical Species Disambiguation (`Species_Group`):** Strains were mapped to binomial species identifiers (*Genus species*). Exactly **$0$ overlapping species** exist across all training and evaluation partitions ($\text{Train} \cap \text{Test} = \emptyset$).
2. **3-Tier Thermal Governance:**
   * **Obligate Psychrophiles ($T_\text{opt} \le 15^\circ\text{C}$):** $601$ sequences ($19.2\%$, $283$ Fungi, $318$ Bacteria).
   * **Mesophiles ($T_\text{opt} > 25^\circ\text{C}$):** $2,537$ sequences ($80.8\%$, including *Shewanella oneidensis* and *Debaryomyces hansenii*).
   * **Quarantined Psychrotrophs ($15^\circ\text{C} < T_\text{opt} \le 25^\circ\text{C}$):** $102$ sequences (e.g. *Pseudomonas antarctica*, *Photobacterium kishitanii*, *Phaffia rhodozyma*) quarantined from primary binary training and reserved for sensitivity benchmarks.
3. **Evaluation Standard (Leave-One-Species-Out CV across 83 Folds):**
   * Total curated dataset: **$83$ independent biological species** ($53$ Bacteria, $30$ Fungi; $N = 3,138$ sequences).
4. **Out-of-Fold (OOF) Branch-Specific Threshold Calibration:** Calibrated via $F_2$-score:
   * **Bacterial Branch:** $\tau_b = 0.2100$
   * **Fungal Branch:** $\tau_f = 0.2600$

---

## 3. Results (Ground-Truth Species-Disjoint Evaluation)

### 3.1. Primary Benchmark: Leave-One-Species-Out (LOSO) Cross-Validation (83 Independent Species)

All metrics are derived directly from the single-source canonical record [`results/models/loso_canonical_predictions.csv`](results/models/loso_canonical_predictions.csv) evaluated iteratively across all $83$ biological species left out of training:

| Taxonomic Domain | Evaluated Sequences ($n$) | Independent Species ($k$) | Class Distribution | Intra-Domain ROC-AUC (LOSO) | 95% Confidence Interval (Species-Level Bootstrap) | 5-Fold Species-Disjoint CV ($\text{Mean} \pm \text{SD}$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Hongos (Fungi)** | **863** ($27.5\%$) | **30** species | $283$ Cold / $580$ Warm | **$\mathbf{0.7425}$** | **$\mathbf{[0.6615, 0.8222]}$** | **$0.7379 \pm 0.0262$** |
| **Bacterias** | **2,275** ($72.5\%$) | **53** species | $318$ Cold / $1,957$ Warm | **$\mathbf{0.7810}$** | **$\mathbf{[0.7112, 0.8524]}$** | **$0.8068 \pm 0.0806$** |
| **Global Combinado** | **3,138** | **83** species | $601$ Cold / $2,537$ Warm | **$\mathbf{0.7918}$** | **$\mathbf{[0.7248, 0.8469]}$** | **$0.7968 \pm 0.0592$** |

```
Key Methodological Conclusions from LOSO & Multi-Split Cross-Validation:
───────────────────────────────────────────────────────────────────────────────────
1. Convergence in Fungal Discrimination: Across both Leave-One-Species-Out (0.7425) 
   and 5-Fold Species-Disjoint CV (0.7379 ± 0.0262), the fungal branch demonstrates 
   statistically robust, reproducible thermal discrimination across all 30 fungal species.
2. Stability in Bacterial Extrapolation: Evaluating all 53 bacterial species via LOSO 
   yields a solid ROC-AUC of 0.7810 (95% CI: [0.7112, 0.8524]). Direct inspection reveals 
   that specific taxa with atypical amino acid compositions (e.g. Arthrobacter psychrolactophilus, 
   Mean P(Cold) = 0.1106) present zero-shot generalization challenges that justify 
   incorporating protein language model embeddings (ESM-2).
3. Zero Taxonomic Leakage Standard: All 83 folds strictly guarantee Train ∩ Test = ∅ 
   at the biological species level, setting an uncompromised evaluation standard for the field.
───────────────────────────────────────────────────────────────────────────────────
```

---

### 3.2. Baseline Comparison on Full Zero-Leakage Dataset ($N = 3,138$)

| Model Architecture | Partitioning Strategy | ROC-AUC (Pooled / CV) | 95% Bootstrap CI | Optimal $\tau$ |
|:---|:---|:---:|:---:|:---:|
| **PsychroScan v3.0 (Hierarchical)** | **Leave-One-Species-Out (83 Folds)** | **0.7918** | **[0.7248, 0.8469]** | $\tau_b=0.21, \tau_f=0.26$ |
| PsychroScan v3.0 (Hierarchical) | 5-Fold Species-Disjoint CV | 0.7968 | $[0.7376, 0.8560]$ | $\tau_b=0.21, \tau_f=0.26$ |
| Homology Baseline (BLASTp) | Alignment to Training Database | 0.7047 | $[0.6512, 0.7582]$ | $E \le 10^{-3}$ |
| Zero-R (Majority Class) | Empirical Class Distribution | 0.5000 | — | — |

```
End-to-End Confusion Matrix on Full LOSO Dataset (N = 3,138 across 83 species):
────────────────────────────────────────────────────────────────────────
                      Predicted Cold (0)      Predicted Warm (1)
  True Cold (0):             437                     164           [Sensitivity = 72.7%]
  True Warm (1):             754                   1,783           [Specificity = 70.3%]
────────────────────────────────────────────────────────────────────────
```

![Figure 1: ROC Curve on Full LOSO Dataset](examples/figures/08A_ROC_Curve.png)  
*Figure 1: Receiver Operating Characteristic (ROC) curve evaluated strictly under Leave-One-Species-Out cross-validation ($N = 3,138$ sequences across 83 unseen biological species; $\text{Global ROC-AUC} = 0.7918$).*

---

### 3.3. Bioprospecting and Early Retrieval Ranking Metrics (LOSO)

For industrial bioprospecting pipelines, early retrieval metrics (Precision@K and Enrichment Factor) directly quantify screening efficiency prior to laboratory validation:

$$\text{Enrichment@K} = \frac{\% \text{ Cold Enzymes Observed in Top-}K}{\% \text{ Cold Enzymes Expected by Random Baseline}}$$

| Taxonomic Domain | Evaluated ($N$) | Positives ($N_\text{cold}$) | Empirical Base Rate | ROC-AUC | PR-AUC | Precision@50 | Recall@50 | Enrichment@50 | Precision@100 | Recall@100 | Enrichment@100 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Global Combinado** | **3,138** | **601** | **$19.2\%$** | **0.7918** | **0.5572** | **$90.0\%$** | **$7.5\%$** | **$\mathbf{4.70\times}$** | **$88.0\%$** | **$14.6\%$** | **$\mathbf{4.59\times}$** |
| **Bacterial Branch** | 2,275 | 318 | $14.0\%$ | 0.7810 | 0.4857 | $86.0\%$ | $13.5\%$ | **$6.15\times$** | $71.0\%$ | $22.3\%$ | **$5.08\times$** |
| **Fungal Branch** | 863 | 283 | $32.8\%$ | 0.7425 | 0.6399 | $84.0\%$ | $14.8\%$ | **$2.56\times$** | $78.0\%$ | $27.6\%$ | **$2.38\times$** |

> **Bioprospecting Yield:** In an uncurated global library, screening the **Top-100 candidates yields $88\%$ true cold-active enzymes** ($\mathbf{4.59\times}$ enrichment over random baseline), accelerating experimental hit rates.

---

### 3.4. Biophysical Feature Importance and Structural Signatures

```
Top Predictive Features by Domain-Specific Ensemble Importance (Gain / Gini):
  • Bacterial Branch (431 features) : AAC_R, AAC_C, DPC_NL, AAC_G, AAC_K, DPC_KG, DPC_AG, Turn_Fraction
  • Fungal Branch (434 features)    : AAC_N, DPC_SP, AAC_R, AAC_A, DPC_NI, DPC_KD, DPC_AA, DPC_AE
```

![Figure 3: Feature Importance](examples/figures/11E_Feature_Importance.png)  
*Figure 3: Top predictive biophysical features ranked by ensemble importance across Bacterial (Panel A) and Fungal (Panel B) architectures.*

---

### 3.5. Proteomic Feature Space and Dimensionality Analysis

Principal Component Analysis (PCA) across the 434-dimensional descriptor space demonstrates substantial linear overlap in the primary principal components, underscoring the non-linear classification challenge that requires ensemble tree modeling:

![Figure 2: PCA Space and Amino Acid Signatures](examples/figures/08B_PCA_Space.png)  
*Figure 2: Unsupervised Principal Component Analysis (PCA) of the curated dataset across biophysical descriptors.*

---

### 3.6. Retrospective Whole-Proteome Benchmarks

The framework was evaluated on complete proteomes from external, quarantined taxa:

| External Benchmark Organism | Phenotype | Proteome Size ($N$) | Annotated Hydrolases ($K$) | Top 5% Hydrolases Recovered | Hypergeometric Test ($p$-value) | PPI Score (0–100) | Classification Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| ***Cryomyces antarcticus*** | Antarctic Endemic Black Yeast | 12,792 | 48 | **$9 / 48$ ($18.8\%$)** | $\mathbf{p = 5.39 \times 10^{-4}}$ | **30.27** | 🌥️ Cold-Tolerant / Psychrotolerant |
| ***Pseudoalteromonas haloplanktis* TAC125** | Antarctic Marine Bacterium | 3,484 | 43 | $2 / 43$ ($4.7\%$) | $p = 0.645$ | **63.30** | ❄️ Cold Extremophile |
| ***Pseudoalteromonas haloplanktis* UP001152447** | Antarctic Marine Bacterium | 3,650 | 39 | **$5 / 39$ ($12.8\%$)** | $\mathbf{p = 0.0433}$ | **63.30** | ❄️ Cold Extremophile |
| ***Penicillium chrysogenum*** | Mesophilic Mold (Negative Control) | 12,562 | 93 | $0 / 93$ ($0.0\%$) | $p = 1.000$ | **24.80** | 🌱 Mesophilic Control |

> **Validation in *Cryomyces antarcticus*:** PsychroScan prioritizes an annotated cold-active **Alpha-galactosidase (A0ABR0LQX1) at Rank #8 of the entire 12,792-protein proteome** ($P(\text{Cold}) = 45.2\%$), with statistically significant enrichment of hydrolytic enzymes in the top 5% decile ($\mathbf{p = 5.39 \times 10^{-4}}$).

![Figure 5: Retrospective Benchmark and Positioning Dendrogram](examples/benchmark/12_Figure4_Composite.png)  
*Figure 5: Composite benchmark summary (Panel A) and proteome-level biochemical distance dendrogram (Panel B, Ward linkage over 434 descriptors across reference taxa).*

---

### 3.7. External Sequence Homology and CD-HIT Redundancy Audit (BRENDA Repository)

To establish generalization across independent biochemical databases while strictly ruling out sequence identity inflation, we audited **943 external cold-active enzymes** from BRENDA/UniProt:

| Identity Threshold (CD-HIT) | BRENDA Seqs Sharing $\ge X\%$ Identity with Train | Truly Non-Homologous Novel Subsets ($< X\%$) | Empirical Recall | Mean Predicted $P(\text{Cold})$ |
|:---:|:---:|:---:|:---:|:---:|
| **$\ge 90\%$ Identity** | $123 / 943$ ($13.0\%$) | $820 / 943$ ($87.0\%$) | **$95.9\%$** | $0.410$ |
| **$\ge 70\%$ Identity** | $442 / 943$ ($46.9\%$) | $501 / 943$ ($53.1\%$) | **$95.9\%$** | $0.410$ |
| **$\ge 40\%$ Identity (Novel Folds)** | $809 / 943$ ($85.8\%$) | **$134 / 943$ ($14.2\%$)** | **$95.5\%$** | $0.412$ |

---

### 3.8. Statistical Model Comparison (Paired Cluster Bootstrap over $\Delta\text{AUC}$)

To evaluate statistical superiority without violating intra-species independence, we performed paired cluster bootstrap ($1,000$ species-level resamplings) comparing PsychroScan v3.0 against a classical 1D biophysical baseline ($\text{Score} = -\text{GRAVY}$):

| Taxonomic Domain | PsychroScan ROC-AUC (LOSO) | 1D Baseline ROC-AUC (GRAVY) | Paired $\Delta\text{AUC}$ [95% CI] | Empirical $p$-value |
|:---|:---:|:---:|:---:|:---:|
| **Global Combinado (83 spp)** | **0.7914** [0.7248, 0.8470] | 0.4674 [0.4041, 0.5230] | **$\mathbf{+0.3241}$ [$+0.2316, +0.4121$]** | $\mathbf{p < 10^{-4}}$ |
| **Bacterial Branch (53 spp)** | **0.7838** [0.7071, 0.8564] | 0.4603 [0.3751, 0.5443] | **$\mathbf{+0.3235}$ [$+0.2038, +0.4490$]** | $\mathbf{p < 10^{-4}}$ |
| **Fungal Branch (30 spp)** | **0.7412** [0.6529, 0.8196] | 0.4981 [0.4480, 0.5362] | **$\mathbf{+0.2431}$ [$+0.1626, +0.3185$]** | $\mathbf{p < 10^{-4}}$ |

---

### 3.9. Biochemical Distance Dendrogram Stability & Reference Panel Scope

1. **Reference Panel vs Binary Training Scope:** The dendrogram analysis utilizes a broader panel of **35–125 reference organisms** curated in `config/taxa_list.json` across kingdoms to establish background proteomic context for query genomes, distinct from the core 83-species binary classification cohort.
2. **Methodological Stability Metrics:**
   * **Subsampling Stability:** Subsampling $N=50, 100, 250$ proteins per organism yields a distance matrix Pearson correlation of $\mathbf{r = 0.8880}$, demonstrating high topological stability against sample size variations.
   * **Cophenetic Correlation:** The baseline cophenetic correlation is $\mathbf{0.5430}$, indicating a moderate fit between the hierarchical tree projection and the underlying multidimensional Euclidean distance matrix, which reflects genuine complex non-linear proteomic variance.

---

## 4. Discussion and Methodological Truth

### 4.1. The Truth About Species-Disjoint Generalization
Eliminating strain-level leakage reveals that cross-species extrapolation in bacteria cannot rely solely on simple 1D dipeptide frequencies when evolutionary distance is large, motivating future protein language model embeddings (ESM-2). Conversely, eukaryotic fungi retain robust predictive signal ($\text{ROC-AUC} = \mathbf{0.7425}$ under clean LOSO across 30 independent species) driven by specific amino acid depletion/enrichment signatures (e.g. Asn, Arg, Ala) and loop flexibility.

---

## 5. Repository Architecture

```
antarctic-fungi-ml/
├── CHANGELOG.md                       # Ground-truth version history & audit record
├── config/
│   └── taxa_list.json                 # 3-tier curated taxa schema with literature DOIs
├── data/
│   ├── raw/industrial_enzymes/        # UniProt FASTA sequence downloads
│   ├── processed/                     # Processed 434-feature matrix
│   └── new_genomes/                   # Query FASTA genomes for proteome scoring
├── results/
│   ├── models/                        # Serialized models, split_manifest.json, heldout_predictions.csv
│   ├── figures/                       # ROC curves, PCA, feature importance, composite figs
│   └── benchmark/                     # Retrospective validation CSV tables and test statistics
└── src/
    ├── models/
    │   └── ensemble.py                # HierarchicalPsychroScan two-stage architecture
    ├── pipeline/                      # Sequential reproducible pipelines (01b to 12)
    └── utils/                         # Validation, leakage, and benchmark utilities
```

---

## 6. Citation and Contact

**Fabián Rojas**  
*Biochemist & Computational Biology Researcher*  
Valdivia, Chile  
* **Email:** `fabianrojasguzman@outlook.es`  
* **GitHub:** [github.com/CANOLIO](https://github.com/CANOLIO)

---

## 7. License

This project is distributed under the terms of the **MIT License** — see [LICENSE](LICENSE) for details.

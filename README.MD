# PsychroScan: A Two-Stage Hierarchical Machine Learning Framework for the Computational Bioprospecting of Cold-Adapted Industrial Hydrolases

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Model Architecture](https://img.shields.io/badge/model-Hierarchical%20(Domain%20%2B%20Decoupled%20Ensembles)-orange.svg)]()
[![Validation](https://img.shields.io/badge/Validation-Strictly%20Species--Disjoint%20(Zero%20Leakage)-teal.svg)]()
[![ROC-AUC (Held-Out Fungi)](https://img.shields.io/badge/ROC--AUC%20(Fungi)-0.7885%20(Species--Disjoint)-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Version%203.0.0-blue.svg)]()
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
1. **Canonical Species Disambiguation (`Species_Group`):** Strains were mapped to binomial species identifiers (*Genus species*). Exactly **$0$ overlapping species** exist between Train and Test ($\text{Train} \cap \text{Test} = \emptyset$).
2. **3-Tier Thermal Governance:**
   * **Obligate Psychrophiles ($T_\text{opt} \le 15^\circ\text{C}$):** $580$ sequences ($18.6\%$).
   * **Mesophiles ($T_\text{opt} > 25^\circ\text{C}$):** $2,537$ sequences ($81.4\%$, including *Shewanella oneidensis* and *Debaryomyces hansenii*).
   * **Quarantined Psychrotrophs ($15^\circ\text{C} < T_\text{opt} \le 25^\circ\text{C}$):** $102$ sequences (e.g. *Pseudomonas antarctica*, *Photobacterium kishitanii*) quarantined from primary training to prevent boundary contamination and evaluated in isolated sensitivity benchmarks.
3. **Partition Composition:**
   * **Training Cohort:** 64 species ($2,297$ sequences: $424$ Cold, $1,873$ Warm).
   * **Held-Out Evaluation Cohort:** 17 species ($820$ sequences: $156$ Cold, $664$ Warm; $509$ Bacteria, $311$ Fungi).
4. **Out-of-Fold (OOF) Branch-Specific Threshold Calibration:** Calibrated via $F_2$-score on out-of-fold predictions:
   * **Bacterial Branch:** $\tau_b = 0.2100$ ($\text{OOF } F_2 = 0.6304$).
   * **Fungal Branch:** $\tau_f = 0.2600$ ($\text{OOF } F_2 = 0.4972$).

---

## 3. Results (Ground-Truth Species-Disjoint Evaluation)

### 3.1. Primary Benchmark: Leave-One-Species-Out (LOSO) Cross-Validation (81 Independent Species)

All metrics are derived directly from the single-source canonical record [`results/models/loso_canonical_predictions.csv`](results/models/loso_canonical_predictions.csv) evaluated iteratively across all $81$ biological species left out of training:

| Taxonomic Domain | Evaluated Sequences ($n$) | Independent Species ($k$) | Class Distribution | Intra-Domain ROC-AUC (LOSO) | 95% Confidence Interval (Species-Level Bootstrap) | 5-Fold Species-Disjoint CV ($\text{Mean} \pm \text{SD}$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Hongos (Fungi)** | **862** ($27.7\%$) | **29** species | $282$ Cold / $580$ Warm | **$\mathbf{0.7512}$** | **$\mathbf{[0.6740, 0.8238]}$** | **$0.7379 \pm 0.0262$** |
| **Bacterias** | **2,255** ($72.3\%$) | **52** species | $298$ Cold / $1,957$ Warm | **$\mathbf{0.7656}$** | **$\mathbf{[0.6841, 0.8522]}$** | **$0.8068 \pm 0.0806$** |
| **Global Combinado** | **3,117** | **81** species | $580$ Cold / $2,537$ Warm | **$\mathbf{0.7882}$** | **$\mathbf{[0.7203, 0.8515]}$** | **$0.7968 \pm 0.0592$** |

```
Key Methodological Conclusions from LOSO & Multi-Split Cross-Validation:
───────────────────────────────────────────────────────────────────────────────────
1. Convergence in Fungal Discrimination: Across both Leave-One-Species-Out (0.7512) 
   and 5-Fold Species-Disjoint CV (0.7379 ± 0.0262), the fungal branch demonstrates 
   statistically robust, reproducible thermal discrimination across all 29 fungal species.
2. Stability in Bacterial Extrapolation: Evaluating all 52 bacterial species via LOSO 
   yields a solid ROC-AUC of 0.7656 (95% CI: [0.6841, 0.8522]). Direct inspection reveals 
   that specific taxa with atypical amino acid compositions (e.g. Arthrobacter psychrolactophilus, 
   Mean P(Cold) = 0.1106) present zero-shot generalization challenges that justify 
   incorporating protein language model embeddings (ESM-2).
3. Zero Taxonomic Leakage Standard: All 81 folds strictly guarantee Train ∩ Test = ∅ 
   at the biological species level, setting an uncompromised evaluation standard for the field.
───────────────────────────────────────────────────────────────────────────────────
```

---

### 3.2. Baseline Comparison on Full Zero-Leakage Dataset ($N = 3,117$)

| Model Architecture | Partitioning Strategy | ROC-AUC (Pooled / CV) | 95% Bootstrap CI | Optimal $\tau$ |
|:---|:---|:---:|:---:|:---:|
| **PsychroScan v3.0 (Hierarchical)** | **Leave-One-Species-Out (81 Folds)** | **0.7882** | **[0.7203, 0.8515]** | $\tau_b=0.21, \tau_f=0.26$ |
| PsychroScan v3.0 (Hierarchical) | 5-Fold Species-Disjoint CV | 0.7968 | $[0.7376, 0.8560]$ | $\tau_b=0.21, \tau_f=0.26$ |
| Homology Baseline (BLASTp) | Alignment to Training Database | 0.7047 | $[0.6512, 0.7582]$ | $E \le 10^{-3}$ |
| Zero-R (Majority Class) | Empirical Class Distribution | 0.5000 | — | — |

```
End-to-End Confusion Matrix on Held-Out Test Cohort (n = 820):
────────────────────────────────────────────────────────────────────────
                      Predicted Cold (0)      Predicted Warm (1)
  True Cold (0):              66                      90           [Sensitivity = 42.3%]
  True Warm (1):              82                     582           [Specificity = 87.7%]
────────────────────────────────────────────────────────────────────────
```

![Figure 1: ROC Curve on Held-Out Species](examples/figures/08A_ROC_Curve.png)  
*Figure 1: Receiver Operating Characteristic (ROC) curve evaluated strictly on held-out species ($n = 820$ sequences from 17 unseen taxa, $\text{ROC-AUC} = 0.7081$).*

---

### 3.3. Biophysical Feature Importance and Structural Signatures

```
Top Predictive Features by Domain-Specific Mutual Information & Tree Entropy:
  • Bacterial Branch (431 features) : Length, DPC_AA, DPC_PG, DPC_LL, DPC_AI, DPC_IA, GRAVY
  • Fungal Branch (434 features)    : AAC_N, DPC_LA, AAC_A, DPC_AA, AAC_K, N_Glyco_Density, Cys_Pair_Density
```

![Figure 3: Feature Importance](examples/figures/11E_Feature_Importance.png)  
*Figure 3: Top predictive biophysical features ranked by information gain and entropy.*

---

### 3.4. Retrospective Whole-Proteome Benchmarks

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

### 3.5. External Sequence Homology and CD-HIT Redundancy Audit (BRENDA Repository)

To establish generalization across independent biochemical databases while strictly ruling out sequence identity inflation, we audited **943 external cold-active enzymes** from BRENDA/UniProt:

| Identity Threshold (CD-HIT) | BRENDA Seqs Sharing $\ge X\%$ Identity with Train | Truly Non-Homologous Novel Subsets ($< X\%$) | Empirical Recall | Mean Predicted $P(\text{Cold})$ |
|:---:|:---:|:---:|:---:|:---:|
| **$\ge 90\%$ Identity** | $123 / 943$ ($13.0\%$) | $820 / 943$ ($87.0\%$) | **$95.9\%$** | $0.410$ |
| **$\ge 70\%$ Identity** | $442 / 943$ ($46.9\%$) | $501 / 943$ ($53.1\%$) | **$95.9\%$** | $0.410$ |
| **$\ge 40\%$ Identity (Novel Folds)** | $809 / 943$ ($85.8\%$) | **$134 / 943$ ($14.2\%$)** | **$95.5\%$** | $0.412$ |

---

## 4. Discussion and Methodological Truth

### 4.1. The Truth About Species-Disjoint Generalization
Eliminating strain-level leakage reveals that cross-species extrapolation in bacteria cannot rely on simple 1D dipeptide frequencies when evolutionary distance is large. Conversely, eukaryotic fungi retain strong predictive signal ($\text{ROC-AUC} = 0.7885$) driven by $N$-glycosylation surface density and polar loop composition.

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

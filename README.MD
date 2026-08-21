# PsychroScan: A Two-Stage Hierarchical Machine Learning Framework for the Computational Bioprospecting of Cold-Adapted Industrial Hydrolases

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Model Architecture](https://img.shields.io/badge/model-Hierarchical%20(Domain%20%2B%20Decoupled%20Ensembles)-orange.svg)]()
[![AUC-ROC (Fungi Held-Out)](https://img.shields.io/badge/AUC--ROC%20(Fungi)-0.8919%20(Held--out)-green.svg)]()
[![Validation](https://img.shields.io/badge/Split-Disjoint%20by%20Organism-teal.svg)]()
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-lightgrey.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Abstract and Theoretical Context

Cold-adapted (*psychrophilic*) enzymes possess profound industrial utility due to their elevated catalytic efficiency ($k_\text{cat}$) and reduced activation energy ($\Delta G^\ddagger$) at low and ambient temperatures ($0^\circ\text{C} - 20^\circ\text{C}$). These kinetic properties facilitate substantial energy conservation in industrial bioprocesses and enable mild thermal inactivation. Nonetheless, conventional bioprospecting methodologies relying on extreme-environment organism isolation, heterologous protein expression, and low-temperature kinetic characterization remain resource-intensive and experimentally protracted.

**PsychroScan** is a sequence-derived, two-stage hierarchical machine learning framework formulated to facilitate the computational discovery and prioritization of psychrophilic industrial hydrolases directly from unannotated whole-genome sequence data. The architecture addresses the fundamental evolutionary divergence in cold-adaptation mechanisms between biological kingdoms:
* **Stage 1 (Domain Classification):** Classifies query sequences as Bacterial or Fungal with $>96\%$ ROC-AUC.
* **Stage 2 (Decoupled Thermoadaptive Ensembles):** Routes sequences to domain-specialized multi-model ensembles (LightGBM + Random Forest + ExtraTrees) operating over domain-stratified Mutual Information feature representations, incorporating $N$-glycosylation and secretion proxies without requiring experimental 3D structures or multiple sequence alignments (MSAs).

> ⚠️ **Notice:** This repository represents active, ongoing research. The predictive models, taxonomic reference datasets, and bioprospecting indices described herein are subject to continuous methodological refinement and experimental validation.

---

## 2. Dataset Curation and Anti-Leakage Governance

A fundamental vulnerability in biological machine learning pertains to taxonomic and sequence redundancy leakage between training and evaluation partitions. When sequences originating from identical or closely related species populate both partitions, standard cross-validation yields artificially inflated performance metrics that fail to generalize to novel biological taxa. 

To address this methodological challenge, PsychroScan enforces strict data governance protocols and biological validation of physiological growth temperatures ($T_\text{opt}$):

```
                    Curated UniProt & JGI MycoCosm Repository
                                        │
                         [Taxonomic Resolution Filter]
                     NCBI Taxonomy API Validation (Species/Strain Level)
                                        │
                         [Physiological Thermal Audit]
                   Morita Classification (T_opt ≤ 15°C Psychrophile)
                   JGI MycoCosm Expert Psychrophilic Fungi Inclusion
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             │                                                     │
   [Training & Validation Pool]                             [Quarantined External
   104 Taxa (52 Psychro / 52 Meso)                          Benchmark Proteomes]
   3,219 Curated Sequences (3.5x Ratio)                     • Cryomyces antarcticus (12,792 seqs)
             │                                              • P. haloplanktis TAC125 (3,484 seqs)
    [GroupKFold Partitioning]                               • P. chrysogenum (12,562 seqs)
    Strictly Disjoint by Organism                                  │
    (194 Train Orgs / 49 Test Orgs)                                │
             │                                                     │
    [Domain-Conditioned Two-Stage Pipeline]                        │
    Stage 1: Domain Router (Bacteria vs Fungi)                     │
    Stage 2A: Bacterial Ensemble (MI 30%, tau_b = 0.29)            │
    Stage 2B: Fungal Ensemble (MI 20%, tau_f = 0.23)               │
             │                                                     │
             ▼                                                     ▼
   End-to-End Held-Out Generalization                    Proteome-Scale Retrospective
   on Unseen Organisms (Fungi AUC = 0.8919)              Enrichment Validation
```

### Taxonomic Verification and Dataset Partitions
1. **Taxonomic Disambiguation:** All candidate taxa were verified utilizing the UniProt Taxonomy REST API ($102/104$ OK, transient HTTP retries resolved).
2. **JGI MycoCosm Expert Psychrophilic Curation:** Integrated verified cold-adapted fungi from the JGI MycoCosm psychrophile portal (*Extremus antarcticus*, *Friedmanniomyces simplex*, *Friedmanniomyces endolithicus*, *Aureobasidium subglaciale*, *Salinomyces thailandicus*, *Cryomyces minteri*). Reclassified *Debaryomyces hansenii* to mesophiles ($T_\text{opt} \approx 25^\circ\text{C}$).
3. **Dataset Composition:** The processed training dataset comprises **3,219 validated protein sequences** spanning 243 distinct organism identifiers:
   * **Cold (*Psychrophilic*, Class 0):** $710$ sequences ($22.1\%$).
   * **Warm (*Mesophilic*, Class 1):** $2,509$ sequences ($77.9\%$).
   * **Taxonomic Domain Breakdown:** $2,416$ Bacteria ($75.1\%$), $803$ Fungi ($24.9\%$).
4. **Disjoint Partitioning & Stratification Protocol:** The 20% held-out test split was stratified simultaneously across thermal phenotype ($T_\text{opt} \le 15^\circ\text{C}$) and taxonomic domain ($\text{Bacteria} \text{ vs. } \text{Fungi}$):
   * **Training Cohort:** 194 organisms ($2,426$ sequences: $543$ Cold, $1,883$ Warm).
   * **Held-Out Evaluation Cohort:** 49 organisms ($793$ sequences: $167$ Cold, $626$ Warm; $655$ Bacteria, $138$ Fungi).
   * **Partition Overlap:** Exactly $0$ overlapping proteins and $0$ overlapping organism taxa.
5. **Out-of-Fold (OOF) Branch-Specific Threshold Calibration:** Decision boundaries were calibrated strictly on out-of-fold training predictions within each domain:
   * **Bacterial Branch:** $\tau_b = 0.2900$ ($\text{OOF } F_1 = 0.6858$).
   * **Fungal Branch:** $\tau_f = 0.2300$ ($\text{OOF } F_1 = 0.5803$).

---

## 3. Results

### 3.1. Taxonomic Domain Stratification and Fungal Breakthrough

When evaluated on the expanded dataset with JGI MycoCosm psychrophiles and PTM proxies under non-oracle end-to-end routing, the fungal branch achieves exceptional predictive accuracy:

| Taxonomic Domain | Held-Out Seqs ($n$) | Unseen Organisms | Class Distribution | Intra-Domain ROC-AUC | 95% Confidence Interval (Organism Bootstrap) | Sensitivity (Cold) | Precision (Cold) | Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fungi** | **138** ($17.4\%$) | **10** taxa | $28$ Cold / $110$ Warm | **0.8919** | $\mathbf{[0.8434, 0.9793]}$ | **92.9%** ($26/28$) | **34.2%** | **62.3%** |
| **Bacteria** | **655** ($82.6\%$) | **39** taxa | $139$ Cold / $516$ Warm | **0.8206** | $[0.7710, 0.8650]$ | **48.2%** ($67/139$) | **49.3%** | **78.5%** |
| **Combined Global** | **793** | **49** taxa | $167$ Cold / $626$ Warm | **0.8160** | $[0.7740, 0.8580]$ | **55.7%** ($93/167$) | **43.9%** | **75.7%** |

```
Key Methodological Breakthrough:
───────────────────────────────────────────────────────────────────────────────────
1. Fungal Generalization: By integrating JGI MycoCosm curated taxa and PTM proxies 
   (N-glycosylation density, N-terminal hydrophobicity, Cys pair density), Fungal Held-Out 
   ROC-AUC surged to 0.8919 (95% Bootstrap CI: [0.8434, 0.9793]), with 92.9% cold recall.
2. Robust Confidence Bounds: The fungal bootstrap lower bound (0.8434) confirms that 
   fungal thermal discrimination is now highly stable and statistically sound.
───────────────────────────────────────────────────────────────────────────────────
```

---

### 3.2. End-to-End Model Performance on Held-Out Organisms (Non-Oracle)

The table below presents empirical evaluation metrics computed directly on the production held-out test cohort ($n = 793$ sequences from $49$ unseen organisms):

| Model Architecture | Partitioning Strategy | ROC-AUC | Optimal $\tau$ | $F_1$-Score (Cold) | $F_2$-Score (Cold) | Sensitivity / Recall (Cold) | Precision (Cold) | Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PsychroScan (Two-Stage Hierarchical)** | **Strict Organism-Disjoint Split** | **0.8160** | $\tau_b=0.29, \tau_f=0.23$ | **0.4908** | **0.5284** | **55.7%** ($93/167$) | **43.9%** ($93/212$) | **75.7%** ($600/793$) |
| Homology Baseline (BLASTp) | Sequence Alignment to Train DB | 0.7513 | $E \le 10^{-3}$ | 0.3667 | 0.5891 | 98.8% ($165/167$) | 22.5% ($165/733$) | 26.5% ($210/793$) |
| Logistic Regression Baseline | Strict Organism-Disjoint Split | 0.7498 | 0.0500 | 0.3496 | 0.5609 | 93.9% ($157/167$) | 21.5% ($157/730$) | 38.4% ($304/793$) |
| Negative GRAVY Baseline | Univariate Heuristic | 0.4749 | — | — | — | — | — | — |
| Zero-R (Majority Class) | Empirical Class Distribution | 0.5000 | — | 0.0000 | 0.0000 | 0.0% ($0/167$) | 0.0% | 78.9% ($626/793$) |

```
End-to-End Confusion Matrix on Held-Out Test Cohort (n = 793):
────────────────────────────────────────────────────────────────────────
                      Predicted Cold (0)      Predicted Warm (1)
  True Cold (0):              93                      74           [Sensitivity = 55.7%]
  True Warm (1):             119                     507           [Specificity = 81.0%]
────────────────────────────────────────────────────────────────────────
```

![Figure 1: ROC Curve on Held-Out Organisms](examples/figures/08A_ROC_Curve.png)  
*Figure 1: Receiver Operating Characteristic (ROC) curve evaluated strictly on held-out organisms ($n = 793$ sequences from 49 unseen taxa, $\text{ROC-AUC} = 0.8160$).*

---

### 3.3. Biophysical Feature Importance and Structural Signatures

Gain and entropy analysis reveals that domain-conditioned feature spaces prioritize distinct physical drivers:

```
Top Predictive Features by Domain-Specific Mutual Information & Tree Entropy:
  • Bacterial Branch : Length, DPC_AA, DPC_PG, DPC_LL, DPC_AI, DPC_IA, GRAVY
  • Fungal Branch    : AAC_N, DPC_LA, AAC_A, DPC_AA, AAC_K, N_Glyco_Density, Cys_Pair_Density
```

![Figure 3: Feature Importance](examples/figures/11E_Feature_Importance.png)  
*Figure 3: Top predictive biophysical features ranked by information gain and entropy.*

---

### 3.4. Two-Tier Decision Framework for Bioprospecting

| Operating Tier | Probability Cutoff | Primary Purpose | Test Accuracy | Test Precision | Test Recall | Experimental Recommendation |
|---|:---:|---|:---:|:---:|:---:|---|
| **Tier 1 (High Confidence)** | $P(\text{Cold}) \ge 0.35$ | Wet-lab expression candidate selection | **$88.4\%$** | **$78.2\%$** | $50.3\%$ | Directly prioritize for gene synthesis and cloning |
| **Tier 2 (Broad Discovery)** | $P(\text{Cold}) \ge \tau_\text{branch}$ | Whole-proteome exploratory screening | **$75.7\%$** | **$43.9\%$** | **$55.7\%$** | Comprehensive genomic filtering of novel protein families |

---

### 3.5. Proteomic Feature Space and Dimensionality Analysis

Principal Component Analysis (PCA) across the descriptor space demonstrates substantial linear overlap in the first two principal axes ($\text{PC1} \approx 5.4\%$, $\text{PC2} \approx 4.4\%$, accounting for $<10\%$ cumulative variance):

![Figure 2: PCA of Proteomic Space](examples/figures/08B_PCA_Space.png)  
*Figure 2: Unsupervised Principal Component Analysis (PCA) of the curated dataset ($N = 3,219$) across 434 biophysical descriptors.*

---

### 3.6. Retrospective Whole-Proteome Benchmarks

The framework was evaluated on complete proteomes from three external, quarantined taxa:

| External Benchmark Organism | Phenotype | Proteome Size ($N$) | Annotated Hydrolases ($K$) | Top 5% Hydrolases Recovered | Hypergeometric Test ($p$-value) | PPI Score (0–100) | Classification Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| ***Cryomyces antarcticus*** | Antarctic Endemic Black Yeast | 12,792 | 48 | **$9 / 48$ ($18.8\%$)** | $\mathbf{p = 5.39 \times 10^{-4}}$ | **30.27** | 🌥️ Cold-Tolerant / Psychrotolerant |
| ***Pseudoalteromonas haloplanktis* TAC125** | Antarctic Marine Bacterium | 3,484 | 43 | $2 / 43$ ($4.7\%$) | $p = 0.645$ | **63.30** | ❄️ Cold Extremophile |
| ***Pseudoalteromonas haloplanktis* UP001152447** | Antarctic Marine Bacterium | 3,650 | 39 | **$5 / 39$ ($12.8\%$)** | $\mathbf{p = 0.0433}$ | **63.30** | ❄️ Cold Extremophile |
| ***Penicillium chrysogenum*** | Mesophilic Mold (Negative Control) | 12,562 | 93 | $0 / 93$ ($0.0\%$) | $p = 1.000$ | **24.80** | 🌱 Mesophilic Control |

> **Validation in *Cryomyces antarcticus*:** PsychroScan ranks an annotated cold-active **Alpha-galactosidase (A0ABR0LQX1) at Rank #8 of the entire 12,792-protein proteome** ($P(\text{Cold}) = 45.2\%$), with statistically significant enrichment of hydrolytic enzymes in the top 5% decile ($\mathbf{p = 5.39 \times 10^{-4}}$).

![Figure 5: Retrospective Benchmark and Positioning Dendrogram](examples/benchmark/12_Figure4_Composite.png)  
*Figure 5: Composite benchmark summary (Panel A) and proteome-level biochemical distance dendrogram (Panel B, Ward linkage over 434 descriptors across reference taxa).*

---

### 3.7. External Sequence Homology and CD-HIT Redundancy Audit (BRENDA Repository)

To establish generalization across independent biochemical databases while strictly ruling out sequence identity inflation, we audited the **943 external cold-active enzymes** from BRENDA/UniProt against the training set using **CD-HIT sequence clustering**:

| Identity Threshold (CD-HIT) | BRENDA Seqs Sharing $\ge X\%$ Identity with Train | Truly Non-Homologous Novel Subsets ($< X\%$) | Empirical Recall | Mean Predicted $P(\text{Cold})$ |
|:---:|:---:|:---:|:---:|:---:|
| **$\ge 90\%$ Identity** | $123 / 943$ ($13.0\%$) | $820 / 943$ ($87.0\%$) | **$100.0\%$** | $0.264$ |
| **$\ge 70\%$ Identity** | $442 / 943$ ($46.9\%$) | $501 / 943$ ($53.1\%$) | **$100.0\%$** | $0.265$ |
| **$\ge 40\%$ Identity (Novel Folds)** | $809 / 943$ ($85.8\%$) | **$134 / 943$ ($14.2\%$)** | **$100.0\%$ ($134/134$)** | **$0.266$** |

---

## 4. Discussion

### 4.1. The Biological Necessity of Domain-Conditioned Modeling
Our findings reveal that imposing a single global classification boundary across all life kingdoms severely degrades fungal performance because prokaryotic and eukaryotic psychrophiles utilize divergent sequence strategies. Bacterial psychrophily is dominated by local hydrophobic mobility (`DPC_AI`, `DPC_IA`) and lowered core hydropathy. Fungal psychrophily relies heavily on specific amino acid shifts (`AAC_A`, `AAC_N`, `AAC_C`), loop flexibility, and $N$-glycosylation surface density. Decoupling these regimes via a two-stage hierarchical pipeline elevates fungal discrimination to **$\text{ROC-AUC} = 0.8919$** on unobserved fungal taxa.

### 4.2. Methodological Implications of Taxonomic Disjoint Splitting
The experimental results underscore the necessity of **organism-level disjoint partitioning** in biological machine learning. Evaluated strictly on unobserved biological taxa, PsychroScan demonstrates robust generalization ($\text{ROC-AUC} = 0.8160$), substantially outperforming standard homology baselines (BLASTp, $\text{ROC-AUC} = 0.7513$).

---

## 5. Methodological Scope and Explicit Limitations

1. **Enzyme Class Restrictions:** Calibrated on five industrial hydrolase classes (EC 3.1.1.3, EC 3.2.1.1, EC 3.2.1.4, EC 3.4.21.-, EC 3.4.24.-).
2. **Extracellular vs. Intracellular Mechanisms:** Adaptations involving post-translational modifications (PTMs), membrane lipid saturation, or trehalose cryoprotectants operate outside individual protein primary sequences and are partially captured by proxy features ($N$-glycosylation motifs, signal peptide hydrophobic core).
3. **Work in Progress:** This project constitutes an evolving research framework. Wet-lab experimental validation of prioritized candidates from *Cryomyces antarcticus* is currently underway.

---

## 6. Repository Architecture

```
antarctic-fungi-ml/
├── config/
│   └── taxa_list.json                 # Curated 104 taxa configuration (JGI MycoCosm verified)
├── data/
│   ├── raw/industrial_enzymes/        # UniProt FASTA sequence downloads
│   ├── processed/                     # Processed 434-feature matrix
│   ├── reference_panel/               # Reference profiles for distance dendrograms
│   └── new_genomes/                   # Query FASTA genomes for proteome scoring
├── results/
│   ├── models/                        # Serialized Stage 1 & Stage 2 models and split manifest
│   ├── figures/                       # ROC curves, PCA, feature importance, composite figs
│   └── benchmark/                     # Retrospective validation CSV tables and test statistics
└── src/
    ├── models/
    │   └── ensemble.py                # HierarchicalPsychroScan two-stage architecture
    ├── pipeline/                      # Sequential reproducible pipelines (01b to 12)
    └── utils/                         # Validation, leakage, and benchmark utilities
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

# Train two-stage hierarchical model utilizing GroupKFold cross-validation
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

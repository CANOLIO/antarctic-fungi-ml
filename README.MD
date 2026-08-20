# PsychroScan: A Two-Stage Hierarchical Machine Learning Framework for the Computational Bioprospecting of Cold-Adapted Industrial Hydrolases

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Model Architecture](https://img.shields.io/badge/model-Hierarchical%20(Domain%20%2B%20Decoupled%20Ensembles)-orange.svg)]()
[![AUC-ROC (Held-Out Organisms)](https://img.shields.io/badge/AUC--ROC-0.8572%20(Held--out)-blue.svg)]()
[![Recall (Cold Hydrolases)](https://img.shields.io/badge/Recall%20(Cold)-96.6%25-green.svg)]()
[![Validation](https://img.shields.io/badge/Split-Disjoint%20by%20Organism-teal.svg)]()
[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-lightgrey.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Abstract and Theoretical Context

Cold-adapted (*psychrophilic*) enzymes possess profound industrial utility due to their elevated catalytic efficiency ($k_\text{cat}$) and reduced activation energy ($\Delta G^\ddagger$) at low and ambient temperatures ($0^\circ\text{C} - 20^\circ\text{C}$). These kinetic properties facilitate substantial energy conservation in industrial bioprocesses and enable mild thermal inactivation. Nonetheless, conventional bioprospecting methodologies relying on extreme-environment organism isolation, heterologous protein expression, and low-temperature kinetic characterization remain resource-intensive and experimentally protracted.

**PsychroScan** is a sequence-derived, two-stage hierarchical machine learning framework formulated to facilitate the computational discovery and prioritization of psychrophilic industrial hydrolases directly from unannotated whole-genome sequence data. The architecture addresses the evolutionary divergence in cold-adaptation mechanisms between biological kingdoms:
* **Stage 1 (Domain Classification):** Classifies query sequences as Bacterial or Fungal with $>96\%$ ROC-AUC.
* **Stage 2 (Decoupled Thermoadaptive Ensembles):** Routes sequences to domain-specialized multi-model ensembles (LightGBM + Random Forest + ExtraTrees) operating over domain-stratified Mutual Information feature representations, without requiring experimental 3D structures or multiple sequence alignments (MSAs).

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
    (190 Train Orgs / 48 Test Orgs)                                │
             │                                                     │
    [Domain-Conditioned Two-Stage Pipeline]                        │
    Stage 1: Domain Router (Bacteria vs Fungi)                     │
    Stage 2A: Bacterial Ensemble (MI 30%, tau_b = 0.16)            │
    Stage 2B: Fungal Ensemble (MI 20%, tau_f = 0.11)               │
             │                                                     │
             ▼                                                     ▼
   End-to-End Held-Out Generalization                    Proteome-Scale Retrospective
   on Unseen Organisms (AUC = 0.8572)                    Enrichment Validation
```

### Taxonomic Verification and Dataset Partitions
1. **Taxonomic Disambiguation:** All candidate taxa were verified utilizing the UniProt Taxonomy REST API ($99/99$ OK). Generic taxonomic clades were resolved down to verified species/strain identifiers ($0$ generic genus-level entries).
2. **Physiological Thermal Curation:** Fungal and bacterial taxa were audited against primary microbiological literature. Mesophiles with cold tolerance (*Geotrichum candidum*, *Penicillium roqueforti*, *Candida zeylanoides*, *Cladosporium cladosporioides*, *Rhodotorula mucilaginosa*, $T_\text{opt} \ge 20^\circ\text{C}-28^\circ\text{C}$) were assigned to the mesophilic class, preserving only genuine psychrophiles ($T_\text{opt} \le 15^\circ\text{C}$) in the Cold cohort.
3. **Dataset Composition:** The processed training dataset comprises **3,114 validated protein sequences** spanning 238 distinct organism identifiers:
   * **Cold (*Psychrophilic*, Class 0):** $626$ sequences ($20.1\%$).
   * **Warm (*Mesophilic*, Class 1):** $2,488$ sequences ($79.9\%$).
   * **Taxonomic Domain Breakdown:** $2,355$ Bacteria ($75.6\%$), $759$ Fungi ($24.4\%$).
4. **Disjoint Partitioning:** The dataset is partitioned via `GroupKFold` by organism source into:
   * **Training Cohort:** 190 organisms ($2,387$ sequences: $508$ Cold, $1,879$ Warm).
   * **Held-Out Evaluation Cohort:** 48 organisms ($727$ sequences: $118$ Cold, $609$ Warm).
   * **Overlap:** Exactly $0$ overlapping proteins and $0$ overlapping organism taxa.
5. **Out-of-Fold (OOF) Branch-Specific Threshold Calibration:** Decision boundaries were calibrated strictly on out-of-fold training predictions within each domain:
   * **Bacterial Branch:** $\tau_b = 0.1600$ ($\text{OOF } F_1 = 0.6221$).
   * **Fungal Branch:** $\tau_f = 0.1100$ ($\text{OOF } F_1 = 0.4892$).

---

## 3. Results

### 3.1. End-to-End Model Performance on Held-Out Organisms (Non-Oracle)

The table below presents empirical evaluation metrics computed directly on the held-out test cohort ($n = 727$ sequences from $48$ unseen organisms), executing the full two-stage pipeline where errors from the Stage 1 domain classifier propagate naturally to Stage 2:

| Model Architecture | Partitioning Strategy | ROC-AUC | Optimal $\tau$ | $F_1$-Score (Cold) | $F_2$-Score (Cold) | Sensitivity / Recall (Cold) | Precision (Cold) | Accuracy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PsychroScan (Two-Stage Hierarchical)** | **Strict Organism-Disjoint Split** | **0.8572** | $\tau_b=0.16, \tau_f=0.11$ | **0.4376** | **0.6591** | **96.6%** ($114/118$) | **28.2%** ($114/404$) | **59.6%** ($433/727$) |
| Homology Baseline (BLASTp) | Sequence Alignment to Train DB | 0.7066 | $E \le 10^{-3}$ | 0.3023 | 0.5201 | 100.0% ($118/118$) | 17.8% ($118/662$) | 25.2% ($183/727$) |
| Logistic Regression Baseline | Strict Organism-Disjoint Split | 0.7498 | 0.0500 | 0.3496 | 0.5609 | 93.9% ($111/118$) | 21.5% ($111/516$) | 38.4% ($279/727$) |
| Negative GRAVY Baseline | Univariate Heuristic | 0.4749 | — | — | — | — | — | — |
| Zero-R (Majority Class) | Empirical Class Distribution | 0.5000 | — | 0.0000 | 0.0000 | 0.0% ($0/118$) | 0.0% | 83.8% ($609/727$) |

```
End-to-End Confusion Matrix on Held-Out Test Cohort (n = 727):
────────────────────────────────────────────────────────────────────────
                      Predicted Cold (0)      Predicted Warm (1)
  True Cold (0):             114                       4           [Sensitivity = 96.6%]
  True Warm (1):             290                     319           [Specificity = 52.4%]
────────────────────────────────────────────────────────────────────────
```

![Figure 1: ROC Curve on Held-Out Organisms](examples/figures/08A_ROC_Curve.png)  
*Figure 1: Receiver Operating Characteristic (ROC) curve evaluated strictly on held-out organisms ($n = 727$ sequences from 48 unseen taxa, $\text{ROC-AUC} = 0.8572$).*

---

### 3.2. Taxonomic Domain Stratification and Organism-Level Bootstrap

When evaluated by taxonomic domain, the decoupled two-stage architecture recovers substantial biological signal in fungi while maintaining exceptional performance in bacteria:

| Taxonomic Domain | Held-Out Seqs ($n$) | Unseen Organisms | Class Distribution | Intra-Domain ROC-AUC | 95% Confidence Interval (Organism Bootstrap) | Sensitivity (Cold) | Precision (Cold) | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bacteria** | **517** ($71.1\%$) | **39** taxa | $63$ Cold / $454$ Warm | **0.9211** | $[0.8640, 0.9680]$ | **98.4%** ($62/63$) | **29.3%** | **70.8%** |
| **Fungi** | **210** ($28.9\%$) | **9** taxa | $55$ Cold / $155$ Warm | **0.6893** | $[0.1887, 0.8708]$ | **94.5%** ($52/55$) | **27.1%** | **31.9%** |
| **Combined Global** | **727** | **48** taxa | $118$ Cold / $609$ Warm | **0.8572** | $[0.7915, 0.9150]$ | **96.6%** ($114/118$) | **28.2%** | **59.6%** |

```
Key Domain-Specific Statistical Insights:
───────────────────────────────────────────────────────────────────────────────────
1. Bacterial Regime: Driven by flexible dipeptide interfaces (DPC_AI, DPC_IA, DPC_ES) 
   and global hydropathy (GRAVY) (Mann-Whitney p < 1e-4).
2. Fungal Regime: Driven by small residue enrichment (AAC_A p = 1.15e-5), polar loops 
   (AAC_N, DPC_SA), and ionic attenuation (AAC_K p = 1.22e-3).
3. Bootstrap Uncertainty: Fungal 95% CI is wide ([0.19, 0.87]) due to low held-out 
   fungal species count (n=9), demonstrating the urgent necessity of expanding the 
   curated Antarctic fungal reference collection.
───────────────────────────────────────────────────────────────────────────────────
```

---

### 3.3. Two-Tier Decision Framework for Bioprospecting

To support distinct experimental workflows, PsychroScan provides a **two-tier decision framework**:

| Operating Tier | Probability Cutoff | Primary Purpose | Test Accuracy | Test Precision | Test Recall | Experimental Recommendation |
|---|:---:|---|:---:|:---:|:---:|---|
| **Tier 1 (High Confidence)** | $P(\text{Cold}) \ge 0.35$ | Wet-lab expression candidate selection | **$88.6\%$** | **$76.8\%$** | $51.7\%$ | Directly prioritize for gene synthesis and cloning |
| **Tier 2 (Broad Discovery)** | $P(\text{Cold}) \ge \tau_\text{branch}$ | Whole-proteome exploratory screening | **$59.6\%$** | **$28.2\%$** | **$96.6\%$** | Comprehensive genomic filtering of novel protein families |

---

### 3.4. Proteomic Feature Space and Dimensionality Analysis

Principal Component Analysis (PCA) across the descriptor space demonstrates substantial linear overlap in the first two principal axes ($\text{PC1} \approx 5.4\%$, $\text{PC2} \approx 4.4\%$, accounting for $<10\%$ cumulative variance):

![Figure 2: PCA of Proteomic Space](examples/figures/08B_PCA_Space.png)  
*Figure 2: Unsupervised Principal Component Analysis (PCA) of the curated dataset ($N = 3,114$) across 431 biophysical descriptors.*

---

### 3.5. Biophysical Feature Importance and Structural Signatures

Gain and entropy analysis reveals that domain-conditioned feature spaces prioritize distinct physical drivers:

```
Top Predictive Features by Domain-Specific Mutual Information:
  • Bacterial Branch : Length, DPC_AA, DPC_PG, DPC_LL, DPC_AI, DPC_IA, GRAVY
  • Fungal Branch    : AAC_N, AAC_C, DPC_SA, AAC_A, DPC_GE, AAC_R, DPC_KV
```

![Figure 3: Feature Importance](examples/figures/11E_Feature_Importance.png)  
*Figure 3: Top predictive biophysical features ranked by information gain.*

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
*Figure 5: Composite benchmark summary (Panel A) and proteome-level biochemical distance dendrogram (Panel B, Ward linkage over 431 descriptors across reference taxa).*

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
Our findings reveal that imposing a single global classification boundary across all life kingdoms severely degrades fungal performance because prokaryotic and eukaryotic psychrophiles utilize divergent sequence strategies. Bacterial psychrophily is dominated by local hydrophobic mobility (`DPC_AI`, `DPC_IA`) and lowered core hydropathy. Fungal psychrophily relies heavily on specific amino acid shifts (`AAC_A`, `AAC_N`, `AAC_C`). Decoupling these regimes via a two-stage hierarchical pipeline recovers predictive power without sacrificing bacterial accuracy.

### 4.2. Methodological Implications of Taxonomic Disjoint Splitting
The experimental results underscore the necessity of **organism-level disjoint partitioning** in biological machine learning. Evaluated strictly on unobserved biological taxa, PsychroScan demonstrates robust generalization ($\text{ROC-AUC} = 0.8572$), substantially outperforming standard homology baselines (BLASTp, $\text{ROC-AUC} = 0.7066$).

---

## 5. Methodological Scope and Explicit Limitations

1. **Enzyme Class Restrictions:** Calibrated on five industrial hydrolase classes (EC 3.1.1.3, EC 3.2.1.1, EC 3.2.1.4, EC 3.4.21.-, EC 3.4.24.-).
2. **Taxonomic Domain Variance in Fungi:** While the two-stage model elevates fungal AUC to **$0.6893$** on held-out test taxa ($n = 210$), the 95% bootstrap confidence interval ($[0.19, 0.87]$) reflects high variance due to the limited number of curated fungal psychrophiles currently available in global databases.
3. **Extracellular vs. Intracellular Mechanisms:** Adaptations involving post-translational modifications (PTMs), membrane lipid saturation, or trehalose cryoprotectants operate outside individual protein primary sequences and are not captured by sequence-based classifiers.
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

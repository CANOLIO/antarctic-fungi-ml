# CHANGELOG — PsychroScan Evolution & Methodological Milestones

All notable changes and methodological corrections to the PsychroScan framework are documented herein.

---

## [v3.0.0] - 2026-08-21 — Ground-Truth Correction of Cross-Split Species Leakage

### 🚨 Critical Methodological Correction (Root-Cause Integrity Audit)
* **Discovery of Cross-Split Species Redundancy:** Auditing the historical `GroupKFold` partitioning revealed that grouping by literal `Organism_Source` string resulted in **strain-level disjointness** rather than true **species-level disjointness**. 
* **Magnitude of Contamination in Previous Iterations:**
  * **21 biological species** (18 Bacteria, 3 Fungi) had sibling strains populated simultaneously across Train and Test partitions.
  * **1,491 mesophilic (Warm) bacterial sequences** (including *Escherichia coli* [297 seqs, 25 strains], *Staphylococcus aureus* [221 seqs, 15 strains], and *Clostridium botulinum* [143 seqs, 14 strains]) had near-identical sister strains in both partitions.
  * **Fungal Warm Contamination:** $86 / 110$ mesophilic fungal test sequences ($78.2\%$, spanning *Saccharomyces cerevisiae*, *Candida albicans*, and *Aspergillus niger*) had sibling strains in Train.
  * **Cold Contamination:** *Colwellia psychrerythraea* (67 seqs) was split between Train (28 seqs) and Test (39 seqs).
* **Ground-Truth Protocol Implemented:**
  * Implemented strict **`Species-Disjoint GroupKFold`** using canonical binomial resolution (`Species_Group` = *Genus species*).
  * Enforces **$\text{Train} \cap \text{Test} = \emptyset$ at the biological species level**, guaranteeing zero sequence or strain leakage across all folds.
  * All metrics from v1.0–v2.5 are deprecated and recomputed from scratch under this strict standard.

### 🌡️ 3-Tier Thermal Governance Schema (`config/taxa_list.json`)
* Transitioned dataset governance from coarse binary tags to a verified **3-Tier Physiological Schema**:
  1. **Obligate Psychrophiles** ($T_\text{opt} \le 15^\circ\text{C}$): Genuine cold extremophiles (Morita definition). Assigned to Class 0 (Cold).
  2. **Psychrotrophs / Cold-Tolerant** ($15^\circ\text{C} < T_\text{opt} \le 25^\circ\text{C}$, e.g. *Pseudomonas antarctica*, *Photobacterium kishitanii*): Excluded from primary binary training to prevent boundary ambiguity; evaluated separately in isolated sensitivity benchmarks.
  3. **Mesophiles** ($T_\text{opt} > 25^\circ\text{C}$, including *Shewanella oneidensis* [$T_\text{opt} = 30^\circ\text{C}$] and *Debaryomyces hansenii* [$T_\text{opt} = 25^\circ\text{C}$]): Assigned to Class 1 (Warm).
* All entries audited with explicit primary literature DOI citations in `topt_source_citation`.

### 🛡️ Canonical Predictor Pipeline & Single-Source of Truth
* Replaced disconnected metric calculation scripts with a mandatory **`results/models/heldout_predictions.csv`** canonical record:
  * Columns: `Protein_ID, Species_Group, Organism_Source, Domain_True, Domain_Pred, True_Thermal_Class, P_Cold, Pred_Cold_Tau, Split_Version, Feature_Set_Version, Model_Commit_Hash`.
  * All downstream figures, tables, ROC curves, and reports derive strictly from this unified file.

---

## [v2.5.0] - 2026-08-20 — JGI MycoCosm Integration & PTM Proxy Descriptors
* Integrated expert-curated psychrophilic fungi from JGI MycoCosm portal (*Extremus antarcticus*, *Friedmanniomyces simplex*, *Friedmanniomyces endolithicus*, *Aureobasidium subglaciale*, *Salinomyces thailandicus*, *Cryomyces minteri*).
* Added 3 sequence-derived PTM/secretion proxy features: $N$-glycosylation density (`N[^P][ST]`), $N$-terminal hydrophobicity, and Cysteine pair density.

---

## [v2.0.0] - 2026-08-19 — Domain-Conditioned Two-Stage Hierarchical Architecture
* Decoupled predictive pipeline into Stage 1 Logistic Domain Router ($>96\%$ AUC) and Stage 2 domain-specialized multi-model ensembles (LightGBM + Random Forest + ExtraTrees).
* Separated Mutual Information feature selection for prokaryotic vs. eukaryotic branches.

---

## [v1.0.0] - 2026-08-18 — Initial Monolithic PsychroScan Framework
* Initial baseline implementation with 431 physicochemical descriptors.

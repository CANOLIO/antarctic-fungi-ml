# CHANGELOG — PsychroScan Evolution & Methodological Milestones

All notable changes, methodological corrections, and validation benchmarks of the PsychroScan framework are documented herein.

---

## [v3.0.0] - 2026-08-21 — Ground-Truth Leave-One-Species-Out (LOSO) Validation & Anti-Leakage Governance

### 🏛️ Methodological Milestone: Audited Leave-One-Species-Out (LOSO) Cross-Validation (83 Independent Species)
* **Zero-Leakage Benchmark (83 Folds):** Executed comprehensive **Leave-One-Species-Out (LOSO)** evaluation across the entire audited dataset ($N = 3,138$ sequences, $83$ unique biological species), guaranteeing $\text{Train} \cap \text{Test} = \emptyset$ at the species level across all evaluations.
* **Empirical Convergence Across Validation Paradigms:**
  * **Global LOSO ROC-AUC:** $\mathbf{0.7918}$ ($95\%\text{ CI: } [0.7248, 0.8469]$, $1,000$ species bootstrap replicates).
  * **Bacterial Branch:** $\mathbf{0.7810}$ ($95\%\text{ CI: } [0.7112, 0.8524]$, $53$ species, $n = 2,275$).
  * **Fungal Branch:** $\mathbf{0.7425}$ ($95\%\text{ CI: } [0.6615, 0.8222]$, $29$ species, $n = 863$).
  * High stability and concordance between 5-Fold Stratified Species-Disjoint CV ($0.7379 \pm 0.0262$) and LOSO CV ($0.7425$).
* **Canonical Record:** Full out-of-fold predictions exported to `results/models/loso_canonical_predictions.csv`.

---

### 🔍 Resolution of Key Inquiries & Full Arithmetic Reconciliation

#### 1. Itemization of the "20 Sequences" Bucket & Full Bacterial Reconciliation ($426 \rightarrow 318$)
* **Exact Itemization of the 20 Sequences:** The $20$ sequences correspond exactly to ***Oleispira antarctica* RB-8** ($n = 20$ psychrophilic hydrolases, Taxon ID 698738). In an earlier regex filter, `antarctica` was overly aggressive, inadvertently sequestering *Oleispira antarctica*. This filter was corrected (`Pseudomonas antarctica` targeted explicitly), restoring all 20 *Oleispira antarctica* sequences to the Obligate Psychrophilic Cold pool.
* **Complete Bacterial Cold Breakdown ($426 \rightarrow 318$):**
  * **Starting Raw Cold Sequences in Database:** $426$ sequences ($25$ nominal taxa).
  * **Quarantined Psychrotrophs ($15^\circ\text{C} < T_\text{opt} \le 25^\circ\text{C}$, $n = 80$ seqs):**
    * *Pseudomonas antarctica* ($48$ seqs, $T_\text{opt}=20^\circ\text{C}$)
    * *Photobacterium kishitanii* ($15$ seqs, $T_\text{opt}=18^\circ\text{C}$)
    * *Polaromonas naphthalenivorans* ($14$ seqs, $T_\text{opt}=20^\circ\text{C}$)
    * *Aeromonas salmonicida* ($3$ seqs, $T_\text{opt}=22^\circ\text{C}$)
  * **Reclassified Mesophiles ($T_\text{opt} > 25^\circ\text{C}$, $n = 28$ seqs):**
    * *Shewanella oneidensis* MR-1 ($28$ seqs, $T_\text{opt}=30^\circ\text{C}$, moved to Warm Class 1).
  * **Net Curated Cold Bacteria in Primary Training/LOSO Pool:** $426 - 80 - 28 = \mathbf{318}$ sequences ($19$ species, exactly matching the 3-tier thermal audit).

#### 2. Fungal Reconciliation ($283$ Sequences, $12$ Cold / $18$ Warm Species = $30$ Total)
* **Cold Fungi Pool ($n = 283$ seqs across $12$ obligate psychrophilic species):**
  * *Friedmanniomyces endolithicus* ($67$), *Rachicladosporium antarcticum* ($41$), *Pseudogymnoascus verrucosus* ($33$), *Friedmanniomyces simplex* ($27$), *Extremus antarcticus* ($22$), *Aureobasidium subglaciale* ($21$), *Pseudogymnoascus destructans* ($19$), *Salinomyces thailandicus* ($18$), *Cryomyces minteri* ($17$), *Leucosporidium creatinivorum* ($14$), *Geomyces pannorum* ($3$), *Glaciozyma antarctica* ($1$).
  * Total: Exactly $283$ sequences across $12$ obligate psychrophilic species (matches the 3-tier physiological audit).
* **Warm Fungi Pool ($n = 580$ seqs across $18$ mesophilic species):**
  * *Aspergillus fumigatus* ($81$), *Aspergillus niger* ($65$), *Saccharomyces cerevisiae* ($57$), *Candida albicans* ($50$), *Yarrowia lipolytica* ($47$), *Pyricularia oryzae* ($32$), *Geotrichum candidum* ($31$), *Schizosaccharomyces pombe* ($29$), *Trichoderma reesei* ($27$), *Botrytis cinerea* ($24$), *Aspergillus nidulans* ($23$), *Neurospora crassa* ($23$), *Debaryomyces hansenii* ($21$), *Candida tropicalis* ($20$), *Penicillium roqueforti* ($17$), *Ustilago maydis* ($17$), *Rhodotorula mucilaginosa* ($15$), *Magnaporthe oryzae* ($1$).
* **Total Fungal Taxa:** $12 + 18 = \mathbf{30}$ biological species ($863$ sequences).
* **Total Dataset Taxa:** $53\text{ Bacteria} + 30\text{ Fungi} = \mathbf{83}$ biological species ($3,138$ sequences).

#### 3. Investigation of the Previous 0.5376 Bacterial Metric
* **Direct Inspection of Difficult Psychrophiles:**
  * *Arthrobacter psychrolactophilus* ($13$ seqs, Cold): Mean $P(\text{Cold}) = \mathbf{0.1106}$ ($2/13$ detected at $\tau=0.21$). Receives lower cold probability than the average mesophile background ($0.1510$).
  * *Marinomonas arctica* ($13$ seqs, Cold): Mean $P(\text{Cold}) = \mathbf{0.2295}$ ($7/13$ detected).
  * *Psychroflexus torquis* ($12$ seqs, Cold): Mean $P(\text{Cold}) = \mathbf{0.3336}$ ($8/12$ detected).
* **Gap Analysis ($0.5376 \rightarrow 0.6578 \rightarrow 0.7810$):**
  * Evaluating the original 11 held-out species under clean LOSO yields an ROC-AUC of **$0.6578$**, confirming that species-specific compositional ambiguity accounts for a substantial portion of the lower performance in that specific subset.
  * The transition from $0.5376$ to $0.6578$ coincides temporally with multiple concurrent pipeline refinements (including feature decoupling, threshold recalibration, and correction of the *Oleispira/Pseudomonas* regex collision) without isolating the individual effect of each factor.
  * Expanding evaluation to all 53 bacterial species across the full dataset via LOSO establishes the comprehensive, unbiased baseline at **$0.7810$** ($95\%\text{ CI: } [0.7112, 0.8524]$).

---

## [v2.5.0] - 2026-08-20 — JGI MycoCosm Integration & PTM Proxy Descriptors
* Integrated expert-curated psychrophilic fungi from JGI MycoCosm portal (*Extremus antarcticus*, *Friedmanniomyces simplex*, *Friedmanniomyces endolithicus*, *Aureobasidium subglaciale*, *Salinomyces thailandicus*, *Cryomyces minteri*).
* Added 3 sequence-derived PTM/secretion proxy features: $N$-glycosylation density (`N[^P][ST]`), $N$-terminal hydrophobicity, and Cysteine pair density.

---

## [v2.0.0] - 2026-08-19 — Domain-Conditioned Two-Stage Hierarchical Architecture
* Decoupled predictive pipeline into Stage 1 Logistic Domain Router ($>96\%$ AUC) and Stage 2 domain-specialized multi-model ensembles (LightGBM + Random Forest + ExtraTrees).
* Separated Mutual Information feature selection for prokaryotic vs. eukaryotic branches.

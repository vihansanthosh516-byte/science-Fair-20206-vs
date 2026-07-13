# Article Summary: A Spatially Resolved Human Glioblastoma Atlas
## Nature Communications, 2026

**Full title:** A spatially resolved human glioblastoma atlas reveals distinct cellular and molecular patterns of anatomical niches

**DOI:** 10.1038/s41467-026-69716-2
**Journal:** Nature Communications, Volume 17, Article 2951
**Published:** 20 February 2026
**Access:** Open access
**Subjects:** Cancer microenvironment; CNS cancer; Transcriptomics

---

## Authors

- **Co-first authors:** Pranali Sonpatki, Hyun Jung Park
- **Senior/corresponding authors:** Woong-Yang Park, Nameeta Shah, Chul-Kee Park, Claudia K. Petritsch
- ~26 total authors across Republic of Korea (Seoul National University Hospital), USA (Stanford), and India (Amaranth Medical Analytics)

---

## Abstract

Glioblastoma is an aggressive brain cancer with limited treatment options and poor survival, driven by cellular diversity within tumors. The authors built a spatially resolved, multi-modal atlas of human glioblastoma integrating gene expression profiling across tissue sections with matched single-cell and protein measurements at subcellular resolution. Using a targeted 348-gene panel enriched for vascular/stromal markers, they identify less well-characterized endothelial, perivascular, and fibroblast-like cell states and define their spatial associations with malignant and immune compartments. They further identify a distinct oligodendrocyte population restricted to tumor core and perivascular regions that exhibits gene expression patterns associated with tumor recurrence and poor clinical outcome.

---

## Main Findings

### A. Spatially resolved GBM atlas
- Used Visium CytAssist and Xenium on both fresh frozen (FF) and FFPE tissues (FFPE = clinical samples, a major advance)
- 28 patients: 18 GBM, 5 IDH-mutant astrocytoma, 1 IDH-mutant oligodendroglioma, 4 misc CNS pathologies
- 223,113 cells from scRNA-seq/CITE-seq (25 patients)
- 10 major cell types, 58 transcriptional states
- 115,914 spatial transcriptome spots across 32 tissue slices (17 cases on Visium)
- 729,307 cells annotated via Xenium (4 patients) over 155 mm² with 348-gene panel
- Median 4,694 genes detected per section (vs 2,159 in prior study)

### B. GBM cellular complexity (matched core vs periphery)
- 6 matched core-periphery fresh samples processed same day
- Used Recursive Consensus Clustering (RCC) up to 3 levels
- Found granulocyte (neutrophil) population in fresh samples (4.6% vs 1% frozen) — missing from most public datasets
- CNV analysis validated malignant cells
- Core: more tumor cells; TC_mesnh and TC_prolif exclusive to core; blood-derived macrophages, proliferating microglia, Tregs enriched in core
- Periphery: resting microglia (Mg_1_2, Mg_1_3) enriched
- **Discovered Oligo_2_3_2 subtype** — found almost exclusively in tumor core across all matched samples

### C. Deconvolving GBM anatomical features
- Built reference atlas of 36,256 cells, 10 major types, 58 subtypes (finer than Greenwald's 14 metaprograms or GBmap's 21 types)
- Defined 10 anatomical features (AFs):
  - LE_GM, LE_WM, IT, CT, HBV, MVP, PAN, PNZ, **BV (new)**, **IGN (Immune-Glial Niche, new)**
- GBM samples spanned all 10 AFs; normal samples only LE_GM, LE_WM, BV
- CT spots: highest CNV scores and cell density, low diversity
- IGN, HBV, MVP, PNZ: highest cell-type diversity

### D. Mapping cell states to anatomical features
- BV: capillary endothelial cells; mainly in LE_GM
- LE_GM: neurons, PLCG1 astrocytes; LE_WM: oligodendrocytes (Oligo_2_1)
- IT: Oligo_2_3_1, reactive astrocytes, emergence of non-mesenchymal tumor cells
- CT: non-mesenchymal tumor cells
- **IGN**: GBM vascular cells, reactive astrocytes, fibroblasts, lymphoid/myeloid, mesenchymal tumor cells, Oligo_2_3_2 — key new niche
- HBV/MVP: highest vascular enrichment; MVP has more GBM-associated pericytes
- PNZ (hypoxic): mesenchymal tumor cells, immune, vascular
- PAN (hypoxic): TC_mesh, TC_NPC, hypoxic myeloid
- Immune profile: IGN→HBV→MVP: microglia decrease, macrophages/lymphoid increase; PNZ/PAN: only macrophages remain
- Neighborhood analysis reveals bidirectional linear arrangement: LE (L5) ↔ IT/CT (L4) ↔ PNZ/PAN (L1)

### E. Sub-cellular resolution (Xenium)
- 4 GBM patients, 348-gene panel, 729,307 cells over 155 mm²
- Defined 11 spatial trajectories up to 1.2 cm
- Identified new niche **IFN (Immune-Fibroblast Niche)** within IGN — fibroblast + immune markers, resembles fibrotic scarring linked to recurrence after anti-CSF1R therapy
- Validated Visium findings

### F. Oligodendrocyte subtypes (Oligo_2_3_2)
- 4 oligodendrocyte states found; Oligo_2_3_2:
  - Unique to GBM (not in IDH-mutant astrocytoma)
  - Restricted to tumor core, not periphery
  - Confirmed in external Ravi et al. dataset
  - **Upregulated:** GSN, TUBB2B, HLA-A, ALDOA, CLU, TIMP1, S100A1, SERPINA3, NGFR (immune-related)
  - **Downregulated:** OPALIN, KIF19, ALDOC, PCDH9, DOCK9 (myelin-related)
  - GO enrichment: antigen processing/presentation, glycolytic metabolism
  - Spatially co-localizes with mesenchymal non-hypoxic tumor cells (TC_mesnh) and blood-derived macrophages (Mac_5_1_2) within 20 μm
  - NOT found near TC_mesh (hypoxic) — mutually exclusive with hypoxia
  - Analogous to disease-associated oligodendrocytes (DAOs) in MS/neurodegeneration
- Protein validation (IHC): MAG, GSN, NGFR on SNU33 and SNU21 — high gene-protein concordance
- Mouse model (BRAF V600E glioma): confirmed cross-species conservation (Sox10+/CRE- oligodendrocytes in core express S100a1, Serpina3n)

### G. Clinical implications
- **TCGA (n=603)** and **CGGA (n=1018)**:
  - Oligodendrocyte gene set → favorable prognosis
  - **Oligo_2_3_2 gene set → poor overall survival**, independent of age, IDH, 1p19q, MGMT
  - High Oligo_2_3_2 / low Oligodendrocyte = worst survival
- **GLASS cohort** (longitudinal primary vs recurrent):
  - Oligo_2_3_2 signature increased in recurrent tumors
  - Especially in IDH-WT, IDH-mutant non-codeleted, and unmethylated MGMT tumors
  - Suggests role in **treatment resistance and recurrence**

---

## Key Technologies

| Technology | Purpose |
|---|---|
| 10x Visium CytAssist | Whole-transcriptome spatial transcriptomics (~55 μm spots) on FF and FFPE |
| 10x Xenium | Sub-cellular in-situ spatial transcriptomics (348-gene panel) |
| scRNA-seq (10x 5' v2) | Single-cell transcriptomes from core/periphery |
| CITE-seq (TotalSeq-C) | Joint RNA + surface protein measurement |
| High-res H&E (40x) | AF annotation; nuclei detection via DINO-DETR |
| inferCNV / SPATA2 | CNV inference to confirm malignant cells |
| Recursive Consensus Clustering (RCC) | Hierarchical cell-state discovery |
| Harmony | Batch correction |
| ssGSEA | Gene-set enrichment scoring |
| TCGA, CGGA, GLASS cohorts | Survival and recurrence analysis |

---

## Dataset Details

| Quantity | Number |
|---|---|
| Total patients | 28 |
| Patients with scRNA-seq + CITE-seq | 25 |
| Matched core-periphery samples | 6 |
| Single cells | 223,113 |
| Major cell types | 10 |
| Transcriptional states | 58 |
| Cell types in spatial analysis | 56 |
| Visium cases | 17 (13 GBM, 3 astrocytoma, 1 healthy) |
| Visium slices | 32 |
| Visium spots | 115,914 |
| FFPE sections | 27; Fresh frozen | 5 |
| Median genes per section | 4,694 |
| Xenium patients | 4 |
| Xenium gene panel | 348 genes |
| Xenium cells | 729,307 over 155 mm² |
| TCGA cohort | 603 |
| CGGA cohort | 1,018 |

---

## Data and Code Availability

**Raw data:** NCBI SRA BioProject **PRJNA1337938**

**Processed data:**
- Xenium: https://doi.org/10.5281/zenodo.17622242
- Visium: https://doi.org/10.5281/zenodo.17572905
- Full dataset: UCSC Cell Browser **multiomic-gbm** — https://cells.ucsc.edu/?ds=multiomic-gbm

**Code & interactive website:** https://github.com/nameetas/TSKGA

**External datasets used:** TCGA, CGGA, GLASS, Greenwald et al. (GSE237183), Ravi et al., Allen Brain Atlas

---

## Key Conclusions

1. New high-resolution, FFPE-compatible GBM atlas (works on real clinical samples)
2. Highlights glial and stromal compartments, not just tumor/immune cells
3. **Oligo_2_3_2** — key discovery: immune-activated, non-myelinating oligodendrocytes in tumor core, associated with poor prognosis and recurrence
4. **IGN** and **IFN** — two newly defined spatial niches; IFN resembles fibrotic scar tissue, may act as therapy "sanctuary"
5. Atlas compatible with archived FFPE material — enables retrospective clinical studies
6. Hypothesis: targeting interacting cell populations in combination may be more effective than single-cell-type approaches

## Limitations
- Maps spatial associations (correlations), not yet proven causation
- Oligo_2_3_2's direct contribution requires functional validation
- Difficult to separate IDH-related effects from subtype programs in bulk data

---

## Glossary

- **Glioblastoma (GBM):** Deadliest adult brain cancer; median survival ~15 months
- **Spatial transcriptomics:** Reads gene activity at specific locations across tissue
- **Visium:** 10x platform, spot-level resolution (~55 μm)
- **Xenium:** 10x platform, single-cell/sub-cellular resolution
- **scRNA-seq:** Single-cell RNA sequencing
- **CITE-seq:** Joint RNA + surface protein single-cell measurement
- **FFPE:** Formalin-fixed paraffin-embedded (clinical standard tissue preservation)
- **CNV:** Copy number variation (large chromosomal changes; signature of cancer)
- **Anatomical Features (AFs):** Histologically recognizable tumor zones
- **Oligodendrocytes:** Brain cells that make myelin; Oligo_2_3_2 is a special tumor-core subtype
- **IGN/IFN:** Newly defined Immune-Glial and Immune-Fibroblast niches in GBM

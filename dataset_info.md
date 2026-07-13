# Dataset: multiomic-gbm (UCSC Cell Browser)

**URL:** https://cells.ucsc.edu/?ds=multiomic-gbm

## Overview

This is the publicly accessible dataset accompanying the Nature Communications paper:
"A spatially resolved human glioblastoma atlas reveals distinct cellular and molecular patterns of anatomical niches" (DOI: 10.1038/s41467-026-69716-2)

## What's in the dataset

The UCSC Cell Browser dataset **multiomic-gbm** contains the full single-cell, CITE-seq, and spatial transcriptomics data with annotations from the study.

### Contents

| Data type | Details |
|---|---|
| scRNA-seq | 223,113 cells from 25 patients |
| CITE-seq | 130 surface antigens on 62,973 cells (11 patients) |
| Visium spatial transcriptomics | 115,914 spots across 32 tissue slices (17 cases) |
| Xenium subcellular ST | 729,307 cells over 155 mm² (4 patients, 348-gene panel) |
| Reference atlas | 36,256 cells, 10 major cell types, 58 transcriptional states |
| Cell type annotations | 56 cell types used for spatial analysis |

### Additional processed data locations

- **Raw scRNA-seq & Visium ST:** NCBI SRA BioProject PRJNA1337938
- **Processed Xenium data:** https://doi.org/10.5281/zenodo.17622242
- **Processed Visium data:** https://doi.org/10.5281/zenodo.17572905
- **Analysis code & interactive viewer:** https://github.com/nameetas/TSKGA

## Patient cohort

- 28 patients total
- 18 GBM (IDH-wildtype)
- 5 IDH-mutant astrocytoma
- 1 IDH-mutant oligodendroglioma (1p19q co-deleted)
- 4 miscellaneous CNS pathologies
- 1 healthy brain tissue control (for Visium)
- 6 patients with matched tumor core and periphery samples

## Cell types identified (10 major, 58 states)

1. Astrocytes
2. Neurons
3. Oligodendrocytes/OPCs (including the key Oligo_2_3_2 subtype)
4. Neoplastic (tumor cells)
5. Microglia
6. Lymphoid (T/B cells)
7. Myeloid (macrophages, dendritic cells, granulocytes)
8. Endothelial
9. Fibroblast
10. Pericytes

## How to access

1. Visit https://cells.ucsc.edu/?ds=multiomic-gbm in a web browser
2. The Cell Browser provides an interactive viewer for exploring:
   - UMAP/t-SNE visualizations
   - Gene expression heatmaps
   - Cell type annotations
3. Data can typically be downloaded in standard formats (AnnData h5ad, loom, CSV metadata)

## Key finding highlighted by this dataset

The **Oligo_2_3_2 oligodendrocyte subtype** — immune-activated, non-myelinating, restricted to tumor core and IGN (Immune-Glial Niche), associated with tumor recurrence and poor patient survival.

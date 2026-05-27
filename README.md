# HCM Multi-Omics Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-4.2%2B-276DC3?style=flat-square&logo=r&logoColor=white)
![Disease](https://img.shields.io/badge/Disease-Hypertrophic%20Cardiomyopathy-red?style=flat-square)

A multi-modal transcriptomic analysis suite for Hypertrophic Cardiomyopathy, spanning bulk RNA-seq, single-cell RNA-seq, and spatial transcriptomics.

---

## Pipeline Overview

```
Bulk RNA-seq (R)
  limma DE → GSEA → TF activity → WGCNA → Regulon modules
                                                    │
                        ┌───────────────────────────┘
                        ▼
scRNA-seq Script 01 (Python)
  QC & normalisation → Reference mapping → Clustering → PAGA trajectory
                                                    │
                        ┌───────────────────────────┘
                        ▼
scRNA-seq Script 02 (Python)
  Pathway enrichment → LIANA CCC → Epigenetic regulator profiling
                                                    │
                        ┌───────────────────────────┘
                        ▼
Spatial (Python)
  Core spatial analyses → Bulk validation → Extension modules → Synthesis
```

Bulk pipeline outputs (regulons, WGCNA modules, TF scores) feed into the spatial pipeline for cross-modal validation. scRNA-seq Script 02 takes the annotated object produced by Script 01 as its input.

---

## Scripts

### `01_hcm_bulk_rnaseq_v4.R`
**Bulk RNA-seq | GSE141910 (183 HCM / 183 Control)**

The entry point of the suite. Merges raw count files, runs differential expression with limma, and identifies upregulated and downregulated genes between HCM and control. Performs GSEA against MSigDB Hallmark and C2 gene sets to characterise dysregulated pathways. Infers transcription factor activity using decoupleR and DoRothEA regulons, then builds co-expression modules via WGCNA to find gene clusters correlated with HCM trait.

The main deliverable is a set of five regulon modules (REG1-REG5) that integrate DEGs, TF activity, and optionally ATAC-seq chromatin accessibility data. These regulons are reused downstream in the spatial pipeline for cross-modal scoring. Includes a curated HCM gene panel check (sarcomere, Ca2+ handling, fibrosis, metabolic genes) as a biological sanity check.

---

### `01_hcm_preprocessing_trajectory.py`
**scRNA-seq | Multi-patient plate data (P1-P4)**

Loads and merges raw plate TSV files across four patients, cleans gene names, and applies standard QC filtering (removing mitochondrial, ribosomal, and spike-in genes). After normalisation and log-transformation, cells are projected onto a healthy ventricular cardiomyocyte reference atlas using `sc.tl.ingest`, which assigns each cell a reference state label (vCM1-vCM4) reflecting proximity to healthy or disease-like transcriptional programmes.

Clusters cells via Leiden at an auto-selected resolution (first stabilisation point across a resolution scan), then computes a PAGA trajectory rooted at the healthiest cell in the vCM1 population. The trajectory orders clusters from healthy to disease by HCM score, producing pseudotime values and a disease progression axis for all downstream analyses. Also scores HCM-core, immune, and epigenetic gene modules per cell.

---

### `02_hcm_enrichment_ccc_epigenetics.py`
**scRNA-seq | Trajectory: clusters 11 → 5 → 16 → 10 → 8 → 13**

Takes the annotated object from Script 01 and characterises the disease trajectory in biological depth across three areas.

**Differential expression and enrichment:** Runs Wilcoxon rank-sum per trajectory cluster, then queries Enrichr (GO BP/MF/CC, KEGG) on the top upregulated genes. Results are exported as per-cluster functional summaries.

**Cell-cell communication:** Runs LIANA rank-aggregate (consensus of CellPhoneDB, NATMI, SingleCellSignalR, and others) to identify ligand-receptor pairs active at each trajectory step, highlighting how intercellular signalling shifts as disease progresses.

**Epigenetic profiling:** Screens 63 chromatin and DNA methylation regulators across trajectory clusters. Identifies epifactors upregulated in terminal vs early clusters, correlates epifactor expression with pseudotime per vCM subtype, and exports an epigenetic trajectory summary linking regulator changes to biological processes.

---

### `spatial_hcm_pipeline.py`
**Spatial transcriptomics | Visium / ST data**

The most comprehensive script in the suite. Analyses spatially-resolved gene expression across 15 core analyses, including QC metric mapping, cluster topology, disease stage progression, HCM score gradients, vCM subtype segregation, master regulator expression (GATA6, PPARGC1A, TEAD1, HAND2), ETC complex depletion, UPR activation boundaries, and a random forest-based spatial risk map that zones tissue into therapeutic priority regions.

Cross-validates bulk pipeline findings spatially: projects WGCNA modules, regulon scores, and TF activity onto tissue coordinates to confirm bulk signals are spatially coherent. Eight extension analyses add metabolic state mapping, fibrosis gradients, proximity-based ligand-receptor hotspots, immune infiltration, sarcomere disarray indexing, and radial/transmural gradient analysis.

Concludes with three follow-up analyses (fibrosis-contractile antagonism, border-zone characterisation, foetal gene re-expression programme) and a synthesis figure that integrates all major findings into a concentric remodelling model of HCM tissue architecture.

---

## Dependencies

**R**
```r
limma, ggplot2, ggrepel, fgsea, msigdbr, ComplexHeatmap,
decoupleR, dorothea, WGCNA, clusterProfiler, org.Hs.eg.db
# optional: motifmatchr, JASPAR2022 (ATAC integration)
```

**Python**
```
scanpy, anndata, numpy, pandas, scipy, scikit-learn,
matplotlib, seaborn, statsmodels, networkx
# optional: gseapy, liana, pyranges
```

---

## Notes

- Optional steps (ATAC integration, LIANA, ChIP-seq overlap via pyranges) degrade gracefully if dependencies are absent.
- Checkpoints are saved after major stages so runs can be resumed without recomputing from scratch.
- A manual dotplot implementation is used in the scRNA-seq scripts to avoid a known scanpy crash in `rank_genes_groups_dotplot`.
- LIANA requires `adata.raw` to be set to `None` before running to prevent a gene-count shape mismatch between HVGs and the full feature set.

#!/usr/bin/env python3
"""
================================================================================
HCM scRNA-seq Multi-Omics Analysis Pipeline
================================================================================
A comprehensive single-cell analysis pipeline for Hypertrophic Cardiomyopathy
(HCM) ventricular cardiomyocyte (vCM) data integrating:
    - scRNA-seq processing and reference mapping
    - Trajectory inference (PAGA + pseudotime)
    - Pseudo-bulk differential expression + GSEA
    - Epigenetic factor analysis
    - H3K27ac ChIP-seq integration
    - TF-Epigenetic Gene Regulatory Network (GRN)
    - WGCNA module and regulon scoring
    - Per-patient statistical analysis

Author: Generated from scanafinal pipeline
Date: 2026-05-27
Requirements: scanpy, anndata, pandas, numpy, scipy, matplotlib, seaborn,
              networkx, gseapy, statsmodels, liana, pyranges
================================================================================
"""

import os
import gc
import warnings
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

from scipy import stats
from scipy.sparse import issparse
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance as scipy_distance
from statsmodels.stats.multitest import multipletests
import networkx as nx

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    warnings.warn("gseapy not installed - GSEA modules will be skipped")

try:
    import liana as li
    LIANA_AVAILABLE = True
except ImportError:
    LIANA_AVAILABLE = False
    warnings.warn("liana not installed - cell-cell communication will be skipped")

try:
    import pyranges as pr
    PYRANGES_AVAILABLE = True
except ImportError:
    PYRANGES_AVAILABLE = False
    warnings.warn("pyranges not installed - ChIP-seq integration will be skipped")

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Pipeline configuration and file paths."""

    OUT_DIR: str = "./hcm_pipeline_outputs"
    PREFIX: str = "hcm_"

    REFERENCE_H5AD: str = "healthy_vCM1_4_sparse_pca_umap.h5ad"
    EPIFACTORS_FILE: str = "epifactors_human.txt"
    CHIP_FILE: str = "GSE165303_H3K27ac_ChIP-Seq_peaks_master_table.txt"
    PROMOTER_BED: str = "genomewide_TSS_pm5kb.bed"
    WGCNA_MODULE_CSV: str = "rna_counts_raw/outputs/tables/WGCNA_module_summary_GO.csv"
    REGULON_CSV: str = "rna_counts_raw/outputs/regulons/ALL_REGULONS.csv"

    GMT_FILES: Dict[str, str] = {
        "GO_BP": "c5.go.bp.v2023.1.Hs.symbols.gmt",
        "GO_MF": "c5.go.mf.v2023.1.Hs.symbols.gmt",
        "KEGG": "c2.cp.kegg.v2023.1.Hs.symbols.gmt",
    }

    N_TOP_HVG: int = 5000
    LEIDEN_RESOLUTIONS: np.ndarray = np.arange(0.3, 2.1, 0.2)
    CHOSEN_RESOLUTION: float = 0.9
    N_NEIGHBORS: int = 15
    N_PSEUDOTIME_BINS: int = 10
    GRN_CORR_THRESHOLD: float = 0.35

    PAL_PATIENT: Dict[str, str] = {
        "P1": "#E63946", "P2": "#457B9D",
        "P3": "#2A9D8F", "P4": "#E9C46A"
    }
    PAL_VCM_BASE: Dict[str, str] = {
        "vCM1": "#264653", "vCM2": "#B8B8B8",
        "vCM3": "#E76F51", "vCM4": "#F4A261",
        "vCM5": "#A8DADC",
    }


# ==============================================================================
# GENE SETS
# ==============================================================================

HCM_CORE_GENES: List[str] = [
    "NPPA", "NPPB", "FN1", "COL1A1", "FGF12", "CREB5", "MYH7", "POSTN"
]

IMMUNE_GENES: List[str] = [
    "PTPRC", "LST1", "TYROBP", "FCER1G", "AIF1", "HLA-DRA", "HLA-DRB1",
    "HLA-DPA1", "HLA-DPB1", "HLA-A", "HLA-B", "HLA-C", "B2M", "CD74",
    "CD3D", "CD3E", "CD3G", "TRAC", "TRBC1", "TRBC2", "CD2", "CD4", "CD8A",
    "CD8B", "IL7R", "LTB", "MALAT1", "MAL", "CCR7", "TCF7", "LEF1", "SATB1",
    "MKI67", "LCK", "LAT", "ZAP70", "ICOS", "CTLA4", "PDCD1", "LAG3", "TIGIT",
    "NKG7", "GNLY", "PRF1", "GZMA", "GZMB", "GZMH", "GZMK", "KLRB1", "KLRD1",
    "KLRK1", "KLRC1", "KLRF1", "FCGR3A", "TRDC", "TRGC1", "XCL1", "XCL2",
    "IFNG", "TBX21", "EOMES", "CX3CR1", "CST7",
    "CD79A", "CD79B", "MS4A1", "CD19", "BANK1", "HVCN1", "CD22", "BLNK",
    "FCRL1", "FCRL2", "IGHM", "IGHD", "CD37", "PAX5", "SPIB",
    "MZB1", "JCHAIN", "SDC1", "XBP1", "DERL3", "TNFRSF17", "PRDM1", "IGKC",
    "IGLC2", "IGHG1", "IGHA1",
    "LYZ", "S100A8", "S100A9", "CTSS", "CTSB", "FCN1", "SAT1", "VCAN",
    "MSR1", "MARCO", "CD68", "CD163", "C1QA", "C1QB", "C1QC", "CSF1R",
    "TLR2", "TLR4", "ITGAM", "ITGAX", "SPI1", "MAFB", "APOE", "LGALS3",
    "IL1B", "TNF", "NLRP3", "CXCL8", "CXCL2", "CCL3", "CCL4",
    "CLEC10A", "CLEC9A", "FCER1A", "CD1C", "CD1E", "IRF8", "BATF3",
    "CCR7", "LAMP3", "IDO1", "CCL17", "CCL22",
    "FCGR3B", "CXCR1", "CXCR2", "CSF3R", "FPR1", "FPR2", "MNDA", "MMP8",
    "MMP9", "ELANE", "CTSG", "AZU1", "LCN2", "OLFM4", "BPI", "PGLYRP1",
    "TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2", "HDC", "CLC", "GATA2", "HPGDS",
    "IL1A", "IL1B", "IL6", "IL10", "IL18", "TGFB1", "IFNG", "TNF", "CXCL9",
    "CXCL10", "CXCL11", "CCL2", "CCL3", "CCL4", "CCL5", "CCL7", "CCL8",
    "CCL19", "CCL21", "CCL22", "CCL24", "CXCL1", "CXCL2", "CXCL3", "CXCL5",
    "CXCL6", "CXCL8", "CXCL12", "XCL1", "XCL2",
    "ISG15", "IFI6", "IFI27", "IFI35", "IFI44", "IFI44L", "IFIH1", "IFIT1",
    "IFIT2", "IFIT3", "IRF1", "IRF7", "MX1", "MX2", "OAS1", "OAS2", "OAS3",
    "RSAD2", "STAT1", "STAT2", "BST2", "DDX58", "SAMD9", "GBP1", "GBP5",
    "NFKB1", "NFKB2", "RELA", "REL", "JUN", "FOS", "FOSB", "JUNB", "JUND",
    "TNFAIP3", "BCL3", "NFKBIA", "NFKBIZ", "SOCS1", "SOCS3",
    "CIITA", "PSMB8", "PSMB9", "TAP1", "TAP2", "CD74", "HLA-E", "HLA-F",
    "HLA-G",
    "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1", "BTLA",
    "CD274", "PDCD1LG2",
    "C1QA", "C1QB", "C1QC", "C2", "C3", "CFB", "CFD", "CFH", "SERPING1",
    "POSTN", "VCAM1", "ICAM1", "SELE", "SELL", "PECAM1", "COL1A1", "FN1",
    "TGFB2", "TGFBI", "THBS1", "SPP1", "PLAUR", "VWF", "MMP2", "MMP14"
]

EPI_GENES: List[str] = [
    "CHD3", "HDAC1", "HDAC2", "HDAC3", "HDAC5", "DNMT1", "DNMT3A", "DNMT3B",
    "EZH2", "EZH1", "BRD2", "BRD4", "KDM6B", "KDM5C", "EP300", "CREBBP",
    "SMARCA4", "ARID1A", "SIRT1", "KMT2A", "KMT2D", "SETD1A", "KDM1A", "NCOR1"
]

EPI_CATEGORIES: Dict[str, List[str]] = {
    "Writers_KMT": ["KMT2A", "KMT2B", "KMT2C", "KMT2D", "SETD1A", "SETD1B",
                     "SETDB1", "EZH1", "EZH2", "NSD1", "NSD2", "SMYD2",
                     "PRMT1", "PRMT5"],
    "Writers_KAT": ["EP300", "CREBBP", "KAT2A", "KAT2B", "KAT6A", "KAT6B",
                     "KAT7", "KAT8", "TAF1"],
    "Writers_DNMT": ["DNMT1", "DNMT3A", "DNMT3B", "DNMT3L"],
    "Erasers_HDAC": ["HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6",
                      "HDAC7", "HDAC8", "HDAC9", "SIRT1", "SIRT2", "SIRT3",
                      "SIRT6"],
    "Erasers_KDM": ["KDM1A", "KDM2A", "KDM4A", "KDM4B", "KDM5A", "KDM5B",
                     "KDM5C", "KDM6A", "KDM6B", "KDM7A"],
    "Erasers_TET": ["TET1", "TET2", "TET3"],
    "Readers_BRD": ["BRD2", "BRD3", "BRD4", "BRD7", "BRD9", "BRDT"],
    "Readers_PHD": ["PHF2", "PHF8", "PHF10", "ING1", "ING2", "ING3"],
    "Remodellers_SWI_SNF": ["SMARCA4", "SMARCA2", "SMARCB1", "ARID1A",
                              "ARID1B", "ARID2", "SMARCC1", "SMARCC2",
                              "SMARCD1"],
    "Remodellers_CHD": ["CHD1", "CHD2", "CHD3", "CHD4", "CHD7", "CHD8", "CHD9"],
    "Remodellers_NuRD": ["CHD3", "CHD4", "HDAC1", "HDAC2", "MBD2", "MBD3",
                          "RBBP4", "RBBP7", "GATAD2A", "GATAD2B"],
    "PRC_Polycomb": ["EZH2", "EED", "SUZ12", "RING1", "RNF2", "CBX2",
                      "CBX4", "CBX6", "CBX7", "CBX8"],
    "Activators_MED": ["MED1", "MED12", "MED13", "MED14", "MED24",
                        "NCOA1", "NCOA2", "NCOA3"],
    "Silencers": ["NCOR1", "NCOR2", "SIN3A", "SIN3B", "RCOR1",
                   "NR2C2", "NR2F1", "NR2F2"],
}

BULK_TFS: List[str] = [
    "ATF1", "NR2C2", "SREBF1", "SREBF2", "FLI1", "E2F2", "KLF3",
    "ELF3", "FOSL2", "ZFX", "NR1H3", "NR5A2", "MYOD1", "NR1H2", "KLF1"
]

CARDIAC_TFS: List[str] = [
    "GATA4", "GATA6", "NKX2-5", "TBX5", "TBX20", "MEF2A", "MEF2C", "MEF2D",
    "HAND1", "HAND2", "SRF", "MYOCD", "MKL1", "MKL2", "NFATC1", "NFATC2",
    "NFATC3", "NFATC4", "STAT3", "STAT5B", "HIF1A", "EPAS1", "ATF3", "ATF6",
    "DDIT3", "XBP1", "SMAD2", "SMAD3", "SMAD4", "SMAD6", "SMAD7", "TGIF1",
    "TGIF2", "TEAD1", "TEAD4", "YAP1", "WWTR1", "PPARGC1A", "PPARGC1B",
    "PPARA", "ESRRA", "ESRRG", "KLF15", "KLF4", "KLF5", "KLF9", "SP1", "SP3",
    "RXRA", "RXRB", "THRB", "NR3C1", "NR4A1", "NR4A2", "NR4A3", "TCF7L2",
    "LEF1", "CTNNB1", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "ZEB2", "FOS",
    "FOSB", "JUN", "JUNB", "JUND", "CREB1", "CREB3", "ATF1", "ATF2", "E2F1",
    "E2F3", "MYC", "MYCN", "ERG", "ETS1", "ETV4", "SOX17", "RBPJ"
]

BULK_MODULES: Dict[str, List[str]] = {
    "Anti_Fibrotic_Stromal": ["SFRP4", "FRZB", "LUM", "ASPN", "LEFTY2",
                               "COL14A1", "FMOD", "SCUBE2"],
    "Immune_Exclusion": ["CCR7", "GZMB", "CXCL9", "CXCL10", "PRF1",
                          "NKG7", "CD3D", "STAT4"],
    "Cytokine_Innate": ["IL6", "IL10", "CCL2", "IL1RL1", "SAA1", "SELE", "EREG"],
    "mTORC1_MYC": ["MYC", "CCND1", "RPS3", "EIF4E", "RPS6KB1", "LDHA", "HK2", "PKM"],
    "Wnt_NonCanonical": ["WNT5A", "WNT9B", "FZD5", "FZD6", "ROR1", "ROR2",
                          "RHOA", "DAAM1"],
    "HCM_Core": HCM_CORE_GENES,
    "NuRD_Epigenetic": ["CHD3", "CHD4", "HDAC1", "HDAC2", "MBD3",
                          "RBBP4", "GATAD2A", "DNMT1"],
}

PATIENTS: Dict[str, List[str]] = {
    "P1": ["GSM4103885_Patient1Plate1", "GSM4103886_Patient1Plate2",
           "GSM4103887_Patient1Plate3", "GSM4103888_Patient1Plate4"],
    "P2": ["GSM4103889_Patient2Plate1", "GSM4103890_Patient2Plate2",
           "GSM4103891_Patient2Plate3", "GSM4103892_Patient2Plate4"],
    "P3": ["GSM4103893_Patient3Plate1", "GSM4103894_Patient3Plate2",
           "GSM4103895_Patient3Plate3", "GSM4103896_Patient3Plate4"],
    "P4": ["GSM4103897_Patient4Plate1", "GSM4103898_Patient4Plate2"],
}


# ==============================================================================
# LOGGING & CHECKPOINTING
# ==============================================================================

def setup_logging(out_dir: str, prefix: str) -> str:
    """Initialize log file and return path."""
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f"{prefix}pipeline_log.txt")
    open(log_file, "w").close()
    return log_file


def log(msg: str, log_file: str, also_print: bool = True) -> None:
    """Write timestamped message to log file and optionally print."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with open(log_file, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line)


def checkpoint(adata: ad.AnnData, tag: str, out_dir: str, prefix: str,
               log_file: str) -> None:
    """Save AnnData checkpoint to h5ad file."""
    path = os.path.join(out_dir, f"{prefix}checkpoint_{tag}.h5ad")
    adata.write_h5ad(path)
    log(f"  Checkpoint saved -> {path}", log_file)


def savefig(fig: plt.Figure, name: str, out_dir: str, prefix: str,
            log_file: str, dpi: int = 200) -> None:
    """Save matplotlib figure to PDF."""
    path = os.path.join(out_dir, f"{prefix}{name}.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {path}", log_file)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_clean_plate(tsv_path: str, patient_id: str,
                         plate_id: str) -> sc.AnnData:
    """Load a single plate TSV file and create an AnnData object."""
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    adata = sc.AnnData(df.T)
    adata.var_names = adata.var_names.str.split("__").str[0]
    adata.var_names_make_unique()
    adata.obs["patient_id"] = patient_id
    adata.obs["plate_id"] = plate_id
    return adata


def load_all_data(config: Config, log_file: str) -> sc.AnnData:
    """Load all patient plates and concatenate into master AnnData."""
    log("Loading all patient data...", log_file)
    all_adatas = []

    for patient_id, plate_list in PATIENTS.items():
        for plate_file in plate_list:
            folder = f"patient{patient_id[1]}"
            path = f"{folder}/{plate_file}.TranscriptCounts.tsv.gz"
            if not os.path.exists(path):
                log(f"  WARNING: File not found: {path}", log_file)
                continue
            log(f"  Loading: {path}", log_file)
            adata = load_and_clean_plate(path, patient_id, plate_file)
            all_adatas.append(adata)

    if not all_adatas:
        raise FileNotFoundError("No data files found. Check paths.")

    adata_hcm = ad.concat(all_adatas, label="batch", join="inner")
    adata_hcm.obs_names_make_unique()
    log(f"Master object: {adata_hcm.n_obs} cells x {adata_hcm.n_vars} genes", log_file)
    return adata_hcm


# ==============================================================================
# PREPROCESSING
# ==============================================================================

def filter_junk_genes(adata: sc.AnnData, log_file: str) -> sc.AnnData:
    """Remove mitochondrial, ERCC spike-in, and ribosomal genes."""
    junk_mask = (
        adata.var_names.str.startswith("MT-") |
        adata.var_names.str.startswith("ERCC-") |
        adata.var_names.str.startswith("RPS") |
        adata.var_names.str.startswith("RPL")
    )
    adata = adata[:, ~junk_mask].copy()
    log(f"After junk filtering: {adata.shape}", log_file)
    return adata


def normalize_and_log(adata: sc.AnnData, log_file: str) -> sc.AnnData:
    """Normalize to 10,000 counts and log1p transform."""
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    max_val = float(adata.X.max())
    mean_val = float(adata.X.mean())
    log(f"Max: {max_val:.4f}, Mean: {mean_val:.4f}", log_file)
    assert 1 <= max_val <= 10, "Normalization check failed"
    log("Normalization OK", log_file)
    return adata


def align_genes(adata: sc.AnnData, adata_ref: sc.AnnData,
                log_file: str) -> Tuple[sc.AnnData, sc.AnnData]:
    """Align query and reference to common genes."""
    common_genes = adata_ref.var_names.intersection(adata.var_names)
    log(f"Aligning on {len(common_genes)} genes", log_file)
    adata_ref = adata_ref[:, common_genes].copy()
    adata = adata[:, common_genes].copy()
    return adata, adata_ref


# ==============================================================================
# REFERENCE MAPPING
# ==============================================================================

def run_ingest(adata: sc.AnnData, adata_ref: sc.AnnData,
               log_file: str) -> sc.AnnData:
    """Map query cells to reference using scanpy ingest."""
    sc.tl.ingest(adata, adata_ref, obs="cell_states")
    log("Reference mapping complete", log_file)

    alignment_table = pd.crosstab(adata.obs["patient_id"], adata.obs["cell_states"])
    alignment_pct = pd.crosstab(
        adata.obs["patient_id"], adata.obs["cell_states"], normalize="index"
    ) * 100
    log("\nCounts:", log_file)
    log(str(alignment_table), log_file)
    log("\nPercentages:", log_file)
    log(str(alignment_pct.round(2)), log_file)
    return adata


# ==============================================================================
# GENE MODULE SCORING
# ==============================================================================

def score_module(adata: sc.AnnData, genes: List[str], key: str,
                 log_file: str) -> None:
    """Score gene module using scanpy score_genes."""
    present = [g for g in genes if g in adata.var_names]
    log(f"  {key}: {len(present)}/{len(genes)} genes present", log_file)
    if len(present) == 0:
        adata.obs[key] = 0.0
        return
    sc.tl.score_genes(adata, gene_list=present, score_name=key, use_raw=False)


def compute_module_scores(adata: sc.AnnData, log_file: str) -> None:
    """Compute HCM, immune, and epigenetic module scores."""
    log("Computing gene module scores...", log_file)
    score_module(adata, HCM_CORE_GENES, "HCM_score", log_file)
    score_module(adata, IMMUNE_GENES, "Immune_score", log_file)
    score_module(adata, EPI_GENES, "Epi_score", log_file)


# ==============================================================================
# CLUSTERING & TRAJECTORY
# ==============================================================================

def run_resolution_scan(adata: sc.AnnData, config: Config,
                        log_file: str) -> float:
    """Scan Leiden resolutions and auto-select optimal."""
    log("Resolution scan...", log_file)
    use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"

    if "X_pca" not in adata.obsm:
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver="arpack")

    sc.pp.neighbors(adata, n_neighbors=config.N_NEIGHBORS, use_rep=use_rep)
    sc.tl.umap(adata)

    n_clusters = {}
    for res in config.LEIDEN_RESOLUTIONS:
        res_rounded = round(res, 1)
        sc.tl.leiden(adata, resolution=res_rounded,
                     key_added=f"leiden_{res_rounded:.1f}")
        n_clusters[res_rounded] = adata.obs[f"leiden_{res_rounded:.1f}"].nunique()

    log(f"  Clusters: {n_clusters}", log_file)

    res_vals = sorted(n_clusters.keys())
    chosen = config.CHOSEN_RESOLUTION
    for i in range(1, len(res_vals) - 1):
        delta = abs(n_clusters[res_vals[i + 1]] - n_clusters[res_vals[i]])
        if delta <= 2:
            chosen = res_vals[i]
            break

    sc.tl.leiden(adata, resolution=chosen, key_added="leiden")
    n_clust = adata.obs["leiden"].nunique()
    log(f"  Chosen resolution: {chosen} -> {n_clust} clusters", log_file)
    return chosen


def build_trajectory(adata: sc.AnnData, log_file: str) -> List[str]:
    """Build disease trajectory using PAGA connectivity and HCM scores."""
    log("Building PAGA trajectory...", log_file)

    sc.tl.paga(adata, groups="leiden")
    cluster_hcm = adata.obs.groupby("leiden")["HCM_score"].mean().sort_values()

    root_cluster = cluster_hcm.index[0]
    root_mask = adata.obs["leiden"] == root_cluster
    root_idx = int(np.where(root_mask)[0][
        np.argmin(adata.obs.loc[root_mask, "HCM_score"].values)
    ])
    adata.uns["iroot"] = root_idx
    log(f"  Root cluster: {root_cluster} (HCM={cluster_hcm.iloc[0]:.3f})", log_file)

    use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    adata.obs["pseudotime"] = adata.obs["dpt_pseudotime"]

    paga_conn = pd.DataFrame(
        adata.uns["paga"]["connectivities"].toarray(),
        index=adata.obs["leiden"].cat.categories,
        columns=adata.obs["leiden"].cat.categories
    )

    def _build_path(start: str, conn: pd.DataFrame, scores: pd.Series,
                    n_steps: int) -> List[str]:
        path = [start]
        visited = {start}
        current = start
        for _ in range(n_steps - 1):
            row = conn.loc[current].drop(labels=list(visited), errors="ignore")
            if row.empty:
                break
            candidates = {nb: w for nb, w in row.items()
                         if w > 0.05 and scores[nb] > scores[current]}
            nxt = max(candidates, key=candidates.get) if candidates else row.idxmax()
            path.append(nxt)
            visited.add(nxt)
            current = nxt
        return path

    trajectory = _build_path(root_cluster, paga_conn, cluster_hcm,
                            n_steps=min(8, len(cluster_hcm)))
    log(f"  Trajectory: {' -> '.join(trajectory)}", log_file)
    return trajectory


# ==============================================================================
# PSEUDO-BULK ANALYSIS
# ==============================================================================

def run_pseudobulk(adata: sc.AnnData, trajectory: List[str],
                   config: Config, log_file: str) -> pd.DataFrame:
    """Compute pseudo-bulk expression for early vs late trajectory halves."""
    log("Pseudo-bulk analysis...", log_file)

    half = len(trajectory) // 2
    early_clusters = trajectory[:half]
    late_clusters = trajectory[half:]
    log(f"  Early: {early_clusters} | Late: {late_clusters}", log_file)

    traj_mask = adata.obs["leiden"].isin(trajectory)
    adata_traj = adata[traj_mask].copy()
    adata_traj.obs["traj_half"] = "Late"
    adata_traj.obs.loc[adata_traj.obs["leiden"].isin(early_clusters), "traj_half"] = "Early"

    def _pseudobulk(adata_sub: sc.AnnData, group_col: str) -> pd.DataFrame:
        groups = adata_sub.obs[group_col].unique()
        pb = {}
        for g in groups:
            mask = adata_sub.obs[group_col] == g
            X = adata_sub[mask].raw.X if adata_sub.raw is not None else adata_sub[mask].X
            if issparse(X):
                X = X.toarray()
            pb[g] = X.mean(axis=0)
        var_names = adata_sub.raw.var_names if adata_sub.raw is not None else adata_sub.var_names
        return pd.DataFrame(pb, index=var_names)

    pb_df = _pseudobulk(adata_traj, "traj_half")
    pb_df.to_csv(os.path.join(config.OUT_DIR, f"{config.PREFIX}pseudobulk_early_vs_late.csv"))

    pb_lfc = pd.DataFrame({
        "Early_mean": pb_df["Early"],
        "Late_mean": pb_df["Late"],
        "logFC": np.log2(pb_df["Late"] + 1e-6) - np.log2(pb_df["Early"] + 1e-6)
    }).sort_values("logFC", ascending=False)
    pb_lfc.to_csv(os.path.join(config.OUT_DIR, f"{config.PREFIX}pseudobulk_logFC.csv"))
    return pb_lfc


# ==============================================================================
# GSEA
# ==============================================================================

def run_gsea(ranked_series: pd.Series, config: Config, log_file: str) -> Dict[str, pd.DataFrame]:
    """Run preranked GSEA on GO BP, GO MF, and KEGG."""
    if not GSEAPY_AVAILABLE:
        log("gseapy not available - skipping GSEA", log_file)
        return {}

    log("Running GSEA...", log_file)
    gsea_results = {}

    for name, gmt in config.GMT_FILES.items():
        if not os.path.exists(gmt):
            log(f"  {name}: GMT not found, skipping", log_file)
            continue
        try:
            res = gp.prerank(
                rnk=ranked_series, gene_sets=gmt,
                min_size=10, max_size=500, permutation_num=500,
                outdir=None, seed=42, verbose=False
            )
            df = res.res2d.copy()
            pvals = df["NOM p-val"].fillna(1.0).values
            _, padj, _, _ = multipletests(pvals, method="fdr_bh")
            df["BH_FDR"] = padj
            df_sig = df[df["BH_FDR"] < 0.25].sort_values("NES", ascending=False)
            gsea_results[name] = df_sig
            df_sig.to_csv(os.path.join(config.OUT_DIR, f"{config.PREFIX}GSEA_{name}.csv"))
            log(f"  {name}: {len(df_sig)} significant terms", log_file)
        except Exception as e:
            log(f"  {name} GSEA failed: {e}", log_file)
    return gsea_results


# ==============================================================================
# MARKER GENES
# ==============================================================================

def run_marker_analysis(adata: sc.AnnData, trajectory: List[str],
                        config: Config, log_file: str) -> None:
    """Find marker genes per cluster and create manual dotplot."""
    log("Per-cluster marker analysis...", log_file)
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                           key_added="rank_genes_leiden", use_raw=True)

    n_genes_per_cluster = 4
    marker_genes, cluster_labels = [], []
    valid_traj = [c for c in trajectory if c in adata.obs["leiden"].cat.categories]

    for cl in valid_traj:
        df_mg = sc.get.rank_genes_groups_df(adata, group=cl,
                                           key="rank_genes_leiden").head(n_genes_per_cluster)
        top = df_mg["names"].tolist()
        marker_genes.extend(top)
        cluster_labels.extend([f"Cl{cl}"] * len(top))

    seen, unique_genes, unique_labels = set(), [], []
    for g, l in zip(marker_genes, cluster_labels):
        if g not in seen and g in adata.var_names:
            seen.add(g)
            unique_genes.append(g)
            unique_labels.append(l)

    if not unique_genes:
        log("  No valid marker genes found", log_file)
        return

    cluster_order = [f"Cl{c}" for c in valid_traj]
    mean_expr = pd.DataFrame(index=cluster_order, columns=unique_genes, dtype=float)
    pct_expr = pd.DataFrame(index=cluster_order, columns=unique_genes, dtype=float)

    for cl in valid_traj:
        mask = adata.obs["leiden"] == cl
        sub = adata[mask]
        X = sub[:, unique_genes].X
        if issparse(X):
            X = X.toarray()
        mean_expr.loc[f"Cl{cl}"] = X.mean(axis=0)
        pct_expr.loc[f"Cl{cl}"] = (X > 0).mean(axis=0) * 100

    mean_norm = mean_expr.copy().astype(float)
    for g in unique_genes:
        col = mean_expr[g].astype(float)
        rng = col.max() - col.min()
        mean_norm[g] = (col - col.min()) / (rng + 1e-8)

    n_genes = len(unique_genes)
    n_clust = len(cluster_order)
    fig, ax = plt.subplots(figsize=(max(12, n_genes * 0.35), max(4, n_clust * 0.55)))

    for ci, cl_label in enumerate(cluster_order):
        for gi, gene in enumerate(unique_genes):
            dot_size = float(pct_expr.loc[cl_label, gene])
            dot_color = float(mean_norm.loc[cl_label, gene])
            ax.scatter(gi, ci, s=dot_size * 3,
                      c=[[plt.cm.Reds(dot_color)]],
                      edgecolors="grey", linewidths=0.3, alpha=0.85)

    boundary = 0
    group_sizes = [sum(1 for l in unique_labels if l == f"Cl{c}") for c in valid_traj]
    for sz in group_sizes[:-1]:
        boundary += sz
        ax.axvline(boundary - 0.5, color="lightgrey", lw=0.8)

    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(unique_genes, rotation=90, fontsize=7)
    ax.set_yticks(range(n_clust))
    ax.set_yticklabels(cluster_order, fontsize=9)
    ax.set_xlabel("Marker genes (trajectory order)")
    ax.set_ylabel("Leiden cluster")
    ax.set_title("Top Markers per Trajectory Cluster")
    ax.grid(False)

    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Norm. mean expression", shrink=0.6, pad=0.02)
    for pct, label in [(20, "20%"), (50, "50%"), (80, "80%")]:
        ax.scatter([], [], s=pct * 3, c="grey", alpha=0.6, label=label)
    ax.legend(title="% cells expressing", loc="upper left",
             bbox_to_anchor=(1.08, 1), fontsize=8, title_fontsize=8)
    plt.tight_layout()
    savefig(fig, "trajectory_markers_dotplot", config.OUT_DIR, config.PREFIX, log_file)


# ==============================================================================
# EPIGENETIC ANALYSIS
# ==============================================================================

def run_epigenetic_analysis(adata: sc.AnnData, trajectory: List[str],
                            config: Config, log_file: str) -> None:
    """Analyze epigenetic factor expression across trajectory."""
    log("Epigenetic analysis...", log_file)

    if os.path.exists(config.EPIFACTORS_FILE):
        epi_db = pd.read_csv(config.EPIFACTORS_FILE, sep="\t")
        sym_col = next((c for c in epi_db.columns
                       if "symbol" in c.lower() or "gene" in c.lower()),
                      epi_db.columns[0])
        epi_symbols = epi_db[sym_col].dropna().unique().tolist()
    else:
        epi_symbols = EPI_GENES

    all_epi_from_cats = list(set(g for genes in EPI_CATEGORIES.values() for g in genes))
    expr_epi = [g for g in all_epi_from_cats if g in adata.var_names]
    log(f"  Expressed epifactors: {len(expr_epi)}/{len(all_epi_from_cats)}", log_file)

    if not expr_epi:
        log("  No epigenetic factors expressed - skipping", log_file)
        return

    cat_map = {g: cat for cat, genes in EPI_CATEGORIES.items() for g in genes}

    # Per-cluster heatmap
    clust_means = {}
    for cl in trajectory:
        mask = adata.obs["leiden"] == cl
        sub = adata[mask]
        X = sub[:, expr_epi].X
        if issparse(X):
            X = X.toarray()
        clust_means[f"Cl{cl}"] = X.mean(axis=0)

    epi_heat = pd.DataFrame(clust_means, index=expr_epi).astype(float)
    epi_heat_z = ((epi_heat.T - epi_heat.T.mean()) / (epi_heat.T.std() + 1e-8)).T.astype(float)

    row_cats = [cat_map.get(g, "Other") for g in epi_heat_z.index]
    unique_cats = list(set(row_cats))
    cat_colors = dict(zip(unique_cats, sns.color_palette("tab20", len(unique_cats))))
    row_colors = pd.Series(row_cats, index=epi_heat_z.index).map(cat_colors)

    fig_h = max(12, len(expr_epi) * 0.22)
    g = sns.clustermap(
        epi_heat_z, row_colors=row_colors,
        cmap="RdBu_r", center=0,
        figsize=(max(8, len(trajectory)), fig_h),
        linewidths=0.2, row_cluster=True, col_cluster=False,
        yticklabels=True, xticklabels=True,
        cbar_kws={"label": "Z-score expression"}
    )
    g.ax_heatmap.set_title("Epigenetic Factor Expression Across Trajectory Clusters")
    plt.savefig(os.path.join(config.OUT_DIR,
                              f"{config.PREFIX}epigenetic_heatmap_trajectory.pdf"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Category-level activity
    cat_activity = defaultdict(dict)
    for cat, genes in EPI_CATEGORIES.items():
        expressed = [g for g in genes if g in adata.var_names]
        if not expressed:
            continue
        for cl in trajectory:
            mask = adata.obs["leiden"] == cl
            X = adata[mask][:, expressed].X
            if issparse(X):
                X = X.toarray()
            cat_activity[cat][f"Cl{cl}"] = float(X.mean())

    cat_act_df = pd.DataFrame(cat_activity).T
    cat_act_df = cat_act_df.loc[cat_act_df.any(axis=1)].astype(float)
    cat_act_norm = cat_act_df.div(cat_act_df.max(axis=1) + 1e-8, axis=0).astype(float)
    cat_act_df.to_csv(os.path.join(config.OUT_DIR,
                                    f"{config.PREFIX}epi_category_activity_per_cluster.csv"))

    fig, ax = plt.subplots(figsize=(max(8, len(trajectory) * 0.8),
                                     max(6, len(cat_act_norm) * 0.5)))
    sns.heatmap(cat_act_norm, ax=ax, cmap="viridis", linewidths=0.3,
               xticklabels=True, yticklabels=True,
               cbar_kws={"label": "Normalised mean expression"})
    ax.set_title("Epigenetic Category Activity - Trajectory Clusters")
    ax.set_xlabel("Leiden Cluster (trajectory order)")
    plt.tight_layout()
    savefig(fig, "epi_category_heatmap", config.OUT_DIR, config.PREFIX, log_file)


# ==============================================================================
# H3K27ac ChIP-seq INTEGRATION
# ==============================================================================

def integrate_chipseq(adata: sc.AnnData, trajectory: List[str],
                      config: Config, log_file: str) -> None:
    """Integrate H3K27ac ChIP-seq data with scRNA-seq trajectory."""
    if not PYRANGES_AVAILABLE:
        log("pyranges not available - skipping ChIP-seq integration", log_file)
        return
    if not (os.path.exists(config.CHIP_FILE) and os.path.exists(config.PROMOTER_BED)):
        log("ChIP-seq files not found - skipping", log_file)
        return

    log("H3K27ac ChIP-seq integration...", log_file)

    chip_df = pd.read_csv(config.CHIP_FILE, sep="\t", low_memory=False)
    col_up = {c: c.upper() for c in chip_df.columns}
    dcm_cols = [c for c in chip_df.columns
                if any(x in col_up[c] for x in ["DCM", "FAIL", "HF"])]
    nf_cols = [c for c in chip_df.columns
               if any(x in col_up[c] for x in ["NF", "NORMAL", "CTRL"])]

    if not dcm_cols or not nf_cols:
        log("  Could not detect condition columns", log_file)
        return

    chip_df[dcm_cols] = chip_df[dcm_cols].apply(pd.to_numeric, errors="coerce")
    chip_df[nf_cols] = chip_df[nf_cols].apply(pd.to_numeric, errors="coerce")
    chip_df["delta_H3K27ac"] = chip_df[dcm_cols].mean(axis=1) - chip_df[nf_cols].mean(axis=1)

    chip_sig = chip_df[chip_df["FDR"] < 0.05] if "FDR" in chip_df.columns else chip_df
    chip_gained = chip_sig[chip_sig["delta_H3K27ac"] > 0].copy()
    chip_lost = chip_sig[chip_sig["delta_H3K27ac"] < 0].copy()
    log(f"  Gained: {len(chip_gained)} | Lost: {len(chip_lost)}", log_file)

    prom_df = pd.read_csv(config.PROMOTER_BED, sep="\t", header=None).iloc[:, :4]
    prom_df.columns = ["Chromosome", "Start", "End", "Gene"]

    def _std_chr(series):
        s = series.astype(str).str.strip().str.replace("^chr", "", regex=True)
        return "chr" + s

    chip_gained["chrom"] = _std_chr(chip_gained["chrom"])
    chip_lost["chrom"] = _std_chr(chip_lost["chrom"])
    prom_df["Chromosome"] = _std_chr(prom_df["Chromosome"])

    def _get_gene_scores(peak_df, label):
        peaks_gr = pr.PyRanges(peak_df.rename(columns={"chrom": "Chromosome",
                                                        "start": "Start",
                                                        "end": "End"}))
        prom_gr = pr.PyRanges(prom_df)
        overlaps = peaks_gr.join(prom_gr).df
        if overlaps.empty:
            log(f"  WARNING [{label}]: No overlaps", log_file)
            return pd.Series(dtype=float)
        return overlaps.groupby("Gene")["delta_H3K27ac"].mean().sort_values(ascending=False)

    gained_scores = _get_gene_scores(chip_gained, "GAINED")
    lost_scores = _get_gene_scores(chip_lost, "LOST")

    adata_genes_upper = [g.upper() for g in adata.var_names]

    def _clean_genes(series, n=500):
        genes = series.index.astype(str).str.upper().unique()
        return [g for g in genes[:n] if g in adata_genes_upper]

    gained_list = _clean_genes(gained_scores)
    lost_list = _clean_genes(abs(lost_scores).sort_values(ascending=False))
    log(f"  Gained genes in adata: {len(gained_list)} | Lost: {len(lost_list)}", log_file)

    if len(gained_list) >= 5:
        sc.tl.score_genes(adata, gained_list, score_name="H3K27ac_gained_score")
    if len(lost_list) >= 5:
        sc.tl.score_genes(adata, lost_list, score_name="H3K27ac_lost_score")

    if "H3K27ac_gained_score" in adata.obs.columns and \
       "H3K27ac_lost_score" in adata.obs.columns:
        adata.obs["H3K27ac_net_score"] = (adata.obs["H3K27ac_gained_score"] -
                                            adata.obs["H3K27ac_lost_score"])
    log("  ChIP-seq integration complete", log_file)


# ==============================================================================
# WGCNA & REGULON INTEGRATION
# ==============================================================================

def load_and_score_wgcna(adata: sc.AnnData, config: Config, log_file: str) -> List[str]:
    """Load WGCNA modules and score on single cells."""
    if not os.path.exists(config.WGCNA_MODULE_CSV):
        log("WGCNA file not found - skipping", log_file)
        return []

    log("Loading WGCNA modules...", log_file)
    wgcna_df = pd.read_csv(config.WGCNA_MODULE_CSV)
    log(f"  Loaded {len(wgcna_df)} modules", log_file)

    score_cols = []
    for _, row in wgcna_df.iterrows():
        mod_id = int(row["Module"])
        key = f"WGCNA_M{mod_id}"
        gene_str = str(row.get("Top50_Genes", ""))
        genes = [g.strip() for g in gene_str.split(",") if g.strip()]
        score_module(adata, genes, key, log_file)
        score_cols.append(key)

    log(f"  Scored {len(score_cols)} WGCNA modules", log_file)
    return score_cols


def load_and_score_regulons(adata: sc.AnnData, config: Config,
                            log_file: str) -> List[Tuple[str, str]]:
    """Load regulons and score on single cells."""
    if not os.path.exists(config.REGULON_CSV):
        log("Regulon file not found - skipping", log_file)
        return []

    log("Loading regulons...", log_file)
    reg_raw = pd.read_csv(config.REGULON_CSV)

    score_cols = []
    for col in reg_raw.columns:
        genes = [str(g).strip() for g in reg_raw[col].dropna() if str(g).strip()]
        if not genes:
            continue
        key = f"score_{col}"
        score_module(adata, genes, key, log_file)
        score_cols.append((col, key))

    log(f"  Scored {len(score_cols)} regulons", log_file)
    return score_cols


# ==============================================================================
# PER-PATIENT STATISTICS
# ==============================================================================

def run_patient_statistics(adata: sc.AnnData, config: Config,
                           log_file: str) -> pd.DataFrame:
    """Run per-patient statistical tests comparing disease vs healthy states."""
    log("Per-patient statistical analysis...", log_file)

    score_cols = ([c for c in adata.obs.columns if c.startswith("WGCNA_M")] +
                  [c for c in adata.obs.columns if c.startswith("score_REG")])
    score_cols = sorted(set(score_cols))

    keep_cols = ["patient_id", "cell_states"] + score_cols
    df = adata.obs[keep_cols].copy()
    df = df[df["cell_states"].isin(["vCM1", "vCM3", "vCM4"])].copy()
    df["is_disease"] = df["cell_states"].isin(["vCM3", "vCM4"]).astype(int)

    log(f"  Cells: {len(df)} (vCM1={sum(df['is_disease']==0)}, "
        f"vCM3/4={sum(df['is_disease']==1)})", log_file)

    rows = []
    for pat in config.PAL_PATIENT.keys():
        sub = df[df["patient_id"] == pat]
        if sub.empty:
            continue
        grp0 = sub[sub["is_disease"] == 0]
        grp1 = sub[sub["is_disease"] == 1]

        for col in score_cols:
            v0 = grp0[col].dropna().values
            v1 = grp1[col].dropna().values
            if len(v0) < 3 or len(v1) < 3:
                continue

            all_x = np.concatenate([np.zeros(len(v0)), np.ones(len(v1))])
            all_y = np.concatenate([v0, v1])
            r_pb, p_pb = stats.pointbiserialr(all_x, all_y)

            U_stat, p_mw = stats.mannwhitneyu(v1, v0, alternative="two-sided")
            auroc = U_stat / (len(v1) * len(v0))

            pooled_std = np.sqrt((v0.std() ** 2 + v1.std() ** 2) / 2)
            cohens_d = (v1.mean() - v0.mean()) / (pooled_std + 1e-9)

            rows.append({
                "patient": pat, "feature": col,
                "n_vCM1": len(v0), "n_disease": len(v1),
                "mean_vCM1": v0.mean(), "mean_disease": v1.mean(),
                "r_pb": r_pb, "p_pb": p_pb, "p_mw": p_mw,
                "auroc": auroc, "cohens_d": cohens_d,
            })

    stats_df = pd.DataFrame(rows)
    corrected_p = []
    for _, sub in stats_df.groupby("patient"):
        _, padj, _, _ = multipletests(sub["p_mw"].values, method="fdr_bh")
        corrected_p.extend(padj)
    stats_df["p_mw_fdr"] = corrected_p

    stats_df.to_csv(os.path.join(config.OUT_DIR,
                                  f"{config.PREFIX}stats_per_patient.csv"), index=False)
    log(f"  Stats table: {len(stats_df)} rows", log_file)
    return stats_df


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main() -> Tuple[sc.AnnData, sc.AnnData]:
    """Run the complete HCM multi-omics analysis pipeline."""

    config = Config()
    os.makedirs(config.OUT_DIR, exist_ok=True)
    log_file = setup_logging(config.OUT_DIR, config.PREFIX)

    log("=" * 60, log_file)
    log("HCM scRNA-seq Multi-Omics Analysis Pipeline", log_file)
    log("=" * 60, log_file)

    # STAGE 1: Data Loading
    log("\n[STAGE 1] Data Loading", log_file)
    adata_hcm = load_all_data(config, log_file)

    # STAGE 2: Preprocessing
    log("\n[STAGE 2] Preprocessing", log_file)
    adata_hcm = filter_junk_genes(adata_hcm, log_file)
    adata_hcm = normalize_and_log(adata_hcm, log_file)

    # STAGE 3: Reference Mapping
    log("\n[STAGE 3] Reference Mapping", log_file)
    if not os.path.exists(config.REFERENCE_H5AD):
        raise FileNotFoundError(f"Reference file not found: {config.REFERENCE_H5AD}")
    adata_ref = sc.read_h5ad(config.REFERENCE_H5AD)
    adata_hcm, adata_ref = align_genes(adata_hcm, adata_ref, log_file)
    adata_hcm = run_ingest(adata_hcm, adata_ref, log_file)

    # STAGE 4: HVG Selection
    log("\n[STAGE 4] HVG Selection", log_file)
    sc.pp.highly_variable_genes(adata_hcm, n_top_genes=config.N_TOP_HVG,
                                flavor="seurat_v3", span=1.0)
    n_hvg = adata_hcm.var["highly_variable"].sum()
    log(f"  HVGs selected: {n_hvg}", log_file)
    adata_hcm_full = adata_hcm.copy()
    adata_hcm = adata_hcm[:, adata_hcm.var["highly_variable"]].copy()
    log(f"  Working object: {adata_hcm.n_obs} x {adata_hcm.n_vars}", log_file)

    # STAGE 5: Module Scores
    log("\n[STAGE 5] Module Scores", log_file)
    compute_module_scores(adata_hcm, log_file)

    # STAGE 6: Clustering & Trajectory
    log("\n[STAGE 6] Clustering & Trajectory", log_file)
    chosen_res = run_resolution_scan(adata_hcm, config, log_file)
    checkpoint(adata_hcm, "post_leiden", config.OUT_DIR, config.PREFIX, log_file)
    trajectory = build_trajectory(adata_hcm, log_file)
    checkpoint(adata_hcm, "post_paga", config.OUT_DIR, config.PREFIX, log_file)

    traj_mask = adata_hcm.obs["leiden"].isin(trajectory)
    adata_traj = adata_hcm[traj_mask].copy()

    # STAGE 7: Pseudotime Analysis
    log("\n[STAGE 7] Pseudotime Analysis", log_file)
    adata_traj = adata_traj[adata_traj.obs["pseudotime"] < 1.0].copy()
    adata_traj.obs["pt_bin"] = pd.cut(
        adata_traj.obs["pseudotime"], bins=config.N_PSEUDOTIME_BINS,
        labels=[f"Bin{i+1}" for i in range(config.N_PSEUDOTIME_BINS)]
    )

    # STAGE 8: Pseudo-bulk & GSEA
    log("\n[STAGE 8] Pseudo-bulk Differential Expression", log_file)
    pb_lfc = run_pseudobulk(adata_traj, trajectory, config, log_file)
    ranked = pb_lfc["logFC"].dropna().sort_values(ascending=False)
    ranked = ranked[~ranked.index.duplicated()]
    gsea_results = run_gsea(ranked, config, log_file)

    # STAGE 9: Marker Genes
    log("\n[STAGE 9] Marker Gene Analysis", log_file)
    run_marker_analysis(adata_traj, trajectory, config, log_file)
    checkpoint(adata_traj, "post_markers", config.OUT_DIR, config.PREFIX, log_file)

    # STAGE 10: LIANA
    if LIANA_AVAILABLE:
        log("\n[STAGE 10] Cell-Cell Communication (LIANA)", log_file)
        try:
            adata_liana = adata_traj.copy()
            adata_liana.raw = None
            li.mt.rank_aggregate(adata_liana, groupby="leiden", use_raw=False,
                                verbose=False, n_perms=100)
            liana_res = adata_liana.uns["liana_res"].copy()
            liana_res.to_csv(os.path.join(config.OUT_DIR,
                                          f"{config.PREFIX}liana_interactions.csv"),
                            index=False)
            log("  LIANA complete", log_file)
        except Exception as e:
            log(f"  LIANA failed: {e}", log_file)

    # STAGE 11: Epigenetic Analysis
    log("\n[STAGE 11] Epigenetic Analysis", log_file)
    run_epigenetic_analysis(adata_traj, trajectory, config, log_file)

    # STAGE 12: ChIP-seq Integration
    log("\n[STAGE 12] ChIP-seq Integration", log_file)
    integrate_chipseq(adata_traj, trajectory, config, log_file)

    # STAGE 13: WGCNA & Regulon Scoring
    log("\n[STAGE 13] WGCNA & Regulon Integration", log_file)
    wgcna_cols = load_and_score_wgcna(adata_hcm, config, log_file)
    regulon_cols = load_and_score_regulons(adata_hcm, config, log_file)

    # STAGE 14: Per-Patient Statistics
    log("\n[STAGE 14] Per-Patient Statistics", log_file)
    stats_df = run_patient_statistics(adata_hcm, config, log_file)

    # STAGE 15: GRN Network
    log("\n[STAGE 15] TF-Epigenetic GRN", log_file)
    grn_tfs = [g for g in CARDIAC_TFS if g in adata_traj.var_names]
    grn_epi = [g for g in list(set(g for gs in EPI_CATEGORIES.values() for g in gs))
               if g in adata_traj.var_names]
    grn_genes = list(dict.fromkeys(grn_tfs + grn_epi))

    if len(grn_genes) > 5:
        X_grn = adata_traj[:, grn_genes].X
        if issparse(X_grn):
            X_grn = X_grn.toarray()
        corr_mat = np.corrcoef(X_grn.T)
        corr_df = pd.DataFrame(corr_mat, index=grn_genes, columns=grn_genes)

        G = nx.Graph()
        G.add_nodes_from(grn_genes)
        for i, g1 in enumerate(grn_genes):
            for j, g2 in enumerate(grn_genes):
                if j <= i:
                    continue
                r = corr_df.loc[g1, g2]
                if abs(r) >= config.GRN_CORR_THRESHOLD:
                    G.add_edge(g1, g2, weight=abs(r), sign=1 if r > 0 else -1)

        fig, ax = plt.subplots(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42, k=1.5)
        node_colors = ["#E63946" if g in grn_tfs else "#8338EC" for g in G.nodes()]
        edge_colors = ["#2A9D8F" if G[u][v]["sign"] > 0 else "#E9C46A"
                       for u, v in G.edges()]
        edge_widths = [G[u][v]["weight"] * 3 for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=600, alpha=0.9)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                               width=edge_widths, alpha=0.7)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="black")

        legend_elements = [
            mpatches.Patch(color="#E63946", label="TF"),
            mpatches.Patch(color="#8338EC", label="Epigenetic factor"),
            mpatches.Patch(color="#2A9D8F", label="Positive correlation"),
            mpatches.Patch(color="#E9C46A", label="Negative correlation"),
        ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=9)
        ax.set_title(f"TF-Epigenetic Co-expression GRN (|r| >= {config.GRN_CORR_THRESHOLD})")
        ax.axis("off")
        plt.tight_layout()
        savefig(fig, "GRN_TF_epi_network", config.OUT_DIR, config.PREFIX, log_file)

    # FINAL SUMMARY
    log("\n" + "=" * 60, log_file)
    log("PIPELINE COMPLETE", log_file)
    log("=" * 60, log_file)

    outputs = sorted([f for f in os.listdir(config.OUT_DIR) if f.startswith(config.PREFIX)])
    log(f"Generated {len(outputs)} output files:", log_file)
    for f in outputs:
        size = os.path.getsize(os.path.join(config.OUT_DIR, f))
        log(f"  {f:60s} {size / 1024:.1f} KB", log_file)

    return adata_hcm, adata_traj


if __name__ == "__main__":
    adata_hcm, adata_traj = main()

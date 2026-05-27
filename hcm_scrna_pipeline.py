"""
hcm_scrna_pipeline.py
=====================
HCM (Hypertrophic Cardiomyopathy) single-cell RNA-seq analysis pipeline.

Modules
-------
  Phase 0  : Imports & global configuration
  Phase 1  : Load, clean, concatenate plates
  Phase 2  : QC gene filtering, normalization, reference mapping
  Module 1 : Gene module scores (HCM / Immune / Epigenetic)
  Module 2 : Leiden resolution scan + clustering
  Module 3 : UMAP visualisations
  Module 4 : PAGA trajectory + diffusion pseudotime
  Module 5 : Pseudotime bin dynamics
  Module 6 : Pseudo-bulk (early vs late)
  Module 7 : GSEA on pseudo-bulk ranked list
  Module 8 : Per-cluster marker genes + manual dotplot
  Module 9 : LIANA cell–cell communication
  Module 10: Epigenetic factor analysis
  Module 11: TF vs epigenetic pseudotime correlation  [NaN fix applied]
  Module 12: TF–Epigenetic co-expression GRN (basic)
  Module 13: H3K27ac ChIP-seq integration + scoring
  Module 13B: GSEA on H3K27ac enhancer genes
  Module A : WGCNA module scoring & visualisation
  Module B : Regulon scoring & visualisation
  Module C : Per-patient statistical analysis
  Module D : Full GRN network analysis

Requirements
------------
  scanpy anndata pandas numpy scipy matplotlib seaborn networkx gseapy
  statsmodels pyranges liana (optional)

Dataset : GSE141910 — HCM patient cardiac scRNA-seq (4 patients, 14 plates)
"""

# ============================================================
# PHASE 0 — IMPORTS & GLOBAL CONFIGURATION
# ============================================================

import os
import gc
import time
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
import gseapy as gp

from scipy import stats
from scipy.sparse import issparse
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance as scipy_distance
from scipy.stats import pointbiserialr, mannwhitneyu, spearmanr  # FIX-15
from statsmodels.stats.multitest import multipletests             # FIX-3

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1
sc.settings.figdir   = "./"

# ── Output directories and file prefix ───────────────────────
OUT_DIR    = "./finaletai"
GRN_DIR    = os.path.join(OUT_DIR, "GRN")
PREFIX     = "final_"
GRN_PREFIX = "0806_GRN_"
LOG_FILE   = "pipeline_log.txt"

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(GRN_DIR,  exist_ok=True)

# ── Colour palettes ───────────────────────────────────────────
PAL_PATIENT = {
    "P1": "#E63946",
    "P2": "#457B9D",
    "P3": "#2A9D8F",
    "P4": "#E9C46A",
}

# Base vCM colour dict; unknown categories fall back to grey (FIX-1)
_BASE_VCM_COLORS = {
    "vCM1": "#264653",
    "vCM2": "#B8B8B8",
    "vCM3": "#E76F51",
    "vCM4": "#F4A261",
    "vCM5": "#A8DADC",
}

# Group colours for stripplot / violin comparison plots
PAL_GROUP = {
    "vCM1\n(healthy)":      "#264653",
    "vCM3+vCM4\n(disease)": "#E76F51",
}

# ── File paths for external data ──────────────────────────────
WGCNA_MODULE_CSV = "rna_counts_raw/outputs/tables/WGCNA_module_summary_GO.csv"
REGULON_CSV      = "rna_counts_raw/outputs/regulons/ALL_REGULONS.csv"
CHIP_FILE        = "GSE165303_H3K27ac_ChIP-Seq_peaks_master_table.txt"
PROMOTER_BED     = "genomewide_TSS_pm5kb.bed"
EPI_FILE         = "epifactors_human.txt"

GMT_FILES = {
    "GO_BP": "c5.go.bp.v2023.1.Hs.symbols.gmt",
    "GO_MF": "c5.go.mf.v2023.1.Hs.symbols.gmt",
    "KEGG":  "c2.cp.kegg.v2023.1.Hs.symbols.gmt",
}

# ── Patient / plate manifest ──────────────────────────────────
PATIENTS_PLATES = {
    "P1": [
        "GSM4103885_Patient1Plate1", "GSM4103886_Patient1Plate2",
        "GSM4103887_Patient1Plate3", "GSM4103888_Patient1Plate4",
    ],
    "P2": [
        "GSM4103889_Patient2Plate1", "GSM4103890_Patient2Plate2",
        "GSM4103891_Patient2Plate3", "GSM4103892_Patient2Plate4",
    ],
    "P3": [
        "GSM4103893_Patient3Plate1", "GSM4103894_Patient3Plate2",
        "GSM4103895_Patient3Plate3", "GSM4103896_Patient3Plate4",
    ],
    "P4": [
        "GSM4103897_Patient4Plate1", "GSM4103898_Patient4Plate2",
    ],
}

# ── Trajectory tipping-point clusters (FIX-13) ───────────────
# Clusters identified from PAGA as the disease-phase transition point.
# Update these after running Module 4 if your trajectory differs.
TIPPING_CLUSTERS = ["14", "4"]

# ============================================================
# GENE LISTS
# ============================================================

HCM_CORE_GENES = [
    "NPPA", "NPPB", "FN1", "COL1A1", "FGF12", "CREB5", "MYH7", "POSTN",
]

# --- Curated short list of epigenetic regulators ---------------
EPI_GENES = [
    "CHD3", "HDAC1", "HDAC2", "HDAC3", "HDAC5",
    "DNMT1", "DNMT3A", "DNMT3B", "EZH2", "EZH1",
    "BRD2", "BRD4", "KDM6B", "KDM5C", "EP300",
    "CREBBP", "SMARCA4", "ARID1A", "SIRT1",
    "KMT2A", "KMT2D", "SETD1A", "KDM1A", "NCOR1",
]

# --- Epigenetic category dictionary used in modules 10 and D --
EPI_CATEGORIES = {
    "Writers_KMT":     [
        "KMT2A", "KMT2B", "KMT2C", "KMT2D", "SETD1A", "SETD1B", "SETDB1",
        "EZH1", "EZH2", "NSD1", "NSD2", "SMYD2", "PRMT1", "PRMT5",
    ],
    "Writers_KAT":     [
        "EP300", "CREBBP", "KAT2A", "KAT2B", "KAT6A", "KAT6B", "KAT7", "KAT8", "TAF1",
    ],
    "Writers_DNMT":    ["DNMT1", "DNMT3A", "DNMT3B", "DNMT3L"],
    "Erasers_HDAC":    [
        "HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6", "HDAC7",
        "HDAC8", "HDAC9", "SIRT1", "SIRT2", "SIRT3", "SIRT6",
    ],
    "Erasers_KDM":     [
        "KDM1A", "KDM2A", "KDM4A", "KDM4B", "KDM5A",
        "KDM5B", "KDM5C", "KDM6A", "KDM6B", "KDM7A",
    ],
    "Erasers_TET":     ["TET1", "TET2", "TET3"],
    "Readers_BRD":     ["BRD2", "BRD3", "BRD4", "BRD7", "BRD9", "BRDT"],
    "Readers_PHD":     ["PHF2", "PHF8", "PHF10", "ING1", "ING2", "ING3"],
    "Remodellers_SWI_SNF": [
        "SMARCA4", "SMARCA2", "SMARCB1", "ARID1A", "ARID1B", "ARID2",
        "SMARCC1", "SMARCC2", "SMARCD1",
    ],
    "Remodellers_CHD":  [
        "CHD1", "CHD2", "CHD3", "CHD4", "CHD7", "CHD8", "CHD9",
    ],
    "Remodellers_NuRD": [
        "CHD3", "CHD4", "HDAC1", "HDAC2", "MBD2", "MBD3",
        "RBBP4", "RBBP7", "GATAD2A", "GATAD2B",
    ],
    "PRC_Polycomb":    [
        "EZH2", "EED", "SUZ12", "RING1", "RNF2",
        "CBX2", "CBX4", "CBX6", "CBX7", "CBX8",
    ],
    "Activators_MED":  [
        "MED1", "MED12", "MED13", "MED14", "MED24",
        "NCOA1", "NCOA2", "NCOA3",
    ],
    "Silencers":       [
        "NCOR1", "NCOR2", "SIN3A", "SIN3B", "RCOR1",
        "NR2C2", "NR2F1", "NR2F2",
    ],
}

# Flat gene→category map used across multiple modules
CAT_MAP = {g: cat for cat, genes in EPI_CATEGORIES.items() for g in genes}

# Bulk-derived TF list for Modules 11, 12, and D
BULK_TFS = [
    "ATF1", "NFE2L2", "E2F4", "CEBPB", "NR2C2",
    "STAT4", "SOX9", "IRF9", "MEF2A", "MEF2C", "MEF2D",
    "GATA4", "NFATC1", "SP1", "SP3", "KLF4", "KLF5",
    "TEAD1", "TEAD4", "YAP1", "WWTR1",
    "TP53", "E2F1", "MYC", "MYCN", "MAX",
    "RUNX1", "RUNX2", "ETS1", "FLI1",
    "NKX2-5", "TBX5", "HAND1", "HAND2",
    "PPARA", "PPARG", "RXRA", "NR3C1",
]

# Disease gene sets used in Module 13
SARCOMERIC_GENES = ["MYH7", "MYH6", "MYBPC3", "TNNT2", "TNNI3", "ACTC1", "TPM1"]
STRESS_GENES     = ["NPPA", "NPPB", "ATF1", "DDIT3", "HSPA1A", "HSPB1"]
EPIGENETIC_GENES = ["BRD4", "EZH1", "SUZ12", "SMARCA4", "CHD9", "NCOR1", "BRD3"]
FIBROSIS_GENES   = ["TGFB1", "COL1A1", "COL3A1", "POSTN", "ACTA2", "FN1"]

# WGCNA & regulon static label maps (used in Module C)
WGCNA_LABELS_STATIC = {
    "WGCNA_M0":  "M0: IL-1 Response",
    "WGCNA_M1":  "M1: Translation",
    "WGCNA_M2":  "M2: Ceramide Catabolism",
    "WGCNA_M3":  "M3: ECM Organization",
    "WGCNA_M4":  "M4: ATP Biosynthesis",
    "WGCNA_M5":  "M5: Macrophage/Foam Cell",
    "WGCNA_M6":  "M6: Amino Acid Transport",
    "WGCNA_M7":  "M7: T Cell Activation",
    "WGCNA_M8":  "M8: Cold Thermogenesis",
    "WGCNA_M9":  "M9: Mitotic Chromatid",
    "WGCNA_M10": "M10: BCR Signaling",
}
REG_LABELS_STATIC = {
    "score_REG1_UP_TFactive":   "REG1: UP + TF-active",
    "score_REG2_UP_TFindep":    "REG2: UP + TF-independent",
    "score_REG3_DOWN_TFrep":    "REG3: DOWN + TF-repressed",
    "score_REG5_WGCNA_ECM":     "REG5: WGCNA ECM hub",
}
REGULON_LABELS = {
    "REG1_UP_TFactive":   "REG1: UP + TF-active targets",
    "REG2_UP_TFindep":    "REG2: UP + TF-independent",
    "REG3_DOWN_TFrep":    "REG3: DOWN + TF-repressed",
    "REG4_DOWN_chromclo": "REG4: DOWN + closed chromatin",
    "REG5_WGCNA_ECM":     "REG5: WGCNA ECM/stromal hub",
}
REG_COLORS = {
    "REG1_UP_TFactive":   "#D62728",
    "REG2_UP_TFindep":    "#FF7F0E",
    "REG3_DOWN_TFrep":    "#1F77B4",
    "REG4_DOWN_chromclo": "#9467BD",
    "REG5_WGCNA_ECM":     "#2CA02C",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def log(msg, also_print=True):
    """Append a timestamped message to the log file and optionally print it."""
    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOG_FILE, "a") as fh:
        fh.write(line + "\n")
    if also_print:
        print(line)


def checkpoint(adata, tag):
    """Save a .h5ad checkpoint to disk."""
    path = f"checkpoint_{tag}.h5ad"
    adata.write_h5ad(path)
    log(f"  Checkpoint saved → {path}")


def savefig(name, dpi=200, subdir=None):
    """Save the current figure to OUT_DIR (or a subfolder), then close it."""
    d = os.path.join(OUT_DIR, subdir) if subdir else OUT_DIR
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, PREFIX + name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def savefig_grn(name, dpi=200):
    """Save a GRN figure to GRN_DIR with the GRN prefix."""
    path = os.path.join(GRN_DIR, GRN_PREFIX + name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def build_safe_palette(adata, obs_col, base_palette):
    """
    Build a palette dict covering every category present in adata.obs[obs_col].
    Categories not in base_palette receive a neutral grey fallback. (FIX-1)
    """
    cats = adata.obs[obs_col].cat.categories.tolist()
    return {k: base_palette.get(k, "#CCCCCC") for k in cats}


def score_module(adata, genes, key):
    """
    Score a gene module with sc.tl.score_genes.
    Genes not present in adata.var_names are silently skipped.
    If no genes are present, the score column is set to 0.0.
    """
    present = [g for g in genes if g in adata.var_names]
    log(f"  {key}: {len(present)}/{len(genes)} genes present")
    if len(present) == 0:
        adata.obs[key] = 0.0
        return
    sc.tl.score_genes(adata, gene_list=present, score_name=key, use_raw=False)


def bh_correct(pvals):
    """Apply Benjamini-Hochberg FDR correction to a Series of p-values."""
    if len(pvals) == 0:
        return pvals
    _, padj, _, _ = multipletests(pvals.values, method="fdr_bh")
    return pd.Series(padj, index=pvals.index)


def friendly(col):
    """Return a human-readable label for a score column name."""
    if col in WGCNA_LABELS_STATIC:
        return WGCNA_LABELS_STATIC[col]
    if col in REG_LABELS_STATIC:
        return REG_LABELS_STATIC[col]
    return col.replace("score_", "").replace("_", " ")


def cross_corr_lead(series1, series2, max_lag=3):
    """
    Compute the best cross-correlation lag between two series.
    NaN/Inf values are masked out before correlating. (FIX-8 adjacent)

    Returns
    -------
    best_lag : int
    best_r   : float
    """
    best_lag, best_r = 0, -np.inf
    s1 = np.array(series1, dtype=float)
    s2 = np.array(series2, dtype=float)
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            a, b = s1[:-lag], s2[lag:]
        elif lag < 0:
            a, b = s1[-lag:], s2[:len(s2) + lag]
        else:
            a, b = s1, s2
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            continue
        r = np.corrcoef(a[mask], b[mask])[0, 1]
        if np.isfinite(r) and r > best_r:
            best_r, best_lag = r, lag
    return best_lag, best_r


def pseudobulk(adata, group_col):
    """
    Compute pseudo-bulk mean expression per group.

    Uses raw counts when adata.raw is available, otherwise uses adata.X.
    Returns a DataFrame of shape (n_genes × n_groups).
    """
    groups = adata.obs[group_col].unique()
    pb = {}
    for g in groups:
        mask = adata.obs[group_col] == g
        X = adata[mask].raw.X if adata.raw is not None else adata[mask].X
        if issparse(X):
            X = X.toarray()
        pb[g] = X.mean(axis=0)
    var_names = adata.raw.var_names if adata.raw is not None else adata.var_names
    return pd.DataFrame(pb, index=var_names)


def get_pseudotime_expression(adata, genes, n_bins=10):
    """
    Bin cells by pseudotime and return mean expression per bin.

    Returns a DataFrame of shape (n_bins × n_genes).
    """
    bins   = pd.cut(
        adata.obs["pseudotime"], bins=n_bins,
        labels=[f"B{i+1}" for i in range(n_bins)],
    )
    result = {}
    for gene in genes:
        if gene not in adata.var_names:
            continue
        X = adata[:, gene].X
        X = X.toarray().flatten() if issparse(X) else X.flatten()
        result[gene] = pd.Series(X, index=adata.obs_names).groupby(bins).mean()
    return pd.DataFrame(result)


def build_trajectory(start_cluster, conn_matrix, score_series, n_steps=6):
    """
    Greedily walk the PAGA connectivity graph, always preferring
    neighbours with higher HCM score (disease-progression direction).

    Parameters
    ----------
    start_cluster : str  — Leiden cluster label with lowest HCM score
    conn_matrix   : pd.DataFrame — PAGA connectivity matrix
    score_series  : pd.Series   — mean HCM score per cluster
    n_steps       : int         — maximum trajectory length

    Returns
    -------
    list of cluster-label strings in trajectory order
    """
    path    = [start_cluster]
    visited = {start_cluster}
    current = start_cluster
    for _ in range(n_steps - 1):
        row        = conn_matrix.loc[current].drop(labels=list(visited))
        candidates = {
            nb: w for nb, w in row.items()
            if w > 0.05 and score_series[nb] > score_series[current]
        }
        if not candidates:
            row2 = conn_matrix.loc[current].drop(labels=list(visited))
            if row2.empty:
                break
            nxt = row2.idxmax()
        else:
            nxt = max(candidates, key=candidates.get)
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return path


# ============================================================
# PHASE 1 — LOAD & CLEAN EACH PLATE
# ============================================================

def load_and_clean_plate(tsv_path, patient_id, plate_id):
    """
    Load a single plate TSV, transpose to cells-×-genes,
    strip gene-name suffixes (gene__suffix → gene), and attach metadata.

    Parameters
    ----------
    tsv_path   : str — path to the .TranscriptCounts.tsv.gz file
    patient_id : str — e.g. 'P1'
    plate_id   : str — e.g. 'GSM4103885_Patient1Plate1'

    Returns
    -------
    AnnData with shape (n_cells × n_genes)
    """
    df    = pd.read_csv(tsv_path, sep="\t", index_col=0)
    adata = sc.AnnData(df.T)                             # rows = cells

    # Strip Ensembl/version suffixes from gene names (e.g. ENSG00001.2__GENEID)
    adata.var_names = adata.var_names.str.split("__").str[0]
    adata.var_names_make_unique()

    adata.obs["patient_id"] = patient_id
    adata.obs["plate_id"]   = plate_id
    return adata


log("=" * 60)
log("PHASE 1 — Loading plates")
log("=" * 60)

all_adatas = []
for p_id, plate_list in PATIENTS_PLATES.items():
    for plate_file in plate_list:
        folder = f"patient{p_id[1]}"
        path   = f"{folder}/{plate_file}.TranscriptCounts.tsv.gz"
        log(f"  Loading: {path}")
        adata  = load_and_clean_plate(path, p_id, plate_file)
        all_adatas.append(adata)

# Concatenate all plates into one master AnnData
adata_hcm = ad.concat(all_adatas, label="batch", join="inner")
adata_hcm.obs_names_make_unique()
del all_adatas
gc.collect()

log(f"\nMaster object: {adata_hcm.n_obs} cells × {adata_hcm.n_vars} genes")

# ============================================================
# PHASE 2 — QC FILTERING, NORMALIZATION, REFERENCE MAPPING
# ============================================================

log("\nPhase 2 — QC, normalization, reference mapping")

# Remove mitochondrial, ribosomal, and ERCC spike-in genes
junk_mask = (
    adata_hcm.var_names.str.startswith("MT-")   |
    adata_hcm.var_names.str.startswith("ERCC-") |
    adata_hcm.var_names.str.startswith("RPS")   |
    adata_hcm.var_names.str.startswith("RPL")
)
adata_hcm = adata_hcm[:, ~junk_mask].copy()
log(f"  After junk-gene filter: {adata_hcm.shape}")

# Store raw counts before normalisation
adata_hcm.raw = adata_hcm

# Library-size normalise to 10 000 counts per cell, then log1p
sc.pp.normalize_total(adata_hcm, target_sum=1e4)
sc.pp.log1p(adata_hcm)

# Quick sanity check on normalisation range
max_val  = adata_hcm.X.max()
mean_val = adata_hcm.X.mean()
log(f"  Normalisation check — max: {max_val:.4f}, mean: {mean_val:.4f}")
if max_val > 10:
    log("  WARNING: max expression > 10; check if log1p was applied")
elif max_val < 1:
    log("  WARNING: max expression < 1; possible double-log")
else:
    log("  Normalisation OK")

# ── Reference mapping via sc.tl.ingest ───────────────────────
log("  Loading healthy vCM reference...")
adata_ref = sc.read_h5ad("healthy_vCM1_4_sparse_pca_umap.h5ad")

common_genes = adata_ref.var_names.intersection(adata_hcm.var_names)
log(f"  Aligning on {len(common_genes)} common genes")
adata_ref = adata_ref[:, common_genes].copy()
adata_hcm = adata_hcm[:, common_genes].copy()

sc.tl.ingest(adata_hcm, adata_ref, obs="cell_states")
log("  Reference mapping complete")

# Distribution check
alignment_table = pd.crosstab(
    adata_hcm.obs["patient_id"],
    adata_hcm.obs["cell_states"],
)
alignment_pct = pd.crosstab(
    adata_hcm.obs["patient_id"],
    adata_hcm.obs["cell_states"],
    normalize="index",
) * 100
print("\nCounts:\n",      alignment_table)
print("\nPercentages:\n", alignment_pct.round(2))

# (FIX-1) Build safe palette now that ingest categories are known
PAL_VCM = build_safe_palette(adata_hcm, "cell_states", _BASE_VCM_COLORS)
log(f"  PAL_VCM covers: {list(PAL_VCM.keys())}")

# (FIX-13) Define PATIENTS list from obs for downstream modules
PATIENTS = sorted(adata_hcm.obs["patient_id"].unique().tolist())

# ============================================================
# MODULE 1 — GENE MODULE SCORES
# ============================================================

log("\n[MODULE 1] Computing gene module scores...")

# (FIX-2) Restrict working object to top-5000 HVGs before neighbours / UMAP
sc.pp.highly_variable_genes(adata_hcm, n_top_genes=5000,
                             flavor="seurat_v3", span=1.0)
n_hvg = adata_hcm.var["highly_variable"].sum()
log(f"  HVGs selected: {n_hvg}")

adata_hcm_full = adata_hcm.copy()           # full-gene object kept for scoring
adata_hcm      = adata_hcm[:, adata_hcm.var["highly_variable"]].copy()
log(f"  Working object after HVG filter: {adata_hcm.n_obs} × {adata_hcm.n_vars}")

# Score each module; missing genes are skipped gracefully
score_module(adata_hcm, HCM_CORE_GENES, "HCM_score")
score_module(adata_hcm, EPI_GENES,      "Epi_score")

# ============================================================
# MODULE 2 — RESOLUTION SCAN → OPTIMAL LEIDEN
# ============================================================

log("\n[MODULE 2] Resolution scan...")

# Run PCA if not already present (scales data first)
use_rep = "X_pca_harmony" if "X_pca_harmony" in adata_hcm.obsm else "X_pca"
if "X_pca" not in adata_hcm.obsm:
    sc.pp.scale(adata_hcm, max_value=10)
    sc.tl.pca(adata_hcm, svd_solver="arpack")

# Build neighbour graph once, then compute UMAP
sc.pp.neighbors(adata_hcm, n_neighbors=15, use_rep=use_rep)
sc.tl.umap(adata_hcm)

resolutions    = np.arange(0.3, 2.1, 0.2)
n_clusters_at  = {}
for res in resolutions:
    sc.tl.leiden(adata_hcm, resolution=round(res, 1),
                 key_added=f"leiden_{res:.1f}")
    n_clusters_at[round(res, 1)] = adata_hcm.obs[f"leiden_{res:.1f}"].nunique()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(n_clusters_at.keys()), list(n_clusters_at.values()),
        "o-", color="#2E75B6", lw=2, ms=7)
ax.set_xlabel("Leiden Resolution"); ax.set_ylabel("Number of Clusters")
ax.set_title("Leiden Resolution Scan — HCM scRNA-seq")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"leiden_resolution_scan.pdf", dpi=150)
plt.close()
log(f"  Resolution scan saved. Clusters: {n_clusters_at}")

# Pick resolution at the first plateau (≤2 new clusters per step)
res_vals   = sorted(n_clusters_at.keys())
CHOSEN_RES = 0.8
for i in range(1, len(res_vals) - 1):
    delta = abs(n_clusters_at[res_vals[i + 1]] - n_clusters_at[res_vals[i]])
    if delta <= 2:
        CHOSEN_RES = res_vals[i]
        break

sc.tl.leiden(adata_hcm, resolution=CHOSEN_RES, key_added="leiden")
N_CLUSTERS = adata_hcm.obs["leiden"].nunique()
log(f"  Chosen resolution: {CHOSEN_RES} → {N_CLUSTERS} clusters")
checkpoint(adata_hcm, "post_leiden")

# ============================================================
# MODULE 3 — THREE UMAPs
# ============================================================

log("\n[MODULE 3] Three UMAPs...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata_hcm, color="patient_id", ax=axes[0],
           show=False, title="Patient ID",
           palette=PAL_PATIENT, legend_loc="on data")
sc.pl.umap(adata_hcm, color="leiden", ax=axes[1],
           show=False, title=f"Leiden (res={CHOSEN_RES})",
           legend_loc="on data", legend_fontsize=7)
sc.pl.umap(adata_hcm, color="HCM_score", ax=axes[2],
           show=False, title="HCM Score",
           color_map="RdYlBu_r")
plt.tight_layout()
plt.savefig("three_umaps.pdf", dpi=150, bbox_inches="tight")
plt.close()

# Supplementary panel: vCM reference state + module scores
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sc.pl.umap(adata_hcm, color="cell_states", ax=axes[0],
           show=False, title="Reference vCM State", palette=PAL_VCM)
sc.pl.umap(adata_hcm, color="Epi_score",   ax=axes[1],
           show=False, title="Epigenetic Score",   color_map="Purples")
sc.pl.umap(adata_hcm, color="HCM_score",   ax=axes[2],
           show=False, title="HCM Score",          color_map="RdYlBu_r")
plt.tight_layout()
plt.savefig("three_umaps_vcm.pdf", dpi=150, bbox_inches="tight")
plt.close()
log("  UMAPs saved.")

# ============================================================
# MODULE 4 — PAGA TRAJECTORY + DIFFUSION PSEUDOTIME
# ============================================================

log("\n[MODULE 4] PAGA trajectory...")

sc.tl.paga(adata_hcm, groups="leiden")

# Root = cluster with lowest mean HCM score (healthiest state)
cluster_hcm_mean  = adata_hcm.obs.groupby("leiden")["HCM_score"].mean().sort_values()
TRAJ_ROOT_CLUSTER = cluster_hcm_mean.index[0]
root_mask         = adata_hcm.obs["leiden"] == TRAJ_ROOT_CLUSTER
root_cell_idx     = int(
    np.where(root_mask)[0][
        np.argmin(adata_hcm.obs.loc[root_mask, "HCM_score"].values)
    ]
)
adata_hcm.uns["iroot"] = root_cell_idx
log(f"  Root cluster: {TRAJ_ROOT_CLUSTER} "
    f"(lowest HCM score = {cluster_hcm_mean.iloc[0]:.3f})")

sc.pp.neighbors(adata_hcm, n_neighbors=15, use_rep=use_rep)
sc.tl.diffmap(adata_hcm)
sc.tl.dpt(adata_hcm)
adata_hcm.obs["pseudotime"] = adata_hcm.obs["dpt_pseudotime"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc.pl.paga(adata_hcm, color="HCM_score", ax=axes[0], show=False,
           title="PAGA — coloured by mean HCM score",
           colorbar=True, node_size_scale=2)
sc.pl.paga(adata_hcm, color="leiden",    ax=axes[1], show=False,
           title="PAGA — Leiden clusters", node_size_scale=2)
plt.tight_layout()
plt.savefig("paga_overview.pdf", dpi=150, bbox_inches="tight")
plt.close()

paga_conn = pd.DataFrame(
    adata_hcm.uns["paga"]["connectivities"].toarray(),
    index=adata_hcm.obs["leiden"].cat.categories,
    columns=adata_hcm.obs["leiden"].cat.categories,
)

TRAJECTORY = build_trajectory(
    TRAJ_ROOT_CLUSTER, paga_conn, cluster_hcm_mean,
    n_steps=min(8, N_CLUSTERS),
)
log(f"  Trajectory: {' → '.join(TRAJECTORY)}")

# String version used by violin/scatter functions (FIX-13)
TRAJECTORY_ORDER = [str(c) for c in TRAJECTORY]

traj_df = pd.DataFrame({
    "cluster":        TRAJECTORY,
    "mean_HCM_score": [cluster_hcm_mean[c] for c in TRAJECTORY],
    "stage":          range(1, len(TRAJECTORY) + 1),
})
traj_df.to_csv("trajectory_order.csv", index=False)

ordered_clusters = cluster_hcm_mean.index.tolist()
paga_ordered     = paga_conn.loc[ordered_clusters, ordered_clusters]
fig, ax = plt.subplots(figsize=(max(8, N_CLUSTERS//2), max(6, N_CLUSTERS//2)))
sns.heatmap(paga_ordered, ax=ax, cmap="YlOrRd", linewidths=0.3,
            xticklabels=True, yticklabels=True, square=True,
            cbar_kws={"label": "PAGA connectivity"})
ax.set_title("PAGA Connectivity Matrix (ordered by HCM score)")
plt.tight_layout()
plt.savefig("paga_connectivity_heatmap.pdf", dpi=150, bbox_inches="tight")
plt.close()
log("  PAGA saved.")
checkpoint(adata_hcm, "post_paga")

# ============================================================
# MODULE 5 — PSEUDOTIME BINS + MODULE SCORE DYNAMICS
# ============================================================

log("\n[MODULE 5] Pseudotime bins...")

adata_hcm = adata_hcm[adata_hcm.obs["pseudotime"] < 1.0].copy()
n_bins    = 10
adata_hcm.obs["pt_bin"] = pd.cut(
    adata_hcm.obs["pseudotime"], bins=n_bins,
    labels=[f"Bin{i+1}" for i in range(n_bins)],
)

score_cols_pt = [c for c in ["HCM_score", "Epi_score"] if c in adata_hcm.obs.columns]
bin_scores    = adata_hcm.obs.groupby("pt_bin", observed=True)[score_cols_pt].mean()

fig, ax = plt.subplots(figsize=(10, 5))
colors_pt = ["#E63946", "#8338EC"]
for col, color in zip(score_cols_pt, colors_pt):
    ax.plot(bin_scores.index, bin_scores[col], "o-",
            label=col, color=color, lw=2, ms=6)
ax.set_xlabel("Pseudotime Bin"); ax.set_ylabel("Mean Score")
ax.set_title("Module Score Dynamics Along Pseudotime")
ax.legend(); ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("module_scores_pseudotime.pdf", dpi=150)
plt.close()
log("  Pseudotime bins saved.")

# ── Build trajectory subset object ───────────────────────────
traj_cells  = adata_hcm.obs["leiden"].isin(TRAJECTORY)
adata_traj  = adata_hcm[traj_cells].copy()

# ============================================================
# MODULE 6 — PSEUDO-BULK: EARLY vs LATE
# ============================================================

log("\n[MODULE 6] Pseudo-bulk...")

half           = len(TRAJECTORY) // 2
EARLY_CLUSTERS = TRAJECTORY[:half]
LATE_CLUSTERS  = TRAJECTORY[half:]
log(f"  Early: {EARLY_CLUSTERS}")
log(f"  Late : {LATE_CLUSTERS}")

adata_traj.obs["traj_half"] = "Late"
adata_traj.obs.loc[
    adata_traj.obs["leiden"].isin(EARLY_CLUSTERS), "traj_half"
] = "Early"

pb_df = pseudobulk(adata_traj, "traj_half")
pb_df.to_csv("pseudobulk_early_vs_late.csv")

pb_lfc = pd.DataFrame({
    "Early_mean": pb_df["Early"],
    "Late_mean":  pb_df["Late"],
    "logFC":      np.log2(pb_df["Late"] + 1e-6) - np.log2(pb_df["Early"] + 1e-6),
}).sort_values("logFC", ascending=False)
pb_lfc.to_csv("pseudobulk_logFC.csv")

top_both = pd.concat([pb_lfc.head(30), pb_lfc.tail(30)])
fig, ax  = plt.subplots(figsize=(8, 10))
colors_bar = ["#E63946" if v > 0 else "#457B9D" for v in top_both["logFC"]]
ax.barh(top_both.index, top_both["logFC"], color=colors_bar)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("log2FC (Late vs Early trajectory)")
ax.set_title("Pseudo-bulk: Late vs Early")
plt.tight_layout()
plt.savefig("pseudobulk_lfc_barplot.pdf", dpi=150)
plt.close()
log("  Pseudo-bulk saved.")

# ============================================================
# MODULE 7 — GSEA ON PSEUDO-BULK
# ============================================================

log("\n[MODULE 7] GSEA on pseudo-bulk...")

ranked_gsea = pb_lfc["logFC"].dropna().sort_values(ascending=False)
ranked_gsea = ranked_gsea[~ranked_gsea.index.duplicated()]

for name, gmt in GMT_FILES.items():
    if not os.path.exists(gmt):
        log(f"  {name}: GMT not found, skipping")
        continue
    try:
        res = gp.prerank(
            rnk=ranked_gsea, gene_sets=gmt,
            min_size=10, max_size=500,
            permutation_num=500,
            outdir=None, seed=42, verbose=False,
        )
        df = res.res2d.copy()

        # (FIX-3) BH correction via statsmodels
        pvals = df["NOM p-val"].fillna(1.0).values
        _, padj, _, _ = multipletests(pvals, method="fdr_bh")
        df["BH_FDR"] = padj
        df_sig = df[df["BH_FDR"] < 0.25].sort_values("NES", ascending=False)

        df_sig.to_csv(f"pseudobulk_GSEA_{name}.csv")
        if len(df_sig) == 0:
            log(f"  {name}: no significant terms")
            continue

        top_all = pd.concat([df_sig.head(15), df_sig.tail(15)]).drop_duplicates()
        top_all = top_all.sort_values("NES")
        fig, ax = plt.subplots(figsize=(9, max(5, len(top_all) * 0.38)))
        colors_g = ["#E63946" if n > 0 else "#457B9D" for n in top_all["NES"]]
        ax.hlines(range(len(top_all)), 0, top_all["NES"], color=colors_g, lw=1.5)
        ax.scatter(top_all["NES"], range(len(top_all)), color=colors_g, s=40)  # FIX-9
        ax.set_yticks(range(len(top_all)))
        ax.set_yticklabels(top_all["Term"].str[:60], fontsize=7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("NES")
        ax.set_title(f"GSEA {name}: Early→Late Trajectory")
        plt.tight_layout()
        plt.savefig(f"pseudobulk_GSEA_{name}_lollipop.pdf", dpi=150)
        plt.close()
        log(f"  {name}: {len(df_sig)} significant terms (BH FDR < 0.25)")
    except Exception as e:
        log(f"  {name} GSEA failed: {e}")

# ============================================================
# MODULE 8 — PER-CLUSTER MARKER GENES + MANUAL DOTPLOT
# ============================================================

log("\n[MODULE 8] Per-cluster marker genes + GSEA...")

sc.tl.rank_genes_groups(
    adata_traj, groupby="leiden", method="wilcoxon",
    key_added="rank_genes_leiden", use_raw=True,
)

valid_traj = [c for c in TRAJECTORY if c in adata_traj.obs["leiden"].cat.categories]

# Per-cluster Enrichr
for cluster in valid_traj:
    try:
        df_mg     = sc.get.rank_genes_groups_df(
            adata_traj, group=cluster, key="rank_genes_leiden",
        ).head(50)
        gene_list = df_mg["names"].tolist()
        for db in ["GO_BP", "GO_MF"]:
            gmt_path = GMT_FILES.get(db)
            if gmt_path and os.path.exists(gmt_path):
                try:
                    enr    = gp.enrichr(gene_list=gene_list, gene_sets=gmt_path,
                                        outdir=None, cutoff=0.1)
                    enr_df = enr.results
                    enr_df = enr_df[enr_df["Adjusted P-value"] < 0.1].head(10)
                    enr_df.to_csv(f"cluster_{cluster}_{db}_enrichr.csv", index=False)
                except Exception as e:
                    log(f"    Cluster {cluster} {db} enrichr failed: {e}")
    except Exception as e:
        log(f"  Cluster {cluster} failed: {e}")

# (FIX-4) Manual dotplot — avoids rank_genes_groups_dotplot crash
# (scanpy internally forces dendrogram=True which triggers NaN correlation)
log("  Building manual dotplot of top markers...")

N_GENES_PER_CLUSTER  = 4
marker_genes_ordered = []
cluster_labels_list  = []

for cl in valid_traj:
    try:
        df_mg     = sc.get.rank_genes_groups_df(
            adata_traj, group=cl, key="rank_genes_leiden",
        )
        top_genes = df_mg["names"].head(N_GENES_PER_CLUSTER).tolist()
        marker_genes_ordered.extend(top_genes)
        cluster_labels_list.extend([f"Cl{cl}"] * len(top_genes))
    except Exception as e:
        log(f"  Warning: could not get markers for cluster {cl}: {e}")

# Deduplicate while preserving order
seen, unique_genes, unique_labels = set(), [], []
for g, lab in zip(marker_genes_ordered, cluster_labels_list):
    if g not in seen and g in adata_traj.var_names:
        seen.add(g)
        unique_genes.append(g)
        unique_labels.append(lab)

cluster_order = [f"Cl{c}" for c in valid_traj]
mean_expr = pd.DataFrame(index=cluster_order, columns=unique_genes, dtype=float)
pct_expr  = pd.DataFrame(index=cluster_order, columns=unique_genes, dtype=float)

for cl in valid_traj:
    mask = adata_traj.obs["leiden"] == cl
    sub  = adata_traj[mask]
    X    = sub[:, unique_genes].X
    if issparse(X):
        X = X.toarray()
    mean_expr.loc[f"Cl{cl}"] = X.mean(axis=0)
    pct_expr.loc[f"Cl{cl}"]  = (X > 0).mean(axis=0) * 100

# Min-max normalise mean expression per gene for colour mapping
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
        dot_size  = float(pct_expr.loc[cl_label, gene])
        dot_color = float(mean_norm.loc[cl_label, gene])
        ax.scatter(gi, ci, s=dot_size * 3,
                   c=[[plt.cm.Reds(dot_color)]],
                   edgecolors="grey", linewidths=0.3, alpha=0.85)

boundary    = 0
group_sizes = [sum(1 for lab in unique_labels if lab == f"Cl{c}") for c in valid_traj]
for sz in group_sizes[:-1]:
    boundary += sz
    ax.axvline(boundary - 0.5, color="lightgrey", lw=0.8)

ax.set_xticks(range(n_genes));     ax.set_xticklabels(unique_genes, rotation=90, fontsize=7)
ax.set_yticks(range(n_clust));     ax.set_yticklabels(cluster_order, fontsize=9)
ax.set_xlim(-0.5, n_genes - 0.5); ax.set_ylim(-0.5, n_clust - 0.5)
ax.set_xlabel("Marker genes (trajectory order)")
ax.set_ylabel("Leiden cluster")
ax.set_title("Top Markers per Trajectory Cluster", fontsize=12)
ax.grid(False)

sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, 1))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Norm. mean expression", shrink=0.6, pad=0.02)
for pct_val, label_str in [(20, "20%"), (50, "50%"), (80, "80%")]:
    ax.scatter([], [], s=pct_val * 3, c="grey", alpha=0.6, label=f"{label_str} expressed")
ax.legend(title="% cells expressing", loc="upper left",
          bbox_to_anchor=(1.08, 1), fontsize=8, title_fontsize=8)
plt.tight_layout()
plt.savefig("trajectory_markers_dotplot.pdf", dpi=150, bbox_inches="tight")
plt.close()
log("  Per-cluster markers saved.")
checkpoint(adata_traj, "post_markers")

# ============================================================
# MODULE 9 — LIANA CELL–CELL COMMUNICATION
# ============================================================

log("\n[MODULE 9] LIANA...")
try:
    import liana as li

    adata_traj_liana     = adata_traj.copy()
    adata_traj_liana.raw = None  # (FIX-6) raw shape mismatch pre-HVG

    li.mt.rank_aggregate(
        adata_traj_liana,
        groupby="leiden",
        use_raw=False,
        verbose=False,
        n_perms=100,
        return_all_lrs=False,
    )

    liana_res = adata_traj_liana.uns["liana_res"].copy()
    liana_res.to_csv("liana_trajectory_interactions.csv", index=False)

    for i in range(len(TRAJECTORY) - 1):
        src, tgt = TRAJECTORY[i], TRAJECTORY[i + 1]
        sub = liana_res[
            (liana_res["source"] == src) & (liana_res["target"] == tgt)
        ].head(20)
        sub.to_csv(f"liana_{src}_to_{tgt}.csv", index=False)

    try:
        li.pl.dotplot(
            adata_traj_liana,
            colour="magnitude_rank", size="specificity_rank",
            source_labels=TRAJECTORY, target_labels=TRAJECTORY,
            uns_key="liana_res", top_n=30,
            orderby="magnitude_rank", orderby_ascending=True,
            figure_size=(14, 8),
        )
        plt.savefig("liana_dotplot.pdf", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        log(f"  LIANA dotplot failed: {e}")
    log("  LIANA complete.")
except ImportError:
    log("  LIANA not installed — skipping")
except Exception as e:
    log(f"  LIANA failed: {e}")

# ============================================================
# MODULE 10 — EPIGENETIC FACTOR ANALYSIS
# ============================================================

log("\n[MODULE 10] Epigenetic analysis...")

# Load EpiFactors DB if available, otherwise use curated list
if os.path.exists(EPI_FILE):
    epi_db  = pd.read_csv(EPI_FILE, sep="\t")
    sym_col = next(
        (c for c in epi_db.columns if "symbol" in c.lower() or "gene" in c.lower()),
        epi_db.columns[0],
    )
    epi_symbols = epi_db[sym_col].dropna().unique().tolist()
else:
    log("  epifactors_human.txt not found — using curated list")
    epi_symbols = EPI_GENES

all_epi_from_cats = list(set(g for genes in EPI_CATEGORIES.values() for g in genes))
expr_epi          = [g for g in all_epi_from_cats if g in adata_traj.var_names]
log(f"  Expressed epifactors: {len(expr_epi)}/{len(all_epi_from_cats)}")

# 10C — Per-cluster epifactor expression heatmap
if expr_epi:
    clust_means = {}
    for cl in TRAJECTORY:
        mask = adata_traj.obs["leiden"] == cl
        X    = adata_traj[mask][:, expr_epi].X
        if issparse(X):
            X = X.toarray()
        clust_means[f"Cl{cl}"] = X.mean(axis=0)

    epi_heat   = pd.DataFrame(clust_means, index=expr_epi).astype(float)  # FIX-7
    epi_heat_z = (epi_heat.T - epi_heat.T.mean()) / (epi_heat.T.std() + 1e-8)
    epi_heat_z = epi_heat_z.T.astype(float)                               # FIX-7

    row_cats   = [CAT_MAP.get(g, "Other") for g in epi_heat_z.index]
    unique_cats = list(set(row_cats))
    cat_colors = dict(zip(unique_cats, sns.color_palette("tab20", len(unique_cats))))
    row_colors = pd.Series(row_cats, index=epi_heat_z.index).map(cat_colors)

    fig_h = max(12, len(expr_epi) * 0.22)
    g = sns.clustermap(
        epi_heat_z, row_colors=row_colors,
        cmap="RdBu_r", center=0,
        figsize=(max(8, len(TRAJECTORY)), fig_h),
        linewidths=0.2, row_cluster=True, col_cluster=False,
        yticklabels=True, xticklabels=True,
        cbar_kws={"label": "Z-score expression"},
    )
    g.ax_heatmap.set_title("Epigenetic Factor Expression Across Trajectory Clusters")
    handles = [mpatches.Patch(color=c, label=k) for k, c in cat_colors.items()]
    g.ax_col_dendrogram.legend(handles=handles, loc="upper right",
                               bbox_to_anchor=(1.4, 1.2), fontsize=6, ncol=2)
    plt.savefig("epigenetic_heatmap_trajectory.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log("  Epigenetic heatmap saved.")

# 10D — Category-level activity per cluster
cat_activity = defaultdict(dict)
for cat, genes in EPI_CATEGORIES.items():
    expressed = [g for g in genes if g in adata_traj.var_names]
    if not expressed:
        continue
    for cl in TRAJECTORY:
        mask = adata_traj.obs["leiden"] == cl
        X    = adata_traj[mask][:, expressed].X
        if issparse(X):
            X = X.toarray()
        cat_activity[cat][f"Cl{cl}"] = float(X.mean())  # FIX-7

cat_act_df   = pd.DataFrame(cat_activity).T
cat_act_df   = cat_act_df.loc[cat_act_df.any(axis=1)].astype(float)  # FIX-7
cat_act_norm = cat_act_df.div(cat_act_df.max(axis=1) + 1e-8, axis=0).astype(float)
cat_act_df.to_csv("epi_category_activity_per_cluster.csv")

fig, ax = plt.subplots(figsize=(max(8, len(TRAJECTORY)*0.8),
                                 max(6, len(cat_act_norm)*0.5)))
sns.heatmap(cat_act_norm, ax=ax, cmap="viridis",
            linewidths=0.3, xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Normalised mean expression"})
ax.set_title("Epigenetic Category Activity — Trajectory Clusters")
ax.set_xlabel("Leiden Cluster (trajectory order)")
plt.tight_layout()
plt.savefig("epi_category_heatmap.pdf", dpi=150)
plt.close()

# 10E — vCM-specific epigenetic switch points
log("  Computing vCM-specific epigenetic switch points...")
vcm_types   = adata_traj.obs["cell_states"].unique()
switch_data = []

for vcm in vcm_types:
    vcm_mask  = adata_traj.obs["cell_states"] == vcm
    vcm_adata = adata_traj[vcm_mask]
    if vcm_adata.n_obs < 20:
        continue
    pt = vcm_adata.obs["pseudotime"].values
    for gene in expr_epi:
        X = vcm_adata[:, gene].X
        if issparse(X):
            X = X.toarray()
        x = X.flatten()
        if x.std() < 1e-6:
            continue
        r, p = stats.spearmanr(x, pt)
        switch_data.append({
            "vCM_type": vcm, "gene": gene,
            "spearman_r": r, "pvalue": p,
            "category": CAT_MAP.get(gene, "Other"),
        })

switch_df = pd.DataFrame(switch_data)
switch_df["padj"] = switch_df.groupby("vCM_type")["pvalue"].transform(bh_correct)
switch_df.to_csv("vcm_epigenetic_switch_correlations.csv", index=False)
log("  vCM switch points saved.")

# ============================================================
# MODULE 11 — TF vs EPIGENETIC TIMING (PSEUDOTIME BINS)
# (FIX-8) — pearsonr now called on NaN/Inf-cleaned pair_df
# ============================================================

log("\n[MODULE 11] TF vs Epigenetic timing...")

TF_expr  = [g for g in BULK_TFS  if g in adata_traj.var_names]
Epi_expr = [g for g in expr_epi if g in adata_traj.var_names]

tf_pt  = get_pseudotime_expression(adata_traj, TF_expr,  n_bins=10)
epi_pt = get_pseudotime_expression(adata_traj, Epi_expr, n_bins=10)
tf_pt.to_csv("TF_pseudotime_expression.csv")
epi_pt.to_csv("Epi_pseudotime_expression.csv")

if not tf_pt.empty:
    tf_z  = ((tf_pt - tf_pt.mean()) / (tf_pt.std() + 1e-8)).T
    fig, ax = plt.subplots(figsize=(10, max(4, len(tf_z)*0.4)))
    sns.heatmap(tf_z, ax=ax, cmap="RdBu_r", center=0,
                linewidths=0.2, yticklabels=True)
    ax.set_title("TF Expression across Pseudotime Bins")
    ax.set_xlabel("Pseudotime Bin")
    plt.tight_layout()
    plt.savefig("TF_pseudotime_heatmap.pdf", dpi=150)
    plt.close()

if not epi_pt.empty:
    epi_z        = ((epi_pt - epi_pt.mean()) / (epi_pt.std() + 1e-8)).T
    peak_bin     = epi_z.idxmax(axis=1)
    epi_z_sorted = epi_z.loc[peak_bin.sort_values().index]
    fig, ax = plt.subplots(figsize=(10, max(6, len(epi_z)*0.3)))
    sns.heatmap(epi_z_sorted, ax=ax, cmap="RdBu_r", center=0,
                linewidths=0.15, yticklabels=True)
    ax.set_title("Epigenetic Factor Expression across Pseudotime Bins (sorted by peak)")
    ax.set_xlabel("Pseudotime Bin")
    plt.tight_layout()
    plt.savefig("Epi_pseudotime_heatmap_sorted.pdf", dpi=150)
    plt.close()

# (FIX-8) Build clean pair_df — drop bins with NaN or Inf before pearsonr
if not tf_pt.empty and not epi_pt.empty:
    tf_mean_bin  = tf_pt.mean(axis=1)
    epi_mean_bin = epi_pt.mean(axis=1)
    common_bins  = tf_mean_bin.index.intersection(epi_mean_bin.index)

    pair_df = pd.DataFrame({
        "tf":  tf_mean_bin[common_bins],
        "epi": epi_mean_bin[common_bins],
    }).replace([np.inf, -np.inf], np.nan).dropna()

    if len(pair_df) >= 3:
        r_val, p_val = stats.pearsonr(pair_df["tf"], pair_df["epi"])
    else:
        r_val, p_val = np.nan, np.nan
        log("  WARNING: fewer than 3 valid pseudotime bins — skipping pearsonr")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(len(pair_df)), pair_df["tf"].values,
            "o-", color="#E63946", label="Mean TF activity", lw=2)
    ax.plot(range(len(pair_df)), pair_df["epi"].values,
            "s-", color="#8338EC", label="Mean Epi factor activity", lw=2)
    ax.set_xticks(range(len(pair_df)))
    ax.set_xticklabels(pair_df.index, rotation=45)
    ax.set_xlabel("Pseudotime Bin"); ax.set_ylabel("Mean expression")
    title_r = (f"Pearson r={r_val:.3f}, p={p_val:.3f}"
               if not np.isnan(r_val) else "r: insufficient data")
    ax.set_title(f"TF vs Epigenetic Factor Dynamics\n({title_r})")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("TF_vs_Epi_pseudotime_correlation.pdf", dpi=150)
    plt.close()
    log(f"  TF vs Epi correlation: r={r_val:.3f}, p={p_val:.3f}")

    # Lead-lag analysis (FIX-8 adjacent: cross_corr_lead handles NaN internally)
    lead_results = []
    for tf in tf_pt.columns:
        for ef in epi_pt.columns:
            s1 = tf_pt[tf].fillna(0).values
            s2 = epi_pt[ef].fillna(0).values
            if len(s1) < 5:
                continue
            lag, r = cross_corr_lead(s1, s2, max_lag=2)
            lead_results.append({"TF": tf, "Epi_factor": ef,
                                  "best_lag": lag, "best_r": r})

    lead_df = pd.DataFrame(lead_results)
    lead_df.to_csv("TF_Epi_lead_lag.csv", index=False)

    n_tf_leads  = (lead_df["best_lag"] > 0).sum()
    n_sync      = (lead_df["best_lag"] == 0).sum()
    n_epi_leads = (lead_df["best_lag"] < 0).sum()
    total       = len(lead_df)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["TF leads\n(TF→Epi)", "Synchronous", "Epi leads\n(Epi→TF)"],
           [n_tf_leads, n_sync, n_epi_leads],
           color=["#E63946", "#F4A261", "#457B9D"])
    ax.set_ylabel("Number of TF-Epi factor pairs")
    ax.set_title(f"TF vs Epigenetic Factor Lead/Lag Analysis\n(n={total} pairs)")
    for idx, v in enumerate([n_tf_leads, n_sync, n_epi_leads]):
        ax.text(idx, v + 0.5, f"{v/total*100:.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("TF_Epi_lead_lag_summary.pdf", dpi=150)
    plt.close()
    log(f"  Lead/Lag: TF leads={n_tf_leads}, Sync={n_sync}, Epi leads={n_epi_leads}")

# ============================================================
# MODULE 12 — TF–EPIGENETIC CO-EXPRESSION GRN (BASIC)
# ============================================================

log("\n[MODULE 12] Basic TF–Epigenetic GRN...")

grn_genes   = list(dict.fromkeys(TF_expr + Epi_expr))
X_grn       = adata_traj[:, grn_genes].X
if issparse(X_grn):
    X_grn = X_grn.toarray()
corr_mat = np.corrcoef(X_grn.T)
corr_df  = pd.DataFrame(corr_mat, index=grn_genes, columns=grn_genes)

abs_upper  = np.abs(corr_mat[np.triu_indices_from(corr_mat, k=1)])
THRESH_GRN = float(np.percentile(abs_upper, 80))

G_basic = nx.Graph()
G_basic.add_nodes_from(grn_genes)
for i, g1 in enumerate(grn_genes):
    for j, g2 in enumerate(grn_genes):
        if j <= i:
            continue
        r = corr_df.loc[g1, g2]
        if abs(r) >= THRESH_GRN:
            G_basic.add_edge(g1, g2, weight=abs(r), sign=1 if r > 0 else -1)

if G_basic.number_of_edges() > 0:
    fig, ax  = plt.subplots(figsize=(12, 10))
    pos_grn  = nx.spring_layout(G_basic, seed=42, k=1.5)
    nc       = ["#E63946" if g in BULK_TFS else "#8338EC" for g in G_basic.nodes()]
    ec       = ["#2A9D8F" if G_basic[u][v]["sign"] > 0 else "#E9C46A"
                for u, v in G_basic.edges()]
    ew       = [G_basic[u][v]["weight"] * 3 for u, v in G_basic.edges()]
    nx.draw_networkx_nodes(G_basic, pos_grn, ax=ax, node_color=nc, node_size=600, alpha=0.9)
    nx.draw_networkx_edges(G_basic, pos_grn, ax=ax, edge_color=ec, width=ew, alpha=0.7)
    nx.draw_networkx_labels(G_basic, pos_grn, ax=ax, font_size=8, font_color="black")
    leg_el = [
        mpatches.Patch(color="#E63946", label="TF (bulk convergent)"),
        mpatches.Patch(color="#8338EC", label="Epigenetic factor"),
        mpatches.Patch(color="#2A9D8F", label="Positive correlation"),
        mpatches.Patch(color="#E9C46A", label="Negative correlation"),
    ]
    ax.legend(handles=leg_el, loc="lower left", fontsize=9)
    ax.set_title(f"TF–Epigenetic Co-expression GRN (|r| ≥ {THRESH_GRN:.2f})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("GRN_TF_epi_network.pdf", dpi=150)
    plt.close()
log("  GRN saved.")

# ============================================================
# MODULE 13 — H3K27ac ChIP-seq INTEGRATION
# ============================================================

log("\n[MODULE 13] H3K27ac ChIP-seq integration...")

# Initialise empty containers so later modules don't NameError
gained_gene_list = []
lost_gene_list   = []
gained_scores    = {}
all_scores       = {}
gene_chip        = pd.DataFrame(columns=["Gene", "delta_H3K27ac"])

if os.path.exists(CHIP_FILE) and os.path.exists(PROMOTER_BED):
    import pyranges as pr  # only imported when the data is present

    chip_df = pd.read_csv(CHIP_FILE, sep="\t", low_memory=False)
    log(f"  ChIP shape: {chip_df.shape}")

    # Auto-detect condition columns
    cols_upper = {c: c.upper() for c in chip_df.columns}
    dcm_cols   = [c for c in chip_df.columns
                  if any(x in cols_upper[c] for x in ["DCM", "FAIL", "HF"])]
    nf_cols    = [c for c in chip_df.columns
                  if any(x in cols_upper[c] for x in ["NF", "NORMAL", "CTRL"])]

    chip_df[dcm_cols] = chip_df[dcm_cols].apply(pd.to_numeric, errors="coerce")
    chip_df[nf_cols]  = chip_df[nf_cols].apply( pd.to_numeric, errors="coerce")
    chip_df["H3K27ac_DCM_mean"] = chip_df[dcm_cols].mean(axis=1)
    chip_df["H3K27ac_NF_mean"]  = chip_df[nf_cols].mean(axis=1)
    chip_df["delta_H3K27ac"]    = (
        chip_df["H3K27ac_DCM_mean"] - chip_df["H3K27ac_NF_mean"]
    )

    chip_sig = chip_df[chip_df["FDR"] < 0.05].copy() if "FDR" in chip_df.columns \
        else chip_df.copy()
    log(f"  Significant peaks: {len(chip_sig)}")

    # Promoter BED (format: chr start end gene)
    prom_df = pd.read_csv(PROMOTER_BED, sep="\t", header=None).iloc[:, :4]
    prom_df.columns = ["Chromosome", "Start", "End", "Gene"]
    prom_df["Chromosome"] = (
        "chr" + prom_df["Chromosome"].astype(str).str.replace("^chr", "", regex=True)
    )

    # Peak-promoter overlap via pyranges
    peaks_pr = pr.from_dict({
        "Chromosome": chip_sig["chrom"].astype(str),
        "Start":      chip_sig["start"],
        "End":        chip_sig["end"],
        "delta":      chip_sig["delta_H3K27ac"],
    })
    proms_pr = pr.from_dict({
        "Chromosome": prom_df["Chromosome"],
        "Start":      prom_df["Start"],
        "End":        prom_df["End"],
        "Gene":       prom_df["Gene"],
    })
    overlap = peaks_pr.join(proms_pr, how="left").df.dropna(subset=["Gene"])
    gene_chip = (
        overlap.groupby("Gene")["delta"].mean().reset_index()
        .rename(columns={"delta": "delta_H3K27ac"})
    )

    gained_genes = gene_chip[gene_chip["delta_H3K27ac"] > 0]["Gene"].tolist()
    lost_genes   = gene_chip[gene_chip["delta_H3K27ac"] < 0]["Gene"].tolist()

    gained_gene_list = [g for g in gained_genes if g in adata_traj.var_names]
    lost_gene_list   = [g for g in lost_genes   if g in adata_traj.var_names]

    gained_scores = dict(zip(gene_chip["Gene"], gene_chip["delta_H3K27ac"]))
    all_scores    = gained_scores.copy()

    adata_genes_upper = [g.upper() for g in adata_traj.var_names]

    # Score cells
    if len(gained_gene_list) >= 5:
        sc.tl.score_genes(adata_traj, gained_gene_list, score_name="H3K27ac_gained_score")
    if len(lost_gene_list) >= 5:
        sc.tl.score_genes(adata_traj, lost_gene_list, score_name="H3K27ac_lost_score")
    if ("H3K27ac_gained_score" in adata_traj.obs and
            "H3K27ac_lost_score" in adata_traj.obs):
        adata_traj.obs["H3K27ac_net_score"] = (
            adata_traj.obs["H3K27ac_gained_score"]
            - adata_traj.obs["H3K27ac_lost_score"]
        )

    # UMAP panels for H3K27ac scores
    scores_available = [c for c in
                        ["H3K27ac_gained_score", "H3K27ac_lost_score", "H3K27ac_net_score"]
                        if c in adata_traj.obs.columns]
    if scores_available:
        cmap_map = {
            "H3K27ac_gained_score": "OrRd",
            "H3K27ac_lost_score":   "Blues_r",
            "H3K27ac_net_score":    "RdBu_r",
        }
        titles_map = {
            "H3K27ac_gained_score": "Gained H3K27ac\n(disease enhancers ON)",
            "H3K27ac_lost_score":   "Lost H3K27ac\n(sarcomeric enhancers OFF)",
            "H3K27ac_net_score":    "Net H3K27ac Remodelling\n(gained − lost)",
        }
        fig, axes = plt.subplots(1, len(scores_available),
                                  figsize=(6 * len(scores_available), 5))
        if len(scores_available) == 1:
            axes = [axes]
        for ax, score in zip(axes, scores_available):
            sc.pl.umap(adata_traj, color=score,
                       cmap=cmap_map[score], ax=ax, show=False,
                       title=titles_map[score], frameon=False)
        plt.suptitle("H3K27ac Enhancer Activity — HCM Trajectory",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        savefig("H3K27ac_umap_scores.pdf")

        # Pseudotime scatter for net score (FIX-15: spearmanr imported at top)
        if ("dpt_pseudotime"       in adata_traj.obs and
                "H3K27ac_net_score" in adata_traj.obs):
            pt_vals    = adata_traj.obs["dpt_pseudotime"].values
            score_vals = adata_traj.obs["H3K27ac_net_score"].values
            clust_vals = adata_traj.obs["leiden"].astype(str).values
            colors_sc  = ["#f78166" if c in TIPPING_CLUSTERS else "#aec6cf"
                          for c in clust_vals]
            rho, pval  = spearmanr(pt_vals, score_vals)

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.scatter(pt_vals, score_vals, c=colors_sc,
                       alpha=0.35, s=8, linewidths=0, rasterized=True)
            ax.set_title(
                f"Pseudotime vs Net H3K27ac Remodelling Score\n"
                f"Spearman ρ = {rho:.3f}  (p = {pval:.2e})",
                fontsize=11,
            )
            ax.set_xlabel("Diffusion Pseudotime", fontsize=10)
            ax.set_ylabel("H3K27ac Net Score\n(gained − lost)", fontsize=10)
            ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            legend_elements = [
                mlines.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="#f78166", markersize=8,
                               label="Tipping point (Cl14/Cl4)"),
                mlines.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="#aec6cf", markersize=8,
                               label="Other clusters"),
            ]
            ax.legend(handles=legend_elements, fontsize=9)
            plt.tight_layout()
            savefig("H3K27ac_pseudotime_scatter.pdf")
            log(f"  Pseudotime correlation: ρ={rho:.3f}, p={pval:.2e}")

    # Disease gene enhancer summary table
    all_disease_genes = (SARCOMERIC_GENES + STRESS_GENES +
                         EPIGENETIC_GENES + FIBROSIS_GENES)
    summary_rows = []
    for gene in all_disease_genes:
        gene_u = gene.upper()
        summary_rows.append({
            "Gene":          gene,
            "Category":      ("Sarcomeric" if gene in SARCOMERIC_GENES else
                              "Stress"     if gene in STRESS_GENES     else
                              "Epigenetic" if gene in EPIGENETIC_GENES else "Fibrosis"),
            "delta_H3K27ac": all_scores.get(gene_u, np.nan),
            "direction":     ("GAINED" if all_scores.get(gene_u, 0) > 0 else
                              "LOST"   if all_scores.get(gene_u, 0) < 0 else
                              "not_detected"),
            "in_adata":      gene_u in adata_genes_upper,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("delta_H3K27ac", ascending=False)
    summary_df.to_csv(os.path.join(OUT_DIR, f"{PREFIX}H3K27ac_disease_gene_summary.csv"),
                      index=False)
    log("  [MODULE 13] Complete.")
else:
    log("  CHIP or BED file not found — Module 13 skipped.")

# ============================================================
# MODULE 13B — GSEA ON H3K27ac ENHANCER GENES
# ============================================================

log("\n[MODULE 13B] GSEA on H3K27ac enhancer genes...")

if not gene_chip.empty:
    ranked_chip = (
        gene_chip.dropna()
        .drop_duplicates(subset="Gene")
        .set_index("Gene")["delta_H3K27ac"]
        .sort_values(ascending=False)
    )
    ranked_chip.index = ranked_chip.index.astype(str).str.upper()

    for name, gmt in GMT_FILES.items():
        if not os.path.exists(gmt):
            log(f"  {name}: GMT not found, skipping")
            continue
        try:
            res = gp.prerank(
                rnk=ranked_chip, gene_sets=gmt,
                min_size=10, max_size=500,
                permutation_num=500,
                outdir=None, seed=42, verbose=False,
            )
            df = res.res2d.copy()
            _, padj, _, _ = multipletests(df["NOM p-val"].fillna(1.0).values,
                                          method="fdr_bh")
            df["BH_FDR"] = padj
            df_sig = df[df["BH_FDR"] < 0.25].sort_values("NES", ascending=False)
            df_sig.to_csv(os.path.join(OUT_DIR, f"{PREFIX}H3K27ac_GSEA_{name}.csv"))
            log(f"  {name}: {len(df_sig)} significant terms")

            if len(df_sig) == 0:
                continue
            top_all = pd.concat([df_sig.head(15), df_sig.tail(15)]).drop_duplicates()
            top_all = top_all.sort_values("NES")
            fig, ax = plt.subplots(figsize=(9, max(5, len(top_all) * 0.38)))
            colors_g = ["#E63946" if n > 0 else "#457B9D" for n in top_all["NES"]]
            ax.hlines(range(len(top_all)), 0, top_all["NES"], color=colors_g, lw=1.5)
            ax.scatter(top_all["NES"], range(len(top_all)), color=colors_g, s=40)  # FIX-9
            ax.set_yticks(range(len(top_all)))
            ax.set_yticklabels(top_all["Term"].str[:60], fontsize=7)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("NES")
            ax.set_title(f"GSEA {name}: H3K27ac Enhancer Landscape")
            plt.tight_layout()
            savefig(f"H3K27ac_GSEA_{name}_lollipop.pdf")
        except Exception as e:
            log(f"  {name} GSEA failed: {e}")
else:
    log("  gene_chip is empty — Module 13B skipped.")

# ============================================================
# MODULE A — WGCNA MODULE SCORING & VISUALISATION
# ============================================================

log(f"\n{'='*60}")
log("[MODULE A] Loading all WGCNA modules from bulk RNA...")

assert os.path.exists(WGCNA_MODULE_CSV), f"WGCNA CSV not found: {WGCNA_MODULE_CSV}"

wgcna_df = pd.read_csv(WGCNA_MODULE_CSV)
log(f"  Loaded {len(wgcna_df)} modules: {wgcna_df['Module'].tolist()}")

# Build short human-readable labels from top GO BP term
wgcna_labels = {}
for _, row in wgcna_df.iterrows():
    mod_id  = int(row["Module"])
    bp_term = str(row.get("Top_Enriched_BP", "Unknown"))
    wgcna_labels[mod_id] = " ".join(bp_term.split()[:5]).rstrip("(").strip()

WGCNA_SCORE_COLS = []
for _, row in wgcna_df.iterrows():
    mod_id   = int(row["Module"])
    key      = f"WGCNA_M{mod_id}"
    gene_str = str(row.get("Top50_Genes", ""))
    genes    = [g.strip() for g in gene_str.split(",") if g.strip()]
    log(f"  Scoring {key} ({len(genes)} genes)...")
    score_module(adata_hcm,  genes, key)
    score_module(adata_traj, genes, key)
    WGCNA_SCORE_COLS.append(key)

log(f"  WGCNA scoring done: {len(WGCNA_SCORE_COLS)} modules")

# UMAP grid for all WGCNA modules
n_cols = 5
n_rows = (len(WGCNA_SCORE_COLS) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
axes_flat = axes.flatten()
for i, key in enumerate(WGCNA_SCORE_COLS):
    mod_id = int(key.replace("WGCNA_M", ""))
    title  = f"M{mod_id}: {wgcna_labels.get(mod_id, '')[:30]}"
    sc.pl.umap(adata_hcm, color=key, ax=axes_flat[i],
               show=False, title=title, color_map="RdYlBu_r", frameon=False)
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.suptitle("WGCNA Module Scores (all modules)", y=1.01, fontsize=14)
plt.tight_layout()
savefig("WGCNA_all_module_umaps.pdf")

# Trajectory lineplot
wgcna_traj_means = pd.DataFrame(index=TRAJECTORY, dtype=float)
for key in WGCNA_SCORE_COLS:
    wgcna_traj_means[key] = [
        adata_traj.obs.loc[adata_traj.obs["leiden"] == cl, key].mean()
        for cl in TRAJECTORY
    ]
wgcna_traj_means.index = [f"Cl{c}" for c in TRAJECTORY]

fig, ax = plt.subplots(figsize=(max(10, len(TRAJECTORY) * 0.9), 6))
for key in WGCNA_SCORE_COLS:
    mod_id = int(key.replace("WGCNA_M", ""))
    ax.plot(range(len(TRAJECTORY)), wgcna_traj_means[key],
            "o-", lw=1.8, label=f"M{mod_id}", alpha=0.85)
ax.set_xticks(range(len(TRAJECTORY)))
ax.set_xticklabels(wgcna_traj_means.index, rotation=45)
ax.set_ylabel("Mean WGCNA Module Score")
ax.set_xlabel("Trajectory Cluster (early → late)")
ax.set_title("WGCNA Module Scores Along Disease Trajectory")
ax.legend(bbox_to_anchor=(1.01, 1), fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("WGCNA_trajectory_lineplot.pdf")

# Z-score heatmap
heat_data = wgcna_traj_means[WGCNA_SCORE_COLS].copy()
heat_z    = (heat_data - heat_data.mean()) / (heat_data.std() + 1e-8)
fig, ax   = plt.subplots(figsize=(max(8, len(WGCNA_SCORE_COLS) * 0.7),
                                   max(5, len(TRAJECTORY) * 0.55)))
sns.heatmap(heat_z, ax=ax, cmap="RdBu_r", center=0, linewidths=0.3,
            xticklabels=[f"M{int(c.replace('WGCNA_M',''))}" for c in WGCNA_SCORE_COLS],
            yticklabels=True,
            cbar_kws={"label": "Z-score (module score)"})
ax.set_title("WGCNA Module Activity Across Trajectory Clusters")
ax.set_xlabel("WGCNA Module"); ax.set_ylabel("Trajectory Cluster")
plt.tight_layout()
savefig("WGCNA_trajectory_heatmap.pdf")

wgcna_traj_means.to_csv(os.path.join(OUT_DIR, f"{PREFIX}WGCNA_cluster_means.csv"))
log("  Module A complete")

# ============================================================
# MODULE B — REGULON SCORING & VISUALISATION
# ============================================================

log(f"\n{'='*60}")
log("[MODULE B] Loading regulons from ALL_REGULONS.csv...")

assert os.path.exists(REGULON_CSV), f"Regulon CSV not found: {REGULON_CSV}"

reg_raw = pd.read_csv(REGULON_CSV)
log(f"  Columns: {reg_raw.columns.tolist()}")

REGULON_DEFS      = {}
REGULON_SCORE_COLS = []

for col in reg_raw.columns:
    genes = [str(g).strip() for g in reg_raw[col].dropna().tolist()
             if str(g).strip() != ""]
    if len(genes) == 0:
        log(f"  {col}: EMPTY — skipping scoring")
        REGULON_DEFS[col] = []
        continue
    REGULON_DEFS[col] = genes
    key = f"score_{col}"
    log(f"  Scoring {col} ({len(genes)} genes)...")
    score_module(adata_hcm,  genes, key)
    score_module(adata_traj, genes, key)
    REGULON_SCORE_COLS.append((col, key))

log(f"  Regulon scoring done: {len(REGULON_SCORE_COLS)} non-empty regulons")

# Regulon UMAPs
if REGULON_SCORE_COLS:
    n_reg = len(REGULON_SCORE_COLS)
    fig, axes = plt.subplots(1, n_reg, figsize=(n_reg * 4.5, 4))
    if n_reg == 1:
        axes = [axes]
    for i, (col, key) in enumerate(REGULON_SCORE_COLS):
        sc.pl.umap(adata_hcm, color=key, ax=axes[i],
                   show=False, title=REGULON_LABELS.get(col, col)[:35],
                   color_map="RdYlBu_r", frameon=False)
    plt.suptitle("HCM Regulon Scores (bulk RNA-derived)", y=1.02, fontsize=13)
    plt.tight_layout()
    savefig("Regulon_all_umaps.pdf")

    # Trajectory lineplot
    reg_traj_means = pd.DataFrame(index=TRAJECTORY, dtype=float)
    for col, key in REGULON_SCORE_COLS:
        reg_traj_means[col] = [
            adata_traj.obs.loc[adata_traj.obs["leiden"] == cl, key].mean()
            for cl in TRAJECTORY
        ]
    reg_traj_means.index = [f"Cl{c}" for c in TRAJECTORY]

    fig, ax = plt.subplots(figsize=(max(10, len(TRAJECTORY) * 0.9), 5))
    for col, key in REGULON_SCORE_COLS:
        ax.plot(range(len(TRAJECTORY)), reg_traj_means[col],
                "o-", lw=2, label=REGULON_LABELS.get(col, col),
                color=REG_COLORS.get(col, "#333333"), alpha=0.9)
    ax.set_xticks(range(len(TRAJECTORY)))
    ax.set_xticklabels(reg_traj_means.index, rotation=45)
    ax.set_ylabel("Mean Regulon Score"); ax.set_xlabel("Trajectory Cluster (early → late)")
    ax.set_title("HCM Regulon Scores Along Disease Trajectory")
    ax.legend(bbox_to_anchor=(1.01, 1), fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("Regulon_trajectory_lineplot.pdf")

    # Heatmap: regulon × vCM state
    vcm_reg_means = {
        col: adata_hcm.obs.groupby("cell_states")[key].mean()
        for col, key in REGULON_SCORE_COLS
    }
    vcm_reg_df = pd.DataFrame(vcm_reg_means).T
    vcm_reg_df.columns.name = "vCM state"

    fig, ax = plt.subplots(figsize=(max(6, len(vcm_reg_df.columns) * 1.5),
                                     max(4, len(vcm_reg_df) * 0.7)))
    sns.heatmap(vcm_reg_df, ax=ax, cmap="RdBu_r", center=0,
                linewidths=0.4, annot=True, fmt=".3f",
                yticklabels=[REGULON_LABELS.get(c, c) for c in vcm_reg_df.index],
                cbar_kws={"label": "Mean regulon score"})
    ax.set_title("Regulon Activity per Reference vCM State")
    ax.set_xlabel("vCM State"); ax.set_ylabel("Regulon")
    plt.tight_layout()
    savefig("Regulon_vCM_heatmap.pdf")

    reg_traj_means.to_csv(os.path.join(OUT_DIR, f"{PREFIX}Regulon_cluster_means.csv"))
    vcm_reg_df.to_csv(   os.path.join(OUT_DIR, f"{PREFIX}Regulon_vCM_means.csv"))

log("  Module B complete")

# ============================================================
# MODULE C — PER-PATIENT STATISTICAL ANALYSIS
# ============================================================

log(f"\n{'='*60}")
log("[MODULE C] Per-patient statistical analysis...")

score_cols = sorted(set(
    [c for c in adata_hcm.obs.columns if c.startswith("WGCNA_M")] +
    [c for c in adata_hcm.obs.columns if c.startswith("score_REG")]
))
print(f"  Score columns found: {score_cols}")

keep_cols = ["patient_id", "cell_states"] + score_cols
df = adata_hcm.obs[keep_cols].copy()
df = df[df["cell_states"].isin(["vCM1", "vCM3", "vCM4"])].copy()
df["is_disease"] = (df["cell_states"].isin(["vCM3", "vCM4"])).astype(int)
df["vcm_group"]  = df["cell_states"].apply(
    lambda x: "vCM3+vCM4\n(disease)" if x in ["vCM3", "vCM4"] else "vCM1\n(healthy)"
)
print(f"  Cells: {len(df)}  "
      f"(vCM1={sum(df['is_disease']==0)}, vCM3/4={sum(df['is_disease']==1)})")
df.to_csv(os.path.join(OUT_DIR, f"{PREFIX}working_dataframe.csv"), index=False)

# Per-patient × per-feature statistics
rows = []
for pat in PATIENTS:  # FIX-13: PATIENTS defined from obs
    sub  = df[df["patient_id"] == pat]
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
        r_pb, p_pb = pointbiserialr(all_x, all_y)

        U_stat, p_mw = mannwhitneyu(v1, v0, alternative="two-sided")
        auroc        = U_stat / (len(v1) * len(v0))
        pooled_std   = np.sqrt((v0.std()**2 + v1.std()**2) / 2)
        cohens_d     = (v1.mean() - v0.mean()) / (pooled_std + 1e-9)

        rows.append({
            "patient": pat, "feature": col,
            "feature_label": friendly(col),
            "n_vCM1": len(v0), "n_disease": len(v1),
            "mean_vCM1": v0.mean(), "mean_disease": v1.mean(),
            "delta_mean": v1.mean() - v0.mean(),
            "r_pb": r_pb, "p_pb": p_pb,
            "p_mw": p_mw, "auroc": auroc, "cohens_d": cohens_d,
        })

stats_df = pd.DataFrame(rows)

# BH correction per patient, then global
corrected_p = []
for pat, sub in stats_df.groupby("patient"):
    _, padj, _, _ = multipletests(sub["p_mw"].values, method="fdr_bh")
    corrected_p.extend(padj)
stats_df["p_mw_fdr"] = corrected_p

_, padj_global, _, _ = multipletests(stats_df["p_mw"].values, method="fdr_bh")
stats_df["p_mw_fdr_global"] = padj_global

stats_df = stats_df.sort_values(["patient", "p_mw_fdr"])
stats_df.to_csv(os.path.join(OUT_DIR, f"{PREFIX}stats_per_patient.csv"), index=False)
print(f"  Stats table: {len(stats_df)} rows")

# Build pivot matrices for heatmaps
labels    = stats_df[["feature", "feature_label"]].drop_duplicates()
label_map = dict(zip(labels["feature"], labels["feature_label"]))

r_mat     = stats_df.pivot(index="feature", columns="patient", values="r_pb")
r_mat.index = [label_map.get(i, i) for i in r_mat.index]

fdr_mat   = stats_df.pivot(index="feature", columns="patient", values="p_mw_fdr")
fdr_mat.index = [label_map.get(i, i) for i in fdr_mat.index]
fdr_log   = -np.log10(fdr_mat + 1e-10).clip(upper=4)

d_mat     = stats_df.pivot(index="feature", columns="patient", values="cohens_d")
d_mat.index = [label_map.get(i, i) for i in d_mat.index]

auroc_mat = stats_df.pivot(index="feature", columns="patient", values="auroc")
auroc_mat.index = [label_map.get(i, i) for i in auroc_mat.index]

n_feats = len(r_mat)

# Plot C1 — Point-biserial r heatmap
fig, ax = plt.subplots(figsize=(max(5, len(PATIENTS) * 1.5), max(4, n_feats * 0.55)))
sns.heatmap(r_mat.astype(float), ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, annot=True, fmt=".2f", annot_kws={"size": 9},
            cbar_kws={"label": "Point-biserial r\n(vCM3+vCM4 vs vCM1)"})
ax.set_title("Correlation with Disease State per Patient — Point-biserial r",
             fontsize=12, pad=12)
ax.set_xlabel("Patient"); ax.set_ylabel("")
plt.tight_layout()
savefig("heatmap_r_pb.pdf")

# Plot C2 — -log10(FDR) heatmap with significance stars
fig, ax = plt.subplots(figsize=(max(5, len(PATIENTS) * 1.5), max(4, n_feats * 0.55)))
sns.heatmap(fdr_log.astype(float), ax=ax, cmap="YlOrRd", vmin=0, vmax=4,
            linewidths=0.5, annot=False,
            cbar_kws={"label": "-log10(FDR)\n(capped at 4)"})
for i, feat in enumerate(fdr_log.index):
    for j, pat in enumerate(fdr_log.columns):
        raw_fdr = fdr_mat.loc[feat, pat] if feat in fdr_mat.index else 1.0
        if pd.isna(raw_fdr):
            continue
        star = ("***" if raw_fdr < 0.001 else
                "**"  if raw_fdr < 0.01  else
                "*"   if raw_fdr < 0.05  else "")
        if star:
            ax.text(j + 0.5, i + 0.5, star,
                    ha="center", va="center",
                    fontsize=10, color="black", fontweight="bold")
ax.set_title("Significance per Patient — Mann-Whitney FDR", fontsize=12, pad=12)
ax.set_xlabel("Patient"); ax.set_ylabel("")
plt.tight_layout()
savefig("heatmap_fdr.pdf")

# Plot C3 — Cohen's d heatmap
fig, ax = plt.subplots(figsize=(max(5, len(PATIENTS) * 1.5), max(4, n_feats * 0.55)))
sns.heatmap(d_mat.astype(float), ax=ax, cmap="RdBu_r", center=0,
            linewidths=0.5, annot=True, fmt=".2f", annot_kws={"size": 9},
            cbar_kws={"label": "Cohen's d\n(disease − healthy)"})
ax.set_title("Effect Size: vCM3+vCM4 vs vCM1 per Patient (Cohen's d)",
             fontsize=12, pad=12)
ax.set_xlabel("Patient"); ax.set_ylabel("")
plt.tight_layout()
savefig("heatmap_cohens_d.pdf")

# Plot C4 — AUROC heatmap
fig, ax = plt.subplots(figsize=(max(5, len(PATIENTS) * 1.5), max(4, n_feats * 0.55)))
sns.heatmap(auroc_mat.astype(float), ax=ax, cmap="RdBu_r", center=0.5, vmin=0, vmax=1,
            linewidths=0.5, annot=True, fmt=".2f", annot_kws={"size": 9},
            cbar_kws={"label": "AUROC\n(0.5 = no discrimination)"})
ax.set_title("Discriminability per Patient (AUROC from Mann-Whitney U)",
             fontsize=12, pad=12)
ax.set_xlabel("Patient"); ax.set_ylabel("")
plt.tight_layout()
savefig("heatmap_auroc.pdf")

# Plot C5 — Ranked lollipop: mean Cohen's d
mean_d = stats_df.groupby("feature")["cohens_d"].mean().reset_index()
mean_d["feature_label"] = mean_d["feature"].map(label_map).fillna(mean_d["feature"])
mean_d["se_d"]          = stats_df.groupby("feature")["cohens_d"].sem().values
mean_d = mean_d.sort_values("cohens_d", ascending=True)

fig, ax = plt.subplots(figsize=(8, max(5, len(mean_d) * 0.55)))
colors_c = ["#D62728" if d > 0 else "#1F77B4" for d in mean_d["cohens_d"]]
ax.hlines(range(len(mean_d)), 0, mean_d["cohens_d"], color=colors_c, lw=1.5)
ax.scatter(mean_d["cohens_d"], range(len(mean_d)), color=colors_c, s=60)  # FIX-9
ax.errorbar(mean_d["cohens_d"], range(len(mean_d)),
            xerr=mean_d["se_d"], fmt="none",
            ecolor="grey", elinewidth=1, capsize=3)
ax.axvline(0,    color="black", lw=0.8, linestyle="--")
ax.axvline( 0.2, color="grey",  lw=0.5, linestyle=":", alpha=0.6)
ax.axvline(-0.2, color="grey",  lw=0.5, linestyle=":", alpha=0.6)
ax.axvline( 0.5, color="grey",  lw=0.5, linestyle=":", alpha=0.6)
ax.axvline(-0.5, color="grey",  lw=0.5, linestyle=":", alpha=0.6)
ax.set_yticks(range(len(mean_d)))
ax.set_yticklabels(mean_d["feature_label"], fontsize=9)
ax.set_xlabel("Mean Cohen's d across patients\n(positive = higher in vCM3+vCM4)")
ax.set_title("Ranked Effect Size: Module/Regulon Correlation with Disease State")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
savefig("lollipop_cohens_d_ranked.pdf")

# Plot C6 — Per-patient stripplots for top 4 features
top_feats = (
    stats_df.groupby("feature")["cohens_d"]
    .apply(lambda x: x.abs().mean())
    .nlargest(4).index.tolist()
)
print(f"\n  Top 4 features by |Cohen's d|: {top_feats}")

fig, axes = plt.subplots(
    len(top_feats), len(PATIENTS),
    figsize=(len(PATIENTS) * 3.5, len(top_feats) * 3.5),
    sharey="row",
)
for row_i, feat in enumerate(top_feats):
    feat_label = label_map.get(feat, feat)
    for col_j, pat in enumerate(PATIENTS):
        ax  = axes[row_i, col_j]
        sub = df[df["patient_id"] == pat][[feat, "vcm_group"]].dropna()
        if sub.empty:
            ax.set_visible(False)
            continue
        order = [o for o in ["vCM1\n(healthy)", "vCM3+vCM4\n(disease)"]
                 if o in sub["vcm_group"].values]
        sns.violinplot(data=sub, x="vcm_group", y=feat, order=order, ax=ax,
                       palette=PAL_GROUP, inner=None, linewidth=0.8, alpha=0.7, cut=0)
        sns.stripplot(
            data=sub.sample(min(200, len(sub)), random_state=42),
            x="vcm_group", y=feat, order=order, ax=ax,
            palette=PAL_GROUP, size=2.5, alpha=0.5, jitter=True,
        )
        for k, grp in enumerate(order):
            med = sub[sub["vcm_group"] == grp][feat].median()
            ax.hlines(med, k - 0.3, k + 0.3, color="black", lw=1.5)  # FIX-10
        row_s = stats_df[(stats_df["patient"] == pat) & (stats_df["feature"] == feat)]
        if not row_s.empty:
            fdr_val = row_s["p_mw_fdr"].values[0]
            d_val   = row_s["cohens_d"].values[0]
            star    = ("***" if fdr_val < 0.001 else
                       "**"  if fdr_val < 0.01  else
                       "*"   if fdr_val < 0.05  else "ns")
            ax.text(0.5, 1.01, f"{star}  d={d_val:.2f}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        if row_i == 0:
            ax.set_title(pat, fontsize=11, fontweight="bold")
        if col_j == 0:
            ax.set_ylabel(feat_label[:30], fontsize=8)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels([o.replace("\n", " ") for o in order],
                           fontsize=7, rotation=15)
plt.suptitle("Top Features by Effect Size: vCM3+vCM4 vs vCM1 per Patient",
             y=1.01, fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("top_features_stripplot.pdf")

# Plot C7 — Patient consistency scatter
d_pivot      = stats_df.pivot(index="feature", columns="patient", values="cohens_d")
d_pivot.index = [label_map.get(i, i) for i in d_pivot.index]
median_sign  = np.sign(d_pivot.median(axis=1))
agree_frac   = d_pivot.apply(
    lambda row: (np.sign(row.dropna()) == median_sign[row.name]).mean(), axis=1
)
mean_abs_d   = d_pivot.abs().mean(axis=1)

consistency_df = pd.DataFrame({
    "feature_label": d_pivot.index,
    "mean_abs_d":    mean_abs_d.values,
    "consistency":   agree_frac.values,
    "direction":     ["UP in disease" if s > 0 else "DOWN in disease"
                      for s in median_sign.values],
}).sort_values("mean_abs_d", ascending=False)
consistency_df.to_csv(os.path.join(OUT_DIR, f"{PREFIX}consistency_summary.csv"),
                      index=False)

fig, ax = plt.subplots(figsize=(7, 5))
colors_dir = ["#D62728" if d == "UP in disease" else "#1F77B4"
              for d in consistency_df["direction"]]
ax.scatter(
    consistency_df["mean_abs_d"],
    consistency_df["consistency"],
    c=colors_dir, s=120, alpha=0.85, edgecolors="white", linewidths=0.5,  # FIX-11
)
for _, row in consistency_df.iterrows():
    ax.annotate(row["feature_label"][:28],
                xy=(row["mean_abs_d"], row["consistency"]),
                xytext=(5, 3), textcoords="offset points", fontsize=7)
ax.axhline(0.75, color="grey", lw=0.8, linestyle="--", alpha=0.6)
ax.axvline(0.2,  color="grey", lw=0.8, linestyle="--", alpha=0.6)
ax.set_xlabel("Mean |Cohen's d| across patients", fontsize=10)
ax.set_ylabel("Patient consistency\n(fraction agreeing on direction)", fontsize=10)
ax.set_xlim(-0.05, consistency_df["mean_abs_d"].max() * 1.25)
ax.set_ylim(0.2, 1.05)
ax.set_title("Feature Reliability: Effect Size vs Patient Consistency\n"
             "Top-right = strong + consistent", fontsize=11)
ax.legend(handles=[mpatches.Patch(color="#D62728", label="Higher in vCM3+vCM4"),
                   mpatches.Patch(color="#1F77B4", label="Lower in vCM3+vCM4")],
          loc="lower right", fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
savefig("consistency_scatter.pdf")

# Module C summary table
summary = (
    stats_df.groupby(["feature", "feature_label"])
    .agg(
        mean_r_pb      = ("r_pb",     "mean"),
        mean_cohens_d  = ("cohens_d", "mean"),
        mean_auroc     = ("auroc",    "mean"),
        n_sig_patients = ("p_mw_fdr", lambda x: (x < 0.05).sum()),
    )
    .reset_index()
    .sort_values("mean_cohens_d", ascending=False)
)
summary.to_csv(os.path.join(OUT_DIR, f"{PREFIX}final_summary.csv"), index=False)
print("\nFeature summary (sorted by mean Cohen's d):")
print(summary.to_string(index=False))
log("  Module C complete")

# ============================================================
# MODULE D — FULL TF–EPIGENETIC GRN NETWORK ANALYSIS
# ============================================================

log(f"\n{'='*60}")
log("[MODULE D] TF–Epigenetic GRN analysis...")

tf_present  = [g for g in BULK_TFS if g in adata_traj.var_names]
epi_present = [g for g in list(set(g for gs in EPI_CATEGORIES.values()
                                    for g in gs))
               if g in adata_traj.var_names]
grn_genes   = list(dict.fromkeys(tf_present + epi_present))

print(f"  TFs present : {len(tf_present)}")
print(f"  Epi present : {len(epi_present)}")
print(f"  Total genes : {len(grn_genes)}")

# Correlation matrix
X_full = adata_traj[:, grn_genes].X
if issparse(X_full):
    X_full = X_full.toarray()
corr_mat_d = np.corrcoef(X_full.T)
corr_df_d  = pd.DataFrame(corr_mat_d, index=grn_genes, columns=grn_genes)
corr_df_d.to_csv(os.path.join(GRN_DIR, GRN_PREFIX + "correlation_matrix.csv"))

abs_upper_d = np.abs(corr_mat_d[np.triu_indices_from(corr_mat_d, k=1)])
THRESH = float(np.percentile(abs_upper_d, 80))
print(f"\n  Auto-threshold (80th pct |r|) = {THRESH:.4f}")
for t in [0.10, 0.15, 0.20, 0.25, 0.30]:
    print(f"    |r|>={t:.2f}  →  {int((abs_upper_d >= t).sum())} edges")

# Build weighted graph
G = nx.Graph()
G.add_nodes_from(grn_genes)
for i, g1 in enumerate(grn_genes):
    for j, g2 in enumerate(grn_genes):
        if j <= i:
            continue
        r = corr_df_d.loc[g1, g2]
        if abs(r) >= THRESH:
            G.add_edge(g1, g2, weight=abs(r), r=r, sign=1 if r > 0 else -1)

isolated = list(nx.isolates(G))
G.remove_nodes_from(isolated)
nodes = list(G.nodes())
print(f"\n  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
      f"(removed {len(isolated)} isolated)")

Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()

# Node metrics
degree_c    = dict(G.degree())
degree_w    = dict(G.degree(weight="weight"))
between_c   = nx.betweenness_centrality(G, weight="weight", normalized=True)
closeness_c = nx.closeness_centrality(G, distance="weight")
try:
    eigen_c = nx.eigenvector_centrality_numpy(G, weight="weight")
except Exception:
    eigen_c = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
pagerank_c  = nx.pagerank(G, weight="weight", alpha=0.85)
clust_c     = nx.clustering(G, weight="weight")
hits_h, hits_a = nx.hits(G, max_iter=500)
try:
    constraint_c = nx.constraint(G, weight="weight")
except Exception:
    constraint_c = {n: np.nan for n in nodes}

local_eff = {}
for n in nodes:
    nbrs = list(G.neighbors(n)) + [n]
    local_eff[n] = nx.local_efficiency(G.subgraph(nbrs))

node_df = pd.DataFrame({
    "gene":             nodes,
    "type":             ["TF" if g in tf_present else "Epi" for g in nodes],
    "epi_category":     [CAT_MAP.get(g, "TF_convergent") for g in nodes],
    "degree":           [degree_c[g]    for g in nodes],
    "strength":         [degree_w[g]    for g in nodes],
    "betweenness":      [between_c[g]   for g in nodes],
    "closeness":        [closeness_c[g] for g in nodes],
    "eigenvector":      [eigen_c[g]     for g in nodes],
    "pagerank":         [pagerank_c[g]  for g in nodes],
    "clustering_coeff": [clust_c[g]     for g in nodes],
    "hub_score":        [hits_h[g]      for g in nodes],
    "authority_score":  [hits_a[g]      for g in nodes],
    "local_efficiency": [local_eff[g]   for g in nodes],
    "constraint":       [constraint_c.get(g, np.nan) for g in nodes],
}).sort_values("eigenvector", ascending=False)

node_df.to_csv(os.path.join(GRN_DIR, GRN_PREFIX + "node_metrics.csv"), index=False)
print("  Top 10 by eigenvector centrality:")
print(node_df[["gene", "type", "degree", "betweenness", "eigenvector",
               "pagerank", "hub_score", "clustering_coeff"]].head(10).to_string(index=False))

# Edge metrics
edge_between = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)
bridges      = set(nx.bridges(G)) if nx.is_connected(G) else set()

edge_rows = []
for u, v, d in G.edges(data=True):
    edge_rows.append({
        "gene_A":      u,
        "gene_B":      v,
        "type_A":      "TF"  if u in tf_present else "Epi",
        "type_B":      "TF"  if v in tf_present else "Epi",
        "edge_type":   ("TF-TF"   if u in tf_present and v in tf_present else
                        "Epi-Epi" if u not in tf_present and v not in tf_present else
                        "TF-Epi"),
        "r":           d["r"],
        "weight":      d["weight"],
        "sign":        d["sign"],
        "interaction": "co-activation" if d["sign"] > 0 else "opposition",
        "betweenness": edge_between.get((u, v), edge_between.get((v, u), np.nan)),
        "is_bridge":   (u, v) in bridges or (v, u) in bridges,
    })

edge_df = pd.DataFrame(edge_rows).sort_values("betweenness", ascending=False)
edge_df.to_csv(os.path.join(GRN_DIR, GRN_PREFIX + "edge_metrics.csv"), index=False)
print(f"  Edges: {len(edge_df)}")
print(f"  TF-TF: {(edge_df['edge_type']=='TF-TF').sum()}  "
      f"TF-Epi: {(edge_df['edge_type']=='TF-Epi').sum()}  "
      f"Epi-Epi: {(edge_df['edge_type']=='Epi-Epi').sum()}")
print(f"  Bridge edges: {edge_df['is_bridge'].sum()}")

# Graph-level metrics
n_comp    = nx.number_connected_components(G)
density   = nx.density(G)
avg_clust = nx.average_clustering(G, weight="weight")
avg_deg   = np.mean(list(degree_c.values()))
if nx.is_connected(Gcc):
    avg_path = nx.average_shortest_path_length(Gcc, weight=None)
    diameter = nx.diameter(Gcc)
else:
    avg_path = diameter = np.nan

communities_gen = nx.community.greedy_modularity_communities(G, weight="weight")
communities     = [list(c) for c in communities_gen]
partition       = {node: i for i, comm in enumerate(communities) for node in comm}
modularity      = nx.community.modularity(
    G, [set(c) for c in communities], weight="weight"
)

n_nodes_g = G.number_of_nodes(); n_edges_g = G.number_of_edges()
p_random  = (2 * n_edges_g) / (n_nodes_g * (n_nodes_g - 1)) if n_nodes_g > 1 else 0
rand_path = (np.log(n_nodes_g) / np.log(avg_deg)) if avg_deg > 1 else np.nan
small_world_sigma = (
    ((avg_clust / p_random) / (avg_path / rand_path))
    if (p_random > 0 and not np.isnan(rand_path)
        and rand_path > 0 and not np.isnan(avg_path))
    else np.nan
)
try:
    assortativity = nx.degree_assortativity_coefficient(G)
except Exception:
    assortativity = np.nan

graph_summary = {
    "n_nodes":              n_nodes_g,
    "n_edges":              n_edges_g,
    "n_components":         n_comp,
    "largest_component":    len(Gcc),
    "density":              round(density, 4),
    "avg_degree":           round(avg_deg, 3),
    "avg_clustering_coeff": round(avg_clust, 4),
    "avg_path_length_LCC":  round(avg_path, 3) if not np.isnan(avg_path) else "NA",
    "diameter_LCC":         diameter if not np.isnan(diameter) else "NA",
    "modularity":           round(modularity, 4),
    "n_communities":        len(communities),
    "assortativity":        round(assortativity, 4) if not np.isnan(assortativity) else "NA",
    "small_world_sigma":    round(small_world_sigma, 3) if not np.isnan(small_world_sigma) else "NA",
    "positive_edge_frac":   round((edge_df["sign"] == 1).mean(), 3),
    "negative_edge_frac":   round((edge_df["sign"] == -1).mean(), 3),
    "threshold":            round(THRESH, 4),
    "n_tf_nodes":           len(tf_present),
    "n_epi_nodes":          len(epi_present),
    "n_bridge_edges":       int(edge_df["is_bridge"].sum()),
}
pd.DataFrame.from_dict(graph_summary, orient="index", columns=["value"]).to_csv(
    os.path.join(GRN_DIR, GRN_PREFIX + "graph_summary.csv")
)
print("\n  Graph summary:")
for k, v in graph_summary.items():
    print(f"    {k:30s}: {v}")

# Community enrichment table
comm_rows = []
for i, comm in enumerate(communities):
    tf_count  = sum(1 for g in comm if g in tf_present)
    epi_count = len(comm) - tf_count
    cats      = [CAT_MAP.get(g, "TF") for g in comm if g not in tf_present]
    top_cat   = pd.Series(cats).value_counts().idxmax() if cats else "None"
    comm_rows.append({
        "community":   i,   "size":        len(comm),
        "n_TF":        tf_count,            "n_Epi":       epi_count,
        "TF_fraction": round(tf_count / len(comm), 2),
        "top_epi_cat": top_cat,             "genes":       ", ".join(sorted(comm)),
    })
comm_df = pd.DataFrame(comm_rows).sort_values("size", ascending=False)
comm_df.to_csv(os.path.join(GRN_DIR, GRN_PREFIX + "community_enrichment.csv"), index=False)
print(f"\n  Communities ({len(communities)}):")
print(comm_df[["community", "size", "n_TF", "n_Epi", "top_epi_cat"]].to_string(index=False))

# Layout
try:
    pos = nx.kamada_kawai_layout(G, weight="weight")
except Exception:
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=150)

all_x = [p[0] for p in pos.values()]
all_y = [p[1] for p in pos.values()]
pad   = 0.12

n_comm    = len(communities)
# (FIX-12) plt.get_cmap replaces deprecated cm.get_cmap
cmap_comm  = plt.get_cmap("tab10", max(n_comm, 10))
comm_colors = {node: cmap_comm(partition[node]) for node in G.nodes()}

# Plot D-A — Main GRN
print("  Plotting D-A: Main GRN...")
fig, ax = plt.subplots(figsize=(11, 9))
for u, v, d in G.edges(data=True):
    x0, y0 = pos[u]; x1, y1 = pos[v]
    ec = "#2A9D8F" if d["sign"] > 0 else "#E9C46A"
    ax.plot([x0, x1], [y0, y1], color=ec,
            linewidth=0.5 + d["weight"] * 3.5,
            alpha=min(0.35 + d["weight"] * 0.5, 0.9))  # FIX: was alpha=min(ea,0.9)=1

eig_vals  = np.array([eigen_c[g] for g in G.nodes()])
eig_norm  = (eig_vals - eig_vals.min()) / (eig_vals.ptp() + 1e-9)
node_sz   = 200 + eig_norm * 1200

for g, sz in zip(G.nodes(), node_sz):
    nx.draw_networkx_nodes(
        G, pos, nodelist=[g], ax=ax,
        node_color=[comm_colors[g]], node_size=[sz],
        edgecolors="#333333" if g in tf_present else "white",
        linewidths=2.0       if g in tf_present else 0.5,
        alpha=0.92,
    )

top_nodes = node_df.head(min(20, len(nodes)))["gene"].tolist()
nx.draw_networkx_labels(G, pos, labels={g: g for g in top_nodes},
                        ax=ax, font_size=7, font_color="black", font_weight="bold")

leg_node = [
    mpatches.Patch(color="none", ec="#333333", lw=2, label="TF (bold border)"),
    mpatches.Patch(color="none", ec="white",   lw=0.5, label="Epigenetic factor"),
]
leg_edge = [
    mlines.Line2D([], [], color="#2A9D8F", lw=2, label="Positive co-expr"),
    mlines.Line2D([], [], color="#E9C46A", lw=2, label="Negative co-expr"),
]
ax.legend(handles=leg_node + leg_edge, loc="upper left", fontsize=8, framealpha=0.88)
ax.set_title(
    f"TF–Epigenetic Co-expression GRN  "
    f"(|r|≥{THRESH:.2f} · {G.number_of_nodes()} nodes · "
    f"{G.number_of_edges()} edges · {n_comm} communities)\n"
    f"Node size = eigenvector centrality  |  Border = TF  |  "
    f"Colour = community  |  Edge width = |r|",
    fontsize=10, pad=8,
)
ax.axis("off")
ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
plt.tight_layout(pad=0.5)
savefig_grn("A_main_network.pdf")

# Plot D-B — 4-panel centrality maps
print("  Plotting D-B: Centrality maps...")
metrics_plot = [
    ("degree",      "Degree",                "Reds"),
    ("betweenness", "Betweenness Centrality", "Oranges"),
    ("closeness",   "Closeness Centrality",   "Greens"),
    ("pagerank",    "PageRank",               "Purples"),
]
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, (met, label, cmap_name) in zip(axes.flat, metrics_plot):
    vals  = np.array([node_df.set_index("gene").loc[g, met] for g in G.nodes()])
    vnorm = (vals - vals.min()) / (vals.ptp() + 1e-9)
    cmap  = plt.get_cmap(cmap_name)  # FIX-12
    ncolors = [cmap(v) for v in vnorm]
    sizes   = 150 + vnorm * 900

    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color="grey",
                linewidth=0.4 + d["weight"] * 2, alpha=0.25)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=ncolors,
                           node_size=sizes, alpha=0.9)
    top5 = sorted(G.nodes(),
                  key=lambda g: node_df.set_index("gene").loc[g, met],
                  reverse=True)[:8]
    nx.draw_networkx_labels(G, pos, labels={g: g for g in top5},
                            ax=ax, font_size=7, font_weight="bold")

    sm = cm.ScalarMappable(cmap=cmap_name,
                           norm=mcolors.Normalize(vals.min(), vals.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label=label)
    ax.set_title(f"{label} (top 8 labelled)", fontsize=10)
    ax.axis("off")
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

plt.suptitle("Node Centrality Maps", fontsize=13, y=1.01)
plt.tight_layout(pad=1.0)
savefig_grn("B_centrality_maps.pdf")

# Plot D-C — HITS hub vs authority
print("  Plotting D-C: HITS hub vs authority...")
fig, ax = plt.subplots(figsize=(7, 6))
for g in G.nodes():
    color = "#E63946" if g in tf_present else "#8338EC"
    ax.scatter(hits_h[g], hits_a[g], color=color,
               s=80 + degree_c[g] * 20, alpha=0.8)
    if eigen_c[g] > np.percentile(list(eigen_c.values()), 70):
        ax.annotate(g, (hits_h[g], hits_a[g]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="black")
xm = np.median(list(hits_h.values()))
ym = np.median(list(hits_a.values()))
ax.axvline(xm, color="grey", lw=0.8, ls="--", alpha=0.6)
ax.axhline(ym, color="grey", lw=0.8, ls="--", alpha=0.6)
xlim = ax.get_xlim(); ylim = ax.get_ylim()
ax.text(xlim[1]*0.98, ylim[1]*0.98, "Hub + Authority",
        ha="right", va="top",    fontsize=8, color="darkgreen",  alpha=0.7)
ax.text(xlim[0]+0.001, ylim[1]*0.98, "Authority only",
        ha="left",  va="top",    fontsize=8, color="steelblue",  alpha=0.7)
ax.text(xlim[1]*0.98, ylim[0]+0.001, "Hub only",
        ha="right", va="bottom", fontsize=8, color="firebrick",  alpha=0.7)
ax.set_xlabel("HITS Hub Score", fontsize=10)
ax.set_ylabel("HITS Authority Score", fontsize=10)
ax.set_title("HITS Hub vs Authority Scores (size ∝ degree)", fontsize=11)
ax.legend(handles=[mpatches.Patch(color="#E63946", label="TF"),
                   mpatches.Patch(color="#8338EC", label="Epi")], fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
savefig_grn("C_HITS_hub_authority.pdf")

# Plot D-D — Clustering coefficient vs degree
print("  Plotting D-D: Clustering vs degree...")
fig, ax = plt.subplots(figsize=(7, 5))
for g in G.nodes():
    color = "#E63946" if g in tf_present else "#8338EC"
    ax.scatter(degree_c[g], clust_c[g], color=color,
               s=60 + eigen_c[g] * 800, alpha=0.8)
    if eigen_c[g] > np.percentile(list(eigen_c.values()), 75):
        ax.annotate(g, (degree_c[g], clust_c[g]),
                    xytext=(4, 3), textcoords="offset points", fontsize=7)
deg_arr   = np.array([degree_c[g] for g in G.nodes()])
clust_arr = np.array([clust_c[g]  for g in G.nodes()])
if len(deg_arr) > 3:
    z    = np.polyfit(deg_arr, clust_arr, 1)
    xfit = np.linspace(deg_arr.min(), deg_arr.max(), 100)
    ax.plot(xfit, np.polyval(z, xfit), "k--", lw=1.2, alpha=0.6,
            label=f"Trend (slope={z[0]:.3f})")
    ax.text(0.03, 0.05,
            "Rich-club: hubs interconnect" if z[0] < 0
            else "Hierarchy: hubs bridge sparse nodes",
            transform=ax.transAxes, fontsize=8, color="grey")
ax.set_xlabel("Degree", fontsize=10)
ax.set_ylabel("Local Clustering Coefficient", fontsize=10)
ax.set_title("Degree vs Clustering — Rich-Club Check\n(size ∝ eigenvector centrality)",
             fontsize=11)
ax.legend(handles=[mpatches.Patch(color="#E63946", label="TF"),
                   mpatches.Patch(color="#8338EC", label="Epi")], fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
savefig_grn("D_clustering_vs_degree.pdf")

# Hub gene summary
print("  Top 10 hub genes:")
print(node_df[["gene", "degree", "type", "epi_category"]].head(10).to_string(index=False))
node_df.head(10).to_csv(os.path.join(GRN_DIR, GRN_PREFIX + "hub_genes.csv"), index=False)
checkpoint(adata_traj, "final")
log("  Module D complete")

# ============================================================
# FINAL SUMMARY
# ============================================================

log("\n" + "=" * 60)
log("FULL PIPELINE COMPLETE")
log("=" * 60)

outputs = sorted([
    f for f in os.listdir(OUT_DIR)
    if os.path.isfile(os.path.join(OUT_DIR, f))
])
log(f"Generated {len(outputs)} output files in {OUT_DIR}:")
for f in outputs:
    size = os.path.getsize(os.path.join(OUT_DIR, f))
    log(f"  {f:60s} {size/1024:.1f} KB")

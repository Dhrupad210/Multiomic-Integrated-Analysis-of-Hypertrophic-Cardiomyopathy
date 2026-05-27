#!/usr/bin/env python3
"""
=============================================================================
HCM SPATIAL TRANSCRIPTOMICS PIPELINE
=============================================================================
Comprehensive spatial analysis of Hypertrophic Cardiomyopathy (HCM) tissue
using Visium / Spatial Transcriptomics data.

PIPELINE STRUCTURE
------------------
  PART 0  : Data loading & preprocessing
  PART 1  : Core spatial analyses (Analysis 1–15)
  PART 2  : Bulk RNA-seq integration (BulkVal 1–7)
  PART 3  : Extension analyses (EXT 1–8)
  PART 4  : Follow-up analyses (FU 1–3)
  PART 5  : Synthesis figure & supplementary stats
  PART 6  : Final file checklist

REQUIRED INPUTS (place in the same directory as this script)
-------------------------------------------------------------
  STdata_HCM_processed.h5ad           – processed AnnData (Seurat → h5ad)
  SCT_expression_matrix.csv           – SCTransform normalised counts
  Spatial_coordinates.csv             – barcode, imagerow, imagecol
  spatial_QC_metadata.csv             – per-spot QC metrics
  spatial_Cluster_markers.csv         – Seurat FindAllMarkers output
  spatial_ATAC_correlated_TFs.csv     – TF–ATAC correlation table
  spatial_GRN_hub_scores.csv          – GRN hub connectivity scores
  spatial_marker_cluster.csv          – cluster–marker mapping
  spatial_Cluster_counts.csv          – per-cluster cell counts

  # Bulk RNA-seq integration (optional but recommended)
  WGCNA_module_summary_GO.csv         – WGCNA module–GO table
  ALL_REGULONS.csv                    – REG1-5 gene lists
  TF_activity_HCM_vs_Control.csv      – differential TF activity (limma)
  HCM_gene_panel.csv                  – curated HCM gene panel
  limma_significant_DEGs.csv          – bulk DE genes

OUTPUTS
-------
  ~50 PNG plots  +  ~35 CSV tables  +  1 synthesis PNG

DEPENDENCIES
------------
  pip install scanpy anndata numpy pandas matplotlib seaborn
              scipy scikit-learn
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import contextlib
import traceback

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

from scipy.ndimage import gaussian_filter, sobel
from scipy.spatial import distance_matrix as scipy_distance_matrix
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.interpolate import griddata
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kruskal, linregress
from scipy.sparse import issparse

from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL DISPLAY SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})

SEP  = "=" * 90
SEP2 = "-" * 90

# Sentinel for missing / invalid stage values stored as INT32_MIN in h5ad
INT32_MIN = np.iinfo(np.int32).min


# =============================================================================
# SECTION A: UTILITY & HELPER FUNCTIONS
# =============================================================================

# ── Logging ──────────────────────────────────────────────────────────────────
LOG = []

def log(msg):
    """Print a message and append it to the in-memory log."""
    print(msg)
    LOG.append(str(msg))


# ── Safe analysis context manager ────────────────────────────────────────────
@contextlib.contextmanager
def safe_analysis(name):
    """
    Wraps each analysis block so that any uncaught exception is caught,
    logged as [SKIPPED], and execution continues with the next block.
    """
    log(f"\n{SEP}\n{name}\n{SEP}")
    try:
        yield
        log(f"[OK] {name}")
    except Exception as exc:
        log(f"[SKIPPED] {name} -- {exc}")
        traceback.print_exc()


# ── Interpretation helpers ────────────────────────────────────────────────────
def morans_interpretation(mi):
    """Return a plain-English description of a Moran's I value."""
    if pd.isna(mi):
        return "Undefined (insufficient data)"
    if mi > 0.5:  return "STRONG spatial clustering (highly organised)"
    if mi > 0.3:  return "MODERATE spatial clustering (regional patterning)"
    if mi > 0.1:  return "WEAK spatial clustering (tendency to cluster)"
    if mi > -0.1: return "RANDOM / no spatial structure (CSR)"
    if mi > -0.3: return "WEAK dispersion"
    return "DISPERSED (negative autocorrelation)"


def coherence_grade(c):
    """Grade a cluster's spatial coherence score (0–1) as A–D."""
    if c > 0.8: return "A (Compact)"
    if c > 0.5: return "B (Moderate)"
    if c > 0.3: return "C (Diffuse)"
    return "D (Scattered)"


def effect_size(r):
    """Classify a Pearson r as Large / Medium / Small / Negligible."""
    a = abs(r)
    if a > 0.7: return "Large"
    if a > 0.5: return "Medium"
    if a > 0.3: return "Small"
    return "Negligible"


def specificity_label(s):
    """Categorise a marker specificity score."""
    if s > 0.8: return "HIGHLY specific"
    if s > 0.6: return "MODERATELY specific"
    if s > 0.5: return "WEAKLY specific"
    return "NON-specific"


def flag(label, value, alert=False):
    """Log a key metric with an arrow or alert icon."""
    icon = "!! " if alert else ">> "
    log(f"   {icon}{label}: {value}")


def critical(text):
    """Log a critical / noteworthy finding."""
    log(f"\n[CRITICAL FINDING]\n   {text}")


# ── Core data helpers ─────────────────────────────────────────────────────────
def load_expr_matrix(path, known_barcodes):
    """
    Load the SCT expression CSV and auto-detect orientation.

    The file may be (genes × barcodes) or (barcodes × genes).
    We compare the overlap of rows vs columns with known_barcodes
    and transpose if barcodes are in the row index.
    """
    df = pd.read_csv(path, index_col=0)
    log(f"   Expression matrix raw shape: {df.shape}")
    col_overlap = len(set(df.columns) & set(known_barcodes))
    idx_overlap = len(set(df.index)   & set(known_barcodes))
    log(f"   Barcode overlap -- columns: {col_overlap}  index: {idx_overlap}")
    if idx_overlap > col_overlap:
        log("   Transposing: barcodes were in index rows")
        df = df.T
    else:
        log("   Orientation OK: barcodes are columns, genes are rows")
    return df


def safe_morans_i(coords, values, max_n=600):
    """
    Compute Moran's I for spatial autocorrelation.

    Subsamples to max_n spots for speed when the dataset is large.
    Uses an inverse-distance spatial weights matrix with a 10th-percentile
    distance cut-off to define the local neighbourhood.
    """
    if len(coords) > max_n:
        idx = np.random.choice(len(coords), max_n, replace=False)
        coords, values = coords[idx], values[idx]

    dm     = squareform(pdist(coords))
    cutoff = np.percentile(dm[dm > 0], 10)
    w      = np.where((dm > 0) & (dm <= cutoff), 1.0 / dm, 0.0)

    # Row-standardise weights
    rs      = w.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    w       = w / rs

    n     = len(values)
    dev   = values - values.mean()
    denom = (dev ** 2).sum()
    if denom == 0:
        return 0.0

    return float((n / w.sum()) * (np.sum(w * np.outer(dev, dev)) / denom))


def spatial_smoothing(coords, values, sigma=5):
    """
    Interpolate scattered point data onto a regular 100×100 grid
    (cubic) and apply Gaussian smoothing for contour plots.

    Returns (grid_row, grid_col, smoothed_values).
    """
    gx, gy = np.mgrid[
        coords[:, 0].min():coords[:, 0].max():100j,
        coords[:, 1].min():coords[:, 1].max():100j,
    ]
    gz = griddata(coords, values, (gx, gy), method="cubic")
    gz = np.nan_to_num(gz)
    return gx, gy, gaussian_filter(gz, sigma=sigma)


def detect_boundaries(coords, values):
    """
    Detect expression boundaries using Sobel edge detection on
    the interpolated grid.  Returns (grid_row, grid_col, gradient_magnitude).
    """
    gx, gy = np.mgrid[
        coords[:, 0].min():coords[:, 0].max():100j,
        coords[:, 1].min():coords[:, 1].max():100j,
    ]
    gz = griddata(coords, values, (gx, gy), method="cubic")
    gz = np.nan_to_num(gz)
    return gx, gy, np.hypot(
        sobel(gz, axis=0, mode="constant"),
        sobel(gz, axis=1, mode="constant"),
    )


def scatter_map(ax, coords, values, title, cmap="viridis",
                label="", vmin=None, vmax=None, s=20):
    """
    Draw a spatial scatter plot (imagecol on x-axis, imagerow on y-axis).
    The y-axis is inverted to match image coordinates (row 0 at the top).
    """
    sc2 = ax.scatter(coords[:, 1], coords[:, 0], c=values,
                     cmap=cmap, s=s, alpha=0.8, vmin=vmin, vmax=vmax)
    plt.colorbar(sc2, ax=ax, label=label)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("imagecol")
    ax.set_ylabel("imagerow")
    ax.invert_yaxis()


def save_fig(path):
    """Call tight_layout, save at 300 dpi, and close all figures."""
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close("all")
    log(f"   Saved: {path}")


def get_gene(expr_matrix, gene, obs_names):
    """
    Return a 1-D expression array for a single gene, aligned to obs_names.
    Returns zeros if the gene is absent from the matrix.
    """
    if gene in expr_matrix.index:
        return (
            expr_matrix.loc[gene]
            .reindex(obs_names, fill_value=0)
            .values
        )
    return np.zeros(len(obs_names))


def get_genes_mean(expr_matrix, genes, obs_names):
    """
    Return the mean expression across a list of genes (skipping missing ones).
    Returns (None, []) if no genes are found.
    """
    avail = [g for g in genes if g in expr_matrix.index]
    if not avail:
        return None, []
    mat = (
        expr_matrix.loc[avail]
        .reindex(columns=obs_names, fill_value=0)
        .values
    )
    return mat.mean(axis=0), avail


def get_genes_sum(expr_matrix, genes, obs_names):
    """
    Return the sum of expression across a list of genes (skipping missing).
    Returns (None, []) if no genes are found.
    """
    avail = [g for g in genes if g in expr_matrix.index]
    if not avail:
        return None, []
    mat = (
        expr_matrix.loc[avail]
        .reindex(columns=obs_names, fill_value=0)
        .values
    )
    return mat.sum(axis=0), avail


def score_module(adata, genes, score_name):
    """
    Score a gene module using scanpy.tl.score_genes.
    Only genes present in adata.var_names are used.
    Stores the result in adata.obs[score_name].
    Returns the number of genes actually scored.
    """
    present = [g for g in genes if g in adata.var_names]
    if len(present) == 0:
        adata.obs[score_name] = 0.0
        return 0
    sc.tl.score_genes(adata, gene_list=present, score_name=score_name,
                      use_raw=False)
    return len(present)


def print_table(df, n=None):
    """Print a DataFrame to the log (first n rows if n is given)."""
    if n:
        log(df.head(n).to_string(index=False))
        if len(df) > n:
            log(f"   ... ({len(df) - n} more rows)")
    else:
        log(df.to_string(index=False))


def stage_means_series(values, stage, unique_stages, valid_mask):
    """
    Compute the mean of `values` per disease stage.
    Returns a list aligned to unique_stages (np.nan for empty stages).
    """
    return [
        values[(stage == int(s)) & valid_mask].mean()
        if ((stage == int(s)) & valid_mask).sum() > 0 else np.nan
        for s in unique_stages
    ]


# ── Bulk loader (optional files – all gracefully handled) ────────────────────
def load_bulk_results():
    """
    Attempt to load all five bulk RNA-seq derived CSV files.
    Missing files are logged as warnings; each key defaults to None.

    Returns a dict with keys:
      wgcna_modules, regulons, tf_activity, hcm_panel, bulk_degs
    """
    bulk_data = {}

    # WGCNA module summary with top-50 genes per module
    if os.path.exists("WGCNA_module_summary_GO.csv"):
        bulk_data["wgcna_modules"] = pd.read_csv("WGCNA_module_summary_GO.csv")
        log(f"   Loaded WGCNA modules: {len(bulk_data['wgcna_modules'])} modules")
    else:
        log("   WARNING: WGCNA_module_summary_GO.csv not found")
        bulk_data["wgcna_modules"] = None

    # Regulon gene lists (REG1–5, one column per regulon)
    if os.path.exists("ALL_REGULONS.csv"):
        bulk_data["regulons"] = pd.read_csv("ALL_REGULONS.csv")
        log(f"   Loaded regulons: {bulk_data['regulons'].shape[1]} regulons")
    else:
        log("   WARNING: ALL_REGULONS.csv not found")
        bulk_data["regulons"] = None

    # Differential TF activity table (columns: TF, logFC, adj.P.Val)
    if os.path.exists("TF_activity_HCM_vs_Control.csv"):
        bulk_data["tf_activity"] = pd.read_csv("TF_activity_HCM_vs_Control.csv")
        log(f"   Loaded TF activity: {len(bulk_data['tf_activity'])} TFs")
    else:
        log("   WARNING: TF_activity_HCM_vs_Control.csv not found")
        bulk_data["tf_activity"] = None

    # Curated HCM gene panel (columns: Symbol, logFC, adj.P.Val)
    if os.path.exists("HCM_gene_panel.csv"):
        bulk_data["hcm_panel"] = pd.read_csv("HCM_gene_panel.csv")
        log(f"   Loaded HCM panel: {len(bulk_data['hcm_panel'])} genes")
    else:
        log("   WARNING: HCM_gene_panel.csv not found")
        bulk_data["hcm_panel"] = None

    # limma significant DEGs (columns: Symbol, logFC, adj.P.Val)
    if os.path.exists("limma_significant_DEGs.csv"):
        bulk_data["bulk_degs"] = pd.read_csv("limma_significant_DEGs.csv")
        log(f"   Loaded bulk DEGs: {len(bulk_data['bulk_degs'])} genes")
    else:
        log("   WARNING: limma_significant_DEGs.csv not found")
        bulk_data["bulk_degs"] = None

    return bulk_data


# =============================================================================
# SECTION B: DATA LOADING & SHARED ARRAYS
# =============================================================================
log("\n" + SEP)
log("LOADING DATA")
log(SEP)

# ── AnnData (h5ad) ─────────────────────────────────────────────────────────
adata = sc.read_h5ad("STdata_HCM_processed.h5ad")
log(f"   h5ad: {adata.n_obs} cells x {adata.n_vars} genes")

# ── Spatial coordinates ─────────────────────────────────────────────────────
# Keep only the intersection of barcodes present in both sources
coords_raw = pd.read_csv("Spatial_coordinates.csv").set_index("barcode")
common_bc  = coords_raw.index.intersection(adata.obs_names)
log(f"   Barcodes -- coords: {len(coords_raw)}  h5ad: {adata.n_obs}"
    f"  intersection: {len(common_bc)}")

if len(common_bc) < adata.n_obs:
    adata = adata[list(common_bc)].copy()
    log(f"   adata subsetted to {adata.n_obs} cells with spatial coordinates")

spatial_coords = coords_raw.loc[adata.obs_names, ["imagerow", "imagecol"]].values
log(f"   Coord range: row [{spatial_coords[:,0].min():.1f},"
    f" {spatial_coords[:,0].max():.1f}]"
    f"  col [{spatial_coords[:,1].min():.1f},"
    f" {spatial_coords[:,1].max():.1f}]")

# ── SCT expression matrix ───────────────────────────────────────────────────
expr_matrix = load_expr_matrix("SCT_expression_matrix.csv",
                               known_barcodes=set(adata.obs_names))
keep_cols   = [b for b in adata.obs_names if b in expr_matrix.columns]
expr_matrix = expr_matrix[keep_cols]
log(f"   Expression matrix final:"
    f" {expr_matrix.shape[0]} genes x {expr_matrix.shape[1]} barcodes")

# ── Auxiliary CSVs ──────────────────────────────────────────────────────────
qc_metadata     = pd.read_csv("spatial_QC_metadata.csv",     index_col=0)
cluster_markers = pd.read_csv("spatial_Cluster_markers.csv")
atac_tfs        = pd.read_csv("spatial_ATAC_correlated_TFs.csv")
grn_hubs        = pd.read_csv("spatial_GRN_hub_scores.csv")
marker_cluster  = pd.read_csv("spatial_marker_cluster.csv")
cluster_counts  = pd.read_csv("spatial_Cluster_counts.csv")
log("   All auxiliary CSVs loaded")

# ── Bulk RNA-seq results (optional) ─────────────────────────────────────────
log("\n" + SEP)
log("LOADING BULK RNA-SEQ RESULTS")
log(SEP)
bulk = load_bulk_results()

# ── Shared obs arrays ────────────────────────────────────────────────────────
# Disease stage: filter out INT32_MIN sentinel (= missing)
stage_raw  = adata.obs["stage"].values.astype(np.int64)
valid_mask = stage_raw != INT32_MIN
log(f"   Cells with valid stage: {valid_mask.sum()} / {len(stage_raw)}")

# Build a human-readable stage name mapping
unique_stages  = sorted(np.unique(stage_raw[valid_mask]))
_default_names = ["Compensatory", "Decompensation", "Collapse",
                  "Advanced", "Stage4", "Stage5"]
STAGE_MAP = {
    int(s): (_default_names[i] if i < len(_default_names) else f"Stage_{int(s)}")
    for i, s in enumerate(unique_stages)
}
log(f"   Stage map: {STAGE_MAP}")

stage       = stage_raw
pseudotime  = adata.obs["pseudotime"].values
hcm_score   = adata.obs["HCM_score"].values
clusters    = adata.obs["seurat_clusters"].values.astype(int)
csd         = adata.obs["CSD"].values

# Collect vCM subtype scores (up to vCM_score5)
vcm_scores, vcm_names = [], []
for i in range(1, 6):
    col = f"vCM_score{i}"
    if col in adata.obs.columns:
        vcm_scores.append(adata.obs[col].values)
        vcm_names.append(f"vCM{i}")

# Pre-load master regulator expression
MASTER_REGULATORS = ["GATA6", "PPARGC1A", "TEAD1", "HAND2"]
mr_expr = {}
for gene in MASTER_REGULATORS:
    vals = get_gene(expr_matrix, gene, adata.obs_names)
    mr_expr[gene] = vals
    if vals.sum() == 0:
        log(f"   Warning: {gene} not found / all-zero in expression matrix")

# Placeholders populated by later analyses (used in Analysis 15)
etc_score         = None
upr_score         = None
contractile_score = None
neighborhood_df   = None

# Precompute top-expressed gene ordering (used in EXT1, EXT6, FU2)
gene_means = expr_matrix.mean(axis=1).sort_values(ascending=False)


# =============================================================================
# PART 1 – CORE SPATIAL ANALYSES (1–15)
# =============================================================================

# ── Analysis 1: QC Metrics Spatial Mapping ───────────────────────────────────
with safe_analysis("ANALYSIS 1: SPATIAL QC METRICS MAPPING"):
    metrics = [c for c in ["nCount_Spatial", "nFeature_Spatial"]
               if c in adata.obs.columns]
    fig, axes = plt.subplots(1, max(len(metrics), 1),
                             figsize=(8 * max(len(metrics), 1), 6))
    if len(metrics) == 1:
        axes = [axes]

    rows = []
    for ax, m in zip(axes, metrics):
        v  = adata.obs[m].values
        scatter_map(ax, spatial_coords, v,
                    f"{m} Spatial Distribution", cmap="YlOrRd", label=m)
        mi = safe_morans_i(spatial_coords, v)
        cv = (v.std() / v.mean() * 100) if v.mean() != 0 else 0
        rows.append({
            "Metric": m, "Mean": v.mean(), "Std": v.std(),
            "Min": v.min(), "Max": v.max(), "Morans_I": mi,
            "CV_pct": round(cv, 1),
            "Spatial_Pattern": (
                "Clustered" if mi > 0.3 else
                "Random"    if abs(mi) < 0.1 else "Dispersed"
            ),
        })
        flag(m, f"Mean={v.mean():.1f}, CV={cv:.1f}%, Moran's I={mi:.3f}")
        flag("Pattern", morans_interpretation(mi))
        if cv > 50:
            critical(f"{m} shows high heterogeneity (CV={cv:.1f}%)"
                     " -- check for technical artefacts")

    save_fig("spatial_01_qc_metrics_map.png")
    df1 = pd.DataFrame(rows)
    df1.to_csv("spatial_01_qc_metrics_statistics.csv", index=False)
    print_table(df1)


# ── Analysis 2: Cluster Spatial Distribution ─────────────────────────────────
with safe_analysis("ANALYSIS 2: CLUSTER SPATIAL DISTRIBUTION"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scatter_map(axes[0], spatial_coords, clusters,
                "Seurat Clusters Spatial Distribution",
                cmap="tab10", label="Cluster")
    for cid in np.unique(clusters):
        m = clusters == cid
        axes[1].scatter(spatial_coords[m, 1], spatial_coords[m, 0],
                        label=f"C{cid}", s=15, alpha=0.6)
    axes[1].set_title("Cluster Territories", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("imagecol"); axes[1].set_ylabel("imagerow")
    axes[1].legend(fontsize=8); axes[1].invert_yaxis()
    save_fig("spatial_02_cluster_distribution.png")

    rows = []
    for cid in np.unique(clusters):
        m  = clusters == cid
        cc = spatial_coords[m]
        tri = np.triu_indices(len(cc), k=1)
        md  = scipy_distance_matrix(cc, cc)[tri].mean() if len(cc) > 1 else 0.0
        coh = 1 / (1 + md)
        rows.append({
            "Cluster": int(cid), "Cell_Count": int(m.sum()),
            "Mean_Pairwise_Distance": md, "Spatial_Coherence": coh,
            "Centroid_Row": cc[:, 0].mean(), "Centroid_Col": cc[:, 1].mean(),
        })
        flag(f"Cluster {cid} (n={m.sum()})",
             f"Coherence={coh:.4f} -- Grade {coherence_grade(coh)}")

    df2 = pd.DataFrame(rows)
    df2.to_csv("spatial_02_cluster_topology.csv", index=False)
    print_table(df2)

    if len(df2) >= 2:
        r, p = pearsonr(df2["Cell_Count"], df2["Spatial_Coherence"])
        flag("Pearson r (Cell_Count vs Coherence)",
             f"{r:.3f} (p={p:.4f}) -- {effect_size(r)}")


# ── Analysis 3: Stage Progression Spatial Mapping ────────────────────────────
with safe_analysis("ANALYSIS 3: STAGE PROGRESSION SPATIAL MAPPING"):
    stage_plot = stage.astype(float)
    stage_plot[~valid_mask] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc0 = axes[0].scatter(spatial_coords[:, 1], spatial_coords[:, 0],
                          c=stage_plot, cmap="RdYlBu_r", s=20, alpha=0.8)
    plt.colorbar(sc0, ax=axes[0], label="Stage")
    axes[0].set_title("Disease Stage Spatial Distribution",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("imagecol"); axes[0].set_ylabel("imagerow")
    axes[0].invert_yaxis()
    scatter_map(axes[1], spatial_coords, pseudotime,
                "Pseudotime Spatial Gradient", cmap="plasma", label="Pseudotime")
    save_fig("spatial_03_stage_progression_map.png")

    rows = []
    for s in unique_stages:
        m   = (stage == int(s)) & valid_mask
        sc_ = spatial_coords[m]
        ps_ = pseudotime[m]
        rows.append({
            "Stage":          int(s),
            "Stage_Name":     STAGE_MAP[int(s)],
            "Cell_Count":     int(m.sum()),
            "Mean_Pseudotime": ps_.mean(),
            "Std_Pseudotime":  ps_.std(),
            "Spatial_Spread":  float(np.std(sc_, axis=0).mean()) if len(sc_) > 1 else 0.0,
            "Centroid_Row":    sc_[:, 0].mean() if len(sc_) > 0 else np.nan,
            "Centroid_Col":    sc_[:, 1].mean() if len(sc_) > 0 else np.nan,
        })

    df3 = pd.DataFrame(rows)
    df3.to_csv("spatial_03_stage_transitions.csv", index=False)
    print_table(df3)

    # Centroid trajectory analysis
    df_v = df3[df3["Stage"] >= 0].sort_values("Stage")
    if len(df_v) > 1:
        log("\n   Disease Progression Trajectory (Centroid Displacement):")
        total = 0
        for i in range(1, len(df_v)):
            dr   = df_v.iloc[i]["Centroid_Row"] - df_v.iloc[i-1]["Centroid_Row"]
            dc   = df_v.iloc[i]["Centroid_Col"] - df_v.iloc[i-1]["Centroid_Col"]
            dist = np.sqrt(dr**2 + dc**2)
            total += dist
            direction = "basal/apical" if abs(dr) > abs(dc) else "septal/lateral"
            s0 = df_v.iloc[i-1]["Stage_Name"]; s1 = df_v.iloc[i]["Stage_Name"]
            log(f"   {s0} -> {s1}: {dist:.1f}px [{direction}]")
        flag("Total path length", f"{total:.1f} pixels")
        if total > 100:
            critical(f"Substantial spatial progression ({total:.0f}px)"
                     " -- directional disease spread")


# ── Analysis 4: HCM Score Spatial Gradient ───────────────────────────────────
with safe_analysis("ANALYSIS 4: HCM SCORE SPATIAL GRADIENT"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scatter_map(axes[0], spatial_coords, hcm_score,
                "HCM Score Spatial Distribution (Raw)", cmap="Reds",
                label="HCM Score")
    gx, gy, sm = spatial_smoothing(spatial_coords, hcm_score, sigma=5)
    im = axes[1].contourf(gy, gx, sm, levels=20, cmap="Reds")
    plt.colorbar(im, ax=axes[1], label="HCM Score (Smoothed)")
    axes[1].set_title("HCM Score Spatial Gradient (Smoothed)",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("imagecol"); axes[1].set_ylabel("imagerow")
    axes[1].invert_yaxis()
    save_fig("spatial_04_hcm_score_gradient.png")

    pcts  = np.percentile(hcm_score, [25, 75])
    zones = {"Hotspot": hcm_score > pcts[1], "Coldspot": hcm_score < pcts[0]}
    zones["Moderate"] = ~zones["Hotspot"] & ~zones["Coldspot"]

    rows = []
    for name, m in zones.items():
        zc = spatial_coords[m]; zs = hcm_score[m]
        mi = safe_morans_i(zc, zs) if m.sum() > 10 else np.nan
        rows.append({
            "Zone": name, "Cell_Count": int(m.sum()),
            "Mean_HCM_Score": zs.mean(), "Std_HCM_Score": zs.std(),
            "Morans_I": mi,
        })
        flag(f"{name} (n={m.sum()})",
             f"Moran's I={mi:.3f} -- {morans_interpretation(mi)}")
        if name == "Hotspot" and not np.isnan(mi):
            if mi < 0.1:
                critical("Hotspots are RANDOMLY distributed"
                         " -- diffuse/microvascular mechanism")
            elif mi > 0.3:
                critical("Hotspots are STRONGLY clustered"
                         " -- focal ischaemic/inflammatory aetiology")

    df4 = pd.DataFrame(rows)
    df4.to_csv("spatial_04_hcm_score_zones.csv", index=False)
    print_table(df4)


# ── Analysis 5: vCM Subtype Spatial Segregation ──────────────────────────────
with safe_analysis("ANALYSIS 5: vCM SUBTYPE SPATIAL SEGREGATION"):
    if not vcm_scores:
        raise RuntimeError("No vCM score columns found in adata.obs")

    n = len(vcm_scores)
    ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    rows = []
    for idx, (name, vals) in enumerate(zip(vcm_names, vcm_scores)):
        scatter_map(axes[idx], spatial_coords, vals,
                    f"{name} Spatial Pattern", cmap="coolwarm")
        mi = safe_morans_i(spatial_coords, vals)
        rows.append({
            "vCM_Subtype": name, "Mean_Score": vals.mean(),
            "Std_Score":   vals.std(), "Morans_I": mi,
            "Spatial_Pattern": (
                "Clustered" if mi > 0.2 else
                "Random"    if abs(mi) < 0.1 else "Dispersed"
            ),
        })
        flag(f"{name} (mean={vals.mean():.3f})",
             f"Moran's I={mi:.3f} -- {morans_interpretation(mi)}")

    for ax in axes[n:]:
        ax.axis("off")
    save_fig("spatial_05_vcm_distribution.png")

    df5 = pd.DataFrame(rows)
    df5.to_csv("spatial_05_vcm_distribution.csv", index=False)
    print_table(df5)

    clustered = df5[df5["Morans_I"] > 0.2]["vCM_Subtype"].tolist()
    if clustered:
        critical(f"Spatially clustered subtypes: {', '.join(clustered)}"
                 " -- anatomically restricted")
    else:
        critical("ALL vCM subtypes show random distribution"
                 " -- diffuse, non-compartmentalised remodelling")


# ── Analysis 6: Master Regulator Spatial Expression ──────────────────────────
with safe_analysis("ANALYSIS 6: MASTER REGULATOR SPATIAL EXPRESSION"):
    n = len(MASTER_REGULATORS)
    ncols = min(2, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()

    rows = []
    for idx, gene in enumerate(MASTER_REGULATORS):
        vals     = mr_expr[gene]
        vlo, vhi = np.percentile(vals, [5, 95])
        scatter_map(axes[idx], spatial_coords, vals,
                    f"{gene} Spatial Expression", cmap="viridis",
                    label="Expression", vmin=vlo, vmax=vhi)
        mi = safe_morans_i(spatial_coords, vals)
        rows.append({
            "Gene":            gene,
            "Mean_Expression": vals.mean(), "Std_Expression": vals.std(),
            "Max_Expression":  vals.max(),  "Morans_I": mi,
            "Spatial_Pattern": (
                "Clustered" if mi > 0.2 else
                "Random"    if abs(mi) < 0.1 else "Dispersed"
            ),
        })
        flag(f"{gene} (max={vals.max():.2f})",
             f"Moran's I={mi:.3f} -- {morans_interpretation(mi)}")

    for ax in axes[n:]:
        ax.axis("off")
    save_fig("spatial_06_master_regulator_expression.png")

    df6 = pd.DataFrame(rows)
    df6.to_csv("spatial_06_master_regulator_expression.csv", index=False)
    print_table(df6)

    territorial = df6[df6["Morans_I"].abs() > 0.15]["Gene"].tolist()
    if territorial:
        critical(f"Territorial regulators: {', '.join(territorial)}"
                 " -- regional transcriptional control")


# ── Analysis 7: ETC Complex I Spatial Depletion ──────────────────────────────
with safe_analysis("ANALYSIS 7: ETC COMPLEX I SPATIAL DEPLETION"):
    # All NDUF* genes = mitochondrial Complex I subunits
    etc_genes = [g for g in expr_matrix.index if g.startswith("NDUF")]
    log(f"   Found {len(etc_genes)} NDUF* (Complex I) genes")
    if not etc_genes:
        raise RuntimeError("No NDUF* genes in expression matrix")

    etc_calc, _ = get_genes_mean(expr_matrix, etc_genes, adata.obs_names)
    # Prefer a precomputed ETC score from adata if available
    etc_score = (adata.obs["ETC_score1"].values
                 if "ETC_score1" in adata.obs.columns else etc_calc)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scatter_map(axes[0], spatial_coords, etc_score,
                "ETC Complex I Spatial Pattern", cmap="RdYlGn",
                label="ETC Score")
    xl = np.linspace(hcm_score.min(), hcm_score.max(), 100)
    axes[1].scatter(hcm_score, etc_score, alpha=0.4, s=8)
    axes[1].plot(xl, np.poly1d(np.polyfit(hcm_score, etc_score, 1))(xl),
                 "r--", lw=2)
    r_val = np.corrcoef(hcm_score, etc_score)[0, 1]
    axes[1].set_xlabel("HCM Score"); axes[1].set_ylabel("ETC Score")
    axes[1].set_title(f"ETC vs HCM Score (r={r_val:.3f})",
                      fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    save_fig("spatial_07_etc_complex_depletion.png")

    pcts = np.percentile(etc_score, [25, 75])
    rows = []
    for name, cond in [("Depleted",  etc_score < pcts[0]),
                       ("Moderate",  (etc_score >= pcts[0]) & (etc_score <= pcts[1])),
                       ("Intact",    etc_score > pcts[1])]:
        zh = hcm_score[cond]; ze = etc_score[cond]
        corr = np.corrcoef(zh, ze)[0, 1] if cond.sum() > 1 else np.nan
        rows.append({
            "Metabolic_Zone": name, "Cell_Count": int(cond.sum()),
            "Mean_ETC_Score": ze.mean(), "Mean_HCM_Score": zh.mean(),
            "Correlation_with_Disease": corr,
        })
        flag(f"{name} zone", f"r={corr:.3f} (ETC={ze.mean():.2f},"
             f" HCM={zh.mean():.2f})")
        if not np.isnan(corr) and corr < -0.3:
            critical(f"{name} zone: Significant negative ETC-HCM correlation"
                     " -- metabolic crisis")

    df7 = pd.DataFrame(rows)
    df7.to_csv("spatial_07_etc_complex_depletion.csv", index=False)
    print_table(df7)


# ── Analysis 8: UPR Activation Spatial Boundary Detection ───────────────────
with safe_analysis("ANALYSIS 8: UPR ACTIVATION SPATIAL BOUNDARY DETECTION"):
    upr_genes = ["DDIT3", "XBP1", "ATF4", "HSPA5"]
    upr_vals, upr_avail = get_genes_mean(expr_matrix, upr_genes, adata.obs_names)
    log(f"   UPR genes found: {upr_avail}")
    if upr_vals is None:
        raise RuntimeError("No UPR genes found in expression matrix")
    upr_score = upr_vals

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    scatter_map(axes[0], spatial_coords, upr_score,
                "UPR Activation Spatial Pattern", cmap="hot", label="UPR Score")
    gx, gy, edge = detect_boundaries(spatial_coords, upr_score)
    im = axes[1].contourf(gy, gx, edge, levels=15, cmap="coolwarm")
    plt.colorbar(im, ax=axes[1], label="Gradient Strength")
    axes[1].set_title("UPR Activation Boundaries",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("imagecol"); axes[1].set_ylabel("imagerow")
    axes[1].invert_yaxis()

    for s in unique_stages:
        m = (stage == int(s)) & valid_mask
        axes[2].scatter(stage[m], upr_score[m], alpha=0.3, s=8,
                        label=STAGE_MAP[int(s)])
        axes[2].scatter(int(s), upr_score[m].mean(), s=200, marker="*",
                        edgecolors="black", lw=1.5, zorder=10)
    axes[2].set_xlabel("Disease Stage"); axes[2].set_ylabel("UPR Score")
    axes[2].set_title("UPR Activation by Stage",
                      fontsize=11, fontweight="bold")
    axes[2].set_xticks(unique_stages)
    axes[2].set_xticklabels([STAGE_MAP.get(int(s), str(s))
                              for s in unique_stages], rotation=15)
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    save_fig("spatial_08_upr_activation_zones.png")

    rows = []; prev_upr = None
    for s in unique_stages:
        m  = (stage == int(s)) & valid_mask
        su = upr_score[m]; sc2 = spatial_coords[m]
        mi = safe_morans_i(sc2, su) if m.sum() > 10 else np.nan
        rows.append({
            "Stage":           int(s),  "Stage_Name": STAGE_MAP[int(s)],
            "Cell_Count":      int(m.sum()),
            "Mean_UPR_Score":  su.mean(), "Std_UPR_Score": su.std(),
            "UPR_High_Fraction":
                float((su > np.percentile(upr_score, 75)).mean()),
            "Spatial_Morans_I": mi,
        })
        flag(STAGE_MAP[int(s)],
             f"UPR={su.mean():.3f},"
             f" High fraction="
             f"{float((su > np.percentile(upr_score, 75)).mean()):.1%}")
        if prev_upr is not None and su.mean() - prev_upr > 0.1:
            critical(f"UPR escalation at {STAGE_MAP[int(s)]}"
                     f" (+{su.mean()-prev_upr:.2f}) -- ER stress crisis")
        prev_upr = su.mean()

    df8 = pd.DataFrame(rows)
    df8.to_csv("spatial_08_upr_activation_zones.csv", index=False)
    print_table(df8)


# ── Analysis 9: Contractile Crash Regional Analysis ──────────────────────────
with safe_analysis("ANALYSIS 9: CONTRACTILE CRASH REGIONAL ANALYSIS"):
    con_genes = ["MYH6", "MYBPC3", "MYH7", "TNNT2", "TPM1"]
    con_vals, con_avail = get_genes_mean(expr_matrix, con_genes, adata.obs_names)
    log(f"   Contractile genes found: {con_avail}")
    if con_vals is None:
        raise RuntimeError("No contractile genes found in expression matrix")
    contractile_score = con_vals

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    scatter_map(axes[0], spatial_coords, contractile_score,
                "Sarcomere Gene Expression", cmap="RdYlGn",
                label="Contractile Score")
    scatter_map(axes[1], spatial_coords, contractile_score,
                "Contractile Genes + Stage Overlay", cmap="RdYlGn",
                label="Contractile Score")
    for s in unique_stages:
        m = (stage == int(s)) & valid_mask
        axes[1].scatter(spatial_coords[m, 1], spatial_coords[m, 0],
                        s=5, alpha=0.25, label=f"S{s}")
    axes[1].legend(fontsize=7); axes[1].invert_yaxis()

    groups = [contractile_score[(stage == int(s)) & valid_mask]
              for s in unique_stages]
    bp = axes[2].boxplot(groups,
                         labels=[STAGE_MAP.get(int(s), str(s))
                                 for s in unique_stages],
                         patch_artist=True)
    for p in bp["boxes"]:
        p.set_facecolor("lightblue")
    axes[2].set_ylabel("Contractile Score")
    axes[2].set_title("Contractile Crash by Stage",
                      fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].tick_params(axis="x", rotation=15)

    axes[3].scatter(pseudotime, contractile_score, alpha=0.4, s=8,
                    c=stage.astype(float), cmap="RdYlBu_r")
    xl = np.linspace(pseudotime.min(), pseudotime.max(), 100)
    axes[3].plot(xl,
                 np.poly1d(np.polyfit(pseudotime, contractile_score, 1))(xl),
                 "r--", lw=2)
    r_val = np.corrcoef(pseudotime, contractile_score)[0, 1]
    axes[3].set_xlabel("Pseudotime"); axes[3].set_ylabel("Contractile Score")
    axes[3].set_title(f"Contractile Crash vs Pseudotime (r={r_val:.3f})",
                      fontsize=11, fontweight="bold")
    axes[3].grid(True, alpha=0.3)
    save_fig("spatial_09_contractile_gene_patterns.png")

    rows = []
    for s in unique_stages:
        m   = (stage == int(s)) & valid_mask
        sc_ = contractile_score[m]
        cr  = np.corrcoef(pseudotime[m], sc_)[0, 1] if m.sum() > 2 else np.nan
        sil = float((sc_ < np.percentile(contractile_score, 25)).mean())
        rows.append({
            "Stage":                   int(s),
            "Stage_Name":              STAGE_MAP[int(s)],
            "Cell_Count":              int(m.sum()),
            "Mean_Contractile_Score":  sc_.mean(),
            "Std_Contractile_Score":   sc_.std(),
            "Fraction_Silenced":       sil,
            "Correlation_with_Pseudotime": cr,
        })
        flag(STAGE_MAP[int(s)],
             f"Score={sc_.mean():.2f}, Silenced={sil:.1%}, r_time={cr:.3f}")
        if sc_.mean() < 2.0:
            critical(f"{STAGE_MAP[int(s)]}: Contractile crash detected"
                     f" (score={sc_.mean():.2f})")
        if sil > 0.3:
            critical(f"{STAGE_MAP[int(s)]}: High silencing fraction ({sil:.1%})"
                     " -- regional failure")

    df9 = pd.DataFrame(rows)
    df9.to_csv("spatial_09_contractile_gene_patterns.csv", index=False)
    print_table(df9)


# ── Analysis 10: CSD Spatial Distribution ────────────────────────────────────
with safe_analysis("ANALYSIS 10: CSD SPATIAL DISTRIBUTION"):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    scatter_map(axes[0], spatial_coords, csd,
                "Cell-State Deviation Spatial Pattern",
                cmap="YlOrRd", label="CSD")
    gx, gy, sm_csd = spatial_smoothing(spatial_coords, csd, sigma=5)
    im = axes[1].contourf(gy, gx, sm_csd, levels=20, cmap="YlOrRd")
    plt.colorbar(im, ax=axes[1], label="CSD (Smoothed)")
    axes[1].set_title("CSD Landscape (Smoothed)",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("imagecol"); axes[1].set_ylabel("imagerow")
    axes[1].invert_yaxis()

    groups = [csd[clusters == c] for c in np.unique(clusters)]
    bp = axes[2].boxplot(groups,
                         labels=[f"C{c}" for c in np.unique(clusters)],
                         patch_artist=True)
    for p in bp["boxes"]:
        p.set_facecolor("coral")
    axes[2].set_xlabel("Cluster"); axes[2].set_ylabel("CSD")
    axes[2].set_title("CSD by Cluster", fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")
    save_fig("spatial_10_csd_landscape.png")

    csd_thr = np.percentile(csd, 75)
    rows = []
    for cid in np.unique(clusters):
        m  = clusters == cid; cc = csd[m]
        mi = safe_morans_i(spatial_coords[m], cc) if m.sum() > 10 else np.nan
        rows.append({
            "Cluster": int(cid), "Cell_Count": int(m.sum()),
            "Mean_CSD": cc.mean(), "Std_CSD": cc.std(),
            "High_CSD_Fraction": float((cc > csd_thr).mean()),
            "Spatial_Morans_I": mi,
        })
        flag(f"Cluster {cid}",
             f"CSD={cc.mean():.3f},"
             f" High fraction={(cc > csd_thr).mean():.1%}")

    df10 = pd.DataFrame(rows)
    df10.to_csv("spatial_10_csd_landscape.csv", index=False)
    print_table(df10)

    high_dev = df10[df10["Mean_CSD"] > 1.2]["Cluster"].tolist()
    if high_dev:
        critical(f"High deviation clusters: {high_dev}"
                 " -- major remodelling programmes")


# ── Analysis 11: TF Hub Connectivity Spatial Patterns ────────────────────────
with safe_analysis("ANALYSIS 11: TF HUB CONNECTIVITY SPATIAL PATTERNS"):
    hub_tfs  = grn_hubs["TF"].values
    hub_expr = {
        tf: get_gene(expr_matrix, tf, adata.obs_names)
        for tf in hub_tfs if tf in expr_matrix.index
    }
    if not hub_expr:
        raise RuntimeError("No hub TFs found in expression matrix")

    hub_names = list(hub_expr.keys())
    log(f"   Hub TFs found: {hub_names}")
    cov_mat   = np.corrcoef(np.array([hub_expr[t] for t in hub_names]))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im = axes[0].imshow(cov_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(hub_names)))
    axes[0].set_xticklabels(hub_names, rotation=45, ha="right")
    axes[0].set_yticks(range(len(hub_names)))
    axes[0].set_yticklabels(hub_names)
    axes[0].set_title("Hub TF Co-expression Matrix",
                      fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=axes[0], label="Correlation")
    hub_agg = np.mean([hub_expr[t] for t in hub_names], axis=0)
    scatter_map(axes[1], spatial_coords, hub_agg,
                "Aggregate Hub TF Expression", cmap="magma",
                label="Hub Score")
    save_fig("spatial_11_grn_hub_coexpression.png")

    rows = []
    for tf in hub_names:
        vals = hub_expr[tf]
        mi   = safe_morans_i(spatial_coords, vals)
        conn = grn_hubs.loc[grn_hubs["TF"] == tf, "connectivity"].values[0]
        rows.append({
            "TF": tf, "Connectivity": conn,
            "Mean_Expression": vals.mean(), "Std_Expression": vals.std(),
            "Morans_I": mi,
            "Spatial_Pattern": (
                "Clustered" if mi > 0.2 else
                "Random"    if abs(mi) < 0.1 else "Dispersed"
            ),
        })
        flag(f"{tf} (connectivity={conn:.3f})",
             f"Moran's I={mi:.3f} -- {morans_interpretation(mi)}")

    df11 = pd.DataFrame(rows)
    df11.to_csv("spatial_11_grn_hub_coexpression.csv", index=False)
    print_table(df11)

    if len(df11) >= 2:
        r, p = pearsonr(df11["Connectivity"], df11["Mean_Expression"])
        flag("Connectivity-Expression correlation",
             f"r={r:.3f} (p={p:.4f}) -- {effect_size(r)}")
        if r > 0.5 and p < 0.05:
            critical("Strong connectivity-expression coupling"
                     " -- hub TFs are transcriptionally active")
        elif r < 0:
            critical("Negative connectivity-expression"
                     " -- potential network dysregulation")


# ── Analysis 12: ATAC-RNA Correlation Spatial Validation ─────────────────────
with safe_analysis("ANALYSIS 12: ATAC-RNA CORRELATION SPATIAL VALIDATION"):
    atac_tf_list = atac_tfs["TF"].values
    atac_tf_expr = {
        tf: get_gene(expr_matrix, tf, adata.obs_names)
        for tf in atac_tf_list if tf in expr_matrix.index
    }
    if not atac_tf_expr:
        raise RuntimeError("No ATAC-correlated TFs in expression matrix")
    log(f"   ATAC TFs found: {list(atac_tf_expr.keys())}")

    tfs_plot = list(atac_tf_expr.keys())[:6]
    ncols    = min(3, len(tfs_plot))
    nrows    = (len(tfs_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes      = np.array(axes).flatten()

    rows = []
    for idx, tf in enumerate(tfs_plot):
        vals = atac_tf_expr[tf]
        rho  = atac_tfs.loc[atac_tfs["TF"] == tf, "rho"].values[0]
        scatter_map(axes[idx], spatial_coords, vals,
                    f"{tf} (ATAC rho={rho:.3f})", cmap="plasma")
        mi  = safe_morans_i(spatial_coords, vals)
        t_r = np.corrcoef(pseudotime, vals)[0, 1]
        rows.append({
            "TF": tf, "ATAC_Correlation": rho,
            "Mean_Expression": vals.mean(), "Spatial_Morans_I": mi,
            "Pseudotime_Correlation": t_r,
            "Spatial_Pattern": (
                "Clustered" if mi > 0.2 else
                "Random"    if abs(mi) < 0.1 else "Dispersed"
            ),
        })
        flag(f"{tf}",
             f"ATAC r={rho:.3f}, Spatial MI={mi:.3f}, Time r={t_r:.3f}")

    for ax in axes[len(tfs_plot):]:
        ax.axis("off")
    save_fig("spatial_12_atac_rna_correlation_zones.png")

    df12 = pd.DataFrame(rows)
    df12.to_csv("spatial_12_atac_rna_correlation_zones.csv", index=False)
    print_table(df12)

    strong = df12[df12["ATAC_Correlation"] > 0.5]["TF"].tolist()
    if strong:
        critical(f"Strong ATAC-RNA coupling: {', '.join(strong)}"
                 " -- epigenetic drivers of remodelling")


# ── Analysis 13: Cluster Marker Spatial Specificity ──────────────────────────
with safe_analysis("ANALYSIS 13: CLUSTER MARKER SPATIAL SPECIFICITY"):
    top_n = 10
    rows  = []
    for cid in cluster_markers["cluster"].unique():
        sub = (cluster_markers[cluster_markers["cluster"] == cid]
               .sort_values("avg_log2FC", ascending=False)
               .head(top_n))
        for _, row in sub.iterrows():
            gene = row["gene"]
            if gene not in expr_matrix.index:
                continue
            ge       = get_gene(expr_matrix, gene, adata.obs_names)
            m        = clusters == int(cid)
            in_m     = ge[m].mean(); out_m = ge[~m].mean()
            spec     = in_m / (in_m + out_m) if (in_m + out_m) > 0 else 1.0
            sp       = (safe_morans_i(spatial_coords[m], ge[m])
                        if m.sum() > 10 else np.nan)
            rows.append({
                "Cluster": int(cid), "Gene": gene,
                "avg_log2FC": row["avg_log2FC"],
                "pct.1": row["pct.1"], "pct.2": row["pct.2"],
                "Specificity_Score": spec, "Spatial_Precision": sp,
                "Mean_Expression": ge.mean(),
            })

    if not rows:
        raise RuntimeError("No cluster markers found in expression matrix")

    marker_df = pd.DataFrame(rows)
    marker_df.to_csv("spatial_13_marker_specificity_scores.csv", index=False)

    unique_cids = sorted(cluster_markers["cluster"].unique())
    n_show      = min(4, len(unique_cids))
    fig, axes   = plt.subplots(1, n_show, figsize=(4 * n_show + 1, 4))
    if n_show == 1:
        axes = [axes]
    for idx, cid in enumerate(unique_cids[:n_show]):
        sub  = marker_df[marker_df["Cluster"] == int(cid)]
        if sub.empty:
            axes[idx].axis("off"); continue
        gene = sub.nlargest(1, "avg_log2FC").iloc[0]["Gene"]
        ge   = get_gene(expr_matrix, gene, adata.obs_names)
        scatter_map(axes[idx], spatial_coords, ge,
                    f"C{int(cid)}: {gene}", cmap="Reds", label="Expr")
    save_fig("spatial_13_marker_specificity_spatial.png")

    log(f"\n   Analyzed {len(rows)} markers across {len(unique_cids)} clusters")
    print_table(marker_df, n=20)

    high_spec = marker_df[marker_df["Specificity_Score"] > 0.8]
    low_spec  = marker_df[marker_df["Specificity_Score"] < 0.6]
    if len(high_spec):
        critical(f"{len(high_spec)} highly specific markers (score>0.8)"
                 " -- robust cluster definitions")
    if len(low_spec):
        critical(f"{len(low_spec)} low-specificity markers (score<0.6)"
                 " -- potential transitional states")


# ── Analysis 14: Multi-Scale Spatial Neighbourhood Analysis ──────────────────
with safe_analysis("ANALYSIS 14: MULTI-SCALE SPATIAL NEIGHBORHOOD ANALYSIS"):
    k_values = [10, 20, 50]
    rows     = []
    for k in k_values:
        log(f"   Building k={k} neighborhood graph...")
        nbrs = NearestNeighbors(n_neighbors=k + 1,
                                algorithm="ball_tree").fit(spatial_coords)
        distances, indices = nbrs.kneighbors(spatial_coords)
        for i in range(len(spatial_coords)):
            ni  = indices[i, 1:]
            ns  = stage[ni]
            sf  = {int(s): float((ns == int(s)).sum() / k) for s in unique_stages}
            row = {
                "Cell_Index":            i,  "k": k,
                "Cell_Stage":            int(stage[i]),
                "Cell_HCM_Score":        float(hcm_score[i]),
                "Neighbor_Mean_HCM":     float(hcm_score[ni].mean()),
                "Neighbor_Heterogeneity":float(hcm_score[ni].std()),
            }
            for s in unique_stages:
                row[f"Neighbor_Stage_{int(s)}_Frac"] = sf.get(int(s), 0.0)
            rows.append(row)

    neighborhood_df = pd.DataFrame(rows)
    neighborhood_df.to_csv("spatial_14_microenvironment_communities.csv",
                           index=False)

    k20 = neighborhood_df[neighborhood_df["k"] == 20].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    scatter_map(axes[0], spatial_coords,
                k20["Neighbor_Heterogeneity"].values,
                "Neighborhood Heterogeneity (k=20)",
                cmap="coolwarm", label="Heterogeneity")

    sfrac_cols = [c for c in k20.columns
                  if c.startswith("Neighbor_Stage_") and c.endswith("_Frac")]

    def get_mtype(row):
        """Classify a cell's microenvironment as Homogeneous or Mixed."""
        fracs = [row[c] for c in sfrac_cols]
        mf    = max(fracs)
        return f"Homogeneous_S{fracs.index(mf)}" if mf > 0.7 else "Mixed"

    k20 = k20.copy()
    k20["Microenvironment"] = k20.apply(get_mtype, axis=1)
    mt_types = k20["Microenvironment"].unique()
    cmap3    = dict(zip(mt_types,
                        plt.cm.Set3(np.linspace(0, 1, len(mt_types)))))
    for mt in mt_types:
        m = (k20["Microenvironment"] == mt).values
        axes[1].scatter(spatial_coords[m, 1], spatial_coords[m, 0],
                        c=[cmap3[mt]], label=mt, s=20, alpha=0.7)
    axes[1].set_title("Microenvironment Types (k=20)",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("imagecol"); axes[1].set_ylabel("imagerow")
    axes[1].legend(fontsize=8); axes[1].invert_yaxis()
    save_fig("spatial_14_neighborhood_analysis.png")
    log(f"   Neighborhood table shape: {neighborhood_df.shape}")

    high_het = k20[k20["Neighbor_Heterogeneity"] >
                   k20["Neighbor_Heterogeneity"].quantile(0.9)]
    if len(high_het):
        critical(f"{len(high_het)} high-heterogeneity neighbourhoods"
                 " -- active remodelling fronts")


# ── Analysis 15: Predictive Spatial Risk Mapping (Integrative) ───────────────
with safe_analysis("ANALYSIS 15: PREDICTIVE SPATIAL RISK MAPPING (INTEGRATIVE)"):
    # Build feature matrix from all previously computed scores
    feat = {
        "HCM_score":  hcm_score, "CSD": csd,
        "pseudotime": pseudotime, "stage": stage.astype(float),
    }
    for gene, vals in mr_expr.items():
        feat[f"TF_{gene}"] = vals
    if etc_score         is not None: feat["ETC_score"]          = etc_score
    if upr_score         is not None: feat["UPR_score"]          = upr_score
    if contractile_score is not None: feat["Contractile_score"]  = contractile_score
    for name, vals in zip(vcm_names, vcm_scores):
        feat[name] = vals
    if neighborhood_df is not None:
        k20s = (neighborhood_df[neighborhood_df["k"] == 20]
                .sort_values("Cell_Index")
                .reset_index(drop=True))
        feat["Neighbor_Mean_HCM"]      = k20s["Neighbor_Mean_HCM"].values
        feat["Neighbor_Heterogeneity"] = k20s["Neighbor_Heterogeneity"].values

    X = pd.DataFrame(feat).fillna(0)
    y = hcm_score
    log(f"   Feature matrix: {X.shape}  Features: {list(X.columns)}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y,
                                               test_size=0.3, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    yp       = rf.predict(X)
    tr2      = r2_score(y_tr, rf.predict(X_tr))
    te2      = r2_score(y_te, rf.predict(X_te))
    tr_rmse  = np.sqrt(mean_squared_error(y_tr, rf.predict(X_tr)))
    te_rmse  = np.sqrt(mean_squared_error(y_te, rf.predict(X_te)))
    log(f"   Train R2={tr2:.4f}  Test R2={te2:.4f}")
    log(f"   Train RMSE={tr_rmse:.4f}  Test RMSE={te_rmse:.4f}")

    fi = (pd.DataFrame({"Feature": X.columns,
                        "Importance": rf.feature_importances_})
          .sort_values("Importance", ascending=False))
    fi.to_csv("spatial_15_risk_feature_importance.csv", index=False)

    pred_err = np.abs(y - yp)
    hr = np.percentile(yp, 75); he = np.percentile(pred_err, 75)
    pz = np.zeros(len(spatial_coords))
    pz[(yp >  hr) & (pred_err <  he)] = 3  # high risk, reliable
    pz[(yp >  hr) & (pred_err >= he)] = 2  # high risk, uncertain
    pz[(yp <= hr) & (pred_err >= he)] = 1  # low risk, uncertain

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    scatter_map(axes[0, 0], spatial_coords, yp,
                "Spatial Risk Prediction", cmap="Reds",
                label="Predicted HCM Risk")
    scatter_map(axes[0, 1], spatial_coords, pred_err,
                "Prediction Error Map", cmap="viridis",
                label="Prediction Error")
    top15 = fi.head(15)
    axes[1, 0].barh(range(len(top15)), top15["Importance"].values)
    axes[1, 0].set_yticks(range(len(top15)))
    axes[1, 0].set_yticklabels(top15["Feature"].values, fontsize=8)
    axes[1, 0].set_xlabel("Importance")
    axes[1, 0].set_title("Top 15 Predictive Features",
                          fontsize=11, fontweight="bold")
    axes[1, 0].invert_yaxis(); axes[1, 0].grid(True, alpha=0.3, axis="x")
    scatter_map(axes[1, 1], spatial_coords, pz,
                "Therapeutic Intervention Priority Zones",
                cmap="RdYlGn_r", label="Priority Zone")
    save_fig("spatial_15_risk_prediction_model.png")

    (pd.DataFrame({
        "Cell_Index": range(len(spatial_coords)),
        "imagerow":   spatial_coords[:, 0], "imagecol": spatial_coords[:, 1],
        "True_HCM_Score": y, "Predicted_HCM_Score": yp,
        "Prediction_Error": pred_err, "Priority_Zone": pz,
        "Stage": stage, "Cluster": clusters,
    }).to_csv("spatial_15_risk_prediction_model.csv", index=False))

    log("\n   Top 5 Predictive Features:")
    for _, row in fi.head(5).iterrows():
        flag(row["Feature"], f"{row['Importance']:.4f}")

    zone_names = {0: "Low risk/Reliable",   1: "Low risk/Uncertain",
                  2: "High risk/Uncertain",  3: "High risk/Reliable"}
    log("\n   Priority Zone Distribution:")
    for zone, count in pd.Series(pz).value_counts().sort_index().items():
        pct = count / len(pz) * 100
        flag(f"Zone {int(zone)} ({zone_names.get(int(zone), '?')})",
             f"{int(count)} cells ({pct:.1f}%)")
    if (pz == 3).sum() > 0:
        critical(f"Zone 3 (High risk/Reliable): {int((pz==3).sum())} cells"
                 " -- priority intervention targets")


# =============================================================================
# PART 2 – BULK RNA-SEQ INTEGRATION (BulkVal 1–7)
# =============================================================================

# ── BulkVal 1: WGCNA Module Scores Spatial Distribution ──────────────────────
with safe_analysis("BULK VALIDATION 1: WGCNA Module Scores Spatial Distribution"):
    if bulk["wgcna_modules"] is None:
        log("   SKIPPED: No WGCNA module data available")
    else:
        wgcna_modules = bulk["wgcna_modules"]
        module_genes  = {}
        for _, row in wgcna_modules.iterrows():
            mod_id    = row["Module"]
            genes_str = row["Top50_Genes"]
            if pd.notna(genes_str):
                module_genes[mod_id] = [g.strip()
                                        for g in genes_str.split(",")]
        log(f"   Loaded {len(module_genes)} WGCNA modules")

        wgcna_scores = {}
        for mod_id, genes in module_genes.items():
            score_name  = f"WGCNA_M{mod_id}"
            n_present   = score_module(adata, genes, score_name)
            wgcna_scores[mod_id] = score_name
            log(f"   Module M{mod_id}: {n_present}/{len(genes)} genes present")

        top_modules = list(wgcna_scores.keys())[:9]
        n_cols = min(3, len(top_modules))
        n_rows = (len(top_modules) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows))
        if len(top_modules) == 1:
            axes = [axes]
        axes_flat = np.array(axes).flatten()

        for idx, mod_id in enumerate(top_modules):
            score_name = wgcna_scores[mod_id]
            vals       = adata.obs[score_name].values
            scatter_map(axes_flat[idx], spatial_coords, vals,
                        f"WGCNA M{mod_id}", cmap="RdBu_r",
                        label="Module Score")
            mi = safe_morans_i(spatial_coords, vals)
            log(f"   M{mod_id}: Moran's I={mi:.3f}"
                f" -- {morans_interpretation(mi)}")

        for ax in axes_flat[len(top_modules):]:
            ax.axis("off")
        plt.suptitle("WGCNA Module Scores Spatial Distribution",
                     y=1.02, fontsize=14)
        save_fig("spatial_bulk_01_wgcna_modules.png")

        # Stage trajectory
        fig, ax = plt.subplots(figsize=(12, 6))
        for mod_id in top_modules[:6]:
            vals = adata.obs[wgcna_scores[mod_id]].values
            ax.plot(range(len(unique_stages)),
                    stage_means_series(vals, stage, unique_stages, valid_mask),
                    "o-", label=f"M{mod_id}", lw=2, alpha=0.8)
        ax.set_xticks(range(len(unique_stages)))
        ax.set_xticklabels([STAGE_MAP.get(int(s), str(s))
                            for s in unique_stages], rotation=45)
        ax.set_xlabel("Disease Stage"); ax.set_ylabel("Mean Module Score")
        ax.set_title("WGCNA Module Scores Along Disease Progression")
        ax.legend(bbox_to_anchor=(1.01, 1), fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_fig("spatial_bulk_01_wgcna_trajectory.png")

        score_cols = [s for s in wgcna_scores.values() if s in adata.obs.columns]
        adata.obs[score_cols].to_csv("spatial_bulk_01_wgcna_scores.csv")


# ── BulkVal 2: Regulon Scores Spatial Validation ─────────────────────────────
with safe_analysis("BULK VALIDATION 2: Regulon Scores Spatial Validation"):
    if bulk["regulons"] is None:
        log("   SKIPPED: No regulon data available")
    else:
        regulons      = bulk["regulons"]
        regulon_genes = {}
        for col in regulons.columns:
            genes = [g.strip() for g in regulons[col].dropna().tolist()
                     if isinstance(g, str)]
            if len(genes) > 10:
                regulon_genes[col] = genes
                log(f"   Regulon {col}: {len(genes)} genes")

        REGULON_COLORS = {
            "REG1_UP_TFactive":   "#D62728",
            "REG2_UP_TFindep":    "#FF7F0E",
            "REG3_DOWN_TFrep":    "#1F77B4",
            "REG4_DOWN_chromclo": "#9467BD",
            "REG5_WGCNA_ECM":     "#2CA02C",
        }

        regulon_scores = {}
        for reg_name, genes in regulon_genes.items():
            score_name = f"REG_{reg_name}"
            n_present  = score_module(adata, genes, score_name)
            regulon_scores[reg_name] = score_name
            log(f"   {reg_name}: {n_present}/{len(genes)} genes present")

        n_reg = len(regulon_scores)
        if n_reg > 0:
            n_cols = min(3, n_reg); n_rows = (n_reg + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(5 * n_cols, 4 * n_rows))
            axes_flat = np.array(axes).flatten()
            for idx, (reg_name, score_name) in enumerate(regulon_scores.items()):
                vals = adata.obs[score_name].values
                scatter_map(axes_flat[idx], spatial_coords, vals,
                            f"{reg_name}", cmap="RdBu_r",
                            label="Regulon Score")
                mi = safe_morans_i(spatial_coords, vals)
                log(f"   {reg_name}: Moran's I={mi:.3f}"
                    f" -- {morans_interpretation(mi)}")
            for ax in axes_flat[n_reg:]:
                ax.axis("off")
            plt.suptitle("Regulon Scores Spatial Distribution",
                         y=1.02, fontsize=14)
            save_fig("spatial_bulk_02_regulon_spatial.png")

            # Stage trajectory
            fig, ax = plt.subplots(figsize=(12, 6))
            for reg_name, score_name in regulon_scores.items():
                vals  = adata.obs[score_name].values
                color = REGULON_COLORS.get(reg_name, "#333333")
                ax.plot(range(len(unique_stages)),
                        stage_means_series(vals, stage, unique_stages,
                                           valid_mask),
                        "o-", label=reg_name, lw=2, color=color)
            ax.set_xticks(range(len(unique_stages)))
            ax.set_xticklabels([STAGE_MAP.get(int(s), str(s))
                                for s in unique_stages], rotation=45)
            ax.set_xlabel("Disease Stage")
            ax.set_ylabel("Mean Regulon Score")
            ax.set_title("Regulon Scores Along Disease Progression")
            ax.legend(bbox_to_anchor=(1.01, 1), fontsize=8)
            ax.grid(True, alpha=0.3); plt.tight_layout()
            save_fig("spatial_bulk_02_regulon_trajectory.png")

            # Cross-regulon correlation heatmap
            reg_cols = [s for s in regulon_scores.values()
                        if s in adata.obs.columns]
            if len(reg_cols) > 1:
                corr_mat = adata.obs[reg_cols].corr()
                fig, ax  = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r",
                            center=0, vmin=-1, vmax=1, ax=ax,
                            xticklabels=[r.replace("REG_", "")
                                         for r in corr_mat.index],
                            yticklabels=[r.replace("REG_", "")
                                         for r in corr_mat.index])
                ax.set_title("Regulon Cross-Correlation in Spatial Data")
                plt.tight_layout()
                save_fig("spatial_bulk_02_regulon_correlation.png")

            reg_out = adata.obs[reg_cols].copy()
            reg_out.to_csv("spatial_bulk_02_regulon_scores.csv")


# ── BulkVal 3: TF Activity Spatial Validation ────────────────────────────────
with safe_analysis("BULK VALIDATION 3: TF Activity Spatial Validation"):
    if bulk["tf_activity"] is None:
        log("   SKIPPED: No TF activity data available")
    else:
        tf_df     = bulk["tf_activity"]
        tf_df_sig = tf_df[tf_df["adj.P.Val"] < 0.05].copy()
        top_active    = tf_df_sig[tf_df_sig["logFC"] > 0]\
                            .nlargest(15, "logFC")["TF"].tolist()
        top_repressed = tf_df_sig[tf_df_sig["logFC"] < 0]\
                            .nsmallest(15, "logFC")["TF"].tolist()
        log(f"   Top active TFs (bulk): {top_active[:5]}...")
        log(f"   Top repressed TFs (bulk): {top_repressed[:5]}...")

        tf_scores = {}

        def _plot_tfs(tf_list, file_tag, title_prefix, cmap_name):
            """Helper: spatial map for a list of TFs."""
            n_plot = min(9, len(tf_list))
            if n_plot == 0:
                return
            nc = min(3, n_plot); nr = (n_plot + nc - 1) // nc
            fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr))
            ax_flat   = np.array(axes).flatten()
            for idx, tf in enumerate(tf_list[:n_plot]):
                vals = get_gene(expr_matrix, tf, adata.obs_names)
                scatter_map(ax_flat[idx], spatial_coords, vals,
                            f"{title_prefix} TF: {tf}", cmap=cmap_name)
                mi = safe_morans_i(spatial_coords, vals)
                log(f"   {title_prefix} TF {tf}: Moran's I={mi:.3f}")
                tf_scores[tf] = vals
            for ax in ax_flat[n_plot:]:
                ax.axis("off")
            plt.suptitle(f"Top {title_prefix} TFs (from bulk)",
                         y=1.02, fontsize=14)
            save_fig(file_tag)

        _plot_tfs(top_active,    "spatial_bulk_03_active_tfs_spatial.png",
                  "Active",    "Reds")
        _plot_tfs(top_repressed, "spatial_bulk_03_repressed_tfs_spatial.png",
                  "Repressed", "Blues_r")

        tf_corr = [{"TF": tf,
                    "r_HCM": pearsonr(hcm_score, v)[0],
                    "p_HCM": pearsonr(hcm_score, v)[1]}
                   for tf, v in tf_scores.items() if v.std() > 0]
        if tf_corr:
            tf_corr_df = (pd.DataFrame(tf_corr)
                          .sort_values("r_HCM", ascending=False))
            tf_corr_df.to_csv("spatial_bulk_03_tf_hcm_correlations.csv",
                              index=False)
            log(f"   TF-HCM correlations saved."
                f" Top: {tf_corr_df.iloc[0]['TF']}"
                f" r={tf_corr_df.iloc[0]['r_HCM']:.3f}")

        tf_expr_df = pd.DataFrame(tf_scores, index=adata.obs_names)
        tf_expr_df.to_csv("spatial_bulk_03_tf_expression.csv")


# ── BulkVal 4: HCM Gene Panel Spatial Validation ─────────────────────────────
with safe_analysis("BULK VALIDATION 4: HCM Gene Panel Spatial Validation"):
    if bulk["hcm_panel"] is None:
        log("   SKIPPED: No HCM panel data available")
    else:
        hcm_panel     = bulk["hcm_panel"]
        hcm_panel_sig = hcm_panel[hcm_panel["adj.P.Val"] < 0.05].copy()
        top_up_hcm    = hcm_panel_sig[hcm_panel_sig["logFC"] > 0]\
                            .nlargest(12, "logFC")["Symbol"].tolist()
        top_down_hcm  = hcm_panel_sig[hcm_panel_sig["logFC"] < 0]\
                            .nsmallest(12, "logFC")["Symbol"].tolist()
        log(f"   Top up HCM genes: {top_up_hcm[:5]}...")
        log(f"   Top down HCM genes: {top_down_hcm[:5]}...")

        all_top = list(dict.fromkeys([g for g in top_up_hcm[:8] + top_down_hcm[:8]
                                      if g in expr_matrix.index]))
        if all_top:
            expr_top   = np.array([get_gene(expr_matrix, g, adata.obs_names)
                                   for g in all_top])
            expr_top_z = (expr_top - expr_top.mean(axis=1, keepdims=True)) \
                         / (expr_top.std(axis=1, keepdims=True) + 1e-8)
            sort_idx   = np.argsort(pseudotime)
            fig, ax    = plt.subplots(figsize=(14, 10))
            im = ax.imshow(expr_top_z[:, sort_idx], cmap="RdBu_r",
                           aspect="auto", vmin=-2, vmax=2)
            ax.set_yticks(range(len(all_top)))
            ax.set_yticklabels(all_top, fontsize=9)
            ax.set_xlabel("Cells sorted by pseudotime")
            ax.set_title("Top HCM Panel Genes Expression Along Pseudotime")
            plt.colorbar(im, ax=ax, label="Z-score")
            plt.tight_layout()
            save_fig("spatial_bulk_04_hcm_panel_heatmap.png")

            # Individual spatial maps for top 6 genes
            genes_plot = (top_up_hcm[:3] + top_down_hcm[:3])[:6]
            genes_plot = [g for g in genes_plot if g in expr_matrix.index]
            nc = min(3, len(genes_plot)); nr = (len(genes_plot) + nc - 1) // nc
            fig, axes  = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr))
            ax_flat    = np.array(axes).flatten()
            for idx, gene in enumerate(genes_plot):
                vals = get_gene(expr_matrix, gene, adata.obs_names)
                scatter_map(ax_flat[idx], spatial_coords, vals,
                            f"HCM Gene: {gene}", cmap="RdBu_r",
                            label="Expression")
                mi     = safe_morans_i(spatial_coords, vals)
                bulk_fc = hcm_panel_sig.loc[
                    hcm_panel_sig["Symbol"] == gene, "logFC"].values
                bulk_fc_str = (f", bulk FC={bulk_fc[0]:.2f}"
                               if len(bulk_fc) > 0 else "")
                log(f"   {gene}{bulk_fc_str}: Moran's I={mi:.3f}")
            for ax in ax_flat[len(genes_plot):]:
                ax.axis("off")
            plt.suptitle("Top HCM Panel Genes - Spatial Distribution",
                         y=1.02, fontsize=14)
            save_fig("spatial_bulk_04_hcm_panel_spatial.png")


# ── BulkVal 5: Bulk DEG Spatial Validation ───────────────────────────────────
with safe_analysis("BULK VALIDATION 5: Bulk DEG Spatial Validation"):
    if bulk["bulk_degs"] is None:
        log("   SKIPPED: No bulk DEG data available")
    else:
        bulk_degs     = bulk["bulk_degs"]
        bulk_degs_sig = bulk_degs[bulk_degs["adj.P.Val"] < 0.05].copy()
        top_up_bulk   = bulk_degs_sig[bulk_degs_sig["logFC"] > 0]\
                            .nlargest(25, "logFC")["Symbol"].tolist()
        top_down_bulk = bulk_degs_sig[bulk_degs_sig["logFC"] < 0]\
                            .nsmallest(25, "logFC")["Symbol"].tolist()

        up_pres   = [g for g in top_up_bulk   if g in expr_matrix.index]
        down_pres = [g for g in top_down_bulk  if g in expr_matrix.index]
        log(f"   Bulk UP signature: {len(up_pres)}/{len(top_up_bulk)}"
            " genes in spatial")
        log(f"   Bulk DOWN signature: {len(down_pres)}/{len(top_down_bulk)}"
            " genes in spatial")

        if len(up_pres) > 5:
            score_module(adata, up_pres,   "Bulk_UP_signature")
            score_module(adata, down_pres, "Bulk_DOWN_signature")

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            scatter_map(axes[0], spatial_coords,
                        adata.obs["Bulk_UP_signature"].values,
                        "Bulk UP Signature Score", cmap="Reds",
                        label="Score")
            mi_up = safe_morans_i(spatial_coords,
                                  adata.obs["Bulk_UP_signature"].values)
            log(f"   Bulk UP signature: Moran's I={mi_up:.3f}")
            scatter_map(axes[1], spatial_coords,
                        adata.obs["Bulk_DOWN_signature"].values,
                        "Bulk DOWN Signature Score", cmap="Blues_r",
                        label="Score")
            mi_dn = safe_morans_i(spatial_coords,
                                  adata.obs["Bulk_DOWN_signature"].values)
            log(f"   Bulk DOWN signature: Moran's I={mi_dn:.3f}")
            plt.suptitle("Bulk RNA-seq DEG Signatures in Spatial Data",
                         y=1.02, fontsize=14)
            save_fig("spatial_bulk_05_bulk_degs_spatial.png")

            fig, ax = plt.subplots(figsize=(10, 5))
            for sig_name in ["Bulk_UP_signature", "Bulk_DOWN_signature"]:
                vals = adata.obs[sig_name].values
                ax.plot(range(len(unique_stages)),
                        stage_means_series(vals, stage, unique_stages,
                                           valid_mask),
                        "o-", label=sig_name.replace("_", " "), lw=2)
            ax.set_xticks(range(len(unique_stages)))
            ax.set_xticklabels([STAGE_MAP.get(int(s), str(s))
                                for s in unique_stages], rotation=45)
            ax.set_xlabel("Disease Stage")
            ax.set_ylabel("Mean Signature Score")
            ax.set_title("Bulk DEG Signatures Along Disease Progression")
            ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
            save_fig("spatial_bulk_05_bulk_degs_trajectory.png")

            pd.DataFrame({
                "cell": adata.obs_names,
                "Bulk_UP_signature":   adata.obs["Bulk_UP_signature"].values,
                "Bulk_DOWN_signature": adata.obs["Bulk_DOWN_signature"].values,
                "stage": stage, "pseudotime": pseudotime,
            }).to_csv("spatial_bulk_05_bulk_signature_scores.csv", index=False)


# ── BulkVal 6: Bulk vs Spatial Correlation Analysis ──────────────────────────
with safe_analysis("BULK VALIDATION 6: Bulk vs Spatial Correlation Analysis"):
    if bulk["bulk_degs"] is None or len(expr_matrix.index) == 0:
        log("   SKIPPED: No bulk DEG data available")
    else:
        # Compute Spearman r of spatial expression vs pseudotime for each gene
        genes_test = bulk["bulk_degs"]["Symbol"].dropna().unique()[:200]
        spatial_corr = []
        for gene in genes_test:
            if gene not in expr_matrix.index:
                continue
            vals = get_gene(expr_matrix, gene, adata.obs_names)
            if vals.std() == 0:
                continue
            r_pt,  p_pt  = spearmanr(vals, pseudotime)
            r_st,  p_st  = spearmanr(vals, stage.astype(float))
            spatial_corr.append({
                "Gene": gene,
                "spatial_vs_pseudotime_r": r_pt,
                "spatial_vs_pseudotime_p": p_pt,
                "spatial_vs_stage_r":      r_st,
                "spatial_vs_stage_p":      p_st,
            })

        if spatial_corr:
            sc_df    = pd.DataFrame(spatial_corr)
            bulk_lfc = (bulk["bulk_degs"][["Symbol", "logFC", "adj.P.Val"]]
                        .drop_duplicates("Symbol"))
            merged   = sc_df.merge(bulk_lfc, left_on="Gene",
                                   right_on="Symbol", how="inner")

            if len(merged) > 10:
                colors = ["#D62728" if (abs(x) > 0.3 and abs(y) > 0.5)
                          else "#1F77B4" if abs(x) > 0.3 else "grey"
                          for x, y in zip(merged["spatial_vs_pseudotime_r"],
                                          merged["logFC"])]
                fig, ax = plt.subplots(figsize=(8, 7))
                ax.scatter(merged["logFC"],
                           merged["spatial_vs_pseudotime_r"],
                           c=colors, alpha=0.6, s=50)

                top_genes = (merged.nlargest(10, "spatial_vs_pseudotime_r")
                             ["Gene"].tolist() +
                             merged.nsmallest(10, "spatial_vs_pseudotime_r")
                             ["Gene"].tolist())
                for gene in top_genes[:15]:
                    row = merged[merged["Gene"] == gene].iloc[0]
                    ax.annotate(gene,
                                (row["logFC"], row["spatial_vs_pseudotime_r"]),
                                xytext=(5, 5), textcoords="offset points",
                                fontsize=7)

                ax.axhline(0, color="black", alpha=0.5)
                ax.axvline(0, color="black", alpha=0.5)
                ax.axhline( 0.3, color="grey", linestyle="--", alpha=0.5)
                ax.axhline(-0.3, color="grey", linestyle="--", alpha=0.5)
                ax.set_xlabel("Bulk logFC (HCM vs Control)")
                ax.set_ylabel("Spatial Correlation with Pseudotime")
                ax.set_title("Bulk vs Spatial: Gene Expression Correlation")

                r_overall, p_overall = pearsonr(merged["logFC"],
                                                merged["spatial_vs_pseudotime_r"])
                ax.text(0.05, 0.95, f"r = {r_overall:.3f}\np = {p_overall:.3e}",
                        transform=ax.transAxes, fontsize=10, va="top")
                plt.tight_layout()
                save_fig("spatial_bulk_06_bulk_vs_spatial_correlation.png")

                merged.to_csv("spatial_bulk_06_bulk_spatial_integration.csv",
                              index=False)
                log(f"   Bulk-spatial correlation: r={r_overall:.3f}"
                    f" (p={p_overall:.3e})")
                if r_overall < 0.2:
                    critical(f"Poor agreement between bulk and spatial data"
                             f" (r={r_overall:.3f}) - check data compatibility")


# ── BulkVal 7: Disease Progression Validation (Kruskal-Wallis) ───────────────
with safe_analysis("BULK VALIDATION 7: Disease Progression Validation"):
    sig_cols = [c for c in adata.obs.columns
                if c.startswith(("WGCNA_M", "REG_", "Bulk_"))]
    if len(sig_cols) == 0:
        log("   SKIPPED: no bulk-derived signature columns found")
    else:
        anova_results = []
        for sig in sig_cols:
            groups = [adata.obs[sig][(stage == int(s)) & valid_mask].values
                      for s in unique_stages]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                h_stat, p_val = kruskal(*groups)
                anova_results.append({
                    "Signature": sig, "Kruskal_p": p_val,
                    "Significant": p_val < 0.05,
                })

        if anova_results:
            anova_df = (pd.DataFrame(anova_results)
                        .sort_values("Kruskal_p"))
            anova_df.to_csv("spatial_bulk_07_stage_signature_anova.csv",
                            index=False)
            sig_pct = (anova_df["Significant"].sum() / len(anova_df) * 100)
            log(f"   {sig_pct:.1f}% of bulk-derived signatures show"
                " significant stage variation")


# =============================================================================
# PART 3 – EXTENSION ANALYSES (EXT 1–8)
# =============================================================================

# ── EXT1: Spatially Variable Genes ───────────────────────────────────────────
with safe_analysis("EXT1: Spatially Variable Genes"):
    N_TEST_GENES  = 500   # top-expressed genes to test (speed vs coverage)
    N_TOP_DISPLAY = 12    # number to show in spatial plots
    MI_THRESHOLD  = 0.10  # minimum Moran's I to call "spatially variable"

    genes_to_test = gene_means[gene_means > 0].index[:N_TEST_GENES].tolist()
    svg_results   = []
    for gene in genes_to_test:
        vals = get_gene(expr_matrix, gene, adata.obs_names)
        if vals.std() == 0:
            continue
        mi = safe_morans_i(spatial_coords, vals)
        svg_results.append({"Gene": gene, "MoransI": mi,
                             "MeanExpr": gene_means[gene]})

    svg_df = (pd.DataFrame(svg_results)
              .sort_values("MoransI", ascending=False)
              .reset_index(drop=True))
    svg_df.to_csv("spatial_ext1_spatially_variable_genes.csv", index=False)

    top_svgs = svg_df.head(N_TOP_DISPLAY)["Gene"].tolist()
    log(f"   Top SVGs: {top_svgs[:6]}")
    log(f"   Genes with MI > {MI_THRESHOLD}:"
        f" {(svg_df['MoransI'] > MI_THRESHOLD).sum()} / {len(svg_df)}")

    n_cols = 4; n_rows = (N_TOP_DISPLAY + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()
    for idx, gene in enumerate(top_svgs):
        vals = get_gene(expr_matrix, gene, adata.obs_names)
        mi   = svg_df.loc[svg_df["Gene"] == gene, "MoransI"].values[0]
        scatter_map(axes_flat[idx], spatial_coords, vals,
                    f"{gene}\nMoran's I={mi:.3f}", cmap="magma", label="Expr")
    for ax in axes_flat[len(top_svgs):]:
        ax.axis("off")
    plt.suptitle("Top Spatially Variable Genes (data-driven)",
                 y=1.02, fontsize=14)
    save_fig("spatial_ext1_svgs_spatial.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(svg_df["MoransI"], bins=50, color="#4C72B0", edgecolor="white")
    ax.axvline(MI_THRESHOLD, color="red", linestyle="--",
               label=f"Threshold ({MI_THRESHOLD})")
    ax.set_xlabel("Moran's I"); ax.set_ylabel("Number of genes")
    ax.set_title("Distribution of Spatial Autocorrelation Across Genes")
    ax.legend(); plt.tight_layout()
    save_fig("spatial_ext1_moransI_distribution.png")


# ── EXT2: Metabolic State Spatial Mapping ────────────────────────────────────
with safe_analysis("EXT2: Metabolic State Spatial Mapping"):
    FAO_GENES = [
        "PPARA", "RXRA", "CPT1A", "CPT1B", "CPT2", "ACADL", "ACADM",
        "ACADVL", "HADHA", "HADHB", "ECHS1", "ACAA2", "ETFDH", "ETFA",
        "ACSL1", "ACSL6", "SLC25A20", "FABP3",
    ]
    GLYCOLYSIS_GENES = [
        "HK1", "HK2", "GPI", "PFKM", "PFKL", "ALDOA", "TPI1",
        "GAPDH", "PGK1", "PGAM1", "ENO1", "ENO3", "PKM", "LDHA",
        "LDHB", "SLC2A1", "SLC2A4", "PFKFB3",
    ]
    OXPHOS_GENES = [
        "NDUFA1", "NDUFB3", "NDUFB5", "SDHB", "UQCRB", "COX4I1",
        "COX7A1", "ATP5F1A", "ATP5F1B", "ATP5MC1",
    ]
    KETONE_GENES = ["BDH1", "OXCT1", "HMGCS2", "ACAT1", "ACAT2"]

    for name, genes in [("FAO",       FAO_GENES),
                         ("Glycolysis", GLYCOLYSIS_GENES),
                         ("OXPHOS",    OXPHOS_GENES),
                         ("Ketone",    KETONE_GENES)]:
        score_module(adata, genes, f"Metab_{name}")

    # FAO-glycolysis ratio: positive = oxidative metabolism dominates
    adata.obs["Metab_FAO_Glycolysis_ratio"] = (
        adata.obs["Metab_FAO"] - adata.obs["Metab_Glycolysis"]
    )

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (col, label, cmap) in zip(axes, [
        ("Metab_FAO",               "FAO Score",         "YlOrRd"),
        ("Metab_Glycolysis",        "Glycolysis Score",  "YlGnBu"),
        ("Metab_OXPHOS",            "OXPHOS Score",      "Purples"),
        ("Metab_FAO_Glycolysis_ratio", "FAO/Glycolysis", "RdBu_r"),
    ]):
        vals = adata.obs[col].values
        mi   = safe_morans_i(spatial_coords, vals)
        scatter_map(ax, spatial_coords, vals,
                    f"{label}\nMoran's I={mi:.3f}", cmap=cmap, label=label)
    plt.suptitle("Metabolic State Spatial Mapping", y=1.02, fontsize=14)
    save_fig("spatial_ext2_metabolic_spatial.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, color in [("Metab_FAO", "#D62728"), ("Metab_Glycolysis", "#1F77B4"),
                       ("Metab_OXPHOS", "#9467BD"), ("Metab_Ketone", "#2CA02C")]:
        vals = adata.obs[col].values
        ax.plot(range(len(unique_stages)),
                stage_means_series(vals, stage, unique_stages, valid_mask),
                "o-", label=col.replace("Metab_", ""), lw=2, color=color)
    ax.set_xticks(range(len(unique_stages)))
    ax.set_xticklabels([STAGE_MAP.get(int(s), str(s)) for s in unique_stages],
                       rotation=45)
    ax.set_xlabel("Disease Stage"); ax.set_ylabel("Mean Score")
    ax.set_title("Metabolic Programme Shift Along Disease Progression")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    save_fig("spatial_ext2_metabolic_trajectory.png")

    metab_cols = ["Metab_FAO", "Metab_Glycolysis", "Metab_OXPHOS",
                  "Metab_Ketone", "Metab_FAO_Glycolysis_ratio"]
    adata.obs[metab_cols].to_csv("spatial_ext2_metabolic_scores.csv")
    log(f"   FAO Moran's I : "
        f"{safe_morans_i(spatial_coords, adata.obs['Metab_FAO'].values):.3f}")
    log(f"   Glycolysis MI : "
        f"{safe_morans_i(spatial_coords, adata.obs['Metab_Glycolysis'].values):.3f}")
    log(f"   FAO/Glycolysis correlation with HCM_score: "
        f"{pearsonr(hcm_score, adata.obs['Metab_FAO_Glycolysis_ratio'].values)[0]:.3f}")


# ── EXT3: Fibrosis / ECM Spatial Gradient ────────────────────────────────────
with safe_analysis("EXT3: Fibrosis / ECM Spatial Gradient"):
    FIBROSIS_GENES = [
        "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL5A1", "COL6A1",
        "COL6A2", "COL6A3", "FN1", "POSTN", "THBS1", "THBS2",
        "ACTA2", "TAGLN", "VIM", "DCN", "LUM", "BGN",
        "TGFB1", "TGFB2", "TGFB3", "CTGF", "CCN2",
        "MMP2", "MMP9", "MMP14", "TIMP1", "TIMP3",
        "SPARC", "TNC", "FBLN5", "ELN", "FBN1",
    ]
    MYOFIBROBLAST_GENES = ["ACTA2", "TAGLN", "CNN1", "MYH11",
                           "PDGFRA", "PDGFRB"]
    ANTI_FIBROSIS_GENES = ["SMAD7", "BMP7", "HGF", "KGF", "PTEN", "KLOTHO"]

    score_module(adata, FIBROSIS_GENES,      "Fibrosis_score")
    score_module(adata, MYOFIBROBLAST_GENES, "Myofibroblast_score")
    score_module(adata, ANTI_FIBROSIS_GENES, "AntiFibrosis_score")
    adata.obs["Net_fibrosis"] = (adata.obs["Fibrosis_score"]
                                 - adata.obs["AntiFibrosis_score"])

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (col, cmap, label) in zip(axes, [
        ("Fibrosis_score",      "Oranges",  "Fibrosis Score"),
        ("Myofibroblast_score", "Reds",     "Myofibroblast Score"),
        ("AntiFibrosis_score",  "Greens",   "Anti-Fibrosis Score"),
        ("Net_fibrosis",        "RdYlGn_r", "Net Fibrosis"),
    ]):
        vals = adata.obs[col].values
        mi   = safe_morans_i(spatial_coords, vals)
        scatter_map(ax, spatial_coords, vals,
                    f"{label}\nMI={mi:.3f}", cmap=cmap, label=label)
    plt.suptitle("Fibrosis / ECM Spatial Gradient", y=1.02, fontsize=14)
    save_fig("spatial_ext3_fibrosis_spatial.png")

    threshold_fib = np.percentile(adata.obs["Fibrosis_score"], 90)
    adata.obs["Fibrosis_hotspot"] = (
        adata.obs["Fibrosis_score"] > threshold_fib).astype(int)

    hotspot_by_stage = []
    for s in unique_stages:
        m   = (stage == int(s)) & valid_mask
        pct = adata.obs["Fibrosis_hotspot"][m].mean() * 100 \
              if m.sum() > 0 else np.nan
        hotspot_by_stage.append({"Stage": STAGE_MAP.get(int(s), str(s)),
                                  "Hotspot_%": pct})
    log(f"   Fibrosis hotspot % by stage:\n"
        f"{pd.DataFrame(hotspot_by_stage).to_string(index=False)}")

    contractile_genes = ["MYH7", "MYH6", "TNNT2", "TNNI3", "TPM1", "ACTC1"]
    contractile_vals, _ = get_genes_mean(expr_matrix, contractile_genes,
                                         adata.obs_names)
    if contractile_vals is not None:
        r_fc, p_fc = pearsonr(adata.obs["Fibrosis_score"].values,
                              contractile_vals)
        log(f"   Fibrosis vs Contractile correlation: r={r_fc:.3f}"
            f" (p={p_fc:.3e})")

    adata.obs[["Fibrosis_score", "Myofibroblast_score",
               "AntiFibrosis_score", "Net_fibrosis",
               "Fibrosis_hotspot"]].to_csv("spatial_ext3_fibrosis_scores.csv")


# ── EXT4: Proximity-Based Ligand–Receptor Hotspots ───────────────────────────
with safe_analysis("EXT4: Proximity-Based Ligand-Receptor Hotspots"):
    # Cardiac-relevant L-R pairs from the literature
    LR_PAIRS = [
        ("TGFB1", "TGFBR1"), ("TGFB1", "TGFBR2"),  # fibrosis/hypertrophy
        ("ANGPT2", "TEK"),   ("EDN1",  "EDNRA"),      # angiogenesis/vasoconstrit.
        ("NRG1",  "ERBB4"),  ("IGF1",  "IGF1R"),      # cardioprotection/hypertrophy
        ("VEGFA", "KDR"),    ("PDGFB", "PDGFRB"),     # angiogenesis/fibroblasts
        ("FGF2",  "FGFR1"),  ("WNT5A", "FZD2"),       # remodelling/WNT
        ("BMP2",  "BMPR2"),  ("ADM",   "CALCRL"),     # anti-hypertrophy/adrenomedull
    ]
    K_NEIGHBOURS = 10  # number of spatial neighbours for receptor averaging

    nbrs   = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1,
                               algorithm="ball_tree").fit(spatial_coords)
    nn_idx = nbrs.kneighbors(spatial_coords,
                              return_distance=False)[:, 1:]  # exclude self

    lr_scores   = {}
    valid_pairs = []
    for lig, rec in LR_PAIRS:
        lig_expr = get_gene(expr_matrix, lig, adata.obs_names)
        rec_expr = get_gene(expr_matrix, rec, adata.obs_names)
        if lig_expr.sum() == 0 or rec_expr.sum() == 0:
            continue
        # Communication score = ligand × neighbour-averaged receptor
        rec_neighbour = rec_expr[nn_idx].mean(axis=1)
        pair_name     = f"{lig}_{rec}"
        lr_scores[pair_name] = lig_expr * rec_neighbour
        valid_pairs.append(pair_name)

    log(f"   Valid L-R pairs (both expressed): {len(valid_pairs)}")
    if valid_pairs:
        sorted_pairs = sorted(valid_pairs,
                              key=lambda p: lr_scores[p].mean(), reverse=True)
        n_plot = min(6, len(sorted_pairs))
        n_cols = 3; n_rows = (n_plot + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows))
        axes_flat = np.array(axes).flatten()
        for idx, pair in enumerate(sorted_pairs[:n_plot]):
            vals = lr_scores[pair]
            mi   = safe_morans_i(spatial_coords, vals)
            scatter_map(axes_flat[idx], spatial_coords, vals,
                        f"{pair}\nMI={mi:.3f}", cmap="hot_r",
                        label="LR Score")
        for ax in axes_flat[n_plot:]:
            ax.axis("off")
        plt.suptitle("Spatial Ligand-Receptor Communication Hotspots",
                     y=1.02, fontsize=14)
        save_fig("spatial_ext4_lr_hotspots.png")

        lr_df            = pd.DataFrame(lr_scores, index=adata.obs_names)
        lr_df["cluster"] = clusters
        cluster_lr_mean  = lr_df.groupby("cluster")[sorted_pairs].mean()
        fig, ax = plt.subplots(figsize=(max(8, len(sorted_pairs)), 6))
        sns.heatmap(cluster_lr_mean.T, annot=True, fmt=".2f",
                    cmap="YlOrRd", ax=ax)
        ax.set_title("Mean L-R Communication Score per Cluster")
        ax.set_xlabel("Cluster"); ax.set_ylabel("L-R Pair"); plt.tight_layout()
        save_fig("spatial_ext4_lr_cluster_heatmap.png")

        lr_df.drop(columns="cluster").to_csv("spatial_ext4_lr_scores.csv")


# ── EXT5: Cardiac Stress Response Zones ──────────────────────────────────────
with safe_analysis("EXT5: Cardiac Stress Response Zones"):
    HSP_GENES = [
        "HSPA1A", "HSPA1B", "HSPA5", "HSPA8", "HSPA9",
        "HSPB1",  "HSPB2",  "HSPB7", "HSPB8",
        "HSP90AA1", "HSP90AB1", "HSP90B1",
        "DNAJB1", "DNAJB6", "DNAJB9", "BAG3",
    ]
    AUTOPHAGY_GENES = [
        "BECN1", "ATG5", "ATG7", "ATG12", "ATG16L1",
        "ULK1",  "ULK2", "SQSTM1", "NBR1", "OPTN",
        "MAP1LC3A", "MAP1LC3B", "LAMP1", "LAMP2",
        "BNIP3", "BNIP3L", "PINK1", "PARK2",
    ]
    ISR_GENES = [
        "ATF4", "ATF3", "ATF6", "DDIT3", "ASNS",
        "SLC7A11", "TRIB3", "CHAC1", "HERPUD1",
        "EIF2AK1", "EIF2AK2", "EIF2AK3", "EIF2AK4",
    ]
    PROTEASOME_GENES = [
        "PSMA1", "PSMA2", "PSMB1", "PSMB5", "PSMD1",
        "PSMD2", "PSMD4", "UBA52", "UBB", "UBC",
        "UCHL1", "USP14", "STUB1",
    ]

    for name, genes in [("HSP",        HSP_GENES),
                         ("Autophagy",  AUTOPHAGY_GENES),
                         ("ISR",        ISR_GENES),
                         ("Proteasome", PROTEASOME_GENES)]:
        score_module(adata, genes, f"Stress_{name}")

    # Compute a combined stress PC1 (first principal component of the four scores)
    stress_mat    = np.column_stack([adata.obs[f"Stress_{n}"].values
                                     for n in ["HSP", "Autophagy", "ISR",
                                               "Proteasome"]])
    stress_scaled = StandardScaler().fit_transform(stress_mat)
    stress_pc1    = PCA(n_components=1).fit_transform(stress_scaled)[:, 0]
    adata.obs["Stress_combined_PC1"] = stress_pc1

    stress_panels = [
        ("Stress_HSP",          "YlOrRd",  "Heat-Shock Score"),
        ("Stress_Autophagy",    "PuBuGn",  "Autophagy Score"),
        ("Stress_ISR",          "RdPu",    "ISR Score"),
        ("Stress_Proteasome",   "BuPu",    "Proteasome Score"),
        ("Stress_combined_PC1", "RdBu_r",  "Combined Stress PC1"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    for ax, (col, cmap, label) in zip(axes, stress_panels):
        vals = adata.obs[col].values
        mi   = safe_morans_i(spatial_coords, vals)
        scatter_map(ax, spatial_coords, vals,
                    f"{label}\nMI={mi:.3f}", cmap=cmap, label=label)
    plt.suptitle("Cardiac Stress Response Zones", y=1.02, fontsize=14)
    save_fig("spatial_ext5_stress_zones.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap_s  = plt.cm.tab10
    for k, (col, _, label) in enumerate(stress_panels):
        vals = adata.obs[col].values
        ax.plot(range(len(unique_stages)),
                stage_means_series(vals, stage, unique_stages, valid_mask),
                "o-", label=label, lw=2,
                color=cmap_s(k / len(stress_panels)))
    ax.set_xticks(range(len(unique_stages)))
    ax.set_xticklabels([STAGE_MAP.get(int(s), str(s)) for s in unique_stages],
                       rotation=45)
    ax.set_xlabel("Disease Stage"); ax.set_ylabel("Mean Score")
    ax.set_title("Stress Response Trajectories Across Disease Stages")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3); plt.tight_layout()
    save_fig("spatial_ext5_stress_trajectory.png")

    stress_cols = [c for c in adata.obs.columns if c.startswith("Stress_")]
    adata.obs[stress_cols].to_csv("spatial_ext5_stress_scores.csv")
    r_st, _ = pearsonr(adata.obs["Stress_combined_PC1"].values, hcm_score)
    log(f"   Combined stress PC1 vs HCM_score: r={r_st:.3f}")


# ── EXT6: Transmural / Radial Gradient Analysis ──────────────────────────────
with safe_analysis("EXT6: Transmural / Radial Gradient Analysis"):
    centroid    = spatial_coords.mean(axis=0)
    radial_dist = np.linalg.norm(spatial_coords - centroid, axis=1)
    adata.obs["Radial_distance"] = radial_dist

    r_min, r_max = radial_dist.min(), radial_dist.max()
    radial_norm  = (radial_dist - r_min) / (r_max - r_min + 1e-8)
    adata.obs["Radial_norm"] = radial_norm

    # Ternary radial zones: Core / Mid / Periphery
    zone_labels = pd.cut(radial_norm, bins=[0, 0.33, 0.66, 1.0],
                         labels=["Core", "Mid", "Periphery"])
    adata.obs["Radial_zone"] = zone_labels.astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scatter_map(axes[0], spatial_coords, radial_norm,
                "Normalised Radial Distance", cmap="plasma",
                label="Distance")
    axes[0].scatter(*centroid[::-1], marker="*", c="red", s=200,
                    zorder=5, label="Centroid")
    axes[0].legend()
    scatter_map(axes[1], spatial_coords, hcm_score,
                "HCM Score (reference)", cmap="RdBu_r",
                label="HCM Score")
    zone_colour_map = {"Core": 0, "Mid": 0.5, "Periphery": 1.0}
    scatter_map(axes[2], spatial_coords,
                np.array([zone_colour_map.get(z, 0.5)
                           for z in adata.obs["Radial_zone"]]),
                "Radial Zones", cmap="Set1", label="Zone")
    plt.suptitle("Transmural / Radial Spatial Gradient", y=1.02, fontsize=14)
    save_fig("spatial_ext6_radial_gradient.png")

    # Spearman correlation of each gene with radial distance
    radial_gene_corr = []
    for gene in gene_means.index[:500]:
        vals = get_gene(expr_matrix, gene, adata.obs_names)
        if vals.std() == 0:
            continue
        r, p = spearmanr(vals, radial_dist)
        radial_gene_corr.append({"Gene": gene, "r_radial": r, "p_radial": p})

    rgc_df = (pd.DataFrame(radial_gene_corr)
              .sort_values("r_radial", ascending=False)
              .reset_index(drop=True))
    rgc_df.to_csv("spatial_ext6_radial_gene_correlations.csv", index=False)
    log(f"   Top centripetal genes (enriched at core): "
        f"{rgc_df.tail(6)['Gene'].tolist()}")
    log(f"   Top centrifugal genes (enriched at periphery): "
        f"{rgc_df.head(6)['Gene'].tolist()}")

    # Violin plots of key scores across radial zones
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, score_col in zip(axes, ["HCM_score", "Stress_combined_PC1",
                                     "Metab_FAO_Glycolysis_ratio"]):
        if score_col not in adata.obs.columns:
            ax.axis("off"); continue
        zone_data = [adata.obs[score_col][adata.obs["Radial_zone"] == z].values
                     for z in ["Core", "Mid", "Periphery"]]
        ax.violinplot(zone_data, positions=[0, 1, 2], showmedians=True)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Core", "Mid", "Periphery"])
        ax.set_title(score_col.replace("_", " "))
        ax.set_ylabel("Score"); ax.grid(True, alpha=0.3)
    plt.suptitle("Score Distribution Across Radial Zones", y=1.02, fontsize=14)
    save_fig("spatial_ext6_zone_violins.png")

    adata.obs[["Radial_distance", "Radial_norm",
               "Radial_zone"]].to_csv("spatial_ext6_radial_metadata.csv")


# ── EXT7: Immune & Inflammatory Infiltration Map ─────────────────────────────
with safe_analysis("EXT7: Immune & Inflammatory Infiltration Map"):
    INNATE_GENES = [
        "ITGAM", "CD68", "CD14", "CSF1R", "LYZ", "S100A8", "S100A9",
        "TYROBP", "FCER1G", "FCGR3A", "C1QA", "C1QB", "C1QC",
        "MRC1", "MARCO", "ADGRE1", "CX3CR1",
    ]
    ADAPTIVE_GENES = [
        "CD3D", "CD3E", "CD4", "CD8A", "CD8B", "FOXP3",
        "GZMB", "PRF1", "NKG7", "GNLY",
        "CD19", "MS4A1", "CD79A", "IGHM",
    ]
    INFLAMMASOME_GENES = [
        "NLRP3", "PYCARD", "CASP1", "IL1B", "IL18",
        "GSDMD", "HMGB1", "S100A1", "RAGE", "IL6", "TNF",
        "CCL2", "CXCL10", "CXCL9",
    ]
    COMPLEMENT_GENES = [
        "C1QA", "C1QB", "C1QC", "C3", "C4A", "C4B",
        "CFB", "CFH", "C5AR1", "C3AR1",
    ]

    for name, genes in [("Innate",       INNATE_GENES),
                         ("Adaptive",     ADAPTIVE_GENES),
                         ("Inflammasome", INFLAMMASOME_GENES),
                         ("Complement",   COMPLEMENT_GENES)]:
        score_module(adata, genes, f"Immune_{name}")

    adata.obs["Immune_total"] = (
        adata.obs["Immune_Innate"].values
        + adata.obs["Immune_Adaptive"].values
        + adata.obs["Immune_Inflammasome"].values
    )

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    for ax, (col, cmap, label) in zip(axes, [
        ("Immune_Innate",       "Oranges", "Innate Immune"),
        ("Immune_Adaptive",     "Blues",   "Adaptive Immune"),
        ("Immune_Inflammasome", "Reds",    "Inflammasome"),
        ("Immune_Complement",   "Purples", "Complement"),
        ("Immune_total",        "hot",     "Total Immune"),
    ]):
        vals = adata.obs[col].values
        mi   = safe_morans_i(spatial_coords, vals)
        scatter_map(ax, spatial_coords, vals,
                    f"{label}\nMI={mi:.3f}", cmap=cmap, label=label)
    plt.suptitle("Immune & Inflammatory Infiltration Map", y=1.02, fontsize=14)
    save_fig("spatial_ext7_immune_map.png")

    immune_thresh  = np.percentile(adata.obs["Immune_total"], 95)
    hotspot_immune = adata.obs["Immune_total"] > immune_thresh
    log(f"   Immune hotspot cells (top 5%): {hotspot_immune.sum()}")

    for s in unique_stages:
        m   = (stage == int(s)) & valid_mask
        pct = hotspot_immune[m].mean() * 100 if m.sum() > 0 else 0
        log(f"   {STAGE_MAP.get(int(s))}: {pct:.1f}% immune hotspot cells")

    if "Fibrosis_score" in adata.obs.columns:
        r_if, p_if = pearsonr(adata.obs["Immune_total"].values,
                              adata.obs["Fibrosis_score"].values)
        log(f"   Immune_total vs Fibrosis_score: r={r_if:.3f}"
            f" (p={p_if:.3e})")

    immune_cols = [c for c in adata.obs.columns if c.startswith("Immune_")]
    adata.obs[immune_cols].to_csv("spatial_ext7_immune_scores.csv")


# ── EXT8: Sarcomere Organisation Index ───────────────────────────────────────
with safe_analysis("EXT8: Sarcomere Organisation Index"):
    SARCOMERE_GENES = {
        "Thick_filament": ["MYH6", "MYH7", "MYH9", "MYL2", "MYL3",
                           "MYL4", "MYL7", "MYLPF"],
        "Thin_filament":  ["ACTC1", "ACTA1", "TNNT2", "TNNI3", "TNNI1",
                           "TPM1", "TPM2", "TMOD1", "NEBL"],
        "Z_disc":         ["ACTN2", "LDB3",  "CSRP3", "MYOZ2", "TCAP",
                           "TTN", "ANKRD1", "CARP"],
        "Titin_elastic":  ["TTN", "FHL2", "MYPN", "MYOM1", "MYOM2"],
        "Calcium":        ["ATP2A2", "PLN", "CASQ2", "RYR2", "CACNA1C",
                           "SLC8A1", "CALM1", "S100A1"],
    }

    for comp, genes in SARCOMERE_GENES.items():
        score_module(adata, genes, f"Sarc_{comp}")

    sarc_cols_present = [c for c in adata.obs.columns
                         if c.startswith("Sarc_")]
    adata.obs["Sarc_total"] = adata.obs[sarc_cols_present].mean(axis=1)

    # Local sarcomere coherence via k-nearest-neighbour CV
    K_SARC   = 8
    nbrs_s   = NearestNeighbors(n_neighbors=K_SARC + 1).fit(spatial_coords)
    nn_idx_s = nbrs_s.kneighbors(spatial_coords,
                                  return_distance=False)[:, 1:]

    sarc_total  = adata.obs["Sarc_total"].values
    local_mean  = sarc_total[nn_idx_s].mean(axis=1)
    local_std   = sarc_total[nn_idx_s].std(axis=1)
    local_cv    = np.where(local_mean > 1e-8, local_std / (local_mean + 1e-8), 0)

    adata.obs["Sarc_local_coherence"] = 1 - local_cv   # high → organised
    adata.obs["Sarc_disarray_index"]  = local_cv        # high → disordered

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat  = axes.flatten()
    for idx, comp in enumerate(list(SARCOMERE_GENES.keys()) + ["Sarc_total"]):
        col = f"Sarc_{comp}" if not comp.startswith("Sarc") else comp
        if col not in adata.obs.columns:
            continue
        vals = adata.obs[col].values
        mi   = safe_morans_i(spatial_coords, vals)
        scatter_map(axes_flat[idx], spatial_coords, vals,
                    f"{col.replace('Sarc_', '')}\nMI={mi:.3f}",
                    cmap="YlOrRd", label="Score")

    scatter_map(axes_flat[6], spatial_coords,
                adata.obs["Sarc_disarray_index"].values,
                "Sarcomere Disarray Index\n(high=disordered)",
                cmap="hot_r", label="CV")
    scatter_map(axes_flat[7], spatial_coords,
                adata.obs["Sarc_local_coherence"].values,
                "Local Sarcomere Coherence\n(high=organised)",
                cmap="Blues", label="1-CV")
    plt.suptitle("Sarcomere Organisation Index", y=1.02, fontsize=14)
    save_fig("spatial_ext8_sarcomere_organisation.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label, color in [
        ("Sarc_total",           "Sarcomere Score",  "#1F77B4"),
        ("Sarc_local_coherence", "Coherence",        "#2CA02C"),
        ("Sarc_disarray_index",  "Disarray Index",   "#D62728"),
    ]:
        vals = adata.obs[col].values
        ax.plot(range(len(unique_stages)),
                stage_means_series(vals, stage, unique_stages, valid_mask),
                "o-", label=label, lw=2, color=color)
    ax.set_xticks(range(len(unique_stages)))
    ax.set_xticklabels([STAGE_MAP.get(int(s), str(s)) for s in unique_stages],
                       rotation=45)
    ax.set_xlabel("Disease Stage"); ax.set_ylabel("Score")
    ax.set_title("Sarcomere Organisation Across Disease Stages")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    save_fig("spatial_ext8_sarcomere_trajectory.png")

    r_dis, p_dis = pearsonr(adata.obs["Sarc_disarray_index"].values, hcm_score)
    log(f"   Sarcomere Disarray vs HCM_score: r={r_dis:.3f}"
        f" (p={p_dis:.3e})")

    sarc_out_cols = [c for c in adata.obs.columns if c.startswith("Sarc_")]
    adata.obs[sarc_out_cols].to_csv("spatial_ext8_sarcomere_scores.csv")


# =============================================================================
# PART 4 – FOLLOW-UP ANALYSES (FU 1–3)
# =============================================================================

# ── FU1: Fibrosis–Contractile Spatial Antagonism ─────────────────────────────
with safe_analysis("FU1: Fibrosis-Contractile Spatial Antagonism"):
    CONTRACTILE_GENES = [
        "MYH6", "MYH7", "TNNT2", "TNNI3", "TPM1",
        "ACTC1", "MYL2", "MYL3", "MYBPC3", "TTN",
    ]
    # Score contractile if not already present from Analysis 9
    if "Contractile_score" not in adata.obs.columns:
        score_module(adata, CONTRACTILE_GENES, "Contractile_score")

    fib   = adata.obs["Fibrosis_score"].values
    cont  = adata.obs["Contractile_score"].values

    # Z-normalise both scores for symmetric thresholding
    fib_z  = (fib  - fib.mean())  / (fib.std()  + 1e-8)
    cont_z = (cont - cont.mean()) / (cont.std() + 1e-8)

    # Define four spatial cell states using ±0.5 SD cut-off
    THRESH = 0.5
    state  = np.full(len(fib_z), "Mixed", dtype=object)
    state[(fib_z >  THRESH) & (cont_z <  -THRESH)] = "Fibrotic_lost"
    state[(fib_z < -THRESH) & (cont_z >   THRESH)] = "Contractile_intact"
    state[(fib_z >  THRESH) & (cont_z >   THRESH)] = "Transitional"
    state[(fib_z < -THRESH) & (cont_z <  -THRESH)] = "Depleted"
    adata.obs["FC_state"] = state
    log(f"   FC state counts:\n{pd.Series(state).value_counts().to_string()}")

    STATE_COLORS = {
        "Fibrotic_lost":      "#D62728",
        "Contractile_intact": "#2CA02C",
        "Transitional":       "#FF7F0E",
        "Depleted":           "#7F7F7F",
        "Mixed":              "#AEC7E8",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for s_name, color in STATE_COLORS.items():
        mask = state == s_name
        if mask.sum() == 0:
            continue
        axes[0].scatter(spatial_coords[mask, 1], spatial_coords[mask, 0],
                        c=color, s=20, alpha=0.8, label=s_name)
    axes[0].invert_yaxis()
    axes[0].set_title("Fibrosis vs Contractile Cell States", fontweight="bold")
    axes[0].set_xlabel("imagecol"); axes[0].set_ylabel("imagerow")
    axes[0].legend(fontsize=7, markerscale=1.5)
    scatter_map(axes[1], spatial_coords, fib_z,
                "Fibrosis Score (Z)", cmap="Reds", label="Z-score")
    scatter_map(axes[2], spatial_coords, cont_z,
                "Contractile Score (Z)", cmap="Greens", label="Z-score")
    plt.suptitle("Fibrosis–Contractile Spatial Antagonism", y=1.02, fontsize=14)
    save_fig("spatial_fu1_fc_antagonism_map.png")

    # Stage composition stacked bar
    stage_state_mat = {}
    for s in unique_stages:
        m = (stage == int(s)) & valid_mask
        total = m.sum()
        stage_state_mat[STAGE_MAP.get(int(s), str(s))] = {
            st: (state[m] == st).sum() / total * 100 if total > 0 else 0
            for st in STATE_COLORS
        }
    stage_state_df = pd.DataFrame(stage_state_mat).T
    log(f"   State % per stage:\n{stage_state_df.round(1).to_string()}")

    fig, ax = plt.subplots(figsize=(9, 5))
    stage_state_df.plot(kind="bar", stacked=True, ax=ax,
                        color=[STATE_COLORS[c] for c in stage_state_df.columns],
                        edgecolor="white")
    ax.set_xlabel("Disease Stage"); ax.set_ylabel("% of cells")
    ax.set_title("Fibrosis–Contractile State Composition Per Stage")
    ax.legend(bbox_to_anchor=(1.01, 1), fontsize=8)
    plt.xticks(rotation=30); plt.tight_layout()
    save_fig("spatial_fu1_fc_stage_composition.png")

    pd.DataFrame({
        "FC_state": state,
        "Fibrosis_Z":    fib_z,
        "Contractile_Z": cont_z,
    }, index=adata.obs_names).to_csv("spatial_fu1_fc_states.csv")

    for s in unique_stages:
        m   = (stage == int(s)) & valid_mask
        pct = (state[m] == "Fibrotic_lost").sum() / m.sum() * 100 \
              if m.sum() > 0 else 0
        log(f"   {STAGE_MAP.get(int(s))}: {pct:.1f}% Fibrotic_lost cells")


# ── FU2: Border-Zone Analysis (Fibrosis–Immune Interface) ────────────────────
with safe_analysis("FU2: Border-Zone Analysis (Fibrosis-Immune Interface)"):
    # Strategy: find cells near BOTH a fibrotic core AND an immune zone
    BORDER_RADIUS_PERCENTILE = 10  # radius = 3× the 10th-pctile NN distance

    fib_vals = adata.obs["Fibrosis_score"].values
    imm_vals = adata.obs["Immune_total"].values

    fib_core = fib_vals > np.percentile(fib_vals, 75)
    imm_zone = imm_vals > np.percentile(imm_vals, 75)

    nn_d   = NearestNeighbors(n_neighbors=2).fit(spatial_coords)
    dists, _ = nn_d.kneighbors(spatial_coords)
    radius = np.percentile(dists[:, 1], BORDER_RADIUS_PERCENTILE) * 3

    dist_mat = cdist(spatial_coords, spatial_coords)
    near_fib = (dist_mat[:, fib_core] < radius).any(axis=1)
    near_imm = (dist_mat[:, imm_zone] < radius).any(axis=1)

    border = near_fib & near_imm & (~fib_core) & (~imm_zone)
    log(f"   Cells in border zone: {border.sum()}")
    log(f"   Fibrosis core cells:  {fib_core.sum()}")
    log(f"   Immune zone cells:    {imm_zone.sum()}")

    zone_label             = np.full(len(fib_vals), "Other", dtype=object)
    zone_label[fib_core]   = "Fibrosis_core"
    zone_label[imm_zone]   = "Immune_zone"
    zone_label[border]     = "Border_zone"  # border overwrites when overlapping
    adata.obs["Interface_zone"] = zone_label

    ZONE_COLORS = {
        "Fibrosis_core": "#D62728",
        "Immune_zone":   "#1F77B4",
        "Border_zone":   "#FF7F0E",
        "Other":         "#CCCCCC",
    }
    fig, ax = plt.subplots(figsize=(8, 7))
    for z, color in ZONE_COLORS.items():
        m = zone_label == z
        ax.scatter(spatial_coords[m, 1], spatial_coords[m, 0],
                   c=color, s=20, alpha=0.9,
                   label=f"{z} (n={m.sum()})")
    ax.invert_yaxis()
    ax.set_title("Fibrosis–Immune Border Zone Map", fontweight="bold")
    ax.set_xlabel("imagecol"); ax.set_ylabel("imagerow")
    ax.legend(fontsize=8, markerscale=1.5); plt.tight_layout()
    save_fig("spatial_fu2_border_zone_map.png")

    # Gene enrichment in border zone vs other zones (mean-expression ratio)
    genes_test   = gene_means[gene_means > 0].index[:300].tolist()
    border_enrich = []
    border_idx    = np.where(zone_label == "Border_zone")[0]
    fib_idx       = np.where(zone_label == "Fibrosis_core")[0]
    imm_idx       = np.where(zone_label == "Immune_zone")[0]

    if len(border_idx) > 5:
        for gene in genes_test:
            vals     = get_gene(expr_matrix, gene, adata.obs_names)
            m_fib    = vals[fib_idx].mean()
            m_border = vals[border_idx].mean()
            m_imm    = vals[imm_idx].mean()
            if m_fib + m_border + m_imm == 0:
                continue
            baseline = (m_fib + m_imm) / 2 + 1e-6
            fc       = (m_border + 1e-6) / baseline
            border_enrich.append({
                "Gene": gene,
                "mean_border":          m_border,
                "mean_fibrosis_core":   m_fib,
                "mean_immune_zone":     m_imm,
                "border_vs_baseline_FC": fc,
            })

        border_df = (pd.DataFrame(border_enrich)
                     .sort_values("border_vs_baseline_FC", ascending=False)
                     .reset_index(drop=True))
        border_df.to_csv("spatial_fu2_border_zone_genes.csv", index=False)
        top_border = border_df.head(10)["Gene"].tolist()
        log(f"   Top border-zone enriched genes: {top_border}")

        # Heatmap of top 20 border-enriched genes
        top20 = border_df.head(20)["Gene"].tolist()
        hm_data = pd.DataFrame({
            "Border":       [get_gene(expr_matrix, g, adata.obs_names)[border_idx].mean()
                             for g in top20],
            "Fibrosis_core":[get_gene(expr_matrix, g, adata.obs_names)[fib_idx].mean()
                             for g in top20],
            "Immune_zone":  [get_gene(expr_matrix, g, adata.obs_names)[imm_idx].mean()
                             for g in top20],
        }, index=top20)
        hm_z = hm_data.subtract(hm_data.mean(axis=1), axis=0)\
                       .divide(hm_data.std(axis=1) + 1e-8, axis=0)
        fig, ax = plt.subplots(figsize=(7, 9))
        sns.heatmap(hm_z, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-2, vmax=2, ax=ax)
        ax.set_title("Border Zone Gene Enrichment\n(Z-score vs zone means)")
        plt.tight_layout()
        save_fig("spatial_fu2_border_zone_heatmap.png")
    else:
        log("   WARNING: too few border-zone cells"
            " -- increase radius or lower quartile threshold")

    adata.obs[["Interface_zone"]].to_csv("spatial_fu2_zone_labels.csv")


# ── FU3: Foetal Gene Re-expression Programme ─────────────────────────────────
with safe_analysis("FU3: Foetal Gene Re-expression Programme"):
    # Foetal isoforms upregulated in cardiac failure
    FGP_UP = [
        "MYH7",   # slow β-MHC (foetal/failing)
        "TNNI1",  # slow skeletal TnI (foetal)
        "ACTC1",  # cardiac α-actin
        "ACTA1",  # skeletal α-actin (foetal)
        "NPPB",   # BNP – stress/foetal marker
        "NPPA",   # ANF – atrial/foetal natriuretic peptide
        "MYL4",   # atrial/foetal myosin light chain
        "MYL7",   # atrial/foetal myosin light chain 7
        "XIRP1",  # foetal cardiac protein
        "ANKRD1", # foetal cardiac ankyrin repeat
        "CITED4", # cardiac foetal factor
    ]
    # Adult isoforms downregulated in the foetal switch
    FGP_DOWN = [
        "MYH6",   # fast α-MHC (adult atrial)
        "TNNI3",  # cardiac TnI (adult)
        "ATP2A2", # SERCA2a – adult calcium handling
        "PLN",    # phospholamban – adult
        "S100A1", # adult cardioprotective
        "CASQ2",  # adult calcium storage
        "MYL2",   # adult ventricular regulatory MLC
    ]

    score_module(adata, FGP_UP,   "FGP_up")
    score_module(adata, FGP_DOWN, "FGP_down")
    adata.obs["FGP_index"] = (adata.obs["FGP_up"].values
                               - adata.obs["FGP_down"].values)

    mi_fgp = safe_morans_i(spatial_coords, adata.obs["FGP_index"].values)
    log(f"   FGP index Moran's I: {mi_fgp:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (col, cmap, label) in zip(axes, [
        ("FGP_up",    "Reds",    "Foetal Programme (UP)"),
        ("FGP_down",  "Blues",   "Adult Programme (DOWN in failure)"),
        ("FGP_index", "RdBu_r",  "FGP Index (Foetal – Adult)"),
    ]):
        mi_p = safe_morans_i(spatial_coords, adata.obs[col].values)
        scatter_map(ax, spatial_coords, adata.obs[col].values,
                    f"{label}\nMI={mi_p:.3f}", cmap=cmap, label="Score")
    plt.suptitle("Foetal Gene Re-expression Programme", y=1.02, fontsize=14)
    save_fig("spatial_fu3_fgp_map.png")

    # Stage trajectory with SEM bands
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, label, color in [
        ("FGP_up",    "Foetal Programme",  "#D62728"),
        ("FGP_down",  "Adult Programme",   "#1F77B4"),
        ("FGP_index", "FGP Index",         "#2CA02C"),
    ]:
        vals = adata.obs[col].values
        means = []; sems = []
        for s in unique_stages:
            g = vals[(stage == int(s)) & valid_mask]
            means.append(g.mean() if len(g) > 0 else np.nan)
            sems.append(g.std() / np.sqrt(len(g)) if len(g) > 1 else 0)
        ax.plot(range(len(unique_stages)), means, "o-",
                label=label, lw=2.5, color=color)
        ax.fill_between(range(len(unique_stages)),
                        np.array(means) - np.array(sems),
                        np.array(means) + np.array(sems),
                        alpha=0.15, color=color)
    ax.set_xticks(range(len(unique_stages)))
    ax.set_xticklabels([STAGE_MAP.get(int(s), str(s)) for s in unique_stages],
                       rotation=30)
    ax.set_xlabel("Disease Stage"); ax.set_ylabel("Score (mean ± SEM)")
    ax.set_title("Foetal vs Adult Isoform Programme Across Disease Stages")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    save_fig("spatial_fu3_fgp_trajectory.png")

    # Correlations with other spatial signals
    correlations = {}
    for other_col in ["Fibrosis_score", "Immune_total", "Radial_norm",
                      "HCM_score", "Stress_combined_PC1"]:
        if other_col in adata.obs.columns:
            r, p = pearsonr(adata.obs["FGP_index"].values,
                            adata.obs[other_col].values)
            correlations[other_col] = {"r": r, "p": p}
            log(f"   FGP_index vs {other_col}: r={r:.3f} (p={p:.3e})")

    if correlations:
        corr_summary = pd.DataFrame(correlations).T.reset_index()
        corr_summary.columns = ["Feature", "r", "p"]
        corr_summary["sig"] = corr_summary["p"] < 0.05
        corr_summary = corr_summary.sort_values("r")
        fig, ax = plt.subplots(figsize=(8, 4))
        colors  = ["#D62728" if r > 0 else "#1F77B4"
                   for r in corr_summary["r"]]
        bars    = ax.barh(corr_summary["Feature"], corr_summary["r"],
                          color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Pearson r with FGP Index")
        ax.set_title("Foetal Gene Programme — Correlation with Spatial Features")
        for bar, sig in zip(bars, corr_summary["sig"]):
            if sig:
                ax.text(bar.get_width() + 0.005 * np.sign(bar.get_width()),
                        bar.get_y() + bar.get_height() / 2,
                        "*", va="center", fontsize=12)
        ax.grid(True, alpha=0.3, axis="x"); plt.tight_layout()
        save_fig("spatial_fu3_fgp_correlations.png")

    # FGP by radial zone
    if "Radial_zone" in adata.obs.columns:
        fgp_by_zone = {z: adata.obs["FGP_index"][adata.obs["Radial_zone"] == z].mean()
                       for z in ["Core", "Mid", "Periphery"]}
        log(f"   FGP index by radial zone: {fgp_by_zone}")

    adata.obs[["FGP_up", "FGP_down", "FGP_index"]].to_csv(
        "spatial_fu3_fgp_scores.csv")


# =============================================================================
# PART 5 – SYNTHESIS FIGURE: Integrated Spatial Disease Model
# =============================================================================
with safe_analysis("SYNTHESIS: Integrated Spatial Disease Model"):
    # ── Guard: ensure all required obs columns are present ──────────────
    required_cols = ["FGP_index", "FC_state", "Interface_zone",
                     "Fibrosis_score", "HCM_score"]
    missing = [c for c in required_cols if c not in adata.obs.columns]
    if missing:
        raise RuntimeError(f"Missing obs columns from FU runs: {missing}")

    STATE_COLORS_SYN = {
        "Fibrotic_lost":      "#D62728",
        "Contractile_intact": "#2CA02C",
        "Transitional":       "#FF7F0E",
        "Depleted":           "#7F7F7F",
        "Mixed":              "#AEC7E8",
    }
    ZONE_COLORS_SYN = {
        "Fibrosis_core": "#D62728",
        "Immune_zone":   "#1F77B4",
        "Border_zone":   "#FF7F0E",
        "Other":         "#DDDDDD",
    }
    STAGE_PALETTE = ["#4393C3", "#F4A582", "#B2182B"]

    fgp   = adata.obs["FGP_index"].values
    fc_st = adata.obs["FC_state"].values
    iz    = adata.obs["Interface_zone"].values
    hcm   = hcm_score

    fig = plt.figure(figsize=(22, 18))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
    ax_A = fig.add_subplot(gs[0, 0])   # FGP spatial map
    ax_B = fig.add_subplot(gs[0, 1])   # FC state spatial map
    ax_C = fig.add_subplot(gs[0, 2])   # Border zone spatial map
    ax_D = fig.add_subplot(gs[0, 3])   # Stage composition bar
    ax_E = fig.add_subplot(gs[1, 0:2]) # FGP trajectory (wide)
    ax_F = fig.add_subplot(gs[1, 2:4]) # FGP vs HCM scatter (wide)
    ax_G = fig.add_subplot(gs[2, 0:2]) # Radial zone box plots (wide)
    ax_H = fig.add_subplot(gs[2, 2:4]) # Summary schematic

    # Panel A – FGP spatial
    vabs  = np.percentile(np.abs(fgp), 97)
    sc_A  = ax_A.scatter(spatial_coords[:, 1], spatial_coords[:, 0],
                         c=fgp, cmap="RdBu_r", s=15, alpha=0.85,
                         vmin=-vabs, vmax=vabs)
    plt.colorbar(sc_A, ax=ax_A, label="FGP Index", shrink=0.8)
    ax_A.invert_yaxis()
    ax_A.set_title(f"A  Foetal Gene Programme\n(Moran's I = {mi_fgp:.3f})",
                   fontweight="bold")
    ax_A.set_xlabel("col"); ax_A.set_ylabel("row")

    # Panel B – FC state
    for st, col in STATE_COLORS_SYN.items():
        m = fc_st == st
        ax_B.scatter(spatial_coords[m, 1], spatial_coords[m, 0],
                     c=col, s=15, alpha=0.8, label=st)
    ax_B.invert_yaxis()
    ax_B.set_title("B  Fibrosis–Contractile States", fontweight="bold")
    ax_B.set_xlabel("col"); ax_B.set_ylabel("row")
    ax_B.legend(fontsize=6, markerscale=1.2, loc="lower right", framealpha=0.7)

    # Panel C – Border zone
    for z, col in ZONE_COLORS_SYN.items():
        m = iz == z
        ax_C.scatter(spatial_coords[m, 1], spatial_coords[m, 0],
                     c=col, s=15, alpha=0.8 if z != "Other" else 0.3,
                     label=z)
    ax_C.invert_yaxis()
    ax_C.set_title("C  Fibrosis–Immune Border Zone", fontweight="bold")
    ax_C.set_xlabel("col"); ax_C.set_ylabel("row")
    ax_C.legend(fontsize=6, markerscale=1.2, loc="lower right", framealpha=0.7)

    # Panel D – Stacked bar: state composition per stage
    state_order = ["Contractile_intact", "Transitional", "Mixed",
                   "Depleted", "Fibrotic_lost"]
    bottom = np.zeros(len(unique_stages))
    for st in state_order:
        vals_bar = []
        for s in unique_stages:
            m = (stage == int(s)) & valid_mask
            pct = (fc_st[m] == st).sum() / m.sum() * 100 if m.sum() > 0 else 0
            vals_bar.append(pct)
        ax_D.bar(range(len(unique_stages)), vals_bar, bottom=bottom,
                 color=STATE_COLORS_SYN[st], label=st,
                 edgecolor="white", linewidth=0.5)
        bottom += np.array(vals_bar)
    ax_D.set_xticks(range(len(unique_stages)))
    ax_D.set_xticklabels([STAGE_MAP.get(int(s), str(s))
                          for s in unique_stages], rotation=30, ha="right")
    ax_D.set_ylabel("% of cells")
    ax_D.set_title("D  State Composition Per Stage", fontweight="bold")
    ax_D.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax_D.set_ylim(0, 100)

    # Panel E – FGP trajectory with SEM
    for col_name, label, lw, ls in [
        ("FGP_up",    "Foetal isoforms ↑", 2.5, "-"),
        ("FGP_down",  "Adult isoforms ↓",  2.5, "--"),
        ("FGP_index", "FGP Index (net)",   3.0, "-"),
    ]:
        if col_name not in adata.obs.columns:
            continue
        vals_e = adata.obs[col_name].values
        means_e, sems_e = [], []
        for s in unique_stages:
            g = vals_e[(stage == int(s)) & valid_mask]
            means_e.append(g.mean() if len(g) > 0 else np.nan)
            sems_e.append(g.std() / np.sqrt(len(g)) if len(g) > 1 else 0)
        color = ("#D62728" if "up"    in col_name else
                 "#1F77B4" if "down"  in col_name else "#2CA02C")
        ax_E.plot(range(len(unique_stages)), means_e, "o" + ls,
                  label=label, lw=lw, color=color)
        ax_E.fill_between(range(len(unique_stages)),
                          np.array(means_e) - np.array(sems_e),
                          np.array(means_e) + np.array(sems_e),
                          alpha=0.15, color=color)
    ax_E.set_xticks(range(len(unique_stages)))
    ax_E.set_xticklabels([STAGE_MAP.get(int(s), str(s)) for s in unique_stages],
                         rotation=30, ha="right")
    ax_E.set_xlabel("Disease Stage"); ax_E.set_ylabel("Score (mean ± SEM)")
    ax_E.set_title("E  Foetal vs Adult Isoform Programme — Disease Trajectory",
                   fontweight="bold")
    ax_E.legend(fontsize=9); ax_E.grid(True, alpha=0.3)

    # Panel F – FGP vs HCM scatter with regression line
    n_sub = min(1500, len(fgp))
    idx_sub = np.random.choice(len(fgp), n_sub, replace=False)
    for i, s in enumerate(unique_stages):
        m_sub = stage[idx_sub] == int(s)
        ax_F.scatter(fgp[idx_sub][m_sub], hcm[idx_sub][m_sub],
                     c=STAGE_PALETTE[i % len(STAGE_PALETTE)],
                     s=15, alpha=0.5,
                     label=STAGE_MAP.get(int(s), str(s)))
    slope, intercept, r_val, _, _ = linregress(fgp, hcm)
    x_line = np.linspace(fgp.min(), fgp.max(), 100)
    ax_F.plot(x_line, slope * x_line + intercept, "k-", lw=2,
              label=f"r = {r_val:.3f}")
    ax_F.set_xlabel("FGP Index"); ax_F.set_ylabel("HCM Score")
    ax_F.set_title(f"F  Foetal Programme vs HCM Score (r = {r_val:.3f})",
                   fontweight="bold")
    ax_F.legend(fontsize=8); ax_F.grid(True, alpha=0.3)

    # Panel G – Radial zone FGP box plots
    zone_order = ["Core", "Mid", "Periphery"]
    if "Radial_zone" in adata.obs.columns:
        fgp_by_z = [fgp[adata.obs["Radial_zone"] == z] for z in zone_order]
    else:
        fgp_by_z = [np.array([]) for _ in zone_order]
    bp = ax_G.boxplot(fgp_by_z, patch_artist=True, notch=False,
                      showfliers=False,
                      medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"],
                             ["#4393C3", "#92C5DE", "#F4A582"]):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    ax_G.set_xticks([1, 2, 3]); ax_G.set_xticklabels(zone_order)
    ax_G.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax_G.set_ylabel("FGP Index")
    ax_G.set_xlabel("Radial Zone (Centre → Periphery)")
    ax_G.set_title("G  Foetal Programme Gradient: Core → Periphery",
                   fontweight="bold")
    ax_G.grid(True, alpha=0.3, axis="y")
    for xi, (z, data) in enumerate(zip(zone_order, fgp_by_z), start=1):
        if len(data) > 0:
            ax_G.text(xi, data.mean(), f"  μ={data.mean():.3f}",
                      va="center", fontsize=8)

    # Panel H – Summary schematic (text-only, no external art)
    ax_H.set_xlim(0, 10); ax_H.set_ylim(0, 10); ax_H.axis("off")
    ax_H.set_title("H  Integrated Spatial Disease Model", fontweight="bold")
    ax_H.add_patch(plt.Circle((3, 5), 1.8, color="#4393C3", alpha=0.3))
    ax_H.text(3, 5, "CORE\nAdult\nFGP−\nContractile",
              ha="center", va="center", fontsize=8,
              fontweight="bold", color="#1A3F6F")
    ax_H.add_patch(plt.Circle((3, 5), 3.0, color="#92C5DE", alpha=0.15))
    ax_H.text(3, 8.3, "MID\nTransitional",
              ha="center", va="center", fontsize=7, color="#2166AC")
    ax_H.add_patch(plt.Circle((3, 5), 3.8, color="#F4A582", alpha=0.1))
    ax_H.text(6.5, 5, "PERIPHERY\nFoetal\nFGP+\nFibrotic",
              ha="center", va="center", fontsize=8,
              fontweight="bold", color="#8B1A1A")
    ax_H.annotate("", xy=(6.2, 5), xytext=(4.8, 5),
                  arrowprops={"arrowstyle": "->", "color": "#D62728", "lw": 2})
    ax_H.text(5.5, 5.3, "FGP↑\nFibrosis↑", ha="center",
              fontsize=7, color="#D62728")

    plt.suptitle(
        "Integrated Spatial Model of HCM Disease Progression\n"
        "Foetal re-expression · Fibrosis–contractile antagonism"
        " · Metabolic border zone",
        fontsize=14, fontweight="bold", y=1.01,
    )
    save_fig("spatial_SYNTHESIS_integrated_model.png")

    # Supplementary stats table
    supp_rows = []
    for other_col, label_s in [
        ("Fibrosis_score",      "Fibrosis Score"),
        ("Immune_total",        "Immune Infiltration"),
        ("Radial_norm",         "Radial Distance"),
        ("HCM_score",           "HCM Score"),
        ("Stress_combined_PC1", "Stress PC1"),
    ]:
        if other_col in adata.obs.columns:
            r, p = pearsonr(fgp, adata.obs[other_col].values)
            supp_rows.append({
                "Analysis": "FU3 FGP Index",
                "Variable_A": "FGP_index",
                "Variable_B": label_s,
                "Pearson_r":  round(r, 3),
                "p_value":    f"{p:.3e}",
                "n_cells":    len(fgp),
            })

    supp_df = pd.DataFrame(supp_rows)
    supp_df.to_csv("spatial_SYNTHESIS_supplementary_stats.csv", index=False)
    log(f"   Supplementary stats table: {len(supp_df)} rows saved")


# =============================================================================
# PART 6 – FINAL FILE CHECKLIST
# =============================================================================
log("\n" + SEP)
log("COMPREHENSIVE SPATIAL ANALYSIS COMPLETE")
log(SEP)

EXPECTED_FILES = [
    # Core analyses
    "spatial_01_qc_metrics_map.png",
    "spatial_01_qc_metrics_statistics.csv",
    "spatial_02_cluster_distribution.png",
    "spatial_02_cluster_topology.csv",
    "spatial_03_stage_progression_map.png",
    "spatial_03_stage_transitions.csv",
    "spatial_04_hcm_score_gradient.png",
    "spatial_04_hcm_score_zones.csv",
    "spatial_05_vcm_distribution.png",
    "spatial_05_vcm_distribution.csv",
    "spatial_06_master_regulator_expression.png",
    "spatial_06_master_regulator_expression.csv",
    "spatial_07_etc_complex_depletion.png",
    "spatial_07_etc_complex_depletion.csv",
    "spatial_08_upr_activation_zones.png",
    "spatial_08_upr_activation_zones.csv",
    "spatial_09_contractile_gene_patterns.png",
    "spatial_09_contractile_gene_patterns.csv",
    "spatial_10_csd_landscape.png",
    "spatial_10_csd_landscape.csv",
    "spatial_11_grn_hub_coexpression.png",
    "spatial_11_grn_hub_coexpression.csv",
    "spatial_12_atac_rna_correlation_zones.png",
    "spatial_12_atac_rna_correlation_zones.csv",
    "spatial_13_marker_specificity_scores.csv",
    "spatial_13_marker_specificity_spatial.png",
    "spatial_14_microenvironment_communities.csv",
    "spatial_14_neighborhood_analysis.png",
    "spatial_15_risk_prediction_model.png",
    "spatial_15_risk_prediction_model.csv",
    "spatial_15_risk_feature_importance.csv",
    # Bulk validation
    "spatial_bulk_01_wgcna_modules.png",
    "spatial_bulk_01_wgcna_trajectory.png",
    "spatial_bulk_01_wgcna_scores.csv",
    "spatial_bulk_02_regulon_spatial.png",
    "spatial_bulk_02_regulon_trajectory.png",
    "spatial_bulk_02_regulon_correlation.png",
    "spatial_bulk_02_regulon_scores.csv",
    "spatial_bulk_03_active_tfs_spatial.png",
    "spatial_bulk_03_repressed_tfs_spatial.png",
    "spatial_bulk_03_tf_hcm_correlations.csv",
    "spatial_bulk_03_tf_expression.csv",
    "spatial_bulk_04_hcm_panel_heatmap.png",
    "spatial_bulk_04_hcm_panel_spatial.png",
    "spatial_bulk_05_bulk_degs_spatial.png",
    "spatial_bulk_05_bulk_degs_trajectory.png",
    "spatial_bulk_05_bulk_signature_scores.csv",
    "spatial_bulk_06_bulk_vs_spatial_correlation.png",
    "spatial_bulk_06_bulk_spatial_integration.csv",
    "spatial_bulk_07_stage_signature_anova.csv",
    # Extensions
    "spatial_ext1_spatially_variable_genes.csv",
    "spatial_ext1_svgs_spatial.png",
    "spatial_ext1_moransI_distribution.png",
    "spatial_ext2_metabolic_spatial.png",
    "spatial_ext2_metabolic_trajectory.png",
    "spatial_ext2_metabolic_scores.csv",
    "spatial_ext3_fibrosis_spatial.png",
    "spatial_ext3_fibrosis_scores.csv",
    "spatial_ext4_lr_hotspots.png",
    "spatial_ext4_lr_cluster_heatmap.png",
    "spatial_ext4_lr_scores.csv",
    "spatial_ext5_stress_zones.png",
    "spatial_ext5_stress_trajectory.png",
    "spatial_ext5_stress_scores.csv",
    "spatial_ext6_radial_gradient.png",
    "spatial_ext6_zone_violins.png",
    "spatial_ext6_radial_gene_correlations.csv",
    "spatial_ext6_radial_metadata.csv",
    "spatial_ext7_immune_map.png",
    "spatial_ext7_immune_scores.csv",
    "spatial_ext8_sarcomere_organisation.png",
    "spatial_ext8_sarcomere_trajectory.png",
    "spatial_ext8_sarcomere_scores.csv",
    # Follow-up analyses
    "spatial_fu1_fc_antagonism_map.png",
    "spatial_fu1_fc_stage_composition.png",
    "spatial_fu1_fc_states.csv",
    "spatial_fu2_border_zone_map.png",
    "spatial_fu2_border_zone_heatmap.png",
    "spatial_fu2_border_zone_genes.csv",
    "spatial_fu2_zone_labels.csv",
    "spatial_fu3_fgp_map.png",
    "spatial_fu3_fgp_trajectory.png",
    "spatial_fu3_fgp_correlations.png",
    "spatial_fu3_fgp_scores.csv",
    # Synthesis
    "spatial_SYNTHESIS_integrated_model.png",
    "spatial_SYNTHESIS_supplementary_stats.csv",
]

ok = 0
for i, f in enumerate(EXPECTED_FILES, 1):
    status = "OK" if os.path.exists(f) else "MISSING"
    if status == "OK":
        ok += 1
    log(f"   {i:3d}. [{status}] {f}")

log(f"\n   {ok}/{len(EXPECTED_FILES)} files present")
log(SEP)

# Write the in-memory log to disk for reproducibility
with open("spatial_pipeline_log.txt", "w") as fh:
    fh.write("\n".join(LOG))
log("   Log saved to spatial_pipeline_log.txt")

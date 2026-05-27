"""
01_hcm_preprocessing_trajectory.py
====================================
Single-cell RNA-seq preprocessing, quality control, reference mapping,
unsupervised clustering, and PAGA trajectory analysis for HCM (Hypertrophic
Cardiomyopathy) left-ventricular cardiomyocyte data.

Dataset : 4 patients, 14 plates (GSM4103885 – GSM4103898)
Reference: healthy_vCM1_4_sparse_pca_umap.h5ad

Pipeline
--------
1.  Load & clean each plate TSV  (strip chromosome suffixes, add metadata)
2.  Concatenate all plates into a single AnnData object
3.  Remove junk genes (MT, ERCC, ribosomal)
4.  Normalize → log1p → sanity-check data range
5.  Align to healthy reference and project via sc.tl.ingest
6.  Compute HCM disease score (NPPA, NPPB, ANKRD1, ACTA1, MYH7)
7.  Leiden clustering → PAGA trajectory graph
8.  Export PAGA adjacency matrix

Author : <your-name>
Date   : 2026-02-11
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import gc
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy.plotting import _utils as sc_plot_utils


# ---------------------------------------------------------------------------
# 1. Data Loading Helpers
# ---------------------------------------------------------------------------

def load_and_clean_plate(tsv_path: str, patient_id: str, plate_id: str) -> ad.AnnData:
    """
    Read a gzipped TSV count matrix and return a clean AnnData object.

    The raw files are Genes × Cells (transposed relative to AnnData convention).
    Gene names carry a chromosome suffix (e.g. 'A2M__chr12') that is stripped
    so they match the reference atlas.

    Parameters
    ----------
    tsv_path   : Path to the .TranscriptCounts.tsv.gz file.
    patient_id : Patient label ('P1' … 'P4') stored in obs['patient_id'].
    plate_id   : Full plate filename stem stored in obs['plate_id'].

    Returns
    -------
    AnnData with shape (n_cells, n_genes).
    """
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)

    # Transpose: raw files are Genes × Cells
    adata = sc.AnnData(df.T)

    # Strip chromosome suffix so gene names match the reference atlas
    adata.var_names = adata.var_names.str.split("__").str[0]
    adata.var_names_make_unique()

    # Attach sample-level metadata for downstream grouping
    adata.obs["patient_id"] = patient_id
    adata.obs["plate_id"]   = plate_id

    return adata


def load_all_plates(patients: dict[str, list[str]]) -> ad.AnnData:
    """
    Iterate over every patient / plate combination, load each TSV, and
    concatenate into a single AnnData using an inner-join on shared genes.

    Parameters
    ----------
    patients : dict mapping patient ID → list of plate file stems.
               Example: {'P1': ['GSM4103885_Patient1Plate1', ...], ...}

    Returns
    -------
    Concatenated AnnData (cells × genes, inner join).
    """
    all_adatas = []

    for p_id, plate_list in patients.items():
        # Folder convention: 'patient1', 'patient2', …
        folder = f"patient{p_id[1]}"

        for plate_file in plate_list:
            path = os.path.join(folder, f"{plate_file}.TranscriptCounts.tsv.gz")
            all_adatas.append(load_and_clean_plate(path, p_id, plate_file))

    # Inner join: keeps only genes present in every plate
    adata_combined = ad.concat(all_adatas, label="batch", join="inner")
    adata_combined.obs_names_make_unique()

    return adata_combined


# ---------------------------------------------------------------------------
# 2. Quality-Control Gene Filtering
# ---------------------------------------------------------------------------

def remove_junk_genes(adata: ad.AnnData) -> ad.AnnData:
    """
    Hard-filter non-informative genes before any modelling step.

    Removed categories
    ------------------
    - Mitochondrial genes    : prefix 'MT-'
    - ERCC spike-ins         : prefix 'ERCC-'
    - Ribosomal small subunit: prefix 'RPS'
    - Ribosomal large subunit: prefix 'RPL'

    Returns a *copy* of the filtered object so the original is not mutated.
    """
    junk_mask = (
        adata.var_names.str.startswith("MT-")   |
        adata.var_names.str.startswith("ERCC-") |
        adata.var_names.str.startswith("RPS")   |
        adata.var_names.str.startswith("RPL")
    )

    adata_clean = adata[:, ~junk_mask].copy()
    print(
        f"[QC] Master object after junk-gene removal: "
        f"{adata_clean.n_obs} cells × {adata_clean.n_vars} genes."
    )
    return adata_clean


# ---------------------------------------------------------------------------
# 3. Normalization & Log-Transformation
# ---------------------------------------------------------------------------

def normalize_and_log(adata: ad.AnnData) -> ad.AnnData:
    """
    Standard scanpy normalization pipeline:

    1. Store raw counts in .raw for downstream DE testing.
    2. Library-size normalize every cell to 10,000 counts (CPM-style).
    3. Log1p-transform to stabilize variance.
    4. Sanity-check the data range (expected: max ~ 9–10, mean ~ 0.1–0.5).

    Returns the modified object in-place (also returned for convenience).
    """
    # Preserve raw integer counts — needed for rank_genes_groups later
    adata.raw = adata

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Range sanity check — catches double-normalization or un-normalized data
    max_val  = adata.X.max()
    mean_val = adata.X.mean()
    print(f"[Norm] Max = {max_val:.4f} | Mean = {mean_val:.4f}")

    if max_val > 10:
        raise ValueError("Data does NOT appear normalized — max value too high.")
    if max_val < 1:
        raise ValueError("Data may be double-log-transformed — max value too low.")

    print("[Norm] Data range looks correct for reference mapping.")
    return adata


# ---------------------------------------------------------------------------
# 4. Reference Mapping (Healthy Atlas Projection)
# ---------------------------------------------------------------------------

def align_and_ingest(
    adata_query : ad.AnnData,
    ref_path    : str = "healthy_vCM1_4_sparse_pca_umap.h5ad",
) -> ad.AnnData:
    """
    Project query cells onto a pre-built healthy cardiomyocyte reference atlas.

    Steps
    -----
    1. Load the reference AnnData (already normalized + PCA + UMAP).
    2. Restrict both objects to their shared gene set.
    3. Use sc.tl.ingest to transfer 'cell_states' labels and PCA/UMAP coords.

    Returns the annotated query object.
    """
    adata_ref = sc.read_h5ad(ref_path)

    # Intersection ensures both objects share exactly the same feature space
    common_genes = adata_ref.var_names.intersection(adata_query.var_names)
    print(f"[Ref] Aligning on {len(common_genes)} common genes …")

    adata_ref   = adata_ref[:, common_genes].copy()
    adata_query = adata_query[:, common_genes].copy()

    # ingest: learns the reference PCA space and projects query cells,
    # then assigns the nearest reference 'cell_states' label via kNN
    sc.tl.ingest(adata_query, adata_ref, obs="cell_states")
    print("[Ref] Reference mapping complete — 'cell_states' column added.")

    # Free the large reference from memory
    del adata_ref
    gc.collect()

    return adata_query


# ---------------------------------------------------------------------------
# 5. HCM Disease Scoring
# ---------------------------------------------------------------------------

def score_hcm_genes(adata: ad.AnnData) -> ad.AnnData:
    """
    Calculate a per-cell HCM disease severity score using five canonical
    hypertrophic-cardiomyopathy marker genes.

    Genes
    -----
    NPPA, NPPB  – natriuretic peptides (wall-stress markers)
    ANKRD1      – mechanosensory response
    ACTA1       – foetal sarcomeric actin (re-expression in HCM)
    MYH7        – β-myosin heavy chain (HCM causal gene)

    The score is stored in adata.obs['HCM_score'].
    Genes missing from the dataset are silently skipped (prevents crashes if
    they were removed during QC).
    """
    hcm_markers    = ["NPPA", "NPPB", "ANKRD1", "ACTA1", "MYH7"]
    present_markers = [g for g in hcm_markers if g in adata.var_names]

    print(f"[Score] Using {len(present_markers)}/{len(hcm_markers)} HCM markers: "
          f"{present_markers}")

    sc.tl.score_genes(adata, gene_list=present_markers, score_name="HCM_score")

    # Report per-patient average to give a quick severity overview
    per_patient = (
        adata.obs
        .groupby("patient_id", observed=True)["HCM_score"]
        .mean()
        .sort_values()
    )
    print(f"[Score] Mean HCM score per patient:\n{per_patient.round(4)}")

    return adata


# ---------------------------------------------------------------------------
# 6. Clustering & PAGA Trajectory
# ---------------------------------------------------------------------------

def cluster_and_paga(
    adata       : ad.AnnData,
    n_neighbors : int  = 15,
    resolution  : float = 0.5,
    paga_threshold : float = 0.15,
) -> ad.AnnData:
    """
    Build a k-nearest-neighbour graph, run Leiden clustering, and compute
    the PAGA (Partition-based Graph Abstraction) trajectory.

    The root cell for pseudotime is set to the vCM1 cell with the lowest
    HCM_score — i.e. the 'healthiest' observed cell.

    Parameters
    ----------
    n_neighbors     : k for kNN graph construction.
    resolution      : Leiden resolution — higher → more clusters.
    paga_threshold  : Minimum PAGA edge weight to display.

    Returns the object with 'leiden' labels and PAGA connectivity in .uns.
    """
    # kNN graph uses the ingest-derived PCA embedding
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution)

    # ------------------------------------------------------------------
    # Set pseudotime root: healthiest vCM1 cell (lowest HCM_score)
    # ------------------------------------------------------------------
    basal_cells  = adata[adata.obs["cell_states"] == "vCM1"]
    root_cell_id = basal_cells.obs["HCM_score"].idxmin()
    adata.uns["iroot"] = int(np.where(adata.obs_names == root_cell_id)[0][0])

    root_score = adata.obs.loc[root_cell_id, "HCM_score"]
    print(f"[Traj] Root cell: {root_cell_id}  |  HCM score: {root_score:.4f}")

    # ------------------------------------------------------------------
    # PAGA: requires a valid colour palette for Leiden categories
    # ------------------------------------------------------------------
    sc_plot_utils._set_default_colors_for_categorical_obs(adata, "leiden")
    sc.tl.paga(adata, groups="leiden")

    # Rank clusters by HCM score to confirm healthy → diseased ordering
    cluster_scores = (
        adata.obs
        .groupby("leiden", observed=True)["HCM_score"]
        .mean()
        .sort_values()
    )
    print(f"[PAGA] Mean HCM score per Leiden cluster (healthy → diseased):\n"
          f"{cluster_scores.round(4)}")

    return adata


# ---------------------------------------------------------------------------
# 7. Export PAGA Adjacency Matrix
# ---------------------------------------------------------------------------

def export_paga_matrix(adata: ad.AnnData, out_path: str = "paga_connectivity_matrix.tsv") -> pd.DataFrame:
    """
    Convert the sparse PAGA connectivity matrix to a labelled dense DataFrame
    and write it to a tab-separated file.

    The values represent the statistical confidence of a direct connection
    between each pair of Leiden clusters (0 = no evidence, 1 = invariant).

    Returns the DataFrame for downstream inspection.
    """
    categories = adata.obs["leiden"].cat.categories

    paga_df = pd.DataFrame(
        adata.uns["paga"]["connectivities"].todense(),
        index=categories,
        columns=categories,
    )

    paga_df.to_csv(out_path, sep="\t")
    print(f"[Export] PAGA adjacency matrix saved to '{out_path}'.")
    return paga_df


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Patient / plate manifest
    # ------------------------------------------------------------------
    patients = {
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

    # ------------------------------------------------------------------
    # Step 1 – Load & concatenate
    # ------------------------------------------------------------------
    adata = load_all_plates(patients)

    # ------------------------------------------------------------------
    # Step 2 – Remove junk genes
    # ------------------------------------------------------------------
    adata = remove_junk_genes(adata)

    # ------------------------------------------------------------------
    # Step 3 – Normalize & log-transform
    # ------------------------------------------------------------------
    adata = normalize_and_log(adata)

    # ------------------------------------------------------------------
    # Step 4 – Project onto healthy reference atlas
    # ------------------------------------------------------------------
    adata = align_and_ingest(adata)

    # ------------------------------------------------------------------
    # Step 5 – Score HCM disease severity
    # ------------------------------------------------------------------
    adata = score_hcm_genes(adata)

    # ------------------------------------------------------------------
    # Step 6 – Cluster (Leiden) + PAGA trajectory
    # ------------------------------------------------------------------
    adata = cluster_and_paga(adata)

    # ------------------------------------------------------------------
    # Step 7 – Export PAGA adjacency matrix
    # ------------------------------------------------------------------
    export_paga_matrix(adata)

    # ------------------------------------------------------------------
    # Save the fully annotated object for use in script 02
    # ------------------------------------------------------------------
    adata.write_h5ad("hcm_annotated.h5ad")
    print("[Done] Annotated AnnData saved to 'hcm_annotated.h5ad'.")

    return adata


if __name__ == "__main__":
    adata_hcm = main()

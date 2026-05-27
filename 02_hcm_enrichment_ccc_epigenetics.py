"""
02_hcm_enrichment_ccc_epigenetics.py
======================================
Downstream analysis of the HCM trajectory identified in script 01:

  A. Differential expression along the optimal path (clusters 11→5→16→10→8→13)
  B. Gene-set enrichment (Enrichr / gseapy) for GO-BP, GO-CC, and KEGG
  C. Cell–cell communication analysis (LIANA consensus pipeline)
  D. Epigenetic regulator (epifactor) profiling along the trajectory

Requires
--------
- 'hcm_annotated.h5ad' produced by script 01
- gseapy, liana, seaborn, matplotlib

Author : <your-name>
Date   : 2026-02-11
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os

import gseapy as gp
import liana as li
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Optimal disease progression path (low → high HCM score)
PATH_CLUSTERS = ["11", "5", "16", "10", "8", "13"]

# Stage labels for human-readable plots
STAGE_LABELS = {
    "11": "I  – Basal",
    "5":  "II – Alert",
    "16": "III – Pause",
    "10": "IV – Commit",
    "8":  "V  – Hypertrophy",
    "13": "VI – Terminal",
}

# Enrichr databases to query
GENE_SET_DATABASES = [
    "GO_Biological_Process_2021",
    "GO_Molecular_Function_2021",
    "GO_Cellular_Component_2021",
    "KEGG_2021_Human",
]

# Core epigenetic regulators (chromatin remodellers, writers, erasers, readers)
EPIFACTORS = [
    "SMARCA4","SMARCA2","SMARCB1","ARID1A","ARID1B","ARID2",
    "CHD1","CHD2","CHD3","CHD4","CHD7","CHD8",
    "EP300","CREBBP","KAT2A","KAT2B","KAT6A","KAT6B",
    "HDAC1","HDAC2","HDAC3","HDAC4","HDAC5","HDAC7","HDAC9",
    "SIRT1","SIRT3","SIRT6",
    "EZH2","EZH1","EED","SUZ12",
    "KMT2A","KMT2B","KMT2C","KMT2D","SETD1A","SETD1B",
    "KDM1A","KDM5A","KDM5B","KDM5C","KDM6A","KDM6B",
    "DNMT1","DNMT3A","DNMT3B","TET1","TET2","TET3",
    "BRD2","BRD3","BRD4","BRD7","BRD9",
    "MED1","MED12","MED13","MED24",
    "NCOA1","NCOA2","NCOA3","NCOR1","NCOR2",
]


# ---------------------------------------------------------------------------
# A. Differential Expression Along the Path
# ---------------------------------------------------------------------------

def run_differential_expression(adata: sc.AnnData) -> dict[str, pd.DataFrame]:
    """
    Run Wilcoxon rank-sum test (cluster vs. rest) on the full dataset, then
    export per-cluster DE tables for each cluster in the optimal path.

    Files written
    -------------
    cluster_<ID>_up_vs_rest.csv   – genes upregulated in cluster vs. rest
    cluster_<ID>_down_vs_rest.csv – genes downregulated

    Returns
    -------
    dict mapping cluster ID → full DE DataFrame (all genes, both directions).
    """
    print("[DE] Running Wilcoxon rank-sum test (global, cluster vs. rest) …")
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

    de_results = {}

    for cl in PATH_CLUSTERS:
        df_all = sc.get.rank_genes_groups_df(adata, group=cl)

        # Split by direction and save
        df_all[df_all["logfoldchanges"] >  0].to_csv(f"cluster_{cl}_up_vs_rest.csv",   index=False)
        df_all[df_all["logfoldchanges"] <  0].to_csv(f"cluster_{cl}_down_vs_rest.csv", index=False)

        de_results[cl] = df_all
        print(f"  Cluster {cl}: {len(df_all)} genes tested.")

    return de_results


def print_terminal_markers(de_results: dict[str, pd.DataFrame], n: int = 5) -> None:
    """
    Print the top-n upregulated markers for cluster 13 (terminal failure state).
    These are the genes most specific to end-stage HCM.
    """
    df13 = de_results.get("13", pd.DataFrame())
    if df13.empty:
        print("[DE] Cluster 13 DE results not found.")
        return

    top = df13.nlargest(n, "logfoldchanges")[["names", "logfoldchanges", "pvals_adj"]]
    print(f"\n[DE] Top {n} markers for Terminal Cluster 13:\n{top.to_string(index=False)}")


def plot_path_dotplot(adata: sc.AnnData, n_markers: int = 3) -> None:
    """
    Visualize per-stage marker gene expression as a scanpy DotPlot.

    For each cluster in the path, the top-n markers (by Wilcoxon score) are
    selected. Dot size encodes fraction of expressing cells; colour encodes
    mean expression (0–1 scaled per gene).
    """
    # Collect unique top markers across all path stages
    top_markers = []
    for cl in PATH_CLUSTERS:
        genes = (
            sc.get.rank_genes_groups_df(adata, group=cl)
            .head(n_markers)["names"]
            .tolist()
        )
        top_markers.extend(genes)

    # Remove duplicates while preserving order
    unique_markers = list(dict.fromkeys(top_markers))

    # Subset to path clusters only to avoid category mismatch
    adata_path = adata[adata.obs["leiden"].isin(PATH_CLUSTERS)].copy()
    adata_path.obs["leiden"] = adata_path.obs["leiden"].cat.remove_unused_categories()

    sc.pl.dotplot(
        adata_path,
        var_names=unique_markers,
        groupby="leiden",
        categories_order=PATH_CLUSTERS,
        standard_scale="var",
        title="Molecular Ramp-up: Basal (11) → Terminal (13)",
        show=False,
    )
    plt.savefig("dotplot_path_markers.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[DE] Dotplot saved to 'dotplot_path_markers.png'.")


# ---------------------------------------------------------------------------
# B. Gene-Set Enrichment (Enrichr)
# ---------------------------------------------------------------------------

def _safe_enrichr(gene_list: list[str], gene_sets: list[str], out_dir: str) -> gp.enrichr:
    """
    Wrapper around gseapy.enrichr that silently skips databases that return
    no significant terms (cutoff = 0.05), avoiding hard crashes.
    """
    try:
        return gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism="human",
            outdir=out_dir,
            cutoff=0.05,
        )
    except Exception as exc:
        print(f"  [Enrichr] Warning for {out_dir}: {exc}")
        return None


def run_pathway_enrichment(de_results: dict[str, pd.DataFrame], top_n_genes: int = 50) -> None:
    """
    For each path cluster, extract the top-n upregulated genes and query
    Enrichr with GO-BP, GO-MF, GO-CC, and KEGG databases.

    Results are written under enrichr_results/cluster_<ID>/.
    A per-cluster top-50 gene list is also saved as a plain text file.
    """
    os.makedirs("enrichr_results", exist_ok=True)

    for cl in PATH_CLUSTERS:
        df = de_results.get(cl, pd.DataFrame())
        if df.empty:
            print(f"  [Enrichr] No DE data for cluster {cl}, skipping.")
            continue

        # Top-n upregulated genes by Wilcoxon score
        de_genes = df.head(top_n_genes)["names"].tolist()

        # Save gene list for reproducibility
        gene_file = f"cluster_{cl}_top{top_n_genes}.txt"
        with open(gene_file, "w") as fh:
            fh.write("\n".join(de_genes))

        out_dir = f"enrichr_results/cluster_{cl}"
        _safe_enrichr(de_genes, GENE_SET_DATABASES, out_dir)
        print(f"  [Enrichr] Cluster {cl} complete → {out_dir}/")


def _load_enrichr_result(cluster_id: str, database: str) -> pd.DataFrame | None:
    """
    Load a single Enrichr result file. Returns None if the file does not exist
    (e.g. no significant terms were found for that database / cluster).
    """
    path = (
        f"enrichr_results/cluster_{cluster_id}/"
        f"{database}.human.enrichr.reports.txt"
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t")


def plot_enrichment_grid(
    database : str   = "GO_Biological_Process_2021",
    top_n    : int   = 10,
    filename : str   = "enrichment_grid.png",
) -> None:
    """
    Create a 2×3 lollipop-plot grid showing the top enriched terms for each
    of the six path stages.

    Visual encoding
    ---------------
    - Horizontal position (x) : −log10(Adjusted P-value)  [significance]
    - Dot colour               : same as x  [viridis gradient]
    - Dot size                 : number of overlapping genes  [gene_count]
    """
    fig, axes = plt.subplots(2, 3, figsize=(26, 14))
    axes = axes.flatten()

    for i, cl in enumerate(PATH_CLUSTERS):
        ax  = axes[i]
        res = _load_enrichr_result(cl, database)

        if res is None:
            ax.text(0.5, 0.5, f"Cluster {cl}: No significant terms",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        top = res.head(top_n).copy()
        top["-log10_p"]   = -np.log10(top["Adjusted P-value"])
        top["gene_count"] = top["Overlap"].str.split("/").str[0].astype(int)

        # Lollipop sticks
        ax.hlines(y=top["Term"], xmin=0, xmax=top["-log10_p"],
                  color="lightgrey", linewidth=2, zorder=1)

        # Lollipop heads
        scatter = ax.scatter(
            top["-log10_p"], top["Term"],
            s=top["gene_count"] * 50,
            c=top["-log10_p"],
            cmap="viridis",
            edgecolors="black",
            zorder=2,
        )

        stage = STAGE_LABELS.get(cl, f"Cluster {cl}")
        ax.set_title(f"Stage {i+1}: {stage}", fontsize=14, fontweight="bold")
        ax.set_xlabel("−log10(Adj. P-value)", fontsize=11)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Add colourbar to the last panel of each row
        if (i + 1) % 3 == 0:
            fig.colorbar(scatter, ax=ax, label="−log10(Adj. P-value)")

    plt.suptitle(f"Functional Progression: {database}", fontsize=20, y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Enrichment] Grid saved to '{filename}'.")


def export_functional_summary() -> None:
    """
    Build a consolidated TSV for each path cluster that merges GO-CC, GO-BP,
    and KEGG results side-by-side (3 × 3 columns = Term, Score, FDR).

    Output files : Cluster_<ID>_Functional_Summary.tsv
    """
    databases = {
        "CC":      "GO_Cellular_Component_2021",
        "BP":      "GO_Biological_Process_2021",
        "Pathway": "KEGG_2021_Human",
    }

    for cl in PATH_CLUSTERS:
        frames = []

        for prefix, db in databases.items():
            res = _load_enrichr_result(cl, db)
            if res is None:
                # Placeholder so the concat still produces aligned columns
                placeholder = pd.DataFrame(columns=[
                    f"{prefix}_Term", f"{prefix}_Score", f"{prefix}_FDR"
                ])
                frames.append(placeholder)
                continue

            tmp = res.head(20)[["Term", "Combined Score", "Adjusted P-value"]].copy()
            tmp.columns = [f"{prefix}_Term", f"{prefix}_Score", f"{prefix}_FDR"]
            frames.append(tmp.reset_index(drop=True))

        merged = pd.concat(frames, axis=1)
        out    = f"Cluster_{cl}_Functional_Summary.tsv"
        merged.to_csv(out, sep="\t", index=False)
        print(f"  [Summary] {out} generated.")


# ---------------------------------------------------------------------------
# C. Cell–Cell Communication (LIANA)
# ---------------------------------------------------------------------------

def run_liana(adata: sc.AnnData) -> pd.DataFrame:
    """
    Run the LIANA consensus ligand–receptor pipeline on the full dataset.

    Method
    ------
    li.mt.rank_aggregate combines CellPhoneDB, Connectome, log2FC, NATMI,
    and SingleCellSignalR into a single consensus magnitude_rank score.

    Key caveats
    -----------
    - Only genes present in the Liana 'consensus' resource AND in the dataset
      contribute (~23 % of the resource in this dataset).
    - use_raw=True expands the feature set from .X (7686) to .raw (8400).
    - Cluster 5 has no target interactions because it contains too few
      qualifying ligand/receptor genes — this is a data limitation, not a bug.

    Returns
    -------
    liana_results DataFrame stored in adata.uns['liana_res'].
    """
    # Ensure leiden column is categorical (required by LIANA internals)
    adata.obs["leiden"] = adata.obs["leiden"].astype("category")

    li.mt.rank_aggregate(
        adata,
        groupby="leiden",
        resource_name="consensus",
        use_raw=True if adata.raw is not None else False,
        verbose=True,
    )

    liana_res = adata.uns["liana_res"].copy()
    print(f"[LIANA] {len(liana_res)} ligand–receptor pairs computed.")
    return liana_res


def extract_transition_signals(
    liana_res : pd.DataFrame,
    source_id : str,
    target_id : str,
    out_csv   : str | None = None,
    n_top     : int = 10,
) -> pd.DataFrame:
    """
    Filter LIANA results for a specific source → target cluster transition
    and return the top interactions sorted by magnitude_rank (lower = better).

    Parameters
    ----------
    liana_res  : Full LIANA results DataFrame.
    source_id  : Leiden cluster label for the sending cluster.
    target_id  : Leiden cluster label for the receiving cluster.
    out_csv    : If given, write the full filtered table to this path.
    n_top      : Number of top interactions to print.

    Returns
    -------
    Filtered DataFrame.
    """
    mask = (
        (liana_res["source"] == source_id) &
        (liana_res["target"] == target_id)
    )
    signals = liana_res[mask].sort_values("magnitude_rank")

    if signals.empty:
        print(f"  [LIANA] No interactions found for {source_id} → {target_id}. "
              f"This may reflect limited receptor/ligand coverage in the dataset.")
        return signals

    print(f"\n[LIANA] Top {n_top} interactions: Cluster {source_id} → {target_id}")
    print(
        signals[["ligand_complex", "receptor_complex", "magnitude_rank", "lrscore"]]
        .head(n_top)
        .to_string(index=False)
    )

    if out_csv:
        signals.to_csv(out_csv, index=False)
        print(f"  Saved → {out_csv}")

    return signals


def plot_liana_dotplot(adata: sc.AnnData, liana_res: pd.DataFrame) -> None:
    """
    Generate a LIANA dotplot for the full HCM trajectory.

    Notes
    -----
    - Cluster 5 is excluded from target_labels because it has no qualifying
      incoming interactions (data limitation documented in the analysis).
    - Returns a plotnine ggplot object that is saved directly via .save().
    """
    # Identify which path clusters actually appear as targets in the data
    active_sources = [c for c in PATH_CLUSTERS if c in liana_res["source"].values]
    active_targets = [c for c in PATH_CLUSTERS if c in liana_res["target"].values]

    if not active_sources or not active_targets:
        print("[LIANA] No interactions found for any path cluster — skipping dotplot.")
        return

    p = li.pl.dotplot(
        adata=adata,
        liana_res=liana_res,
        source_labels=active_sources,
        target_labels=active_targets,
        colour="magnitude_rank",
        size="specificity_rank",
        top_n=20,
        orderby="magnitude_rank",
        orderby_ascending=True,
        inverse_colour=True,   # high rank → low colour = misleading → invert
        figure_size=(14, 10),
        return_fig=True,
    )

    p.save("Global_HCM_Trajectory_Dotplot.png", dpi=300)
    print("[LIANA] Dotplot saved to 'Global_HCM_Trajectory_Dotplot.png'.")


# ---------------------------------------------------------------------------
# D. Epigenetic Regulator Profiling
# ---------------------------------------------------------------------------

def profile_epifactors(
    adata   : sc.AnnData,
    de_results : dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Identify which core epigenetic regulators (EPIFACTORS list) are
    differentially upregulated at each trajectory stage.

    Steps
    -----
    1. Compare Terminal (clusters 10, 8, 13) vs Early (clusters 11, 6, 16)
       using Wilcoxon to find globally relevant epifactors.
    2. For each path cluster, extract cluster-specific up-regulated epifactors
       and save to CSV.
    3. Run Enrichr on each cluster's epifactor list (GO-BP + GO-CC).
    4. Generate a matrixplot heatmap of all available epifactors across stages.

    Returns
    -------
    DataFrame of terminal-vs-early epifactor DE results.
    """
    # ------------------------------------------------------------------
    # Global: Terminal vs Early
    # ------------------------------------------------------------------
    adata.obs["progression_group"] = "other"
    adata.obs.loc[adata.obs["leiden"].isin(["11", "6", "16"]), "progression_group"] = "Early"
    adata.obs.loc[adata.obs["leiden"].isin(["10", "8",  "13"]), "progression_group"] = "Terminal"

    sc.tl.rank_genes_groups(
        adata,
        groupby="progression_group",
        groups=["Terminal"],
        reference="Early",
        method="wilcoxon",
    )

    df_terminal = sc.get.rank_genes_groups_df(adata, group="Terminal")
    epi_hits    = (
        df_terminal[df_terminal["names"].isin(EPIFACTORS)]
        .sort_values("logfoldchanges", ascending=False)
    )
    epi_hits.to_csv("epifactors_terminal_vs_early.csv", index=False)
    print(f"[Epi] {len(epi_hits)} epifactors differentially expressed (Terminal vs Early).")

    # ------------------------------------------------------------------
    # Per-cluster: save up-regulated epifactors and run Enrichr
    # ------------------------------------------------------------------
    for cl in PATH_CLUSTERS:
        df = de_results.get(cl, pd.DataFrame())
        if df.empty:
            continue

        epi_up = df[(df["names"].isin(EPIFACTORS)) & (df["logfoldchanges"] > 0)]
        epi_up.to_csv(f"cluster_{cl}_epifactors_up.csv", index=False)
        print(f"  Cluster {cl}: {len(epi_up)} upregulated epifactors.")

        if not epi_up.empty:
            _safe_enrichr(
                gene_list=epi_up["names"].tolist(),
                gene_sets=["GO_Biological_Process_2021", "GO_Cellular_Component_2021"],
                out_dir=f"enrichr_results/epifactors/cluster_{cl}",
            )

    return epi_hits


def plot_epifactor_barplot(epi_hits: pd.DataFrame, n: int = 30) -> None:
    """
    Horizontal barplot of the top-n epifactors ranked by logFC
    (Terminal vs Early). Colour gradient follows the logFC value.
    """
    top = epi_hits.head(n).copy()

    plt.figure(figsize=(8, 12))
    sns.barplot(
        data=top,
        x="logfoldchanges",
        y="names",
        hue="logfoldchanges",   # avoids the deprecated palette-without-hue warning
        palette="magma",
        legend=False,
    )
    plt.title("Top Epigenetic Drivers: Terminal vs Early Stages", fontsize=13)
    plt.xlabel("Log Fold Change")
    plt.tight_layout()
    plt.savefig("terminal_epifactor_logFC.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[Epi] Barplot saved to 'terminal_epifactor_logFC.png'.")


def plot_epifactor_heatmap(adata: sc.AnnData) -> None:
    """
    Matrixplot (mean expression per cluster, 0–1 scaled per gene) showing
    the progressive activation of epigenetic regulators from stage I to VI.

    Only epifactors actually present in the dataset are plotted.
    """
    # Restrict to path clusters and clean up unused categories
    adata_sub = adata[adata.obs["leiden"].isin(PATH_CLUSTERS)].copy()
    adata_sub.obs["leiden"] = adata_sub.obs["leiden"].cat.remove_unused_categories()

    available = [g for g in EPIFACTORS if g in adata_sub.var_names]
    print(f"[Epi] Plotting {len(available)} epifactors present in the dataset.")

    sc.pl.matrixplot(
        adata_sub,
        var_names=available,
        groupby="leiden",
        categories_order=PATH_CLUSTERS,
        standard_scale="var",
        cmap="magma",
        title="Epigenetic Storm: Progressive Activation (Stage I → VI)",
        show=False,
    )
    plt.savefig("Epifactor_RampUp_Heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[Epi] Heatmap saved to 'Epifactor_RampUp_Heatmap.png'.")


def build_epigenetic_trajectory_summary(de_results: dict[str, pd.DataFrame]) -> None:
    """
    Generate a single master TSV summarising the key epigenetic events at
    each trajectory stage:

    Columns: CLUSTER | UPREGULATED_EPIFACTORS | TOP_BIOLOGICAL_PROCESSES

    This table is suitable for inclusion in a manuscript supplementary file.
    """
    rows = []

    for cl in PATH_CLUSTERS:
        epi_file = f"cluster_{cl}_epifactors_up.csv"

        if not os.path.exists(epi_file):
            rows.append({
                "CLUSTER": cl,
                "UPREGULATED_EPIFACTORS": "Not available",
                "TOP_BIOLOGICAL_PROCESSES": "Not available",
            })
            continue

        epi_df = pd.read_csv(epi_file)

        if epi_df.empty:
            rows.append({
                "CLUSTER": cl,
                "UPREGULATED_EPIFACTORS": "None detected",
                "TOP_BIOLOGICAL_PROCESSES": "No enrichment",
            })
            continue

        genes_str = "; ".join(epi_df["names"].tolist())

        # Attempt to load pre-computed Enrichr BP results for this cluster
        bp_res = _load_enrichr_result(cl, "GO_Biological_Process_2021")
        bp_str = "; ".join(bp_res.head(5)["Term"].tolist()) if bp_res is not None else "Run enrichment first"

        rows.append({
            "CLUSTER": cl,
            "UPREGULATED_EPIFACTORS": genes_str,
            "TOP_BIOLOGICAL_PROCESSES": bp_str,
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("Epigenetic_Trajectory_Summary.tsv", sep="\t", index=False)
    print("[Epi] Master summary saved to 'Epigenetic_Trajectory_Summary.tsv'.")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # Load the annotated object from script 01
    # ------------------------------------------------------------------
    print("[Load] Reading hcm_annotated.h5ad …")
    adata = sc.read_h5ad("hcm_annotated.h5ad")
    print(f"       {adata.n_obs} cells × {adata.n_vars} genes")

    # ------------------------------------------------------------------
    # A – Differential expression
    # ------------------------------------------------------------------
    de_results = run_differential_expression(adata)
    print_terminal_markers(de_results)
    plot_path_dotplot(adata)

    # ------------------------------------------------------------------
    # B – Pathway enrichment
    # ------------------------------------------------------------------
    run_pathway_enrichment(de_results)

    # One enrichment grid per major database
    for db in GENE_SET_DATABASES[:2]:           # BP and MF for brevity
        safe_name = db.replace(" ", "_").replace("/", "-")
        plot_enrichment_grid(
            database=db,
            filename=f"enrichment_grid_{safe_name}.png",
        )

    export_functional_summary()

    # ------------------------------------------------------------------
    # C – Cell–cell communication
    # ------------------------------------------------------------------
    liana_res = run_liana(adata)

    # Terminal transition: cluster 8 (hypertrophic) → cluster 13 (failure)
    extract_transition_signals(
        liana_res,
        source_id="8",
        target_id="13",
        out_csv="Terminal_Failure_Interactions.csv",
    )

    plot_liana_dotplot(adata, liana_res)

    # ------------------------------------------------------------------
    # D – Epigenetic regulators
    # ------------------------------------------------------------------
    epi_hits = profile_epifactors(adata, de_results)
    plot_epifactor_barplot(epi_hits)
    plot_epifactor_heatmap(adata)
    build_epigenetic_trajectory_summary(de_results)

    print("\n[Done] All downstream analyses complete.")


if __name__ == "__main__":
    main()

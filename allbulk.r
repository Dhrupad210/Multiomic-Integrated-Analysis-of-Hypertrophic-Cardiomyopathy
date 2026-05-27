## ============================================================
## HCM BULK RNA-seq PIPELINE v4 
## GSE141910 (n=366: 183 HCM / 183 Control)
##
## OUTPUTS:
##   CSVs  — DE results, WGCNA modules, TF activity, regulons
##   PNGs  — PCA, volcano, heatmap, GSEA, WGCNA, TF lollipop
##   TXTs  — Gene lists per regulon (for scRNA import)
##
## REGULON MODULES (4-5 extracted at end):
##   REG1: UP in HCM, chromatin OPEN     (active up-regulon)
##   REG2: UP in HCM, chromatin CLOSED   (paradox / post-transcriptional)
##   REG3: DOWN in HCM, chromatin OPEN   (silenced by TF loss)
##   REG4: DOWN in HCM, chromatin CLOSED (repressed down-regulon)
##   REG5: WGCNA ME3 ECM/stromal module  (co-expression hub)
##
## USAGE with scRNA (after running this pipeline):
##   Use the .txt gene lists with AddModuleScore() in Seurat
##   or score_genes() in scanpy for your 4-patient vCM data.
## ============================================================

## ── 0. LIBRARIES ────────────────────────────────────────────
suppressPackageStartupMessages({
  library(data.table)
  library(limma)
  library(ggplot2)
  library(ggrepel)
  library(dplyr)
  library(tidyr)
  library(fgsea)
  library(msigdbr)
  library(AnnotationDbi)
  library(org.Hs.eg.db)
  library(clusterProfiler)
  library(ComplexHeatmap)
  library(circlize)
  library(decoupleR)
  library(dorothea)
  library(WGCNA)
})

options(stringsAsFactors = FALSE)

## ── Output directories ───────────────────────────────────────
dir.create("outputs/plots",    recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables",   recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/regulons", recursive = TRUE, showWarnings = FALSE)

banner <- function(n, title) {
  msg <- paste0("\n", paste(rep("=", 65), collapse = ""),
                "\n  STEP ", n, ": ", toupper(title),
                "\n", paste(rep("=", 65), collapse = ""))
  message(msg)
}

save_png <- function(plot_obj, filename, width = 10, height = 8, res = 200) {
  path <- file.path("outputs/plots", filename)
  png(path, width = width, height = height, units = "in", res = res)
  if (inherits(plot_obj, "gg")) {
    print(plot_obj)
  } else {
    plot_obj()   # function-based plots (base R)
  }
  dev.off()
  message("  Saved: ", path)
}

save_csv <- function(df, filename) {
  path <- file.path("outputs/tables", filename)
  write.csv(df, path, row.names = FALSE)
  message("  Saved: ", path)
}

save_txt <- function(gene_vec, filename) {
  path <- file.path("outputs/regulons", filename)
  writeLines(as.character(gene_vec[!is.na(gene_vec)]), path)
  message("  Saved: ", path, "  (", length(gene_vec[!is.na(gene_vec)]), " genes)")
}

## ============================================================
## STEP 1: MERGE RAW COUNT FILES
## ============================================================
banner(1, "Merge raw count files")

files <- list.files(pattern = "\\.csv\\.gz$", full.names = TRUE)
message("  Found ", length(files), " files")
if (length(files) == 0) stop("No .csv.gz files found in working directory.")

read_clean <- function(f) {
  dt <- fread(f)
  dt[, .SD, .SDcols = c(1, 2)]
}

dt_list <- lapply(files, read_clean)
genes   <- dt_list[[1]][[1]]
dt_list <- lapply(dt_list, function(x) x[match(genes, x[[1]])])
names(dt_list) <- gsub("\\.csv\\.gz$", "", basename(files))

final_matrix <- data.table(GeneID = genes)
final_matrix <- cbind(final_matrix, as.data.table(lapply(dt_list, `[[`, 2)))

for (j in 2:ncol(final_matrix))
  set(final_matrix, which(is.na(final_matrix[[j]])), j, 0L)

fwrite(final_matrix, "outputs/tables/GSE141910_raw_counts_merged.csv")
message("  Merged matrix: ", nrow(final_matrix), " genes x ",
        ncol(final_matrix) - 1, " samples")

## ============================================================
## STEP 2: SETUP — COUNTS + METADATA
## ============================================================
banner(2, "Prepare counts and metadata")

counts           <- as.data.frame(final_matrix)
rownames(counts) <- counts$GeneID
counts$GeneID    <- NULL

cols       <- colnames(counts)
is_ctrl    <- grepl("_C", cols)
is_hcm     <- grepl("_P", cols)

if (sum(is_ctrl) == 0 | sum(is_hcm) == 0)
  stop("Cannot find _C (control) or _P (HCM) in column names.")

coldata <- data.frame(
  row.names = cols,
  condition = factor(
    ifelse(is_ctrl, "Control", ifelse(is_hcm, "HCM", NA)),
    levels = c("Control", "HCM")
  )
)
stopifnot(!any(is.na(coldata$condition)))
message("  Control: ", sum(is_ctrl), "   HCM: ", sum(is_hcm))

## ============================================================
## STEP 3: PCA
## ============================================================
banner(3, "PCA quality check")

pca    <- prcomp(t(counts), scale. = TRUE)
pct    <- round(100 * pca$sdev^2 / sum(pca$sdev^2), 1)

pca_df <- data.frame(
  PC1 = pca$x[, 1], PC2 = pca$x[, 2],
  condition = coldata$condition,
  sample    = rownames(coldata)
)

p_pca <- ggplot(pca_df, aes(PC1, PC2, color = condition)) +
  geom_point(size = 2.5, alpha = 0.8) +
  stat_ellipse(level = 0.90, linetype = "dashed", linewidth = 0.5) +
  scale_color_manual(values = c("Control" = "steelblue", "HCM" = "firebrick")) +
  theme_classic(base_size = 13) +
  labs(
    title    = "PCA — GSE141910 Bulk RNA-seq",
    subtitle = paste0("n=", nrow(pca_df), " samples  |  PC1=", pct[1],
                      "%  PC2=", pct[2], "%"),
    x = paste0("PC1 (", pct[1], "%)"),
    y = paste0("PC2 (", pct[2], "%)"),
    color = NULL
  )

save_png(p_pca, "01_PCA.png", width = 7, height = 6)

## ============================================================
## STEP 4: DIFFERENTIAL EXPRESSION (limma)
## ============================================================
banner(4, "Differential expression — limma")

design <- model.matrix(~ condition, data = coldata)
fit    <- lmFit(counts, design)
fit    <- eBayes(fit)
res    <- topTable(fit, coef = "conditionHCM", number = Inf)
res$GeneID <- rownames(res)

## Annotate with gene symbols
res$clean_GeneID <- sub("\\..*", "", res$GeneID)
res$Symbol <- mapIds(
  org.Hs.eg.db, keys = res$clean_GeneID,
  column = "SYMBOL", keytype = "ENSEMBL", multiVals = "first"
)
res$Symbol[is.na(res$Symbol)] <- res$GeneID[is.na(res$Symbol)]

## Significance flags
res$significant  <- res$adj.P.Val < 0.05 & abs(res$logFC) > 1
res$direction    <- ifelse(res$logFC > 0, "Up", "Down")

message("  DEGs (adj.P<0.05, |logFC|>1): ",
        sum(res$significant), " total  (",
        sum(res$significant & res$logFC > 0), " up / ",
        sum(res$significant & res$logFC < 0), " down)")

save_csv(res, "limma_full_results.csv")
save_csv(res[res$significant, ], "limma_significant_DEGs.csv")

## ── Volcano ─────────────────────────────────────────────────
top_vol <- res[res$significant, ]
top_vol <- top_vol[order(top_vol$adj.P.Val), ][1:min(25, nrow(top_vol)), ]

p_vol <- ggplot(res, aes(logFC, -log10(adj.P.Val))) +
  geom_point(aes(color = significant), size = 0.6, alpha = 0.7) +
  geom_text_repel(
    data = top_vol, aes(label = Symbol),
    size = 2.8, max.overlaps = 20, box.padding = 0.4
  ) +
  scale_color_manual(
    values = c("FALSE" = "grey75", "TRUE" = "firebrick"),
    labels = c("FALSE" = "n.s.", "TRUE" = "adj.P<0.05, |logFC|>1")
  ) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50", linewidth = 0.4) +
  theme_classic(base_size = 13) +
  labs(
    title    = "Volcano — HCM vs Control (limma)",
    subtitle = paste0(sum(res$significant), " DEGs  |  adj.P<0.05  |logFC|>1"),
    x = "log2 Fold Change (HCM vs Control)",
    y = "-log10(adj. P-value)",
    color = NULL
  )

save_png(p_vol, "02_Volcano.png", width = 9, height = 7)

## ============================================================
## STEP 5: HEATMAP — TOP 50 DEGs
## ============================================================
banner(5, "Heatmap — top 50 DEGs")

top_up   <- res[res$logFC >  0 & res$adj.P.Val < 0.05, ]
top_up   <- top_up[order(top_up$adj.P.Val), ][1:min(25, nrow(top_up)), ]
top_dn   <- res[res$logFC <  0 & res$adj.P.Val < 0.05, ]
top_dn   <- top_dn[order(top_dn$adj.P.Val), ][1:min(25, nrow(top_dn)), ]
top50    <- rbind(top_up, top_dn)
top50$clean_GeneID <- sub("\\..*", "", top50$GeneID)

hm_mat <- as.matrix(counts)[rownames(counts) %in% top50$clean_GeneID, ]
hm_mat <- t(scale(t(hm_mat)))
hm_mat[hm_mat >  3] <-  3
hm_mat[hm_mat < -3] <- -3

row_labels <- top50$Symbol[match(rownames(hm_mat), top50$clean_GeneID)]
row_labels[is.na(row_labels)] <- rownames(hm_mat)[is.na(row_labels)]

col_cond <- ifelse(grepl("_C", colnames(hm_mat)), "Control", "HCM")
col_ann  <- HeatmapAnnotation(
  Condition = col_cond,
  col = list(Condition = c("Control" = "steelblue", "HCM" = "firebrick")),
  show_legend = TRUE
)
row_dir <- ifelse(top50$logFC[match(rownames(hm_mat), top50$clean_GeneID)] > 0, "Up", "Down")
row_ann <- rowAnnotation(
  Direction = row_dir,
  col = list(Direction = c("Up" = "firebrick", "Down" = "steelblue"))
)

png("outputs/plots/03_Heatmap_top50_DEGs.png",
    width = 14, height = 10, units = "in", res = 200)
draw(Heatmap(
  hm_mat,
  name             = "Z-score",
  col              = colorRamp2(c(-3, 0, 3), c("steelblue", "white", "firebrick")),
  top_annotation   = col_ann,
  right_annotation = row_ann,
  row_labels       = row_labels,
  row_names_gp     = gpar(fontsize = 8),
  show_column_names = FALSE,
  cluster_rows     = TRUE,
  cluster_columns  = TRUE,
  column_title     = "Top 50 DEGs: HCM vs Control — GSE141910 (n=366)",
  column_title_gp  = gpar(fontsize = 12, fontface = "bold")
))
dev.off()
message("  Saved: outputs/plots/03_Heatmap_top50_DEGs.png")

## ============================================================
## STEP 6: GSEA — HALLMARK + C2
## ============================================================
banner(6, "GSEA — Hallmark + C2 (fgsea)")

ranked <- setNames(res$t, res$clean_GeneID)
ranked <- sort(ranked[!duplicated(names(ranked))], decreasing = TRUE)

get_msig <- function(coll) {
  tryCatch(
    msigdbr(species = "Homo sapiens", collection = coll),
    error = function(e) msigdbr(species = "Homo sapiens", category = coll)
  )
}

hallmark   <- get_msig("H")
c2         <- get_msig("C2")
pathways_H  <- split(hallmark$ensembl_gene, hallmark$gs_name)
pathways_C2 <- split(c2$ensembl_gene,      c2$gs_name)

set.seed(42)
message("  Running GSEA Hallmark...")
gsea_H  <- fgsea(pathways_H,  ranked, minSize = 15, maxSize = 500, nPermSimple = 10000)
message("  Running GSEA C2...")
gsea_C2 <- fgsea(pathways_C2, ranked, minSize = 15, maxSize = 500, nPermSimple = 10000)

flatten_le <- function(dt) {
  dt2 <- as.data.frame(dt)
  dt2$leadingEdge <- sapply(dt2$leadingEdge, paste, collapse = ";")
  dt2
}

gsea_H_sig  <- gsea_H[padj  < 0.05][order(NES)]
gsea_C2_sig <- gsea_C2[padj < 0.05][order(NES)]

save_csv(flatten_le(gsea_H_sig),  "GSEA_Hallmark_significant.csv")
save_csv(flatten_le(gsea_C2_sig), "GSEA_C2_significant.csv")
message("  Significant Hallmark: ", nrow(gsea_H_sig),
        "   C2: ", nrow(gsea_C2_sig))

## ── GSEA barplot ─────────────────────────────────────────────
top_h <- gsea_H_sig[order(abs(NES), decreasing = TRUE)][1:min(25, .N)]
top_h[, pathway_clean := gsub("_", " ", gsub("HALLMARK_", "", pathway))]

p_gsea <- ggplot(top_h, aes(reorder(pathway_clean, NES), NES, fill = NES > 0)) +
  geom_col(width = 0.75) +
  coord_flip() +
  scale_fill_manual(
    values = c("TRUE" = "firebrick", "FALSE" = "steelblue"),
    labels = c("TRUE" = "Up in HCM", "FALSE" = "Down in HCM")
  ) +
  theme_classic(base_size = 11) +
  theme(axis.text.y = element_text(size = 9)) +
  labs(
    title = "GSEA Hallmark Pathways — HCM vs Control",
    x = NULL, y = "Normalized Enrichment Score (NES)", fill = NULL
  )

save_png(p_gsea, "04_GSEA_Hallmark_barplot.png", width = 11, height = 8)

## ── Enrichment curve for top pathway ─────────────────────────
top1 <- gsea_H_sig[which.max(abs(NES))]$pathway
enrich_curve <- plotEnrichment(pathways_H[[top1]], ranked) +
  labs(title = gsub("HALLMARK_", "", top1),
       subtitle = paste0("NES=", round(gsea_H_sig[pathway == top1]$NES, 3),
                         "  padj=", signif(gsea_H_sig[pathway == top1]$padj, 3)))

save_png(enrich_curve, paste0("04b_GSEA_enrichment_", gsub("[/ ]", "_", top1), ".png"),
         width = 7, height = 5)

## ============================================================
## STEP 7: TF ACTIVITY — decoupleR + DoRothEA
## ============================================================
banner(7, "TF activity — decoupleR + DoRothEA")

## Build symbol-mapped expression matrix
expr_raw <- as.matrix(counts)
rownames(expr_raw) <- sub("\\..*", "", rownames(expr_raw))

sym_map       <- setNames(res$Symbol, res$clean_GeneID)
mapped_sym    <- sym_map[rownames(expr_raw)]
keep_sym      <- !is.na(mapped_sym) & !duplicated(mapped_sym)
expr_sym      <- expr_raw[keep_sym, ]
rownames(expr_sym) <- mapped_sym[keep_sym]
message("  Symbol-mapped expression: ", nrow(expr_sym), " genes x ",
        ncol(expr_sym), " samples")

regulon <- dorothea_hs %>%
  dplyr::filter(confidence %in% c("A", "B", "C")) %>%
  dplyr::rename(source = tf) %>%
  dplyr::select(source, target, mor)

n_overlap <- length(intersect(regulon$target, rownames(expr_sym)))
message("  DoRothEA targets in matrix: ", n_overlap, " / ",
        length(unique(regulon$target)))

tf_acts <- run_ulm(
  mat = expr_sym, net = regulon,
  .source = "source", .target = "target", .mor = "mor",
  minsize = 4
)

## ── FIX: avoid column_to_rownames() dependency on tibble ─────
tf_wide <- tf_acts %>%
  dplyr::filter(statistic == "ulm") %>%
  tidyr::pivot_wider(id_cols = source, names_from = condition, values_from = score)

tf_mat              <- as.matrix(tf_wide[, -1])   # drop 'source' column
rownames(tf_mat)    <- tf_wide$source              # TF names as rownames

message("  TF activity matrix: ", nrow(tf_mat), " TFs x ", ncol(tf_mat), " samples")

## Differential TF activity
cols_tf  <- colnames(tf_mat)
cond_tf  <- factor(ifelse(grepl("_C", cols_tf), "Control", "HCM"),
                   levels = c("Control", "HCM"))
des_tf   <- model.matrix(~ cond_tf)
fit_tf   <- eBayes(lmFit(tf_mat, des_tf))
tf_res   <- topTable(fit_tf, coef = "cond_tfHCM", number = Inf)
tf_res$TF <- rownames(tf_res)

save_csv(tf_res, "TF_activity_HCM_vs_Control.csv")
message("  Significant TFs (adj.P<0.05): ", sum(tf_res$adj.P.Val < 0.05))

## ── TF lollipop ──────────────────────────────────────────────
sig_tf   <- tf_res[tf_res$adj.P.Val < 0.05, ]
plot_tf  <- sig_tf[order(abs(sig_tf$t), decreasing = TRUE), ][1:min(40, nrow(sig_tf)), ]
plot_tf  <- plot_tf[order(plot_tf$logFC), ]

p_tf <- ggplot(plot_tf, aes(reorder(TF, logFC), logFC, color = logFC > 0)) +
  geom_segment(aes(xend = TF, y = 0, yend = logFC), linewidth = 0.7) +
  geom_point(size = 2.5) +
  coord_flip() +
  scale_color_manual(
    values = c("TRUE" = "firebrick", "FALSE" = "steelblue"),
    labels = c("TRUE" = "Active in HCM", "FALSE" = "Repressed in HCM")
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  theme_classic(base_size = 11) +
  labs(
    title    = "Differential TF Activity: HCM vs Control",
    subtitle = "decoupleR ULM + DoRothEA (confidence A/B/C)",
    x = NULL, y = "Activity logFC (HCM vs Control)", color = NULL
  )

save_png(p_tf, "05_TF_activity_lollipop.png", width = 9, height = 9)

## ── TF volcano ────────────────────────────────────────────────
tf_res$label <- ifelse(tf_res$adj.P.Val < 0.05 & abs(tf_res$logFC) > 0.3,
                       tf_res$TF, NA)

p_tfv <- ggplot(tf_res, aes(logFC, -log10(adj.P.Val), color = adj.P.Val < 0.05)) +
  geom_point(size = 1.8, alpha = 0.8) +
  geom_text_repel(aes(label = label), size = 2.8, max.overlaps = 25, na.rm = TRUE) +
  scale_color_manual(values = c("FALSE" = "grey70", "TRUE" = "firebrick")) +
  theme_classic(base_size = 12) +
  labs(title = "TF Activity Volcano", x = "logFC", y = "-log10(adj.P)", color = "adj.P<0.05")

save_png(p_tfv, "05b_TF_activity_volcano.png", width = 8, height = 7)

## ============================================================
## STEP 8: WGCNA — SIGNED HYBRID (power=4, 5k genes)
## ============================================================
banner(8, "WGCNA co-expression modules")

## Load WGCNA early and pin cor() to avoid Bioconductor masking
enableWGCNAThreads(nThreads = min(8, parallel::detectCores()))

ery_syms    <- c("HBA1","HBA2","HBB","HBD","HBE1","HBG1","HBG2",
                 "HBM","HBQ1","HBZ","GYPA","GYPB","GYPC","ANK1",
                 "SLC4A1","CA1","CA2","ALAS2","TFRC")
ery_ens     <- res$clean_GeneID[res$Symbol %in% ery_syms]

expr_w      <- as.matrix(counts)
rownames(expr_w) <- sub("\\..*", "", rownames(expr_w))
expr_w      <- expr_w[!rownames(expr_w) %in% ery_ens, ]

top5k       <- names(sort(apply(expr_w, 1, var), decreasing = TRUE))[1:5000]
datExpr0    <- t(expr_w[top5k, ])

gsg      <- WGCNA::goodSamplesGenes(datExpr0, verbose = 0)
datExpr  <- datExpr0[gsg$goodSamples, gsg$goodGenes]
message("  WGCNA input: ", nrow(datExpr), " samples x ", ncol(datExpr), " genes")

## Soft threshold — use power=4 (R²=0.99 from pilot run)
softPower <- 4
message("  Using softPower=", softPower)

## Pin WGCNA::cor before blockwiseModules
cor <- WGCNA::cor

net <- blockwiseModules(
  datExpr,
  power             = softPower,
  TOMType           = "signed",
  networkType       = "signed hybrid",
  minModuleSize     = 30,
  reassignThreshold = 0,
  mergeCutHeight    = 0.25,
  numericLabels     = TRUE,
  pamRespectsDendro = FALSE,
  saveTOMs          = FALSE,
  verbose           = 2
)

## Restore stats::cor
cor <- stats::cor

n_mod <- length(unique(net$colors)) - 1
message("  Modules detected: ", n_mod)

## Module-trait correlation
cond_num <- as.numeric(grepl("_P", rownames(datExpr)))
MEs      <- net$MEs

mod_cor  <- WGCNA::cor(MEs, cond_num, use = "p")
mod_pval <- corPvalueStudent(mod_cor, nrow(datExpr))

mod_sum <- data.frame(
  Module      = rownames(mod_cor),
  Correlation = mod_cor[, 1],
  Pvalue      = mod_pval[, 1]
) %>%
  arrange(Pvalue) %>%
  mutate(Padj = p.adjust(Pvalue, "BH"))

save_csv(mod_sum, "WGCNA_module_HCM_correlation.csv")
message("  Top module: ", mod_sum$Module[1],
        "  r=", round(mod_sum$Correlation[1], 3))

## kME for hub gene ranking
cor <- WGCNA::cor
kME <- signedKME(datExpr, MEs)
cor <- stats::cor

## Module heatmap
png("outputs/plots/06_WGCNA_module_heatmap.png",
    width = 8, height = max(5, n_mod * 0.45), units = "in", res = 200)
labeledHeatmap(
  Matrix        = mod_cor,
  xLabels       = "HCM Status",
  yLabels       = rownames(mod_cor),
  ySymbols      = rownames(mod_cor),
  colorLabels   = FALSE,
  colors        = blueWhiteRed(50),
  textMatrix    = signif(mod_cor, 2),
  setStdMargins = FALSE,
  cex.text      = 0.65,
  zlim          = c(-1, 1),
  main          = paste0("WGCNA Module-HCM Correlation (power=", softPower, ")")
)
dev.off()
message("  Saved: outputs/plots/06_WGCNA_module_heatmap.png")

## Extract all module gene tables
all_mod_genes <- lapply(unique(net$colors), function(m) {
  g <- names(net$colors)[net$colors == m]
  kme_col <- paste0("kME", m)
  df <- data.frame(
    ENSEMBL = g,
    Module  = m,
    kME     = if (kme_col %in% colnames(kME)) kME[g, kme_col] else NA
  )
  df <- merge(df,
              res[, c("clean_GeneID","Symbol","logFC","adj.P.Val","significant")],
              by.x = "ENSEMBL", by.y = "clean_GeneID", all.x = TRUE)
  df[order(-abs(df$kME)), ]
})
names(all_mod_genes) <- paste0("M", unique(net$colors))

all_mod_df <- do.call(rbind, all_mod_genes)
save_csv(all_mod_df, "WGCNA_all_module_genes.csv")

## ── Top module GO enrichment ──────────────────────────────────
top_mod_num   <- as.integer(gsub("ME", "", mod_sum$Module[1]))
top_mod_genes <- all_mod_genes[[paste0("M", top_mod_num)]]
top_mod_syms  <- top_mod_genes$Symbol[!is.na(top_mod_genes$Symbol)]

go_top <- enrichGO(
  gene          = top_mod_syms,
  OrgDb         = org.Hs.eg.db,
  keyType       = "SYMBOL",
  ont           = "BP",
  pAdjustMethod = "BH",
  pvalueCutoff  = 0.05,
  qvalueCutoff  = 0.2,
  readable      = TRUE
)

if (!is.null(go_top) && nrow(go_top) > 0) {
  save_csv(go_top@result, "WGCNA_top_module_GO_BP.csv")
  p_go <- dotplot(go_top, showCategory = 20) +
    labs(title = paste0("GO:BP — WGCNA Top Module (",
                        mod_sum$Module[1], ", r=",
                        round(mod_sum$Correlation[1], 2), ")"))
  save_png(p_go, "06b_WGCNA_top_module_GO.png", width = 9, height = 8)
}

## ============================================================
## STEP 9: SARCOMERE / HCM GENE PANEL
## ============================================================
banner(9, "Known HCM gene panel check")

hcm_genes <- c(
  # Thick filament
  "MYH7","MYH6","MYBPC3","MYL2","MYL3",
  # Thin filament
  "TNNT2","TNNI3","TNNI1","TNNC1","TPM1","ACTC1",
  # Giant / Z-disc
  "TTN","ACTN2","CSRP3","FHL1","ANKRD1",
  # Ca2+ handling
  "PLN","RYR2","CASQ2","ATP2A2",
  # Hypertrophy TFs
  "MEF2A","MEF2C","MEF2D","GATA4","NFATC1",
  # Natriuretic peptides
  "NPPA","NPPB",
  # Fibrosis
  "COL1A1","COL3A1","POSTN","CCN2",
  # Wnt
  "SFRP4","FRZB","WNT5A","WNT9A",
  # Metabolic
  "PDK4","LDHA","HK2","KLF15"
)

panel_df <- res[res$Symbol %in% hcm_genes,
                c("Symbol","logFC","adj.P.Val","significant")]
panel_df <- panel_df[order(panel_df$logFC, decreasing = TRUE), ]
save_csv(panel_df, "HCM_gene_panel.csv")

p_panel <- ggplot(panel_df, aes(reorder(Symbol, logFC), logFC,
                                 color = significant, size = significant)) +
  geom_segment(aes(xend = Symbol, y = 0, yend = logFC),
               color = "grey60", linewidth = 0.6) +
  geom_point() +
  scale_color_manual(values = c("FALSE" = "grey70", "TRUE" = "firebrick")) +
  scale_size_manual(values = c("FALSE" = 2, "TRUE" = 3.5), guide = "none") +
  geom_hline(yintercept = c(-1, 0, 1), linetype = c("dashed","solid","dashed"),
             color = c("grey50","grey30","grey50"), linewidth = 0.4) +
  coord_flip() +
  theme_classic(base_size = 11) +
  labs(title = "Known HCM Gene Panel",
       subtitle = paste0("Significant (adj.P<0.05, |logFC|>1): ",
                         sum(panel_df$significant, na.rm=TRUE), " / ", nrow(panel_df)),
       x = NULL, y = "log2FC (HCM vs Control)", color = "Significant")

save_png(p_panel, "07_HCM_panel_lollipop.png", width = 8, height = 9)

## ============================================================
## STEP 10: ATAC INTEGRATION (if atac_scores.tab exists)
## ============================================================
banner(10, "ATAC motif enrichment (conditional)")

atac_present <- file.exists("atac_scores.tab")
if (!atac_present) {
  message("  atac_scores.tab not found — skipping ATAC step")
  atac_res <- NULL
} else {
  message("  ATAC file found — running motif enrichment")

  suppressPackageStartupMessages({
    library(GenomicRanges)
    library(BSgenome.Hsapiens.UCSC.hg38)
    library(JASPAR2022)
    library(TFBSTools)
    library(motifmatchr)
  })

  atac <- fread("atac_scores.tab")
  colnames(atac) <- gsub("[#']", "", colnames(atac))
  colnames(atac) <- trimws(colnames(atac))
  setnames(atac, c("chr","start","end","ATAC_DCM","ATAC_NF"))
  atac[, chr := paste0("chr", gsub("chr","", trimws(chr), ignore.case=TRUE))]
  atac[, delta_atac := ATAC_DCM - ATAC_NF]
  atac[, mean_atac  := (ATAC_DCM + ATAC_NF)/2]

  atac_sig <- atac[mean_atac > 0]
  open_th  <- quantile(atac_sig$delta_atac, 0.90)
  clos_th  <- quantile(atac_sig$delta_atac, 0.10)
  open_dcm <- atac_sig[delta_atac >= open_th]
  clos_dcm <- atac_sig[delta_atac <= clos_th]

  make_gr <- function(dt)
    GRanges(seqnames = dt$chr,
            ranges   = IRanges(as.integer(dt$start), as.integer(dt$end)))

  gr_open <- make_gr(open_dcm)
  gr_clos <- make_gr(clos_dcm)
  gr_bg   <- make_gr(atac_sig)

  pwm_list <- getMatrixSet(JASPAR2022,
                           list(collection="CORE", tax_group="vertebrates",
                                all_versions=FALSE))

  mot_open <- matchMotifs(pwm_list, gr_open, genome=BSgenome.Hsapiens.UCSC.hg38, out="matches")
  mot_clos <- matchMotifs(pwm_list, gr_clos, genome=BSgenome.Hsapiens.UCSC.hg38, out="matches")
  mot_bg   <- matchMotifs(pwm_list, gr_bg,   genome=BSgenome.Hsapiens.UCSC.hg38, out="matches")

  run_fisher <- function(fg, bg, n_fg, n_bg) {
    lapply(seq_along(pwm_list), function(i) {
      h <- sum(motifMatches(fg)[,i])
      b <- sum(motifMatches(bg)[,i])
      ft <- fisher.test(matrix(c(h, n_fg-h, b, n_bg-b), 2), alternative="greater")
      data.frame(Motif=name(pwm_list)[i], OR=ft$estimate, Pvalue=ft$p.value)
    }) %>% bind_rows() %>%
      mutate(Padj = p.adjust(Pvalue, "BH")) %>%
      arrange(Padj)
  }

  atac_open <- run_fisher(mot_open, mot_bg, nrow(open_dcm), nrow(atac_sig))
  atac_clos <- run_fisher(mot_clos, mot_bg, nrow(clos_dcm), nrow(atac_sig))

  save_csv(atac_open, "ATAC_motif_open_DCM.csv")
  save_csv(atac_clos, "ATAC_motif_closed_DCM.csv")

  atac_res <- list(open = atac_open, closed = atac_clos)
  message("  ATAC motif enrichment complete")
}

## ============================================================
## STEP 11: BUILD REGULON MODULES
## ============================================================
## Strategy:
##   REG1 — UP in HCM + open chromatin TF targets  (active activation)
##   REG2 — UP in HCM + no chromatin support        (cell-intrinsic / post-tx)
##   REG3 — DOWN in HCM + open chromatin            (TF-lost targets)
##   REG4 — DOWN in HCM + closed chromatin          (repressed module)
##   REG5 — WGCNA ME3 hub genes (ECM/stromal co-expression)
##
## For scRNA use: score each regulon gene list with
##   Seurat::AddModuleScore() or scanpy sc.tl.score_genes()
## ============================================================
banner(11, "Build regulon modules for scRNA scoring")

sig_df <- res[res$significant, ]

## Get TF targets in DoRothEA
reg_targets <- dorothea_hs %>%
  filter(confidence %in% c("A","B","C"))

## Active TFs in HCM (significantly up-regulated TF activity)
active_tfs_up  <- tf_res$TF[tf_res$adj.P.Val < 0.05 & tf_res$logFC > 0]
active_tfs_dn  <- tf_res$TF[tf_res$adj.P.Val < 0.05 & tf_res$logFC < 0]

## Targets of active TFs
targets_of_up_tfs <- reg_targets %>%
  filter(tf %in% active_tfs_up) %>%
  pull(target) %>% unique()

targets_of_dn_tfs <- reg_targets %>%
  filter(tf %in% active_tfs_dn) %>%
  pull(target) %>% unique()

## UP DEGs
up_degs <- sig_df$Symbol[sig_df$logFC > 0]
dn_degs  <- sig_df$Symbol[sig_df$logFC < 0]

## ATAC-open motif TFs (if available)
if (!is.null(atac_res)) {
  open_tfs   <- atac_res$open$Motif[atac_res$open$Padj < 0.05]
  closed_tfs <- atac_res$closed$Motif[atac_res$closed$Padj < 0.05]

  ## REG1: UP DEGs + target of active TF + TF motif open
  reg1 <- up_degs[up_degs %in% targets_of_up_tfs &
                    up_degs %in% unlist(lapply(open_tfs, function(tf)
                      reg_targets$target[grepl(tf, reg_targets$tf, ignore.case=TRUE)]))]

  ## REG3: DOWN DEGs + target of repressed TF + TF motif open (de-repressed?)
  reg3 <- dn_degs[dn_degs %in% targets_of_dn_tfs &
                    dn_degs %in% unlist(lapply(open_tfs, function(tf)
                      reg_targets$target[grepl(tf, reg_targets$tf, ignore.case=TRUE)]))]

  ## REG4: DOWN DEGs + target of repressed TF + closed chromatin
  reg4 <- dn_degs[dn_degs %in% targets_of_dn_tfs &
                    dn_degs %in% unlist(lapply(closed_tfs, function(tf)
                      reg_targets$target[grepl(tf, reg_targets$tf, ignore.case=TRUE)]))]

} else {
  ## Without ATAC: use TF activity alone
  reg1 <- intersect(up_degs, targets_of_up_tfs)
  reg3 <- intersect(dn_degs, targets_of_dn_tfs)
  reg4 <- character(0)
}

## REG2: UP DEGs NOT explained by active TF targets (post-tx / other)
reg2 <- up_degs[!up_degs %in% targets_of_up_tfs]

## REG5: WGCNA top module hub genes (ECM/stromal)
reg5 <- top_mod_genes$Symbol[
  !is.na(top_mod_genes$Symbol) & !is.na(top_mod_genes$kME) &
    top_mod_genes$kME > 0.7
]

## ── Fallback: if ATAC-filtered lists are tiny, use TF-only ──
if (length(reg1) < 10) {
  message("  REG1 ATAC-filtered is small (n=", length(reg1),
          "); falling back to TF-activity-only definition")
  reg1 <- intersect(up_degs, targets_of_up_tfs)
}
if (length(reg3) < 10) {
  reg3 <- intersect(dn_degs, targets_of_dn_tfs)
}

## ── Remove overlaps (priority: REG1 > REG2 > REG3 > REG4 > REG5) ──
reg2 <- setdiff(reg2, reg1)
reg3 <- setdiff(reg3, c(reg1, reg2))
reg4 <- setdiff(reg4, c(reg1, reg2, reg3))
reg5 <- setdiff(reg5, c(reg1, reg2, reg3, reg4))

## ── Summary ──────────────────────────────────────────────────
cat("\n", paste(rep("─", 60), collapse=""), "\n")
cat("  REGULON SUMMARY\n")
cat(paste(rep("─", 60), collapse=""), "\n")
cat(sprintf("  REG1 (UP + TF-active targets):         %d genes\n", length(reg1)))
cat(sprintf("  REG2 (UP + TF-independent):            %d genes\n", length(reg2)))
cat(sprintf("  REG3 (DOWN + TF-repressed targets):    %d genes\n", length(reg3)))
cat(sprintf("  REG4 (DOWN + closed chromatin):        %d genes\n", length(reg4)))
cat(sprintf("  REG5 (WGCNA ECM/stromal hub, kME>0.7):%d genes\n", length(reg5)))
cat(paste(rep("─", 60), collapse=""), "\n")

## ── Save gene lists ──────────────────────────────────────────
save_txt(reg1, "REG1_UP_TF_active_targets.txt")
save_txt(reg2, "REG2_UP_TF_independent.txt")
save_txt(reg3, "REG3_DOWN_TF_repressed_targets.txt")
save_txt(reg4, "REG4_DOWN_closed_chromatin.txt")
save_txt(reg5, "REG5_WGCNA_ECM_hub.txt")

## ── Save as single CSV for easy loading ──────────────────────
max_len <- max(length(reg1), length(reg2), length(reg3),
               length(reg4), length(reg5))
pad_na  <- function(x, n) c(x, rep(NA, n - length(x)))
reg_csv <- data.frame(
  REG1_UP_TFactive   = pad_na(reg1, max_len),
  REG2_UP_TFindep    = pad_na(reg2, max_len),
  REG3_DOWN_TFrep    = pad_na(reg3, max_len),
  REG4_DOWN_chromclo = pad_na(reg4, max_len),
  REG5_WGCNA_ECM     = pad_na(reg5, max_len)
)
save_csv(reg_csv, "../regulons/ALL_REGULONS.csv")

## ── Regulon summary plot ─────────────────────────────────────
reg_sizes <- data.frame(
  Regulon     = c("REG1\nUP+TF-active","REG2\nUP+TF-indep",
                  "REG3\nDOWN+TF-rep","REG4\nDOWN+closed",
                  "REG5\nWGCNA-ECM"),
  N           = c(length(reg1), length(reg2), length(reg3),
                  length(reg4), length(reg5)),
  Direction   = c("Up","Up","Down","Down","Mixed")
)

p_reg <- ggplot(reg_sizes, aes(Regulon, N, fill = Direction)) +
  geom_col(width = 0.65) +
  geom_text(aes(label = N), vjust = -0.4, fontface = "bold", size = 4.5) +
  scale_fill_manual(values = c("Up" = "firebrick", "Down" = "steelblue",
                               "Mixed" = "darkorchid4")) +
  theme_classic(base_size = 12) +
  labs(
    title    = "Regulon Module Sizes — HCM Bulk RNA + ATAC + WGCNA",
    subtitle = "Use .txt files with Seurat::AddModuleScore() or sc.tl.score_genes()",
    x = NULL, y = "Number of Genes", fill = "Direction"
  ) +
  theme(axis.text.x = element_text(size = 10))

save_png(p_reg, "08_Regulon_sizes.png", width = 8, height = 5)

## ── Heatmap of REG1 + REG3 genes in bulk samples ─────────────
reg_hm_genes <- c(head(reg1, 20), head(reg3, 20))
reg_hm_genes <- unique(reg_hm_genes[!is.na(reg_hm_genes)])

if (length(reg_hm_genes) >= 5) {
  ## Map back to ENSEMBL for indexing
  sym2ens <- setNames(res$clean_GeneID, res$Symbol)
  reg_hm_ens <- sym2ens[reg_hm_genes]
  reg_hm_ens <- reg_hm_ens[!is.na(reg_hm_ens)]

  hm2 <- expr_raw[rownames(expr_raw) %in% reg_hm_ens, ]
  if (nrow(hm2) >= 3) {
    hm2 <- t(scale(t(hm2)))
    hm2[hm2 >  3] <-  3
    hm2[hm2 < -3] <- -3

    rlab2 <- res$Symbol[match(rownames(hm2), res$clean_GeneID)]
    rlab2[is.na(rlab2)] <- rownames(hm2)[is.na(rlab2)]

    col_c2 <- ifelse(grepl("_C", colnames(hm2)), "Control", "HCM")
    ca2    <- HeatmapAnnotation(
      Condition = col_c2,
      col = list(Condition = c("Control"="steelblue","HCM"="firebrick"))
    )
    reg_dir2 <- ifelse(rlab2 %in% reg1, "REG1 (Up)", "REG3 (Down)")
    ra2 <- rowAnnotation(
      Regulon = reg_dir2,
      col = list(Regulon = c("REG1 (Up)"="firebrick","REG3 (Down)"="steelblue"))
    )

    png("outputs/plots/09_Regulon_heatmap_REG1_REG3.png",
        width = 14, height = 8, units = "in", res = 200)
    draw(Heatmap(
      hm2,
      name              = "Z-score",
      col               = colorRamp2(c(-3,0,3), c("steelblue","white","firebrick")),
      top_annotation    = ca2,
      right_annotation  = ra2,
      row_labels        = rlab2,
      row_names_gp      = gpar(fontsize = 8),
      show_column_names = FALSE,
      cluster_rows      = TRUE,
      cluster_columns   = TRUE,
      column_title      = "REG1 (Up+TF-active) vs REG3 (Down+TF-repressed) — Bulk RNA",
      column_title_gp   = gpar(fontsize = 11, fontface = "bold")
    ))
    dev.off()
    message("  Saved: outputs/plots/09_Regulon_heatmap_REG1_REG3.png")
  }
}

## ── Convergent TF × ATAC plot ─────────────────────────────────
if (!is.null(atac_res)) {
  tf_sig_df  <- tf_res[tf_res$adj.P.Val < 0.05, ]
  tf_sig_df$in_open_motif <- sapply(
    tf_sig_df$TF,
    function(tf) any(grepl(tf, atac_res$open$Motif[atac_res$open$Padj < 0.05],
                           ignore.case = TRUE))
  )
  conv_tfs <- tf_sig_df[tf_sig_df$in_open_motif, ]
  save_csv(conv_tfs, "Convergent_TF_activity_and_ATAC.csv")

  if (nrow(conv_tfs) >= 3) {
    p_conv <- ggplot(conv_tfs, aes(logFC, -log10(adj.P.Val), label = TF)) +
      geom_point(aes(color = logFC > 0), size = 3) +
      geom_text_repel(size = 3, max.overlaps = 20) +
      scale_color_manual(values = c("TRUE"="firebrick","FALSE"="steelblue"),
                         labels = c("TRUE"="Active in HCM","FALSE"="Repressed in HCM")) +
      theme_classic(base_size = 12) +
      labs(title = "Convergent TFs: RNA Activity + ATAC Motif",
           x = "TF Activity logFC", y = "-log10(adj.P)", color = NULL)
    save_png(p_conv, "10_Convergent_TF_ATAC.png", width = 8, height = 7)
  }
}

## ============================================================
## STEP 12: USAGE INSTRUCTIONS FOR scRNA INTEGRATION
## ============================================================
banner(12, "scRNA integration instructions")

instructions <- c(
  "================================================================",
  "  HOW TO USE REGULON GENE LISTS IN YOUR 4-PATIENT scRNA DATA",
  "================================================================",
  "",
  "Your regulon files are in: outputs/regulons/",
  "",
  "── Seurat (R) ─────────────────────────────────────────────────",
  "",
  "# Load gene lists",
  "reg1 <- readLines('outputs/regulons/REG1_UP_TF_active_targets.txt')",
  "reg2 <- readLines('outputs/regulons/REG2_UP_TF_independent.txt')",
  "reg3 <- readLines('outputs/regulons/REG3_DOWN_TF_repressed_targets.txt')",
  "reg4 <- readLines('outputs/regulons/REG4_DOWN_closed_chromatin.txt')",
  "reg5 <- readLines('outputs/regulons/REG5_WGCNA_ECM_hub.txt')",
  "",
  "# Score each regulon (requires Seurat object 'seurat_obj')",
  "seurat_obj <- AddModuleScore(",
  "  seurat_obj,",
  "  features = list(REG1=reg1, REG2=reg2, REG3=reg3, REG4=reg4, REG5=reg5),",
  "  name     = 'HCM_REG'",
  ")",
  "# Scores appear as: HCM_REG1 ... HCM_REG5 in metadata",
  "",
  "# Violin plot across vCM states:",
  "VlnPlot(seurat_obj, features=c('HCM_REG1','HCM_REG3','HCM_REG5'),",
  "        group.by='vCM_label', pt.size=0)",
  "",
  "# UMAP overlay:",
  "FeaturePlot(seurat_obj, features='HCM_REG1',",
  "            order=TRUE, cols=c('lightgrey','firebrick'))",
  "",
  "── Scanpy (Python) ────────────────────────────────────────────",
  "",
  "import pandas as pd, scanpy as sc",
  "",
  "regs = pd.read_csv('outputs/regulons/ALL_REGULONS.csv')",
  "reg_dict = {col: regs[col].dropna().tolist() for col in regs.columns}",
  "",
  "for name, genes in reg_dict.items():",
  "    sc.tl.score_genes(adata, gene_list=genes, score_name=name)",
  "",
  "# Plot on UMAP:",
  "sc.pl.umap(adata, color=['REG1_UP_TFactive','REG3_DOWN_TFrep',",
  "                         'REG5_WGCNA_ECM', 'vCM_label'])",
  "",
  "# Compare across vCM states:",
  "sc.pl.violin(adata, keys=['REG1_UP_TFactive','REG3_DOWN_TFrep'],",
  "             groupby='vCM_label')",
  "",
  "── Expected biology in your vCM states ────────────────────────",
  "",
  "  REG1 (UP + TF-active):   high in disease/stressed vCM states",
  "       Key TFs: ATF1, NFE2L2, E2F4, CEBPB, NR2C2",
  "  REG2 (UP + TF-indep):    fetal metabolic reprogramming genes",
  "  REG3 (DOWN + TF-rep):    lost in disease (STAT4, SOX9, IRF9 targets)",
  "       Expect low in VCM3/4 failing states",
  "  REG4 (DOWN + closed):    chromatin-silenced genes",
  "  REG5 (WGCNA ECM/stromal):SFRP4, FRZB, LUM — Wnt inhibitors + ECM",
  "       Useful for fibroblast contamination check in your scRNA",
  "",
  "================================================================"
)

writeLines(instructions, "outputs/regulons/HOW_TO_USE_IN_scRNA.txt")
cat(paste(instructions, collapse="\n"), "\n")

## ============================================================
## FINAL SUMMARY
## ============================================================
message("\n", paste(rep("=", 65), collapse=""))
message("ALL STEPS COMPLETE")
message("")
message("PLOTS (outputs/plots/):")
message("  01_PCA.png")
message("  02_Volcano.png")
message("  03_Heatmap_top50_DEGs.png")
message("  04_GSEA_Hallmark_barplot.png")
message("  04b_GSEA_enrichment_<top_pathway>.png")
message("  05_TF_activity_lollipop.png")
message("  05b_TF_activity_volcano.png")
message("  06_WGCNA_module_heatmap.png")
message("  06b_WGCNA_top_module_GO.png")
message("  07_HCM_panel_lollipop.png")
message("  08_Regulon_sizes.png")
message("  09_Regulon_heatmap_REG1_REG3.png")
if (!is.null(atac_res)) message("  10_Convergent_TF_ATAC.png")
message("")
message("TABLES (outputs/tables/):")
message("  GSE141910_raw_counts_merged.csv")
message("  limma_full_results.csv")
message("  limma_significant_DEGs.csv")
message("  GSEA_Hallmark_significant.csv")
message("  GSEA_C2_significant.csv")
message("  TF_activity_HCM_vs_Control.csv")
message("  WGCNA_module_HCM_correlation.csv")
message("  WGCNA_all_module_genes.csv")
message("  WGCNA_top_module_GO_BP.csv")
message("  HCM_gene_panel.csv")
if (!is.null(atac_res)) message("  ATAC_motif_open_DCM.csv  ATAC_motif_closed_DCM.csv")
message("  Convergent_TF_activity_and_ATAC.csv")
message("")
message("REGULONS (outputs/regulons/):")
message("  REG1_UP_TF_active_targets.txt")
message("  REG2_UP_TF_independent.txt")
message("  REG3_DOWN_TF_repressed_targets.txt")
message("  REG4_DOWN_closed_chromatin.txt")
message("  REG5_WGCNA_ECM_hub.txt")
message("  ALL_REGULONS.csv")
message("  HOW_TO_USE_IN_scRNA.txt")
message(paste(rep("=", 65), collapse=""))

# ============================================================
# LIBRARIES
# ============================================================
library(data.table)
library(limma)
library(ggplot2)
library(ggrepel)
library(org.Hs.eg.db)
library(AnnotationDbi)

# ============================================================
# 1. LOAD DATA
# ============================================================
counts <- fread("outputs/tables/GSE141910_raw_counts_merged.csv")
counts <- as.data.frame(counts)
rownames(counts) <- counts$GeneID
counts$GeneID <- NULL

# ============================================================
# 2. METADATA
# ============================================================
sample_names <- colnames(counts)
condition    <- factor(ifelse(grepl("_C", sample_names), "Control", "HCM"),
                       levels = c("Control", "HCM"))
coldata <- data.frame(row.names = sample_names, condition = condition)

# ============================================================
# 3. LIMMA DE
# ============================================================
design <- model.matrix(~ condition, data = coldata)
fit    <- lmFit(counts, design)
fit    <- eBayes(fit)
res    <- topTable(fit, coef = "conditionHCM", number = Inf)

# ============================================================
# 4. ANNOTATION
# ============================================================
res$GeneID       <- rownames(res)
res$clean_GeneID <- sub("\\..*", "", res$GeneID)
res$Symbol       <- mapIds(org.Hs.eg.db,
                           keys     = res$clean_GeneID,
                           column   = "SYMBOL",
                           keytype  = "ENSEMBL",
                           multiVals = "first")
res$Symbol[is.na(res$Symbol)] <- res$GeneID[is.na(res$Symbol)]

# ============================================================
# 5. DIRECTION COLUMN (3 levels for clean color mapping)
# ============================================================
res$direction <- "NS"
res$direction[res$adj.P.Val < 0.05 & res$logFC >  0.5] <- "Up"
res$direction[res$adj.P.Val < 0.05 & res$logFC < -0.5] <- "Down"
res$direction <- factor(res$direction, levels = c("Up", "Down", "NS"))

# ============================================================
# 6. PICK TOP 20 UP + TOP 20 DOWN — ONLY THESE GET LABELS
# ============================================================
sig_up   <- res[res$direction == "Up",   ]
sig_down <- res[res$direction == "Down", ]

top20_up   <- sig_up  [order(-sig_up$logFC),    ][1:min(20, nrow(sig_up)),   ]
top20_down <- sig_down[order( sig_down$logFC),   ][1:min(20, nrow(sig_down)), ]

# Subset passed directly to geom_text_repel — no empty labels anywhere
label_df <- rbind(top20_up, top20_down)

# ============================================================
# 7. VOLCANO PLOT
# ============================================================
n_up   <- sum(res$direction == "Up")
n_down <- sum(res$direction == "Down")

p <- ggplot(res, aes(x = logFC, y = -log10(adj.P.Val))) +

  # All background points
  geom_point(aes(color = direction), size = 0.8, alpha = 0.5) +

  # Labels ONLY for the 40 genes — separate data argument prevents segment explosion
  geom_text_repel(
    data          = label_df,
    aes(label = Symbol, color = direction),
    size          = 3,
    fontface      = "bold",
    max.overlaps  = Inf,
    box.padding   = 0.5,
    point.padding = 0.3,
    segment.color = "grey50",
    segment.alpha = 0.6,
    min.segment.length = 0.1,
    show.legend   = FALSE
  ) +

  # 3-level color: Up=red, Down=blue, NS=lightgrey
  scale_color_manual(
    values = c("Up" = "#D62728", "Down" = "#1F77B4", "NS" = "grey80"),
    labels = c(
      "Up"   = paste0("Up (n=", n_up, ")"),
      "Down" = paste0("Down (n=", n_down, ")"),
      "NS"   = "Not significant"
    ),
    name = NULL
  ) +

  # Threshold lines
  geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed",
             color = "black", linewidth = 0.4, alpha = 0.4) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed",
             color = "black", linewidth = 0.4, alpha = 0.4) +

  theme_classic(base_size = 13) +
  theme(
    legend.position   = "top",
    legend.text       = element_text(size = 11),
    plot.title        = element_text(face = "bold", size = 15),
    plot.subtitle     = element_text(size = 11, color = "grey40"),
    panel.grid.major  = element_line(color = "grey95", linewidth = 0.3)
  ) +
  labs(
    title    = "Volcano Plot: HCM vs Control",
    subtitle = paste0("Top 20 Up (red) and Top 20 Down (blue) genes labeled | ",
                      "FDR < 0.05, |logFC| > 0.5"),
    x = "log2 Fold Change",
    y = "-log10 Adjusted P-value"
  )

# ============================================================
# 8. EXPORT
# ============================================================
ggsave("Volcano_TopGenes_Labeled.png", p,
       width = 12, height = 10, dpi = 300)

print(p)

# ============================================================
# BUILD MODULE SUMMARY TABLE
# ============================================================

import pandas as pd
import numpy as np
import gseapy as gp

# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------
INFILE = "WGCNA_all_module_genes.csv"

# ------------------------------------------------------------
# LOAD TABLE
# ------------------------------------------------------------
df = pd.read_csv(INFILE)

print(df.head())

# Expected columns:
# Module | kME | Symbol | logFC | adj.P.Val

# ------------------------------------------------------------
# CLEAN GENE SYMBOLS
# ------------------------------------------------------------
df = df.dropna(subset=["Symbol"])

# Remove ENSG genes
df = df[
    ~df["Symbol"].astype(str).str.startswith("ENSG")
].copy()

# Remove duplicates
df = df.drop_duplicates(subset=["Module", "Symbol"])

# ------------------------------------------------------------
# GET TOP 50 GENES PER MODULE
# ------------------------------------------------------------
# Rank by absolute kME

df["abs_kME"] = np.abs(df["kME"])

top50 = (
    df
    .sort_values(["Module", "abs_kME"], ascending=[True, False])
    .groupby("Module")
    .head(50)
)

# ------------------------------------------------------------
# BUILD SUMMARY TABLE
# ------------------------------------------------------------
summary_rows = []

modules = sorted(top50["Module"].unique())

for mod in modules:

    sub = top50[top50["Module"] == mod]

    genes = sub["Symbol"].tolist()

    # --------------------------------------------------------
    # GENE STRING
    # --------------------------------------------------------
    gene_string = ", ".join(genes)

    # --------------------------------------------------------
    # GSEA: GO BIOLOGICAL PROCESS
    # --------------------------------------------------------
    try:

        bp = gp.enrichr(
            gene_list=genes,
            gene_sets="GO_Biological_Process_2023",
            organism="human",
            outdir=None,
            cutoff=0.5
        )

        bp_res = bp.results

        if bp_res.shape[0] > 0:
            top_bp = bp_res.sort_values(
                "Adjusted P-value"
            ).iloc[0]["Term"]
        else:
            top_bp = "NA"

    except Exception as e:
        print(f"BP failed for module {mod}: {e}")
        top_bp = "NA"

    # --------------------------------------------------------
    # GSEA: GO MOLECULAR FUNCTION
    # --------------------------------------------------------
    try:

        mf = gp.enrichr(
            gene_list=genes,
            gene_sets="GO_Molecular_Function_2023",
            organism="human",
            outdir=None,
            cutoff=0.5
        )

        mf_res = mf.results

        if mf_res.shape[0] > 0:
            top_mf = mf_res.sort_values(
                "Adjusted P-value"
            ).iloc[0]["Term"]
        else:
            top_mf = "NA"

    except Exception as e:
        print(f"MF failed for module {mod}: {e}")
        top_mf = "NA"

    # --------------------------------------------------------
    # STORE ROW
    # --------------------------------------------------------
    summary_rows.append({
        "Module": mod,
        "Top50_Genes": gene_string,
        "Top_Enriched_BP": top_bp,
        "Top_Enriched_MF": top_mf
    })

# ------------------------------------------------------------
# FINAL TABLE
# ------------------------------------------------------------
summary_df = pd.DataFrame(summary_rows)

# Sort modules
summary_df = summary_df.sort_values("Module")

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
OUTFILE = "WGCNA_module_summary_GO.csv"

summary_df.to_csv(OUTFILE, index=False)

print("\nSaved:")
print(OUTFILE)

print("\nShape:")
print(summary_df.shape)

print("\nPreview:")
print(summary_df.head())



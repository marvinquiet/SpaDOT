# --- please change the library path to your own path ---
.libPaths("/net/mulan/home/wenjinma/Rlib") # set library path on mulan
Sys.setenv(RETICULATE_PYTHON = "/net/mulan/home/wenjinma/envs/spatialATAC/bin/python")

# --- to run this script, please install the following packages ---
suppressPackageStartupMessages({
    library(SPARK)
    library(Seurat)
    library(optparse)
    library(reticulate) # for correctly loading anndata object
    library(anndata)
    library(Matrix)
    library(FNN)
    library(igraph)
    library(bluster)
})

# set optparse
option_list = list(
    make_option(c("--data"), type='character', default=NULL, help="Path to anndata object."),
    make_option(c("--result_dir"), type='character', default=NULL, help="Directory to save results. Default: the same as data directory."), 
    make_option(c("--numCores"), type="integer", default=8, 
        help="Number of cores used to run SPARK-X [default %default]",
        metavar="number")
)
parser = OptionParser(option_list=option_list)
opt = parse_args(parser)
if (is.null(opt$data)) {
    print_help(parser)
    stop("Please provide a valid data path.")
}
if (is.null(opt$result_dir)) {
    opt$result_dir = dirname(opt$data)
}

# load data
adata = read_h5ad(opt$data)
obj = CreateSeuratObject(counts = t(adata$X), meta.data = adata$obs)
# remove ribosomal genes if any
ribosomal_genes = grep("^Rp[sl]|^RP[SL]", rownames(obj), value=T)
obj =  obj[!rownames(obj) %in% ribosomal_genes, ]
mt_genes = grep("^mt-|^MT-", rownames(obj), value=T)
obj =  obj[!rownames(obj) %in% mt_genes, ]
obj$timepoint = as.factor(obj$timepoint)
timepoints = levels(obj$timepoint)
for (tp in timepoints) {
    tp_obj = subset(obj, subset=timepoint==tp)
    count_spark = tp_obj@assays$RNA@layers$counts
    rownames(count_spark) = rownames(tp_obj)
    colnames(count_spark) = colnames(tp_obj)
    tp_obj = SCTransform(tp_obj, return.only.var.genes = FALSE, variable.features.n = NULL, variable.features.rv.th = 1.3)
    count_spark = count_spark[rownames(tp_obj), ]
    locations_spark = cbind(tp_obj$pixel_x, tp_obj$pixel_y)
    locations_spark = as.matrix(locations_spark)
    sparkX = sparkx(count_spark, locations_spark,numCores=opt$numCores)
    significant_gene_number = sum(sparkX$res_mtest$adjustedPval<=0.05)
    significant_gene_number = min(nrow(sparkX$res_mtest), max(significant_gene_number, 500)) # avoid too few genes
    SVGs = sparkX$res_mtest[order(sparkX$res_mtest$adjustedPval),][1:significant_gene_number, ]
    write.csv(SVGs, file.path(opt$result_dir, paste0(tp, '_SVG_sparkx.csv')), quote=F)
    SVG_mat = tp_obj@assays$SCT@scale.data[rownames(SVGs), ]
    SVG_pcs = prcomp(SVG_mat, center=T, scale=T)
    SVG_top_pcs = SVG_pcs$x[, 1:30]
    nearest_K = min(nrow(SVG_top_pcs), 100, significant_gene_number)
    knn.norm = FNN::get.knn(as.matrix(SVG_top_pcs), k=nearest_K)
    # knn.norm = FNN::get.knn(SVG_mat, k=30) # use original matrix
    knn.norm = data.frame(from = rep(1:nrow(knn.norm$nn.index),
               k=nearest_K), to = as.vector(knn.norm$nn.index), weight = 1/(1 + as.vector(knn.norm$nn.dist)))
    nw.norm = igraph::graph_from_data_frame(knn.norm, directed = FALSE)
    nw.norm = igraph::simplify(nw.norm)
    # slighly increase to get more clusters
    resolution = 1
    lc.norm = igraph::cluster_louvain(nw.norm, resolution=resolution) # use a larger resolution to derive multiple clusters
    while (length(unique(lc.norm$membership)) < 10) {
        resolution = resolution + 0.1
        lc.norm = igraph::cluster_louvain(nw.norm, resolution=resolution)
    }
    merged = bluster::mergeCommunities(nw.norm, lc.norm$membership, number=10)
    SVGs$cluster = merged
    write.csv(SVGs, file.path(opt$result_dir, paste0(tp, '_SVG_sparkx_clustered_louvain.csv')), quote=F)
}
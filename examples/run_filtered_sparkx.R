# Run R SPARK-X 1.1.1 on the annotation-free, filtered count exports.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
exchange <- normalizePath(args[[1]])
destination <- args[[2]]
suppressPackageStartupMessages({
  library(SPARK)
  library(Matrix)
})
stopifnot(as.character(packageVersion("SPARK")) == "1.1.1")
dir.create(destination, recursive = TRUE, showWarnings = FALSE)

for (timepoint in basename(list.dirs(exchange, full.names = TRUE, recursive = FALSE))) {
  source <- file.path(exchange, timepoint)
  counts <- as(readMM(file.path(source, "counts.mtx")), "CsparseMatrix")
  genes <- readLines(file.path(source, "genes.txt"))
  spots <- readLines(file.path(source, "cells.txt"))
  coordinates <- as.matrix(read.delim(file.path(source, "coordinates.tsv")))
  stopifnot(nrow(counts) == length(genes), ncol(counts) == length(spots),
            nrow(coordinates) == length(spots), ncol(coordinates) == 2,
            !anyDuplicated(genes), !anyDuplicated(spots))
  dimnames(counts) <- list(genes, spots)
  result <- SPARK::sparkx(counts, coordinates, numCores = 1)$res_mtest
  stopifnot(all(is.finite(result$adjustedPval)),
            all(is.finite(result$combinedPval)))
  result$gene <- rownames(result)
  result <- result[order(result$adjustedPval, result$combinedPval,
                         result$gene), , drop = FALSE]
  output <- file.path(destination, timepoint)
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  write.csv(result, file.path(output, "R_SPARKX_all_retained.csv"),
            row.names = FALSE)
}

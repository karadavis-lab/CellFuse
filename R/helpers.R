library(base)

#'  Normalization function
#'
#' @param target Matrix to normalize
#' @param reference Reference matrix
#' @return normalized matrix
#' @keywords internal
Normalize_function <- function(target, reference) {
  target <- as.matrix(target)
  reference <- as.matrix(reference)
  target_sig_Normalize_function <- target
  
  if (nrow(target) == nrow(reference)) {
    for (j in 1:ncol(target)) {
      target_sig_Normalize_function[, j] <- sort(reference[, j])[rank(target[, j])]
    }
  } else if (nrow(target) > nrow(reference)) {
    num_chunks <- ceiling(nrow(target) / nrow(reference))
    chunk_size <- nrow(reference)
    for (i in 1:num_chunks) {
      start_idx <- (i - 1) * chunk_size + 1
      end_idx <- min(i * chunk_size, nrow(target))
      target_chunk <- target[start_idx:end_idx, , drop = FALSE]
      for (j in 1:ncol(target)) {
        ref_sorted <- sort(reference[, j])
        target_rank <- rank(target_chunk[, j])
        target_sig_Normalize_function[start_idx:end_idx, j] <-
          ref_sorted[round((target_rank / max(target_rank)) * (length(ref_sorted) - 1)) + 1]
      }
    }
  } else {
    ref_subset <- reference[sample(nrow(reference), nrow(target), replace = FALSE),, drop = FALSE]
    for (j in 1:ncol(target)) {
      target_sig_Normalize_function[, j] <- sort(ref_subset[, j])[rank(target[, j])]
    }
  }
  
  target_sig_Normalize_function[target == 0] <- 1e-5
  return(target_sig_Normalize_function)
}

#' PCA-space Normalization
#'
#' @param ct1_rep1_i_sig Target matrix
#' @param file_ref_sig Reference matrix
#' @return normalized matrix in original (reference) space
#' @keywords internal
NormalizeData <- function(ct1_rep1_i_sig, file_ref_sig) {
  pca_ref0 <- prcomp(file_ref_sig, center = FALSE, scale. = FALSE)
  pca_ref0_rotation <- pca_ref0$rotation
  pca_ref0_pcs <- pca_ref0$x
  pca_tar1_pcs <- as.matrix(ct1_rep1_i_sig) %*% pca_ref0_rotation
  
  pca_tar1_pcs_qt <- pca_tar1_pcs
  for (i in 1:dim(pca_tar1_pcs)[2]) {
    pca_tar1_pcs_qt[, i] <- Normalize_function(pca_tar1_pcs[, i], pca_ref0_pcs[, i])
  }
  
  tar1PCA_QT <- pca_tar1_pcs_qt %*% t(pca_ref0_rotation)
  return(tar1PCA_QT)
}



#' Integrate query cell types  with reference cell types 
#'
#' This function normalizes query data to reference data using PCA-space
#' quantile normalization, performed separately for each cell type.
#'
#' @param data A data frame containing marker columns, a `Celltype` column, and a `batch` column
#' @param markers A character vector of column names corresponding to marker features
#' @param ref_batch Character value indicating the reference batch label
#' @param query_batch Character value indicating the query batch label
#'
#' @return A data frame with quantile-normalized marker values and original `Celltype` column
#' @export
IntegrateData <- function(ref_path, query_path,Celltype_col) {
  # Define fixed column name for cell type
  Celltype_col <- Celltype_col
  
  # Load and label datasets
  refdata <- read.csv(ref_path)
  refdata$batch <- "Reference"
  
  query_data <- read.csv(query_path)
  query_data$batch <- "QueryPreintegration"
  
  common_cols <- intersect(colnames(refdata),colnames(query_data))
  # Combine data
  data <- rbind(refdata[,common_cols], query_data[,common_cols])
  data$index <- seq_len(nrow(data))
  
  # Check for required celltype column
  if (!(Celltype_col %in% colnames(data))) {
    stop("Column 'Celltype_col' not found in input data.")
  }
  
  # Automatically identify marker columns
  exclude_cols <- c("batch", "index", Celltype_col)
  markers <- setdiff(colnames(data), exclude_cols)
  markers <- markers[sapply(data[markers], is.numeric)]
  
  normalized_list <- list()
  
  for (celltype in unique(data[[Celltype_col]])) {
    ref <- data[data[[Celltype_col]] == celltype & data$batch == "Reference", c(markers, "index")]
    target <- data[data[[Celltype_col]] == celltype & data$batch == "QueryPreintegration", c(markers, "index")]
    
    if (nrow(ref) > 2 && nrow(target) > 0) {
      corrected <- NormalizeData(target[, markers], ref[, markers])
      new_rows <- data.frame(corrected, Celltype = celltype, index = target$index)
      normalized_list[[celltype]] <- new_rows
    }
  }
  
  if (length(normalized_list) == 0) return(NULL)
  combined <- do.call(rbind, normalized_list)
  combined <- combined[order(combined$index), ]
  combined$index <- NULL
  
  return(combined)
}









#' Train CellFuse model using labeled reference data
#'
#'#' @description
#' Trains a supervised contrastive learning model on a labeled reference
#' proteomics dataset to learn a shared embedding
#' space for downstream cross-modality cell type prediction.
#'
#' @details
#' The function uses a Python backend (via \pkg{reticulate}) to implement
#' supervised contrastive learning. The reference dataset must contain:
#' \itemize{
#'   \item Rows corresponding to individual cells
#'   \item Columns corresponding to protein markers
#'   \item A column containing cell type labels (specified by \code{cluster_column})
#' }
#' The trained model, embeddings, and performance metrics are saved to
#' \code{save_path}, including a timestamped folder in
#' \code{Saved_model/}.
#'
#' @param dataset_name Name of the dataset (e.g., "Levine32")
#' @param data_dir Path to the reference dataset directory
#' @param save_path Path to directory where model and outputs will be saved
#' @param device Device to use ("cpu" or "cuda")
#' @param cluster_column Column name for cluster labels in metadata
#' @param lr Learning rate
#' @param margin Margin for contrastive loss
#' @param bs Batch size
#' @param epoch Number of training epochs
#' @param k K for KNN classifier
#' @param min_delta Minimum improvement for early stopping
#' @param patience Number of validation steps with no improvement before stopping
#' @param val_step Number of steps between validation checks
#' @param output_dim Dimension of embedding space
#' @param dropout_prob Dropout probability
#' @param activation_function Activation function (e.g., 'relu', 'leaky_relu')
#' @param alpha Weight for auxiliary loss or regularization term
#'
#' @return
#' Invisibly returns \code{NULL}. The function writes the following outputs to
#' \code{save_path}:
#' \itemize{
#'   \item Trained CellFuse model weights (saved under \code{Saved_model/})
#'   \item Training and validation performance metrics
#'   \item A PNG file containing training curves (loss and accuracy vs epoch)
#'   \item Optional intermediate artifacts required for downstream prediction
#' }
#'
#' @details
#' Model performance is visualized as training and validation loss and accuracy
#' across epochs and saved automatically to the specified output directory.
#' Early stopping is applied based on validation loss.

#' @examples
#' \dontrun{
#' TrainModel(
#'   dataset_name = "CyTOF",
#'   data_dir = "Reference_Data/",
#'   save_path = "Predicted_Data/",
#'   device = "cpu",
#'   cluster_column = "cluster.orig"
#' )
#' }
#'
#' @export
TrainModel <- function(dataset_name, data_dir, save_path, device, cluster_column,
                       lr, margin, bs, epoch, k, min_delta, patience, val_step,
                       output_dim, dropout_prob, activation_function, alpha) {

  reticulate::source_python(system.file("python", "python_funcs.py", package = "CellFuse"))

  py$trainModel(
    dataset_name = dataset_name,
    data_dir = data_dir,
    save_path = save_path,
    device = device,
    cluster_column = cluster_column,
    lr = lr,
    margin = margin,
    bs = bs,
    epoch = epoch,
    k = k,
    min_delta = min_delta,
    patience = patience,
    val_step = val_step,
    output_dim = output_dim,
    dropout_prob = dropout_prob,
    activation_function = activation_function,
    alpha = alpha
  )
}


#' Predict query cell types using a trained CellFuse model
#'
#'
#' @description
#' Projects query cells into a previously learned embedding space and
#' assigns predicted cell type labels using K-nearest neighbors.
#'
#' @details
#' This function loads a trained CellFuse model and applies it to
#' a query dataset
#'
#' The query dataset must:
#' \itemize{
#'   \item Contain the same markers used during training
#'   \item Have rows corresponding to individual cells
#' }
#'
#' Predicted labels and embeddings are saved to the output directory.
#'
#' @param dataset_name Character. Name of training dataset.
#' @param data_dir Character. Path to reference data directory.
#' @param test_data_dir Character. Path to query dataset directory.
#' @param test_data Character. Name of query dataset.
#' @param model_dir Character. Path to saved model directory.
#' @param model_date Character. Timestamp identifying trained model. YYYY-MM-DD
#' @param device Character. \code{"cpu"} or \code{"cuda"}.
#' @param cluster_column Character. Column containing cluster labels.
#' @param lr Numeric. Learning rate.
#' @param margin Numeric. Margin for contrastive loss.
#' @param bs Integer. Batch size.
#' @param epoch Integer. Number of epochs.
#' @param knn_k Integer. Number of neighbors for KNN classification.
#' @param output_dim Integer. Embedding dimension.
#' @param dropout_prob Numeric. Dropout probability.
#' @param activation_function Character. Activation function.
#'
#'
#' @return
#' Invisibly returns \code{NULL}. Prediction results are written to disk.
#'
#' @seealso \code{\link{TrainModel}}, \code{\link{IntegrateData}}
#'
#' @examples
#' \dontrun{
#' PredictCells(
#'   dataset_name = "CyTOF",
#'   data_dir = "Reference_Data/",
#'   test_data_dir = "Query_Data/",
#'   model_dir = "Predicted_Data/Saved_model"
#' )
#' }
#'
#' @export
PredictCells <- function(dataset_name, data_dir, test_data_dir, test_data,
                             model_dir, model_date, device, cluster_column,
                             lr, margin, bs, epoch,
                             knn_k, output_dim,
                             dropout_prob, activation_function) {

  reticulate::source_python(system.file("python", "python_funcs.py", package = "CellFuse"))

  py$predict_cells(
    dataset_name = dataset_name,
    data_dir = data_dir,
    test_data_dir = test_data_dir,
    test_data = test_data,
    model_dir = model_dir,
    model_date = model_date,
    device = device,
    cluster_column = cluster_column,
    lr = lr,
    margin = margin,
    bs = bs,
    epoch = epoch,
    knn_k = knn_k,
    output_dim = output_dim,
    dropout_prob = dropout_prob,
    activation_function = activation_function
  )
}


#' Runtime calculation for PredictCells() function
#'
#'
#'@description
#' Evaluates the computational runtime of the CellFuse query prediction step.
#'
#' @details
#' Useful for benchmarking CPU/GPU performance and scalability analysis.
#'
#' @inheritParams PredictCells
#'
#' @return
#' Numeric value indicating runtime (in seconds).
#'
#' @param dataset_name Name of the training dataset
#' @param data_dir Path to reference data directory
#' @param test_data_dir Path to query/test data directory
#' @param test_data Name of the test dataset
#' @param model_dir Path to saved model directory
#' @param model_date Date when model was created (this helps CellFuse to load correct model)
#' @param device 'cpu' or 'cuda'
#' @param cluster_column Column name containing cluster labels
#' @param lr Learning rate
#' @param margin Margin value for contrastive loss
#' @param bs Batch size
#' @param epoch Number of epochs
#' @param knn_k K for KNN classifier
#' @param output_dim Size of embedding output
#' @param dropout_prob Dropout probability
#' @param activation_function Activation function name
#'
#' @export
predict_cells_time <- function(dataset_name, data_dir, test_data_dir, test_data,
                             model_dir, model_date, device, cluster_column,
                             lr, margin, bs, epoch,
                             knn_k, output_dim,
                             dropout_prob, activation_function) {

  reticulate::source_python(system.file("python", "python_funcs.py", package = "CellFuse"))

  py$predict_cells_time(
    dataset_name = dataset_name,
    data_dir = data_dir,
    test_data_dir = test_data_dir,
    test_data = test_data,
    model_dir = model_dir,
    model_date = model_date,
    device = device,
    cluster_column = cluster_column,
    lr = lr,
    margin = margin,
    bs = bs,
    epoch = epoch,
    knn_k = knn_k,
    output_dim = output_dim,
    dropout_prob = dropout_prob,
    activation_function = activation_function
  )
}

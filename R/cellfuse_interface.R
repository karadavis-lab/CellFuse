


#' Train CellFuse model using reference cell types
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


#' Predict query cell types using trained CellFuse model
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

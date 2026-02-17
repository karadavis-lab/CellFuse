# Train CellFuse model using labeled reference data

\#' @description Trains a supervised contrastive learning model on a
labeled reference proteomics dataset to learn a shared embedding space
for downstream cross-modality cell type prediction.

## Usage

``` r
TrainModel(
  dataset_name,
  data_dir,
  save_path,
  device,
  cluster_column,
  lr,
  margin,
  bs,
  epoch,
  k,
  min_delta,
  patience,
  val_step,
  output_dim,
  dropout_prob,
  activation_function,
  alpha
)
```

## Arguments

- dataset_name:

  Name of the dataset (e.g., "Levine32")

- data_dir:

  Path to the reference dataset directory

- save_path:

  Path to directory where model and outputs will be saved

- device:

  Device to use ("cpu" or "cuda")

- cluster_column:

  Column name for cluster labels in metadata

- lr:

  Learning rate

- margin:

  Margin for contrastive loss

- bs:

  Batch size

- epoch:

  Number of training epochs

- k:

  K for KNN classifier

- min_delta:

  Minimum improvement for early stopping

- patience:

  Number of validation steps with no improvement before stopping

- val_step:

  Number of steps between validation checks

- output_dim:

  Dimension of embedding space

- dropout_prob:

  Dropout probability

- activation_function:

  Activation function (e.g., 'relu', 'leaky_relu')

- alpha:

  Weight for auxiliary loss or regularization term

## Value

Invisibly returns `NULL`. The function writes the following outputs to
`save_path`:

- Trained CellFuse model weights (saved under `Saved_model/`)

- Training and validation performance metrics

- A PNG file containing training curves (loss and accuracy vs epoch)

- Optional intermediate artifacts required for downstream prediction

## Details

The function uses a Python backend (via reticulate) to implement
supervised contrastive learning. The reference dataset must contain:

- Rows corresponding to individual cells

- Columns corresponding to protein markers

- A column containing cell type labels (specified by `cluster_column`)

The trained model, embeddings, and performance metrics are saved to
`save_path`, including a timestamped folder in `Saved_model/`.

Model performance is visualized as training and validation loss and
accuracy across epochs and saved automatically to the specified output
directory. Early stopping is applied based on validation loss.

## Examples

``` r
if (FALSE) { # \dontrun{
TrainModel(
  dataset_name = "CyTOF",
  data_dir = "Reference_Data/",
  save_path = "Predicted_Data/",
  device = "cpu",
  cluster_column = "cluster.orig"
)
} # }
```

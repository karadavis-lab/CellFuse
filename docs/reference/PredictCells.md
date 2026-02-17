# Predict query cell types using a trained CellFuse model

Projects query cells into a previously learned embedding space and
assigns predicted cell type labels using K-nearest neighbors.

## Usage

``` r
PredictCells(
  dataset_name,
  data_dir,
  test_data_dir,
  test_data,
  model_dir,
  model_date,
  device,
  cluster_column,
  lr,
  margin,
  bs,
  epoch,
  knn_k,
  output_dim,
  dropout_prob,
  activation_function
)
```

## Arguments

- dataset_name:

  Character. Name of training dataset.

- data_dir:

  Character. Path to reference data directory.

- test_data_dir:

  Character. Path to query dataset directory.

- test_data:

  Character. Name of query dataset.

- model_dir:

  Character. Path to saved model directory.

- model_date:

  Character. Timestamp identifying trained model. YYYY-MM-DD

- device:

  Character. `"cpu"` or `"cuda"`.

- cluster_column:

  Character. Column containing cluster labels.

- lr:

  Numeric. Learning rate.

- margin:

  Numeric. Margin for contrastive loss.

- bs:

  Integer. Batch size.

- epoch:

  Integer. Number of epochs.

- knn_k:

  Integer. Number of neighbors for KNN classification.

- output_dim:

  Integer. Embedding dimension.

- dropout_prob:

  Numeric. Dropout probability.

- activation_function:

  Character. Activation function.

## Value

Invisibly returns `NULL`. Prediction results are written to disk.

## Details

This function loads a trained CellFuse model and applies it to a query
dataset

The query dataset must:

- Contain the same markers used during training

- Have rows corresponding to individual cells

Predicted labels and embeddings are saved to the output directory.

## See also

[`TrainModel`](TrainModel.md), [`IntegrateData`](IntegrateData.md)

## Examples

``` r
if (FALSE) { # \dontrun{
PredictCells(
  dataset_name = "CyTOF",
  data_dir = "Reference_Data/",
  test_data_dir = "Query_Data/",
  model_dir = "Predicted_Data/Saved_model"
)
} # }
```

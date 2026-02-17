# Runtime calculation for PredictCells() function

Evaluates the computational runtime of the CellFuse query prediction
step.

## Usage

``` r
predict_cells_time(
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

  Name of the training dataset

- data_dir:

  Path to reference data directory

- test_data_dir:

  Path to query/test data directory

- test_data:

  Name of the test dataset

- model_dir:

  Path to saved model directory

- model_date:

  Date when model was created (this helps CellFuse to load correct
  model)

- device:

  'cpu' or 'cuda'

- cluster_column:

  Column name containing cluster labels

- lr:

  Learning rate

- margin:

  Margin value for contrastive loss

- bs:

  Batch size

- epoch:

  Number of epochs

- knn_k:

  K for KNN classifier

- output_dim:

  Size of embedding output

- dropout_prob:

  Dropout probability

- activation_function:

  Activation function name

## Value

Numeric value indicating runtime (in seconds).

## Details

Useful for benchmarking CPU/GPU performance and scalability analysis.

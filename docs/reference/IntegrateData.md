# Integrate query cell types with reference cell types

This function normalizes query data to reference data using PCA-space
quantile normalization, performed separately for each cell type.

## Usage

``` r
IntegrateData(ref_path, query_path, Celltype_col)
```

## Arguments

- ref_path:

  Path to reference CSV file.

- query_path:

  Path to predicted query CSV file.

- Celltype_col:

  Character. Column containing cell-type labels.

## Value

A data.frame containing normalized query marker values.

## Details

Normalization is performed separately per cell type. Only numeric marker
columns are processed.

## Examples

``` r
if (FALSE) { # \dontrun{
corrected <- IntegrateData(
  ref_path = "Reference_Data/CyTOF_train.csv",
  query_path = "Query_Data/CITEseq_test.csv",
  Celltype_col = "cluster.orig"
)
} # }

```

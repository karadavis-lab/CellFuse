# CellFuse
<p align="left">
<img src="figures/logo.png" height="140"/>
</p>

**CellFuse** is an R package for multimodal single-cell and spatial proteomics data integration using deep contrastive learning. Single-cell and spatial proteomic technologies capture complementary biological information, yet no single platform can measure all modalities within the same cell. Most existing integration methods such as Seurat and Harmony are optimized for transcriptomic data and rely on a large set of shared, strongly linked features, an assumption that often fails for low-dimensional proteomic modalities. We present CellFuse, a deep learning-based, modality-agnostic integration framework designed specifically for settings with limited feature overlap. CellFuse leverages supervised contrastive learning to learn a shared embedding space, enabling accurate cell type prediction and seamless integration across modalities and experimental conditions. 

This work has been led by Abhishek Koladiya from [Kara Davis Lab](https://kldavislab.org/) @Stanford

------------------------------------------------------------------------
<img src="figures/Figure1_v2.png" width="800" height="1200"/>

### Python Environment Setup (Required)

Before using `CellFuse`, you must configure Python with required packages.
```shell
conda create -n myenv python=3.10 -y
conda activate myenv

# Install Python dependencies
conda install pytorch pandas scikit-learn matplotlib seaborn
```
R Script to Use the Environment

    # Load and install R dependencies
    required_packages <- c("reticulate", "remotes")
    installed <- required_packages %in% rownames(installed.packages())
    if (any(!installed)) {
      install.packages(required_packages[!installed])
    }
    
    # Use the conda environment from R
    library(reticulate)
    
    use_condaenv("myenv", required = TRUE)
    
    # Confirm Python setup
    py_config()
    
    ## For reviwers: move to the folder where you have zip version of package ##
    devtools::install_local("CellFuse-main.zip")
    
    ## when the package will be available publically ###
    ##devtools::install("AbhivKoladiya/CellFuse")
    
    library(CellFuse)

CellFuse requires data in following formate

    # CellFuseProject/
    # ├── Reference_Data/  (e.g., Reference CyTOF or CITE-seq, rows= cells, columns =markers)
    # ├── Query_Data/  (e.g. Query datasets CODEX, IMC,  CITE-seq, rows= cells, columns =markers)
    # ├── Predicted_Data/  (Output folder where CellFuse will save predicted labels)
    # ├── Predicted_Data/Saved_model  (Folder for saving trained CellFuse models)
      
      
    ### Lets create these folders #####
      
    dir.create("Reference_Data", showWarnings = FALSE)
    dir.create("Query_Data", showWarnings = FALSE)
    dir.create("Predicted_Data", showWarnings = FALSE)
    dir.create("Predicted_Data/Saved_model", showWarnings = FALSE)


### Data Preparation

    ## first split your reference data in 70/30 %
    RefenenceData <- read.csv("Reference_Data/CyTOF.csv")

    trainIndex <- createDataPartition(RefenenceData$cluster.orig, p = 0.7, list = FALSE)
    train_data <- RefenenceData[trainIndex, ]
    validation_data <- RefenenceData[-trainIndex, ]

    # Save the datasets
    setwd("Reference_Data/")
    write_csv(train_data[,c(common_cols)], "CyTOF_train.csv")
    write_csv(validation_data[,c(common_cols)], "CyTOF_val.csv")


### Stage 1 (Model Training): Train the CellFuse model using Reference cell types

    TrainModel(dataset_name = "CyTOF",
      data_dir = "path/to/reference_data/",save_path = "path/to/save_model/",
        device = "cpu",cluster_column = "cluster.orig", 
        lr=as.numeric(0.0009), margin=as.numeric(0.8), bs=as.integer(256), epoch=as.integer(50),
        k=as.integer(5), min_delta=as.numeric(0.01), patience=as.integer(5), val_step=as.integer(5),
        output_dim=as.integer(8), dropout_prob=as.numeric(0.7),
        activation_function='leaky_relu',alpha=as.numeric(0.01))

### Stage 2 (Cell type Prediction): Use trained CellFuse model to predict Query cell types

    PredictCells(dataset_name = "CyTOF",data_dir = "path/to/reference_data/",
      test_data_dir = "path/to/query_data/",
      test_data = "CITEseq",model_dir = "path/to/save_model/Saved_model",
      model_date="YYYY-MM-DD",device="cpu",cluster_column='cluster.orig',
      lr=as.numeric(0.001),margin=0.5,bs=as.integer(256), epoch=as.integer(50),
      knn_k=as.integer(5),output_dim=as.integer(8),
      dropout_prob=as.numeric(0.5),activation_function='leaky_relu')


### Stage 3 (Data Integration): Integrate query cell types with reference cell types

    corrected_data <- IntegrateData(
      ref_path="Reference_Data/CyTOF_train.csv",query_path="Query_Data/CITEseq_test.csv",
      Celltype_col="cluster.orig")


## Vignette

Check out this [vignette](doc/CellFuseQuickstart.html) for integration of CyTOF and CITESeq data.      

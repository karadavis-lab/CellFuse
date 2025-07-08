## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------

# Set Python path manually
Sys.setenv(RETICULATE_PYTHON = "/usr/local/Caskroom/miniforge/base/envs/r-reticulate-env/bin/python")

library(reticulate)
# Create a virtualenv (if needed)
virtualenv_create("cellfuse_env",python = Sys.getenv("RETICULATE_PYTHON"))
use_virtualenv("cellfuse_env", required = TRUE)

# Install required packages
virtualenv_install("cellfuse_env", packages = c("torch", "pandas", "scikit-learn", "matplotlib", "seaborn"))
py_config()


## ----TrainModel, message=FALSE, warning=FALSE, fig.align='center'-------------
library(CellFuse)
#### Train model using Levine32 data ####
TrainModel(dataset_name="Levine32",
              data_dir="/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Figure_2/BM/Reference_Data/", 
              save_path="/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Figure_2/BM/Predicted_Data/",
              device="cpu",cluster_column = "cluster.orig", 
              lr=as.numeric(0.0009), margin=as.numeric(0.8), bs=as.integer(256), epoch=as.integer(50),
              k=as.integer(5), min_delta=as.numeric(0.01), patience=as.integer(5), val_step=as.integer(5),
              output_dim=as.integer(8), dropout_prob=as.numeric(0.7),
              activation_function='leaky_relu',alpha=as.numeric(0.01))

## ----training outpur----------------------------------------------------------
knitr::include_graphics("/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Package/CellFuse/doc/Training_output.png")

## ----PredictCells, message=FALSE, warning=FALSE, fig.align='center'-----------
### Predict CITEseq data ###
setwd("/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Figure_2/BM/Predicted_Data/")
PredictCells(dataset_name="Levine32",
                 data_dir="/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Figure_2/BM/Reference_Data",
                 test_data_dir="/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Figure_2/BM/Query_Data/", 
                 test_data="CITEseq",
                 model_dir="/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Figure_2/BM/Predicted_Data/Saved_model",
                 device="cpu",cluster_column='cluster.orig',
                 lr=as.numeric(0.001),margin=0.5,bs=as.integer(256), epoch=as.integer(50),
                 knn_k=as.integer(5),output_dim=as.integer(8),
                 dropout_prob=as.numeric(0.5),activation_function='leaky_relu')

## ----IntegrateData,message=FALSE, warning=FALSE, fig.align='center'-----------
# Integrate query data with reference
correctd_data= IntegrateData(
  ref_path="/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Figure_2/BM/Reference_Data/Levine32_train.csv",
  query_path="/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Figure_2/BM/Query_Data/CITEseq_test.csv",
  Celltype_col="cluster.orig"
)

## ----post integration---------------------------------------------------------
## Merge pre-integrated data ##
ref= read.csv("/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Figure_2/BM/Reference_Data/Levine32_train.csv")
query=read.csv("/Users/koladiya/Documents/Postdoc/Papers/CellFuse/Figure_2/BM/Query_Data/CITEseq_test.csv")
Preintegration= rbind.data.frame(cbind(ref,batch=rep("Reference (CyTOF)",times=dim(ref)[1])),
                           cbind(query[,1:12],cluster.orig=correctd_data$Celltype,
                                 batch=rep("Query (CITEseq)",times=dim(query)[1])))

## merge post-integration ##
PostIntegration= rbind.data.frame(cbind(ref,batch=rep("Reference (CyTOF)",times=dim(ref)[1])),
                           cbind(correctd_data[,1:12],cluster.orig=correctd_data$Celltype,
                                 batch=rep("Query (CITEseq)",times=dim(correctd_data)[1])))

## Run tSNE ###
library(Rtsne)

e_pre<- Rtsne(Preintegration[,1:12])
colnames(e_pre$Y)= c("tSNE1","tSNE2")
e_post<- Rtsne(PostIntegration[,1:12])
colnames(e_post$Y)= c("tSNE1","tSNE2")

Preintegration=cbind.data.frame(Preintegration,e_pre$Y)
PostIntegration=cbind.data.frame(PostIntegration,e_post$Y)

## ----tsne-visualization, message=FALSE, warning=FALSE,fig.width=10, fig.height=8----
library(ggplot2)
library(patchwork)

# Define plots
P1 <- ggplot(Preintegration, aes(x = tSNE1, y = tSNE2, colour = cluster.orig)) +
  geom_point(size = 0.01) + theme_classic() + ggtitle("Preintegration: Celltypes")

P2 <- ggplot(Preintegration, aes(x = tSNE1, y = tSNE2, colour = batch)) +
  geom_point(size = 0.01) + theme_classic() + ggtitle("Preintegration: Modality")

P3 <- ggplot(PostIntegration, aes(x = tSNE1, y = tSNE2, colour = cluster.orig)) +
  geom_point(size = 0.01) + theme_classic() + ggtitle("CellFuse Integration: Celltypes")

P4 <- ggplot(PostIntegration, aes(x = tSNE1, y = tSNE2, colour = batch)) +
  geom_point(size = 0.01) + theme_classic() + ggtitle("CellFuse Integration: Modality")

# Combine and show plots
(P1 | P2) / (P3 | P4)


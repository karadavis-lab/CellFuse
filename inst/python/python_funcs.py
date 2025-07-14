import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import precision_score, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import time


# Define the Contrastive Loss class
class ContrastiveLoss(nn.Module):
  def __init__(self, margin=1.0):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin
  
  def forward(self, x0, x1, y):
    diff = x0 - x1
    dist_sq = torch.sum(torch.pow(diff, 2), -1)
    dist = torch.sqrt(dist_sq)
    mdist = self.margin - dist
    dist = torch.clamp(mdist, min=0.0)
    loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
    loss = torch.sum(loss) / 2.0 / x0.size()[0]
    return loss

class CellClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5, activation_function='relu',alpha=0.01):
        super(CellClassifier, self).__init__()
        # Define the layers and other components based on the parameters passed
        self.fc1 = nn.Linear(input_dim, 512) ##256
        self.fc2 = nn.Linear(512, 256) #256
        self.fc3 = nn.Linear(256, 128) #256
        self.fc4 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        if activation_function == 'relu':
            self.activation = nn.ReLU()
        elif activation_function == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=alpha)
        elif activation_function == 'elu':
            self.activation = nn.ELU(alpha=alpha)  # You can also apply alpha to ELU if needed
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
       
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.softmax(self.fc4(x))
        #x = F.normalize(self.fc4(x), p=2, dim=1)  # Normalize embeddings only for cosine
        return x

# Function to transform raw representation to embedded representation
def project(encoder, X, device):
    with torch.no_grad():
        encoder.eval()
        if torch.is_tensor(X):
            X = X.to(device=device, dtype=torch.float32)
        else:
            X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
        emb_X = encoder(X).detach().cpu().numpy()
    return emb_X

# Define the Early Stopper class
class EpochController:
    def __init__(self, patience=3, min_delta=0.01):
        """
        Initialize the EpochController.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def StopEarly(self, validation_loss):
        """
        Check if training should stop early.

        Args:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        # Significant improvement threshold
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0  # Reset counter if there's improvement
            print(f"Improvement: Val Loss = {validation_loss:.6f}, Best Loss = {self.best_loss:.6f}")
        else:
            # Increment counter if no significant improvement
            self.counter += 1
            print(f"No significant improvement: Counter = {self.counter}, Val Loss = {validation_loss:.6f}, Best Loss = {self.best_loss:.6f}")

            # Check if patience has been exceeded
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                return True

        return False


class Dataset(Dataset):

    def __init__(self, markers, labels):
        self.markers = markers
        self.labels = self.labels = labels.to_numpy() if isinstance(labels, pd.Series) else labels#labels
        self.one_hot_labels = F.one_hot(torch.from_numpy(labels))
        self.n_samples = markers.shape[0]
        self.n_markers = markers.shape[1]
        self.representative_tensor = None
        self.representative_labels = None
        self.device = torch.device("cpu")

    def __len__(self):
        return self.n_samples


    def __getitem__(self, idx):
        marker = torch.from_numpy(self.markers[idx, :])
        label = self.one_hot_labels[idx]

        marker = marker.to(device=self.device, dtype=torch.float32)
        label = label.to(device=self.device, dtype=torch.float32)
        return marker, label


    def get_marker_profile_dim(self):
        return self.markers.shape[1]


    def to(self, device):
        self.device = device


    def create_representative_tensor(self, markers, labels, n_rep_mat):

        #n_labels = len(np.unique(labels))
        n_labels = len(np.unique(labels))  # Get number of unique labels
        representative_tensor = np.zeros((n_rep_mat, n_labels, markers.shape[1]))
        representative_labels = np.zeros((n_rep_mat, n_labels), np.int32)

        #print(f"Epoch {epoch}: Creating representative tensor")
        print(f"Training data: Number of Cells and Markers: {markers.shape}")

        for i_idx in range(min(n_rep_mat, n_labels)):
        #for i_idx in range(n_rep_mat):
            representative_mat = representative_tensor[i_idx, :, :]
            for i in np.unique(labels):
                repr_idx = np.argwhere(labels == i).flatten()
                repr_vector = markers[repr_idx, :].mean(axis=0)
                representative_mat[i, :] = repr_vector
            representative_labels[i_idx, :] = range(n_labels)
        representative_tensor = torch.from_numpy(representative_tensor)
        self.representative_labels = representative_labels
        self.representative_tensor = representative_tensor.to(
            device=self.device, dtype=torch.float32)


class ContrastiveLossMahalanobis(nn.Module):
    def __init__(self, margin=1.0, covariance_matrix=None):
        super(ContrastiveLossMahalanobis, self).__init__()
        self.margin = margin
        if covariance_matrix is not None:
            self.cov_inv = torch.linalg.inv(covariance_matrix)
        else:
            self.cov_inv = None

    def forward(self, x0, x1, y):
        diff = x0 - x1  # Difference vector

        # Compute Mahalanobis distance
        if self.cov_inv is None:
            # Compute inverse covariance matrix if not provided
            covariance_matrix = torch.cov(torch.cat((x0, x1), dim=0).T)
            self.cov_inv = torch.linalg.inv(covariance_matrix + 1e-6 * torch.eye(covariance_matrix.size(0)).to(x0.device))  # Add small value for stability

        dist_sq = torch.sum(diff @ self.cov_inv * diff, dim=-1)  # Mahalanobis squared distance
        dist = torch.sqrt(dist_sq + 1e-6)  # Add epsilon for numerical stability

        # Compute margin and loss
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.mean(loss)  # Average over the batch
        return loss

def trainModel(dataset_name, data_dir, save_path, device, lr, margin, bs, epoch, k, min_delta, patience,val_step, output_dim, dropout_prob, activation_function, alpha, cluster_column='cluster.orig'):
    train_path = f"{data_dir}/{dataset_name}_train.csv"
    train_data = pd.read_csv(train_path)

    column_names_concatenated = ", ".join(train_data.columns)
    print(column_names_concatenated)

    val_path = f"{data_dir}/{dataset_name}_val.csv"
    val_data = pd.read_csv(val_path)

    train_markers = train_data.loc[:, train_data.columns != cluster_column].to_numpy()
    train_labels = train_data[cluster_column]
    train_source_labels_int = train_labels.rank(method="dense", ascending=True).astype(int) - 1

    train_labels = train_labels.values
    train_source_labels_int = train_source_labels_int.values

    label_map = dict()
    for k in range(train_source_labels_int.max() + 1):
        label_map[train_labels[train_source_labels_int == k][0]] = k

    val_markers = val_data.loc[:, val_data.columns != cluster_column].to_numpy()
    val_labels = val_data[cluster_column]
    val_source_labels_int = val_labels.map(label_map).values


    # Scale the data
    scaler = StandardScaler()
    train_markers = scaler.fit_transform(train_markers)
    val_markers = scaler.transform(val_markers)

    # Compute class weights
    #class_weights = compute_class_weight('balanced', classes=np.unique(train_source_labels_int), y=train_source_labels_int)
    #class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


    train_data = Dataset(train_markers, train_source_labels_int)
    #valid_data = Dataset(val_markers, val_source_labels_int.to_numpy())
    valid_data = Dataset(val_markers, val_source_labels_int)

    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=False)

    train_data.create_representative_tensor(train_data.markers, train_data.labels, epoch)
    representative_tensor = train_data.representative_tensor
    train_data.to(device)
    valid_data.to(device)
    marker_dim = train_data.get_marker_profile_dim()
    embedding_dim = marker_dim  # Define embedding_dim explicitly

    #### Model #######
    model = CellClassifier(marker_dim, output_dim, dropout_prob, activation_function, alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Compute covariance matrix
    with torch.no_grad():
        initial_embeddings = []
        for input_x, _ in train_loader:
            input_x = input_x.to(device)
            initial_embeddings.append(model(input_x).cpu().numpy())
        initial_embeddings = np.vstack(initial_embeddings)
        cov_matrix_np = np.cov(initial_embeddings.T)
        cov_matrix_np += np.eye(cov_matrix_np.shape[0]) * 1e-6  # Add small value for stability
        cov_matrix = torch.from_numpy(cov_matrix_np).float().to(device)

    criterion = ContrastiveLossMahalanobis(margin=margin, covariance_matrix=cov_matrix)
    #criterion = ContrastiveLoss(margin=margin)

    Stopper = EpochController(patience=patience, min_delta=min_delta)
    train_losses, val_losses = [],[]
    #val_losses = []
    train_accuracies,val_accuracies = [],[]
    #val_accuracies = []
    for ep in range(epoch):
        total_train_loss, total_val_loss = 0, 0
        rep_mat = representative_tensor[ep, :, :].to(device)

        # Training loop
        model.train()
        for i, (input_x, label) in enumerate(train_loader):
            #model.train()
            input_x, label = input_x.to(device), label.to(device)
            embed_x = model(input_x)
            embed_rep_mat = model(rep_mat)
            embed_x = embed_x[:, None, :]

            # Compute loss and update model parameters
            loss = criterion(embed_x, embed_rep_mat, label)
            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

         # Track average training loss
        total_train_loss /= len(train_loader)
        train_losses.append(total_train_loss)

        # Perform validation at every epoch
        model.eval()
        with torch.no_grad():
            for i, (input_x, label) in enumerate(valid_loader):
                #model.eval()
                input_x, label = input_x.to(device), label.to(device)
                embed_x = model(input_x)
                embed_rep_mat = model(rep_mat)
                embed_x = embed_x[:, None, :]
                loss = criterion(embed_x, embed_rep_mat, label)
                total_val_loss += loss.item()
            total_val_loss /= len(valid_loader)
            val_losses.append(total_val_loss)

            # Step the scheduler
            scheduler.step(total_val_loss)
            print(f"Epoch: {ep}, Learning Rate: {optimizer.param_groups[0]['lr']}")
            if Stopper.StopEarly(total_val_loss):
              print(f"Stopping Early||epoch:{ep}, training loss:{total_train_loss:.4f}, validation loss:{total_val_loss:.4f}")
              break

        # Evaluate accuracy on training data using KNN
        knn = KNeighborsClassifier(n_neighbors=k)
        train_embeddings = []
        for i, (input_x, _) in enumerate(train_loader):
            model.eval()
            input_x = input_x.to(device)
            embed_x = model(input_x).cpu().detach().numpy()
            train_embeddings.append(embed_x)
        train_embeddings = np.vstack(train_embeddings)
        train_embeddings = np.nan_to_num(train_embeddings)

        knn.fit(train_embeddings, train_source_labels_int)

        # Calculate accuracy on training set
        train_preds = knn.predict(train_embeddings)
        train_accuracy = accuracy_score(train_source_labels_int, train_preds)
        train_accuracies.append(train_accuracy)

        # Evaluate accuracy on validation data
        val_embeddings = []
        for i, (input_x, _) in enumerate(valid_loader):
            model.eval()
            input_x = input_x.to(device)
            embed_x = model(input_x).cpu().detach().numpy()
            val_embeddings.append(embed_x)
        val_embeddings = np.vstack(val_embeddings)
        val_preds = knn.predict(val_embeddings)
        val_accuracy = accuracy_score(val_source_labels_int, val_preds)
        val_accuracies.append(val_accuracy)

        print(f"Epoch:{ep}, training loss:{total_train_loss:.4f}, validation loss:{total_val_loss:.4f}, "
              f"training accuracy:{train_accuracy:.4f}, validation accuracy:{val_accuracy:.4f}")
    train_date = datetime.datetime.now().strftime("%Y-%m-%d")
    model_info = f"{dataset_name}_{train_date}"
    torch.save(model.state_dict(), f"{save_path}/Saved_model/{model_info}.pt")
    
    
    # Plot loss and accuracy vs epoch
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epoch')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

def predict_cells(dataset_name, data_dir, test_data_dir, test_data, model_dir, model_date, model,device, lr, margin, bs, epoch, knn_k, output_dim, dropout_prob, activation_function, cluster_column='cluster.orig', output_dir=None):
    # Load training data
    train_path = f"{data_dir}/{dataset_name}_train.csv"
    cell_train = pd.read_csv(train_path)

    #train_markers = cell_train.loc[:, cell_train.columns != cluster_column].to_numpy()
    train_markers =cell_train.select_dtypes(include=[np.number]).to_numpy()
    train_labels = cell_train[cluster_column]


    train_source_labels_int = train_labels.rank(method="dense", ascending=True).astype(int) - 1
    label_map = {label: idx for idx, label in enumerate(np.unique(train_labels))}
    reverse_label_map = {v: k for k, v in label_map.items()}

    # Load test data
    test_path = f"{test_data_dir}/{test_data}_test.csv"
    cell_test = pd.read_csv(test_path)
    #test_markers = cell_test.loc[:, cell_test.columns != cluster_column].to_numpy()
    test_markers = cell_test.select_dtypes(include=[np.number]).to_numpy()

    # Scale the data
    train_scaler = StandardScaler()
    train_markers = train_scaler.fit_transform(train_markers)

    test_scaler = StandardScaler()
    test_markers = test_scaler.fit_transform(test_markers)

    # Load model
    model_info = f"{dataset_name}_{model_date}"
    model_path = f"{model_dir}/{model_info}.pt"
    print(f"Loading the model: {model_path}")

    model = CellClassifier(train_markers.shape[1], output_dim, dropout_prob, activation_function).to(device)
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError:
        model = torch.load(model_path).to(device)
    model.eval()

    # Project embeddings
    print("Projecting embeddings...")
    emb_X_train = project(model, train_markers, device)
    emb_X_test = project(model, test_markers, device)

    # Train KNN on projected embeddings
    print("Fitting KNN model...")
    knn = KNeighborsClassifier(n_neighbors=knn_k)
    knn.fit(emb_X_train, train_source_labels_int)
    y_predict_test = knn.predict(emb_X_test)
    y_predict_proba_test = knn.predict_proba(emb_X_test)

    # Assign predictions and probabilities
    cell_test["CellFuse_Pred"] = [reverse_label_map[pred] for pred in y_predict_test]
    cell_test["Prediction_Probability"] = y_predict_proba_test.max(axis=1)

    for i, class_name in enumerate(knn.classes_):
        cell_test[f"probability_{reverse_label_map[class_name]}"] = y_predict_proba_test[:, i]

    # Save predictions
    if output_dir is None:
        output_dir = "."
    output_file = os.path.join(output_dir, f"Pred_{test_data}_Ref_{model_info}.csv")
    cell_test.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}.")

def predict_cells_time(dataset_name, data_dir, test_data_dir, test_data, model_dir,model_date, device, lr, margin, bs, epoch, knn_k, output_dim, dropout_prob, activation_function, cluster_column='cluster.orig', output_dir=None):
    # Load training data
    train_path = f"{data_dir}/{dataset_name}_train.csv"
    cell_train = pd.read_csv(train_path)
    train_markers = cell_train.select_dtypes(include=[np.number]).to_numpy()
    train_labels = cell_train[cluster_column]

    train_source_labels_int = train_labels.rank(method="dense", ascending=True).astype(int) - 1
    label_map = {label: idx for idx, label in enumerate(np.unique(train_labels))}
    reverse_label_map = {v: k for k, v in label_map.items()}

    # Load test data
    test_path = f"{test_data_dir}/{test_data}_test.csv"
    cell_test = pd.read_csv(test_path)
    test_markers = cell_test.select_dtypes(include=[np.number]).to_numpy()

    # Scale the data
    train_scaler = StandardScaler()
    train_markers = train_scaler.fit_transform(train_markers)

    test_scaler = StandardScaler()
    test_markers = test_scaler.fit_transform(test_markers)

    # === START timing here
    time_start = time.time()

    # Load model
    model_info = f"{dataset_name}_{model_date}"
    model_path = f"{model_dir}/{model_info}.pt"
    print(f"Loading the model: {model_path}")

    model = CellClassifier(train_markers.shape[1], output_dim, dropout_prob, activation_function).to(device)
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError:
        model = torch.load(model_path).to(device)
    model.eval()

    # Project embeddings
    print("Projecting embeddings...")
    emb_X_train = project(model, train_markers, device)
    emb_X_test = project(model, test_markers, device)

    # Train KNN and predict
    print("Fitting KNN model...")
    knn = KNeighborsClassifier(n_neighbors=knn_k)
    knn.fit(emb_X_train, train_source_labels_int)
    y_predict_test = knn.predict(emb_X_test)
    y_predict_proba_test = knn.predict_proba(emb_X_test)

    # Assign predictions
    cell_test["prediction"] = [reverse_label_map[pred] for pred in y_predict_test]
    cell_test["Prediction_Probability"] = y_predict_proba_test.max(axis=1)
    for i, class_name in enumerate(knn.classes_):
        cell_test[f"probability_{reverse_label_map[class_name]}"] = y_predict_proba_test[:, i]

    # === ⏱️ END timing here
    time_end = time.time()
    total_seconds = round(time_end - time_start, 2)

    # Save timing log
    timing_df = pd.DataFrame([{
        "Dataset": dataset_name,
        "TestData": test_data,
        "StartTime": time_start,
        "EndTime": time_end,
        "InferenceTime_sec": total_seconds
    }])
    timing_log_path = os.path.join(output_dir or ".", f"prediction_timing_{model_info}.csv")
    timing_df.to_csv(timing_log_path, index=False)

    # Save predictions
    if output_dir is None:
        output_dir = "."
    output_file = os.path.join(output_dir, f"Pred_{test_data}_Ref_{model_info}.csv")
    cell_test.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}.")
    print(f"Inference time: {total_seconds} seconds. Timing log saved to {timing_log_path}.")


"""
VITAL: Multi-View Clustering - Unified Single File Implementation
Supports multiple views (more than 2 views).
"""

import os
import sys
import logging
import argparse
import time
import random
import numpy as np
import scipy.io as sio
import sklearn.metrics as metrics
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
from munkres import Munkres
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")


# ======================= Utility Functions =======================

def set_seed(seed=None):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_log(args):
    """Set up logging by creating necessary directories and initializing logging configurations."""
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.exists(args.log_path + args.dataset_name + '/'):
        os.mkdir(args.log_path + args.dataset_name + '/')

    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_folder_path = os.path.join(args.log_path + args.dataset_name + '/' + timestamp)
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    log_file_path = os.path.join(log_folder_path, 'train.log')
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


# ======================= Data Loading =======================

class MultiViewDataset(torch.utils.data.Dataset):
    """Dataset for multi-view data."""
    def __init__(self, data_list, labels):
        """
        Args:
            data_list: List of numpy arrays, each array is data for one view
            labels: numpy array of labels
        """
        self.data_list = data_list
        self.labels = labels
        self.num_views = len(data_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        views = [torch.from_numpy(self.data_list[v][idx]) for v in range(self.num_views)]
        label = self.labels[idx]
        return views, label


class MultiViewDataset_test(torch.utils.data.Dataset):
    """Dataset for multi-view test data with separate labels for each view."""
    def __init__(self, data_list, labels, labels_Y_list):
        """
        Args:
            data_list: List of numpy arrays, each array is data for one view
            labels: numpy array of labels for view 0
            labels_Y_list: List of numpy arrays, labels for each view (view 0's labels, then shuffled labels for other views)
        """
        self.data_list = data_list
        self.labels = labels
        self.labels_Y_list = labels_Y_list  # List of labels for each view
        self.num_views = len(data_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        views = [torch.from_numpy(self.data_list[v][idx]) for v in range(self.num_views)]
        label = self.labels[idx]
        labels_Y = [self.labels_Y_list[v][idx] for v in range(self.num_views)]
        return views, label, labels_Y


def collate_fn(batch):
    """Custom collate function for multi-view data."""
    views_list = [[] for _ in range(len(batch[0][0]))]
    labels = []
    
    for views, label in batch:
        for v, view in enumerate(views):
            views_list[v].append(view)
        labels.append(label)
    
    views_tensor = [torch.stack(v) for v in views_list]
    labels_tensor = torch.tensor(labels)
    return views_tensor, labels_tensor


def collate_fn_test(batch):
    """Custom collate function for multi-view test data with separate labels."""
    num_views = len(batch[0][0])
    views_list = [[] for _ in range(num_views)]
    labels = []
    labels_Y_list = [[] for _ in range(num_views)]  # List of labels for each view
    
    for views, label, labels_y in batch:
        for v, view in enumerate(views):
            views_list[v].append(view)
            labels_Y_list[v].append(labels_y[v])
        labels.append(label)
    
    views_tensor = [torch.stack(v) for v in views_list]
    labels_tensor = torch.tensor(labels)
    labels_Y_tensors = [torch.tensor(ly) for ly in labels_Y_list]
    return views_tensor, labels_tensor, labels_Y_tensors


def load_data(args):
    """Load and preprocess data for multi-view clustering."""
    data_list = []
    mat = sio.loadmat(args.dataset_path + args.dataset_name + '.mat')

    # Load the dataset based on the dataset name
    if args.dataset_name == 'CUB':
        data_list.append(mat['X'][0][0])
        data_list.append(mat['X'][0][1])
        label = np.squeeze(mat['gt'].astype(np.uint8))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # Apply min-max normalization to all views
    data_list = [minmax_scale(data).astype(np.float32) for data in data_list]

    if data_list[0].shape[0] != label.shape[0]:
        raise ValueError("The dataset dimensions are not (num_samples x features_dims)")

    dims_list = [data.shape[1] for data in data_list]
    num_samples = label.shape[0]
    num_classes = len(np.unique(label))
    num_views = len(data_list)

    # Split data based on aligned_rate
    split_idx = np.random.permutation(num_samples)
    aligned_num = int(np.ceil(args.aligned_rate * num_samples))
    aligned_idx = split_idx[:aligned_num]
    unaligned_idx = split_idx[aligned_num:]

    # Separate aligned and unaligned data
    aligned_data = [data[aligned_idx] for data in data_list]
    unaligned_data = [data[unaligned_idx] for data in data_list]
    aligned_labels = label[aligned_idx]
    unaligned_labels = label[unaligned_idx]

    # Prepare training data (only aligned data)
    train_data = aligned_data
    train_labels = aligned_labels

    # Prepare testing data
    if args.aligned_rate == 1.0:
        test_data = aligned_data
        test_labels = aligned_labels
        # All views have the same labels when fully aligned
        test_labels_Y_list = [aligned_labels for _ in range(num_views)]
    else:
        # Shuffle each non-first view independently
        unaligned_data_shuffled = [unaligned_data[0]]  # Keep view 0 unchanged
        shuffle_indices = [None]  # View 0 has no shuffle
        for v in range(1, num_views):
            shuffle_idx_v = np.random.permutation(len(unaligned_idx))
            shuffle_indices.append(shuffle_idx_v)
            unaligned_data_shuffled.append(unaligned_data[v][shuffle_idx_v])
        
        # Concatenate aligned and unaligned data
        test_data = [np.concatenate([aligned_data[v], unaligned_data[v] if v == 0 else unaligned_data_shuffled[v]]) 
                     for v in range(num_views)]
        test_labels = np.concatenate([aligned_labels, unaligned_labels])
        
        # Create labels_Y for each view
        test_labels_Y_list = []
        for v in range(num_views):
            if v == 0:
                # View 0 is not shuffled
                test_labels_Y_list.append(np.concatenate([aligned_labels, unaligned_labels]))
            else:
                # Other views are shuffled independently
                test_labels_Y_list.append(np.concatenate([aligned_labels, unaligned_labels[shuffle_indices[v]]]))

    # Create data loaders for training and testing
    train_dataset = MultiViewDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn
    )
    
    test_dataset = MultiViewDataset_test(test_data, test_labels, test_labels_Y_list)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=collate_fn_test
    )

    return train_loader, test_loader, num_samples, num_classes, num_views, dims_list


# ======================= Clustering & Evaluation =======================

def Purity_score(y_true, y_pred):
    """Calculate purity score for clustering."""
    y_true = y_true.copy()
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    purity = metrics.accuracy_score(y_true, y_voted_labels)
    return purity


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """Computes the predicted labels using Munkres algorithm."""
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def classification_metric(y_true, y_pred, average='macro', decimals=4):
    """Compute classification metrics."""
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    accuracy = np.round(metrics.accuracy_score(y_true, y_pred), decimals)
    precision = np.round(metrics.precision_score(y_true, y_pred, average=average), decimals)
    recall = np.round(metrics.recall_score(y_true, y_pred, average=average), decimals)
    f_score = np.round(metrics.f1_score(y_true, y_pred, average=average), decimals)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix


def clustering_metric(y_true, y_pred, n_clusters, decimals=4):
    """Compute clustering metrics."""
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)
    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)

    ami = np.round(metrics.adjusted_mutual_info_score(y_true, y_pred), decimals)
    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), decimals)
    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), decimals)

    return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari}, **classification_metrics), confusion_matrix


def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    """Generate cluster assignments based on input data."""
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj


def Clustering(x_list, y):
    """Perform multi-view clustering and evaluation."""
    n_clusters = np.size(np.unique(y))
    x_final_concat = np.concatenate(x_list[:], axis=1)
    kmeans_assignments, km = get_cluster_sols(
        x_final_concat, ClusterClass=KMeans, n_clusters=n_clusters,
        init_args={'n_init': 10, 'random_state': 42}
    )
    y_preds = get_y_preds(y, kmeans_assignments, n_clusters)
    if np.min(y) == 1:
        y = y - 1
    scores, _ = clustering_metric(y, kmeans_assignments, n_clusters)
    
    # Calculate purity (not returned, for internal use)
    _ = Purity_score(y, kmeans_assignments)

    ret = {'kmeans': scores}
    return y_preds, ret


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors.
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def inference(model, test_loader, args):
    """Perform inference on the test dataset with realignment support."""
    model.eval()
    aligned_outputs = [[] for _ in range(model.num_views)]
    gt_labels = []

    with torch.no_grad():
        for batch_idx, (views, labels, labels_Y_list) in enumerate(test_loader):
            # Move views to GPU
            views = [v.cuda() for v in views]
            labels = labels.cuda()
            test_num = len(labels)
            
            # Forward pass
            mus, logvars, _ = model(views)
            
            # Process features from each view
            h_list = []
            for v in range(model.num_views):
                mu = mus[v]
                logvar = logvars[v]
                std = (0.5 * logvar).exp()
                
                if args.feats_norm:
                    h = F.normalize(mu) + F.normalize(std)
                else:
                    h = mu + std
                h_list.append(h)
            
            if args.aligned_rate == 1.0:
                # Fully aligned: no realignment needed
                for i in range(test_num):
                    for v in range(model.num_views):
                        aligned_outputs[v].append(h_list[v][i, :].cpu().numpy())
            else:
                # Compute distance matrices and realign each view independently with view 0
                # View 0 stays as is
                for i in range(test_num):
                    aligned_outputs[0].append(h_list[0][i, :].cpu().numpy())
                
                # For each other view, compute distance to view 0 and realign
                for v in range(1, model.num_views):
                    if args.feats_norm:
                        C_v = euclidean_dist(F.normalize(mus[0]), F.normalize(mus[v]))
                    else:
                        C_v = euclidean_dist(mus[0], mus[v])
                    
                    for i in range(test_num):
                        idx = torch.argsort(C_v[i, :])
                        C_v[:, idx[0]] = float("inf")
                        aligned_outputs[v].append(h_list[v][idx[0], :].cpu().numpy())
            
            gt_labels.extend(labels.cpu().numpy())

    # Concatenate all batches
    data = [np.array(outputs) for outputs in aligned_outputs]
    gt_labels = np.array(gt_labels)
    
    _, ret = Clustering(data, gt_labels)
    
    acc = round(ret['kmeans']['accuracy'] * 100, 4)
    nmi = round(ret['kmeans']['NMI'] * 100, 4)
    ari = round(ret['kmeans']['ARI'] * 100, 4)

    return acc, nmi, ari


# ======================= Model Definitions =======================

class RecognitionModel(nn.Module):
    """Recognition model (encoder) for one view."""
    def __init__(self, recognition_model_dim, use_dropout, dropout_rate, activation):
        super(RecognitionModel, self).__init__()

        recognition_model = []
        for i in range(len(recognition_model_dim) - 1):
            recognition_model.append(nn.Linear(recognition_model_dim[i], recognition_model_dim[i + 1]))

            if i < len(recognition_model_dim) - 2:
                recognition_model.append(nn.BatchNorm1d(recognition_model_dim[i + 1]))
                recognition_model.append(nn.ReLU())
                if use_dropout:
                    recognition_model.append(nn.Dropout(dropout_rate))
            else:
                if activation == 'relu':
                    recognition_model.append(nn.ReLU())
                elif activation == 'tanh':
                    recognition_model.append(nn.Tanh())
                elif activation == 'none':
                    recognition_model.append(nn.Identity())

        self.recognition_model = nn.Sequential(*recognition_model)

    def forward(self, x):
        return self.recognition_model(x)


class GenerativeModel(nn.Module):
    """Generative model (decoder) for one view."""
    def __init__(self, generative_model_dim):
        super(GenerativeModel, self).__init__()

        generative_model = []
        for i in range(len(generative_model_dim) - 1):
            generative_model.append(nn.Linear(generative_model_dim[i], generative_model_dim[i + 1]))
            if i < len(generative_model_dim) - 2:
                generative_model.append(nn.ReLU())
            else:
                generative_model.append(nn.Sigmoid())

        self.generative_model = nn.Sequential(*generative_model)

    def forward(self, x):
        return self.generative_model(x)


class VITAL(nn.Module):
    """VITAL model for multi-view clustering."""
    def __init__(self, args, dims_list):
        super(VITAL, self).__init__()
        
        self.num_views = len(dims_list)
        hidden_dims = args.hidden_dims  # e.g., [1024, 1024, 1024]
        latent_dim = args.latent_dim    # e.g., 128
        
        # Build model dimensions for each view
        self.recognition_model_dims = []
        self.generative_model_dims = []
        for dim in dims_list:
            self.recognition_model_dims.append([dim] + hidden_dims + [latent_dim * 2])
            self.generative_model_dims.append([latent_dim] + hidden_dims + [dim])
        
        self.activation = args.activation
        self.use_dropout = args.use_dropout
        self.dropout_rate = args.dropout_rate

        self.temperature = args.temperature
        self.alpha = args.alpha
        self.vcl_epochs = args.vcl_epochs
        self.vcl_dr_epochs = args.vcl_dr_epochs

        self.l2 = F.normalize

        # Initialize recognition and generative models for each view
        self.recognition_models = nn.ModuleList([
            RecognitionModel(self.recognition_model_dims[v], self.use_dropout, self.dropout_rate, self.activation)
            for v in range(self.num_views)
        ])
        self.generative_models = nn.ModuleList([
            GenerativeModel(self.generative_model_dims[v])
            for v in range(self.num_views)
        ])

    def reparameterise(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, views):
        """
        Forward pass for multiple views.
        Args:
            views: list of tensors, each tensor is data for one view
        Returns:
            mus: list of mu tensors for each view
            logvars: list of logvar tensors for each view
            recs: list of reconstructed tensors for each view
        """
        mus = []
        logvars = []
        recs = []
        
        for v in range(self.num_views):
            # Get embedding from recognition model
            embedding = self.recognition_models[v](views[v])
            
            # Split into mu and logvar
            mu, logvar = torch.chunk(embedding, chunks=2, dim=1)
            mus.append(mu)
            logvars.append(logvar)
            
            # Reparameterize
            z = self.reparameterise(mu, logvar)
            
            # Reconstruct
            x_rec = self.generative_models[v](z)
            recs.append(x_rec)
        
        return mus, logvars, recs

    def kl_loss(self, mu, logvar):
        """Calculate the KL divergence loss."""
        return ((mu * mu + logvar.exp() - logvar - 1) / 2).mean()
    
    def intra_loss(self, x_rec, x):
        """Calculate the intra-view loss."""
        x_rec_norm = self.l2(x_rec, p=2, dim=1)
        x_norm = self.l2(x, p=2, dim=1)
        sim = x_rec_norm.mm(x_norm.t()) / self.temperature
        intra_loss = - sim.log_softmax(1).diag().mean()
        return intra_loss

    def inter_loss_pair(self, mu_i, mu_j, epoch):
        """Calculate the inter-view loss between two views."""
        mu_i_norm = self.l2(mu_i, p=2, dim=1)
        mu_j_norm = self.l2(mu_j, p=2, dim=1)

        # Calculate similarity matrices
        sim_ij = mu_i_norm.mm(mu_j_norm.t()) / self.temperature
        sim_ji = sim_ij.t()

        # Get the batch size
        bs = len(sim_ij)

        if epoch <= self.vcl_epochs:    # VCL training stage
            loss_ij = - (torch.pow(sim_ij.softmax(1).diag().clamp_min(1e-7), self.alpha) * (sim_ij.log_softmax(1).diag())).mean()
            loss_ji = - (torch.pow(sim_ji.softmax(1).diag().clamp_min(1e-7), self.alpha) * (sim_ji.log_softmax(1).diag())).mean()
        else:                           # VCL-DR training stage
            # Keep everything on GPU
            per_loss_ij = -(sim_ij.log_softmax(1)).reshape(-1, 1)
            per_loss_ji = -(sim_ji.log_softmax(1)).reshape(-1, 1)
            
            # Normalize on GPU
            per_loss_ij = (per_loss_ij - per_loss_ij.min()) / (per_loss_ij.max() - per_loss_ij.min() + 1e-8)
            per_loss_ji = (per_loss_ji - per_loss_ji.min()) / (per_loss_ji.max() - per_loss_ji.min() + 1e-8)
            
            with torch.no_grad():
                mask_ij, alpha_ij = self.GMM(per_loss_ij, bs)
                mask_ji, alpha_ji = self.GMM(per_loss_ji, bs)

            loss_ij = - (torch.pow(sim_ij.softmax(1)[mask_ij], self.alpha_norm(alpha_ij[mask_ij])) * (sim_ij.log_softmax(1)[mask_ij])).mean()
            loss_ji = - (torch.pow(sim_ji.softmax(1)[mask_ji], self.alpha_norm(alpha_ji[mask_ji])) * (sim_ji.log_softmax(1)[mask_ji])).mean()

        return (loss_ij + loss_ji)

    def inter_loss(self, mus, epoch):
        """Calculate the inter-view loss for all view pairs."""
        total_loss = 0
        num_pairs = 0
        
        # Calculate loss for all pairs of views
        for i, j in combinations(range(self.num_views), 2):
            total_loss += self.inter_loss_pair(mus[i], mus[j], epoch)
            num_pairs += 1
        
        # Average over all pairs
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        
        return total_loss
    
    def GMM(self, input_tensor, bs):
        """
        Fit GMM on GPU matching sklearn's GaussianMixture behavior. (GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4))
        """
        # Parameters matching sklearn
        max_iter = 10
        tol = 1e-2
        reg_covar = 5e-4
        
        # input_tensor: (bs*bs, 1) on GPU
        input_flat = input_tensor.squeeze()  # (bs*bs,)
        n_samples = input_flat.size(0)
        
        # Initialize means using k-means style (similar to sklearn's default 'kmeans' init)
        sorted_vals, _ = torch.sort(input_flat)
        means = torch.stack([
            sorted_vals[:n_samples//4].mean(),
            sorted_vals[-n_samples//4:].mean()
        ])
        
        # Initialize variances
        vars = torch.stack([
            sorted_vals[:n_samples//4].var() + reg_covar,
            sorted_vals[-n_samples//4:].var() + reg_covar
        ])
        vars = torch.clamp(vars, min=reg_covar)
        
        # Initialize weights (uniform)
        weights = torch.tensor([0.5, 0.5], device=input_flat.device)
        
        prev_log_likelihood = float('-inf')
        
        # EM algorithm
        for iteration in range(max_iter):
            # E-step: compute responsibilities
            # log N(x | mu, sigma^2) = -0.5 * log(2*pi*sigma^2) - 0.5 * (x-mu)^2 / sigma^2
            log_prob0 = -0.5 * torch.log(2 * 3.141592653589793 * vars[0]) - 0.5 * (input_flat - means[0]).pow(2) / vars[0]
            log_prob1 = -0.5 * torch.log(2 * 3.141592653589793 * vars[1]) - 0.5 * (input_flat - means[1]).pow(2) / vars[1]
            
            # Add log weights
            log_weighted0 = log_prob0 + torch.log(weights[0])
            log_weighted1 = log_prob1 + torch.log(weights[1])
            
            # Log-sum-exp for numerical stability
            log_weighted_max = torch.maximum(log_weighted0, log_weighted1)
            log_sum = log_weighted_max + torch.log(
                torch.exp(log_weighted0 - log_weighted_max) + 
                torch.exp(log_weighted1 - log_weighted_max) + 1e-10
            )
            
            # Responsibilities (posterior probabilities)
            log_resp0 = log_weighted0 - log_sum
            log_resp1 = log_weighted1 - log_sum
            resp0 = torch.exp(log_resp0)
            resp1 = torch.exp(log_resp1)
            
            # Compute log-likelihood for convergence check
            current_log_likelihood = log_sum.mean().item()
            
            # Check convergence
            if iteration > 0 and abs(current_log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = current_log_likelihood
            
            # M-step: update parameters
            # Update weights
            n0 = resp0.sum()
            n1 = resp1.sum()
            weights[0] = n0 / n_samples
            weights[1] = n1 / n_samples
            
            # Update means
            means[0] = (resp0 * input_flat).sum() / (n0 + 1e-10)
            means[1] = (resp1 * input_flat).sum() / (n1 + 1e-10)
            
            # Update variances with regularization
            vars[0] = (resp0 * (input_flat - means[0]).pow(2)).sum() / (n0 + 1e-10) + reg_covar
            vars[1] = (resp1 * (input_flat - means[1]).pow(2)).sum() / (n1 + 1e-10) + reg_covar
            vars = torch.clamp(vars, min=reg_covar)
        
        # Final E-step to get final responsibilities
        log_prob0 = -0.5 * torch.log(2 * 3.141592653589793 * vars[0]) - 0.5 * (input_flat - means[0]).pow(2) / vars[0]
        log_prob1 = -0.5 * torch.log(2 * 3.141592653589793 * vars[1]) - 0.5 * (input_flat - means[1]).pow(2) / vars[1]
        log_weighted0 = log_prob0 + torch.log(weights[0])
        log_weighted1 = log_prob1 + torch.log(weights[1])
        log_weighted_max = torch.maximum(log_weighted0, log_weighted1)
        log_sum = log_weighted_max + torch.log(
            torch.exp(log_weighted0 - log_weighted_max) + 
            torch.exp(log_weighted1 - log_weighted_max) + 1e-10
        )
        resp0 = torch.exp(log_weighted0 - log_sum)
        resp1 = torch.exp(log_weighted1 - log_sum)
        
        # Determine which component is "clean" (smaller mean = smaller loss = cleaner)
        if means[0] < means[1]:
            prob = resp0
            pair_label = (resp0 > resp1).long()
            clean_idx = 0
        else:
            prob = resp1
            pair_label = (resp1 > resp0).long()
            clean_idx = 1
        
        mask_gmm = (pair_label == 1).reshape(bs, bs)
        
        # VITAL mask scheme
        d = torch.abs(means[0] - means[1])
        input_matrix = input_flat.reshape(bs, bs)
        
        pos_values = input_matrix[mask_gmm]
        if pos_values.numel() > 0:
            loss_upper_bound = torch.quantile(pos_values, min(d.item(), 1.0))
        else:
            loss_upper_bound = input_matrix.max()
        
        confidence_mask = (input_matrix > 0) & (input_matrix <= loss_upper_bound) & mask_gmm
        mask = confidence_mask.clone()
        mask.fill_diagonal_(True)
        
        alpha = prob.reshape(bs, bs)
        return mask, alpha

    def alpha_norm(self, alpha_val):
        """Normalize alpha values."""
        alpha_max = torch.max(alpha_val)
        alpha_min = torch.min(alpha_val)
        if alpha_max == alpha_min:
            alpha_normed = torch.full_like(alpha_val, fill_value=self.alpha)
        else:
            alpha_normed = ((alpha_val - alpha_min) * self.alpha) / (alpha_max - alpha_min)
        return alpha_normed


# ======================= Training Function =======================

def train(train_loader, model, optimizer, epoch):
    """Train the model for one epoch."""
    model.train()
    loss_value = 0

    for _, (views, labels) in enumerate(train_loader):
        # Move views to GPU
        views = [v.cuda() for v in views]
        labels = labels.cuda()
        
        # Forward pass
        mus, logvars, recs = model(views)
        
        # Compute the losses
        inter_loss = model.inter_loss(mus, epoch)
        
        intra_loss = 0
        kl_loss = 0
        for v in range(model.num_views):
            intra_loss += model.intra_loss(recs[v], views[v])
            kl_loss += model.kl_loss(mus[v], logvars[v])
        
        # Total loss
        loss = intra_loss + inter_loss + kl_loss
        loss_value += loss.item()
        
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# ======================= Argument Parser =======================

def get_args_parser():
    parser = argparse.ArgumentParser(description='VITAL: Multi-View Clustering')

    # Dataset info
    parser.add_argument('--dataset_name', default='CUB', type=str, 
                        help='The name of the dataset.')
    parser.add_argument('--aligned_rate', default=1.0, type=float, 
                        help='Aligned rate for PVP setting. 1.0 means fully aligned.')
    parser.add_argument('--train_time', default=5, type=int, 
                        help='Number of training iterations.')
    parser.add_argument('--dataset_path', default='./datasets/', type=str, 
                        help='Path to the dataset.')
    parser.add_argument('--log_path', default='./log/', type=str, 
                        help='Path to save logs.')

    # Network architecture
    parser.add_argument('--hidden_dims', default=[1024, 1024, 1024], type=int, nargs='+',
                        help='Hidden layer dimensions for both encoder and decoder.')
    parser.add_argument('--latent_dim', default=128, type=int,
                        help='Latent dimension (mu and logvar each have this dimension).')
    parser.add_argument('--activation', default='none', type=str, 
                        choices=['none', 'relu', 'tanh'], 
                        help='Activation function used in the adaption layer.')
    parser.add_argument('--use_dropout', default=True, type=lambda x: x.lower() == 'true',
                        help='Whether to use dropout in the model.')
    parser.add_argument('--dropout_rate', default=0.2, type=float, 
                        help='Dropout rate if dropout is used.')
    parser.add_argument('--temperature', default=0.4, type=float, 
                        help='Temperature parameter for model training.')

    # Optimizer parameters
    parser.add_argument('--batch_size', default=1024, type=int, 
                        help='Batch size for training.')
    parser.add_argument('--vcl_epochs', default=100, type=int, 
                        help='Number of epochs for VCL training.')
    parser.add_argument('--vcl_dr_epochs', default=110, type=int, 
                        help='Number of epochs for total training.')
    parser.add_argument('--vcl_lr', default=2e-3, type=float, 
                        help='Learning rate for VCL training.')
    parser.add_argument('--vcl_dr_lr', default=1e-4, type=float, 
                        help='Learning rate for VCL-DR training.')

    # Other parameters
    parser.add_argument('--alpha', default=0.1, type=float, 
                        help='Alpha parameter for focal loss.')
    parser.add_argument('--feats_norm', default=True, type=lambda x: x.lower() == 'true',
                        help='Whether to normalize features.')

    # GPU parameters
    parser.add_argument('--gpu', default='0', type=str, 
                        help='GPU device number to use.')

    return parser.parse_args()


# ======================= Main Function =======================

def main():
    args = get_args_parser()

    # if args.dataset_name in ['MNIST-USPS']:
    #     args.batch_size = 256
    # if args.dataset_name in ['NoisyMNIST']:
    #     args.feats_norm = False
    #     if args.aligned_rate == 1.0:
    #         args.dropout_rate = 0.5
    #         args.batch_size = 2048
    #     else:
    #         args.dropout_rate = 0.1
    # if args.dataset_name in ['wiki_2_view']:
    #     if args.aligned_rate == 1.0:
    #         args.dropout_rate = 0.5
    #     else:
    #         args.dropout_rate = 0.2
    # if args.dataset_name in ['nuswide_deep_2_view']:
    #     args.dropout_rate = 0.5
    # if args.dataset_name in ['AWA-7view-10158sample']:
    #     args.dropout_rate = 0.5
    # if args.dataset_name in ['Scene15']:
    #     args.batch_size = 512

    # Set the CUDA device
    torch.cuda.set_device(f"cuda:{args.gpu}")
    get_log(args)

    acc_list, nmi_list, ari_list = [], [], []
    seed_list = [random.randint(1, 10000) for _ in range(args.train_time)]

    # Begin Training
    for t in range(1, args.train_time + 1):
        set_seed(seed_list[t - 1])

        train_loader, test_loader, num_samples, num_classes, num_views, dims_list = load_data(args)

        # Log dataset information on the first iteration
        if t == 1:
            logging.info(f'''Dataset info.: 
                dataset name: {args.dataset_name}, 
                number of views: {num_views}, 
                feature dimensions: {dims_list}, 
                number of samples: {num_samples}, 
                number of classes: {num_classes},
                aligned rate: {args.aligned_rate}''')

        # Initialize model with dims_list from data loader
        model = VITAL(args, dims_list).cuda()

        if t == 1:
            logging.info("Model info.:")
            logging.info(model)

        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.vcl_lr)

        for epoch in range(args.vcl_dr_epochs + 1):
            if epoch == 0:
                with torch.no_grad():
                    train(train_loader, model, optimizer, epoch)
            else:
                train(train_loader, model, optimizer, epoch)

            # Adjust learning rate for VCL-DR stage
            if epoch == args.vcl_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.vcl_dr_lr

        # Begin Testing
        acc, nmi, ari = inference(model, test_loader, args)

        if t == 1:
            logging.info("Performance:")
        logging.info("Round %d: ACC: %.2f  NMI: %.2f  ARI: %.2f", 
                     t, round(acc, 2), round(nmi, 2), round(ari, 2))

        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)

    # Log the average results and standard deviation if multiple training iterations
    if args.train_time != 1:
        logging.info("   Mean: ACC: %.2f  NMI: %.2f  ARI: %.2f", 
                     round(np.mean(acc_list), 2), round(np.mean(nmi_list), 2), round(np.mean(ari_list), 2))
        logging.info("    std: ACC: %.2f   NMI: %.2f   ARI: %.2f", 
                     round(np.std(acc_list), 2), round(np.std(nmi_list), 2), round(np.std(ari_list), 2))


if __name__ == '__main__':
    main()

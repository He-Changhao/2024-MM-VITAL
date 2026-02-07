import logging
import numpy as np
import time
import os
import sys
import torch
import random
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import norm

def set_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_log(args):
    """
    Set up logging by creating necessary directories and initializing logging configurations.

    Args:
        args (Namespace): Command-line arguments that include paths for logging.
    """
    # Create directories for logs if they do not exist
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.exists(args.log_path + args.dataset_name + '/'):
        os.mkdir(args.log_path + args.dataset_name + '/')

    # Create a timestamped directory for the current log session
    timestamp = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time()))
    log_folder_path = os.path.join(args.log_path + args.dataset_name + '/' + timestamp)
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)

    # Set up logging format and level
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

    # Create a file handler for logging to a file
    log_file_path = os.path.join(log_folder_path, 'train.log')
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def train(train_loader, model, optimizer, epoch):
    """
    Train the model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): Optimizer for updating the model parameters.
        epoch (int): The current epoch number.

    """
    model.train()
    loss_value = 0

    # Iterate through the training data
    for _, (view0, view1, labels) in enumerate(train_loader):
        view0, view1, labels = view0.cuda(), view1.cuda(), labels.cuda()
        
        mu0, mu1, logvar0, logvar1, x0_rec, x1_rec = model(view0, view1)
        
        # Compute the losses
        inter_loss = model.inter_loss(mu0, mu1, epoch)
        intra_loss = model.intra_loss(x0_rec, view0) + model.intra_loss(x1_rec, view1)
        kl_loss = model.kl_loss(mu0, logvar0) + model.kl_loss(mu1, logvar1)
        
        # Total loss
        loss = intra_loss + inter_loss + kl_loss
        loss_value += loss.item()
        
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def density_plot(input, gt_labels, log_path, epoch, name='loss', density=True):
    gt_labels = gt_labels.repeat(gt_labels.shape[0], 1)
    gt_mask = (gt_labels == gt_labels.t()).int().view(-1).cpu().numpy().ravel()

    plt.figure(figsize=(7, 5))
    plt.clf()
    ax = plt.gca()
    clean_index = np.where(gt_mask == 1)[0]
    noisy_index = np.where(gt_mask == 0)[0]

    gmm = GaussianMixture(n_components=2)
    gmm.fit(input.reshape(-1, 1))

    #  Plot mixture PDF
    x = np.linspace(np.min(input), np.max(input), 1000)
    pdf = np.exp(gmm.score_samples(x.reshape(-1, 1)))
    ax.plot(x, pdf, color='black', linestyle='-', label='Mixture PDF')

    # Plot individual GMM components
    for i in range(gmm.n_components):
        mean = gmm.means_[i][0]
        std_dev = np.sqrt(gmm.covariances_[i][0][0])
        component_pdf = norm.pdf(x, mean, std_dev)
        if mean == min(gmm.means_):
            color = 'green'
        else:
            color = 'red'
        ax.plot(x, gmm.weights_[i] * component_pdf, color=color, linestyle='--', label=f'Component {i+1}')

    # Plot density
    hist1, bins1, _ = ax.hist(input[clean_index], bins=100, density=density, histtype='stepfilled', color='green', alpha=0.4, label='Pos Pairs')
    hist2, bins2, _ = ax.hist(input[noisy_index], bins=100, density=density, histtype='stepfilled', color='red', alpha=0.4, label='Neg Pairs')

    plt.yticks(size=15)
    plt.xticks(size=15)
    plt.xlabel(name, fontsize=15)
    if not density:
        plt.ylabel('num', fontsize=15)
    else:
        plt.ylabel('density', fontsize=15)
    plt.legend(loc='upper right', fontsize=12, frameon=True)
    plt.savefig(os.path.join(log_path, str(epoch) + "epoch_visualization.png"))
    plt.close()

def tsne(X, y, num_classes, title='', seed=0):
    tsne = TSNE(n_components=2, random_state=seed)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    if np.min(y)==1:
        for i in range(1,num_classes+1):
            plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=str(i))
    else:
        for i in range(num_classes):
            plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=str(i))

    plt.xticks([])
    plt.yticks([])
    plt.savefig(title+'.png')
    plt.close()


def acc_plot():
    aligned_rate = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10']
    # AE2-Nets
    mean = [54.33, 27.61, 33.11, 30.33, 29.76, 31.32, 27.23, 31.24, 24.30, 26.42]
    std = [7.60, 6.50, 5.65, 9.32, 10.22, 7.26, 9.29, 9.23, 6.88, 7.63]
    plt.plot(aligned_rate, mean, label='AE2-Nets',color='lightcoral')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],color='lightcoral', alpha=0.2)
    # PVC
    mean = [59.57,58.54,54.53,57.33,55.03,55.53,51.77,48.10,45.90,41.00]
    std = [3.69,3.95,2.87,1.71,3.64,2.46,5.65,1.91,2.32,3.79]
    plt.plot(aligned_rate, mean, label='PVC',color='C1')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C1', alpha=0.2)
    # MvCLN
    mean = [64.92, 59.97, 60.00, 58.30, 54.64,52.13, 51.40, 49.85, 42.83, 43.17]
    std = [5.68, 4.87, 4.70, 4.31, 3.69, 5.90, 3.38, 3.05, 3.07, 5.11]
    plt.plot(aligned_rate, mean, label='MvCLN',color='C2')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C2', alpha=0.2)
    # DSIMVC 
    mean = [59.13, 35.07, 35.27, 35.60, 33.97, 35.87, 36.70, 36.80, 34.83, 34.67]
    std = [3.96, 1.97, 2.71, 2.71, 1.27, 2.28, 1.89, 1.24, 2.76, 2.40]
    plt.plot(aligned_rate, mean, label='DSIMVC',color='C4')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C4', alpha=0.2)
    # MFLVC
    mean = [70.03, 42.47, 39.93, 40.23, 40.50, 40.97, 41.23, 40.87, 39.87, 40.97]
    std = [3.14, 0.64, 3.46, 1.86, 1.91, 0.89, 1.30, 1.66, 2.23, 1.23]
    plt.plot(aligned_rate, mean, label='MFLVC',color='C5')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C5', alpha=0.2)
    # DCP
    mean = [61.53, 32.70, 31.33, 30.60, 29.37,31.07, 27.17, 30.53, 31.17, 31.13]
    std = [5.51, 1.65, 1.84, 3.91, 3.25, 2.60, 2.15, 3.85, 0.57, 0.86]
    plt.plot(aligned_rate, mean, label='DCP',color='C6')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C6', alpha=0.2)
    # GCFAgg
    mean = [71.17, 41.13, 39.40, 40.83, 39.73,39.87, 42.40, 39.10, 40.93, 40.70]
    std = [3.26, 3.39, 2.73, 1.52, 2.80, 2.03, 3.82, 2.57, 2.17, 1.80]
    plt.plot(aligned_rate, mean, label='GCFAgg',color='C7')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C7', alpha=0.2)
    # DealMVC
    mean = [53.66, 38.53, 39.03, 38.50, 39.70,40.87, 39.23, 38.40, 39.83, 40.53]
    std = [2.33, 2.34, 2.50, 2.07, 1.66, 1.99, 2.46, 1.53, 2.29, 1.32]
    plt.plot(aligned_rate, mean, label='DealMVC',color='C8')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C8', alpha=0.2)
    # SURE
    mean = [62.70, 56.86, 58.70, 58.57, 56.96,54.47, 56.80, 47.69, 46.00, 38.83]
    std = [7.72, 4.03, 6.18, 4.68, 4.69, 2.56, 5.52, 2.94, 2.32, 2.28]
    plt.plot(aligned_rate, mean, label='SURE',color='C9')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C9', alpha=0.2)
    # ICMVC
    mean = [82.97, 59.17, 59.43, 59.70, 58.20,58.73, 59.23, 59.10, 59.07, 58.37]
    std = [5.23, 5.00, 1.46, 2.33, 4.44, 2.91, 3.23, 1.61, 1.77, 4.14]
    plt.plot(aligned_rate, mean, label='ICMVC',color='C0')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C0', alpha=0.2)

    # Ours
    mean = [85.07,83.67,82.80,81.47,80.00,78.70,79.43,79.27,77.57,70.33]
    std = [0.70,3,1.91,1.1,2.12,2.99,2.71,2.16,2.19,3.06]
    plt.plot(aligned_rate, mean, label='Ours',color='C3')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C3', alpha=0.2)
    plt.grid(True)
    plt.legend(loc='lower left', prop={'size': 12})
    plt.ylabel('ACC', fontsize=17)
    plt.xlabel('aligned rate (%)', fontsize=17)
    plt.savefig('ACC.png')
    plt.close()

def nmi_plot():
    aligned_rate = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10']
    # AE2-Nets
    mean = [49.93, 21.72, 27.75, 23.49, 22.81, 25.55, 19.99, 24.60, 12.96, 18.57]
    std = [4.12, 9.35, 7.39, 12.76, 14.21, 9.76, 12.67, 12.22, 10.37, 10.41]
    plt.plot(aligned_rate, mean, label='AE2-Nets',color='lightcoral')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],color='lightcoral', alpha=0.2)
    # PVC
    mean = [66.69,63.94,61.23,61.33,60.87,59.75,57.13,55.19,52.35,44.51]
    std = [2.96,2.88,1.91,2.34,3.05,2.36,2.70,1.75,1.96,3.10]
    plt.plot(aligned_rate, mean, label='PVC',color='C1')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C1', alpha=0.2)
    # MvCLN
    mean = [59.96, 56.74, 56.41, 57.40, 51.31,49.67, 49.22, 45.16, 36.61, 37.94]
    std = [4.32, 4.31, 3.65, 3.82, 2.70, 5.81, 2.22, 2.64, 1.87, 5.01]
    plt.plot(aligned_rate, mean, label='MvCLN',color='C2')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C2', alpha=0.2)
    # DSIMVC 
    mean = [57.50, 32.21, 31.42, 31.35, 31.91, 31.70, 33.80, 33.49, 30.73, 30.05]
    std = [1.55, 0.70, 3.60, 3.31, 1.31, 1.67, 1.81, 0.96, 2.36, 3.46]
    plt.plot(aligned_rate, mean, label='DSIMVC',color='C4')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C4', alpha=0.2)
    # MFLVC
    mean = [67.04, 39.85, 37.31, 37.19, 37.78, 38.67, 39.77, 38.06, 37.80, 39.06]
    std = [1.32, 1.63, 2.19, 0.67, 1.22, 0.87, 1.13, 2.07, 2.13, 1.32]
    plt.plot(aligned_rate, mean, label='MFLVC',color='C5')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C5', alpha=0.2)
    # DCP
    mean = [68.27, 32.85, 30.18, 30.43, 29.45,28.76, 28.39, 29.33, 30.84, 31.82]
    std = [4.33, 2.50, 1.83, 4.22, 2.78, 3.54, 1.07, 4.51, 1.35, 2.14]
    plt.plot(aligned_rate, mean, label='DCP',color='C6')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C6', alpha=0.2)
    # GCFAgg
    mean = [67.12, 39.58, 38.09, 38.58, 38.07,39.36, 40.54, 37.35, 39.11, 38.97]
    std = [1.63, 1.71, 1.56, 0.49, 1.22, 1.72, 2.37, 0.75, 1.88, 0.88]
    plt.plot(aligned_rate, mean, label='GCFAgg',color='C7')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C7', alpha=0.2)
    # DealMVC
    mean = [62.35, 36.34, 36.73, 35.91, 37.39,37.55, 36.30, 36.04, 36.24, 36.01]
    std = [0.61, 1.22, 1.48, 1.94, 0.94, 1.32, 1.91, 1.94, 3.31, 1.21]
    plt.plot(aligned_rate, mean, label='DealMVC',color='C8')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C8', alpha=0.2)
    # SURE
    mean = [60.06, 56.00, 55.79, 56.59, 55.66,50.26, 52.87, 45.39, 41.61, 32.79]
    std = [3.94, 2.50, 4.87, 3.39, 4.32, 4.92, 3.81, 2.48, 2.33, 2.19]
    plt.plot(aligned_rate, mean, label='SURE',color='C9')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C9', alpha=0.2)
    # ICMVC
    mean = [77.05, 54.46, 54.72, 53.97, 53.06,53.15, 54.30, 54.52, 53.67, 54.74]
    std = [3.73, 2.83, 1.32, 1.31, 2.51, 2.56, 2.36, 0.94, 1.29, 3.03]
    plt.plot(aligned_rate, mean, label='ICMVC',color='C0')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C0', alpha=0.2)
    # Ours
    mean = [79.99,79.69,78.19,77.41,76.63,75.74,74.42,74.68,71.36,65.96]
    std = [0.80,1.87,1.12,1.56,1.81,2.02,2.61,1.15,1.01,2.42]
    plt.plot(aligned_rate, mean, label='Ours',color='C3')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C3', alpha=0.2)
    plt.grid(True)
    plt.legend(loc='lower left', prop={'size': 12})
    plt.ylabel('NMI', fontsize=17)
    plt.xlabel('aligned rate (%)', fontsize=17)
    plt.savefig('NMI.png')
    plt.close()

def ari_plot():
    aligned_rate = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10']
    # AE2-Nets
    mean = [34.88, 10.58, 14.29, 11.86, 11.43, 12.62, 9.32, 13.12, 7.74, 8.89]
    std = [5.20, 5.53, 4.39, 7.61, 8.72, 6.07, 7.80, 7.43, 5.47, 6.61]
    plt.plot(aligned_rate, mean, label='AE2-Nets',color='lightcoral')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],color='lightcoral', alpha=0.2)
    # PVC
    mean = [52.90,50.48,45.40,46.03,45.60,44.47,40.22,36.83,31.74,24.28]
    std = [4.83,4.44,3.80,2.36,4.97,3.22,4.61,2.80,1.39,2.95]
    plt.plot(aligned_rate, mean, label='PVC',color='C1')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C1', alpha=0.2)
    # MvCLN
    mean = [47.84, 42.63, 42.49, 42.71, 37.56,35.54, 33.71, 31.05, 22.20, 23.32]
    std = [3.92, 4.60, 4.39, 3.90, 2.45, 7.63, 2.53, 3.38, 1.82, 4.96]
    plt.plot(aligned_rate, mean, label='MvCLN',color='C2')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C2', alpha=0.2)
    # DSIMVC 
    mean = [41.20, 17.00, 17.22, 17.01, 16.09, 16.71, 17.71, 18.01, 16.09, 15.66]
    std = [2.33, 0.62, 2.84, 2.64, 1.19, 1.29, 1.41, 0.71, 1.84, 2.43]
    plt.plot(aligned_rate, mean, label='DSIMVC',color='C4')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C4', alpha=0.2)
    # MFLVC
    mean = [54.44, 24.60, 21.89, 21.79, 22.13, 22.61, 23.59, 22.50, 22.07, 23.61]
    std = [1.99, 0.98, 2.16, 1.05, 1.60, 0.79, 0.92, 1.73, 1.37, 1.19]
    plt.plot(aligned_rate, mean, label='MFLVC',color='C5')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C5', alpha=0.2)
    # DCP
    mean = [48.47, 9.09, 7.98, 8.90, 7.40,6.65, 5.67, 6.78, 7.36, 8.15]
    std = [7.91, 1.90, 0.90, 4.79, 2.28, 2.89, 1.23, 3.27, 2.26, 1.90]
    plt.plot(aligned_rate, mean, label='DCP',color='C6')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C6', alpha=0.2)
    # GCFAgg
    mean = [54.35, 22.80, 21.06, 22.10, 21.16,22.24, 24.23, 20.54, 22.53, 22.30]
    std = [2.57, 2.15, 2.01, 0.67, 1.69, 2.03, 3.06, 1.11, 2.17, 0.87]
    plt.plot(aligned_rate, mean, label='GCFAgg',color='C7')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C7', alpha=0.2)
    # DealMVC
    mean = [46.07, 20.04, 20.95, 20.52, 21.87,22.18, 20.58, 20.63, 21.22, 21.57]
    std = [1.33, 1.37, 1.49, 1.16, 0.83, 0.84, 2.20, 1.88, 3.28, 1.44]
    plt.plot(aligned_rate, mean, label='DealMVC',color='C8')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C8', alpha=0.2)
    # SURE
    mean = [46.13, 41.46, 41.52, 43.10, 42.46,37.19, 39.43, 30.17, 26.76, 18.07]
    std = [5.65, 3.02, 5.70, 4.36, 5.23, 4.29, 4.31, 3.72, 2.10, 2.31]
    plt.plot(aligned_rate, mean, label='SURE',color='C9')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C9', alpha=0.2)
    # ICMVC
    mean = [69.75, 41.20, 41.62, 40.94, 39.84,40.11, 41.05, 41.26, 40.52, 41.30]
    std = [5.75, 3.94, 1.71, 2.20, 3.60, 2.81, 3.10, 1.53, 1.56, 3.98]
    plt.plot(aligned_rate, mean, label='ICMVC',color='C0')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C0', alpha=0.2)
    # Ours
    mean = [72.40,71.18,69.59,68.60,66.09,65.40,64.95,65.50,61.69,55.31]
    std = [1.14,3.18,2.18,1.28,2.61,2.55,3.97,1.45,1.65,2.96]
    plt.plot(aligned_rate, mean, label='Ours',color='C3')
    plt.fill_between(aligned_rate, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], color='C3', alpha=0.2)
    plt.grid(True)
    plt.legend(loc='lower left', prop={'size': 12})
    plt.ylabel('ARI', fontsize=17)
    plt.xlabel('aligned rate (%)', fontsize=17)
    plt.savefig('ARI.png')
    plt.close()

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture


class RecognitionModel(nn.Module):
    def __init__(self, recognition_model_dim, use_dropout, dropout_rate, activation):
        super(RecognitionModel, self).__init__()

        recognition_model = []

        for i in range(len(recognition_model_dim) - 1):
            recognition_model.append(nn.Linear(recognition_model_dim[i], recognition_model_dim[i + 1]))

            if i < len(recognition_model_dim) - 2:  # Apply batchnorm, activation and dropout for all layers except the last one
                recognition_model.append(nn.BatchNorm1d(recognition_model_dim[i + 1]))
                recognition_model.append(nn.ReLU())
                if use_dropout:
                    recognition_model.append(nn.Dropout(dropout_rate))
            else:  # For the adaption layer, apply the specified activation if specified
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
    def __init__(self, generative_model_dim):
        super(GenerativeModel, self).__init__()

        generative_model = []

        for i in range(len(generative_model_dim) - 1):
            generative_model.append(nn.Linear(generative_model_dim[i], generative_model_dim[i + 1]))
            if i < len(generative_model_dim) - 2:  # Apply ReLU for all layers except the last one
                generative_model.append(nn.ReLU())
            else:
                generative_model.append(nn.Sigmoid())

        self.generative_model = nn.Sequential(*generative_model)

    def forward(self, x):
        return self.generative_model(x)

class VITAL(nn.Module):
    def __init__(self, args):
        super(VITAL, self).__init__()
        self.recognition_model_dims = args.recognition_model_dims
        self.generative_model_dims = args.generative_model_dims
        self.activation = args.activation
        self.use_dropout = args.use_dropout
        self.dropout_rate = args.dropout_rate

        self.temperature = args.temperature
        self.init_alpha = args.init_alpha
        self.vcl_epochs = args.vcl_epochs
        self.vcl_dr_epochs = args.vcl_dr_epochs

        self.fitting_type = args.fitting_type
        self.mask_scheme = args.mask_scheme
        self.fix_alpha = args.fix_alpha

        self.l2 = F.normalize
        self.gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

        # Initialize recognition and generative models
        self.view0_recognition_model = RecognitionModel(self.recognition_model_dims[0], self.use_dropout, self.dropout_rate, self.activation)
        self.view1_recognition_model = RecognitionModel(self.recognition_model_dims[1], self.use_dropout, self.dropout_rate, self.activation)
        self.view0_generative_model = GenerativeModel(self.generative_model_dims[0])
        self.view1_generative_model = GenerativeModel(self.generative_model_dims[1])

    def reparameterise(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x0, x1):
        # Get embeddings from the recognition models
        view0_embedding = self.view0_recognition_model(x0)
        view1_embedding = self.view1_recognition_model(x1)
        
        # Split embeddings into mu and logvar
        mu0, logvar0 = torch.chunk(view0_embedding, chunks=2, dim=1)
        mu1, logvar1 = torch.chunk(view1_embedding, chunks=2, dim=1)
        
        # Reparameterize to get latent variables z0 and z1
        z0 = self.reparameterise(mu0, logvar0)
        z1 = self.reparameterise(mu1, logvar1)
        
        # Reconstruct inputs using the generative models
        x0_rec = self.view0_generative_model(z0)
        x1_rec = self.view1_generative_model(z1)
        
        return mu0, mu1, logvar0, logvar1, x0_rec, x1_rec

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

    def inter_loss(self, mu0, mu1, epoch):
        """Calculate the inter-view loss."""
        mu0_norm = self.l2(mu0, p=2, dim=1)
        mu1_norm = self.l2(mu1, p=2, dim=1)

        # Calculate similarity matrices
        sim_ij = mu0_norm.mm(mu1_norm.t()) / self.temperature
        sim_ji = sim_ij.t()

        # Get the batch size
        self.bs = len(sim_ij)

        if epoch <= self.vcl_epochs:    # VCL training stage
            loss_ij = - (torch.pow(sim_ij.softmax(1).diag().clamp_min(1e-7), self.init_alpha) * (sim_ij.log_softmax(1).diag())).mean()
            loss_ji = - (torch.pow(sim_ji.softmax(1).diag().clamp_min(1e-7), self.init_alpha) * (sim_ji.log_softmax(1).diag())).mean()
        else:                           # VCL-DR training stage
            with torch.no_grad():
                if self.fitting_type == 'loss':
                    # Calculate loss-based mask and alpha values
                    per_loss_ij = -(sim_ij.log_softmax(1)).reshape(-1, 1).cpu().numpy()
                    per_loss_ji = -(sim_ji.log_softmax(1)).reshape(-1, 1).cpu().numpy()
                    per_loss_ij = (per_loss_ij - per_loss_ij.min()) / (per_loss_ij.max() - per_loss_ij.min())
                    per_loss_ji = (per_loss_ji - per_loss_ji.min()) / (per_loss_ji.max() - per_loss_ji.min())
                    mask_ij, alpha_ij = self.GMM(per_loss_ij, self.bs, clean_pos='left')
                    mask_ji, alpha_ji = self.GMM(per_loss_ji, self.bs, clean_pos='left')
                elif self.fitting_type == 'sim':
                    # Calculate similarity-based mask and alpha values
                    per_sim_ij = sim_ij.reshape(-1, 1).cpu().numpy()
                    per_sim_ji = sim_ji.reshape(-1, 1).cpu().numpy()
                    per_sim_ij = (per_sim_ij - per_sim_ij.min()) / (per_sim_ij.max() - per_sim_ij.min())
                    per_sim_ji = (per_sim_ji - per_sim_ji.min()) / (per_sim_ji.max() - per_sim_ji.min())
                    mask_ij, alpha_ij = self.GMM(per_sim_ij, self.bs, clean_pos='right')
                    mask_ji, alpha_ji = self.GMM(per_sim_ji, self.bs, clean_pos='right')

            # Calculate loss with masks and alpha values
            loss_ij = - (torch.pow(sim_ij.softmax(1)[mask_ij], self.alpha_norm(alpha_ij[mask_ij])) * (sim_ij.log_softmax(1)[mask_ij])).mean()
            loss_ji = - (torch.pow(sim_ji.softmax(1)[mask_ji], self.alpha_norm(alpha_ji[mask_ji])) * (sim_ji.log_softmax(1)[mask_ji])).mean()

        return (loss_ij + loss_ji)
    
    def GMM(self, input, bs, clean_pos='left'):
        """Fit GMM and return mask and alpha values."""
        pair_label = self.gmm.fit_predict(input)

        if clean_pos == 'left':
            prob = self.gmm.predict_proba(input)[:, self.gmm.means_.argmin()]
            mask_gmm = (pair_label == self.gmm.means_.argmin()).reshape(bs, -1)
        elif clean_pos == 'right':
            prob = self.gmm.predict_proba(input)[:, self.gmm.means_.argmax()]
            mask_gmm = (pair_label == self.gmm.means_.argmax()).reshape(bs, -1)
        
        if self.mask_scheme == 'vital':
            center1, center2 = self.gmm.means_
            d = abs(center1 - center2)
            pos_indices_by_gmm = np.where(mask_gmm)
            input_matrix = input.reshape(bs, -1)
            loss_pos_by_gmm = input_matrix[pos_indices_by_gmm]
            loss_upper_bound = np.quantile(loss_pos_by_gmm, min(d, 1.0))
            confidence_mask = np.logical_and(0 < (input_matrix * mask_gmm), (input_matrix * mask_gmm) <= loss_upper_bound)
            mask = torch.tensor(confidence_mask).cuda()
            mask.fill_diagonal_(True)   # Rectify the diagonal

            if self.fix_alpha:
                alpha = torch.full((bs, bs), self.init_alpha).cuda()
            else:
                alpha = torch.tensor(prob.reshape(bs, -1)).cuda()
            return mask, alpha
        elif self.mask_scheme == 'gmm':
            if self.fix_alpha:
                alpha = torch.full((bs, bs), self.init_alpha).cuda()
            else:
                alpha = torch.tensor(prob.reshape(bs, -1)).cuda()
            return mask_gmm, alpha

    def alpha_norm(self, alpha):
        """Normalize alpha values."""
        alpha_max = torch.max(alpha)
        alpha_min = torch.min(alpha)
        if alpha_max == alpha_min:
            alpha_norm = torch.full_like(alpha, fill_value=self.init_alpha)
        else:
            alpha_norm = ((alpha - alpha_min) * self.init_alpha) / (alpha_max - alpha_min)
        return alpha_norm


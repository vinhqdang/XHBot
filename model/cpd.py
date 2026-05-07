import torch
import torch.nn as nn
import torch.nn.functional as F

class CPD(nn.Module):
    def __init__(self, in_channels, code_dim, original_feat_dim, num_classes=2, tau=0.5):
        super(CPD, self).__init__()
        self.code_dim = code_dim
        self.tau = tau
        
        # Projections
        self.mlp_b = nn.Sequential(
            nn.Linear(in_channels, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim)
        )
        self.mlp_s = nn.Sequential(
            nn.Linear(in_channels, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim)
        )
        
        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, original_feat_dim)
        )
        
        # Prototypes for classes (bot, human)
        self.prototypes = nn.Parameter(torch.randn(num_classes, code_dim))
        
        # Classifier
        self.classifier = nn.Linear(code_dim, num_classes)
        
    def forward(self, h_v):
        z_b = self.mlp_b(h_v)
        z_s = self.mlp_s(h_v)
        
        z_b_norm = F.normalize(z_b, dim=1)
        logits = self.classifier(z_b)
        
        return z_b, z_b_norm, z_s, logits
        
    def compute_losses(self, z_b, z_b_norm, z_s, logits, y_v, x_v, z_b_norm_aug=None):
        # Prototype contrastive loss
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim_proto = torch.matmul(z_b_norm, proto_norm.t()) / self.tau
        
        mask = (y_v >= 0) & (y_v < 2)
        if mask.sum() > 0:
            labeled_sim = sim_proto[mask]
            labeled_y = y_v[mask]
            numerator = torch.exp(labeled_sim[torch.arange(labeled_sim.size(0)), labeled_y])
            denominator = torch.exp(labeled_sim).sum(dim=1)
            loss_proto = -torch.log(numerator / (denominator + 1e-8)).mean()
        else:
            loss_proto = torch.tensor(0.0, device=z_b.device)
            
        # Disentanglement loss (correlation penalty)
        z_b_centered = z_b - z_b.mean(dim=0)
        z_s_centered = z_s - z_s.mean(dim=0)
        cov = torch.matmul(z_b_centered.t(), z_s_centered) / (z_b.size(0) - 1)
        loss_disent = torch.norm(cov, p='fro')
        
        # Reconstruction loss
        x_recon = self.decoder(z_s)
        loss_recon = F.mse_loss(x_recon, x_v)
        
        # Graph augmentation InfoNCE loss
        loss_infonce = torch.tensor(0.0, device=z_b.device)
        if z_b_norm_aug is not None:
            sim_aug = torch.matmul(z_b_norm, z_b_norm_aug.t()) / self.tau
            numerator = torch.exp(torch.diag(sim_aug))
            denominator = torch.exp(sim_aug).sum(dim=1)
            loss_infonce = -torch.log(numerator / (denominator + 1e-8)).mean()
            
        return loss_proto, loss_disent, loss_recon, loss_infonce

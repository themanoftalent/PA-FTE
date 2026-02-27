import torch
import torch.nn as nn

class MemoryAugmentedRisk(nn.Module):
    def __init__(self, d_model=128, d_mem=64):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_mem)
        )
        self.W_h = nn.Linear(d_model, 1)
        self.W_M = nn.Linear(d_mem, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_t, M_prev):
        psi_h = self.psi(h_t)
        M_new = 0.85 * M_prev + (1 - 0.85) * psi_h   # gamma = 0.85
        r_aug = self.sigmoid(self.W_h(h_t) + self.W_M(M_new))
        return r_aug, M_new

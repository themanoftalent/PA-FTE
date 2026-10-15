import torch
import torch.nn as nn

class CausalTransformerEncoder(nn.Module):
    def __init__(self, input_dim=12, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model))  # learnable

    def forward(self, x, mask=None):
        # x: [B, L, input_dim]
        B, L, _ = x.shape
        x = self.embedding(x) + self.pos_enc[:, :L, :]
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        if mask is not None:
            causal_mask = causal_mask | (~mask.unsqueeze(1).expand(-1, L, L))
        out = self.transformer(x, mask=causal_mask)
        return out[:, -1, :]  # h_t [B, d_model]

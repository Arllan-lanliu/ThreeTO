"""XLSR + CQCC cross-attention fusion followed by AASIST."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.SSL import XLSR
from model.backbone.ASSIST import AASIST

from cqcc_ssl.cqcc import CQCCExtractor


class SSLToCQCCCrossAttention(nn.Module):
    """Fuse SSL and CQCC features with SSL as Query and CQCC as Key/Value."""

    def __init__(
        self,
        ssl_dim: int = 1024,
        cqcc_dim: int = 30,
        out_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ssl_proj = nn.Linear(ssl_dim, out_dim) if ssl_dim != out_dim else nn.Identity()
        self.cqcc_proj = nn.Linear(cqcc_dim, out_dim)
        self.cross_attn = nn.MultiheadAttention(
            out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ssl_feat: torch.Tensor, cqcc_feat: torch.Tensor) -> torch.Tensor:
        q = self.ssl_proj(ssl_feat)
        kv = self.cqcc_proj(cqcc_feat)
        attended, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        fused = self.norm1(q + self.dropout(attended))
        return self.norm2(fused + self.dropout(self.ffn(fused)))


class CQCCXLSRAASIST(nn.Module):
    """audio -> XLSR + CQCC -> Cross-Attention(SSL->CQCC) -> AASIST."""

    def __init__(
        self,
        xlsr_model_dir: str,
        device: str = "cuda",
        freeze_xlsr: bool = False,
        assist_project_choice: int = 0,
        xlsr_selected_layers=None,
        xlsr_layer_fusion: str = "last",
        cqcc_dim: int = 30,
        cqcc_hop_length: int = 320,
        cqcc_n_bins: int = 84,
        cqcc_bins_per_octave: int = 12,
        cqcc_fmin: float = 15.625,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.xlsr = XLSR(
            model_dir=xlsr_model_dir,
            device=device,
            freeze=freeze_xlsr,
            selected_layers=xlsr_selected_layers,
            layer_fusion=xlsr_layer_fusion,
        )
        self.cqcc = CQCCExtractor(
            hop_length=cqcc_hop_length,
            n_bins=cqcc_n_bins,
            bins_per_octave=cqcc_bins_per_octave,
            n_coeffs=cqcc_dim,
            fmin=cqcc_fmin,
        )
        self.fusion = SSLToCQCCCrossAttention(
            ssl_dim=1024,
            cqcc_dim=cqcc_dim,
            out_dim=1024,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )
        self.w2vaasist = AASIST(
            in_dim=1024,
            assist_project_choice=assist_project_choice,
        )

    def forward(self, audio_data: torch.Tensor):
        ssl_feat = self.xlsr.extract_features(audio_data)
        cqcc_feat = self.cqcc(audio_data)
        fused_feat = self.fusion(ssl_feat, cqcc_feat)
        return self.w2vaasist(fused_feat)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self.xlsr, "freeze", False):
            self.xlsr.eval()
        return self


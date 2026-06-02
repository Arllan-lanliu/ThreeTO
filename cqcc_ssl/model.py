"""XLSR + CQCC cross-attention fusion followed by AASIST."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SSL import XLSR
from model.backbone.ASSIST import AASIST

from cqcc_ssl.cqcc import CQCCExtractor


class SSLToCQCCCrossAttention(nn.Module):
    """Fuse SSL and CQCC with SSL as Query and CQCC as Key/Value.

    This is the engineering implementation of the paper's SSL->SF cross-attention
    block. The attention direction is unchanged (query=SSL, key/value=CQCC), and
    LayerNorm plus an FFN are added on top of the residual cross-attention output.
    """

    def __init__(
        self,
        ssl_dim: int = 1024,
        cqcc_dim: int = 60,
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
        # query=SSL, key/value=CQCC (paper SSL->SF direction).
        q = self.ssl_proj(ssl_feat)
        kv = self.cqcc_proj(cqcc_feat)
        attended, _ = self.cross_attn(query=q, key=kv, value=kv, need_weights=False)
        fused = self.norm1(q + self.dropout(attended))
        return self.norm2(fused + self.dropout(self.ffn(fused)))


class CQCCXLSRAASIST(nn.Module):
    """audio -> XLSR + CQCC -> Cross-Attention(SSL->CQCC) -> AASIST.

    Default config (fusion_dim=1024, align_cqcc_to_ssl=False) keeps the current
    engineering setup. For a paper-style setup, use fusion_dim=128 and
    align_cqcc_to_ssl=True. AASIST accepts arbitrary in_dim and projects to 128
    internally, so fusion_dim=128 is supported.
    """

    def __init__(
        self,
        xlsr_model_dir: str,
        device: str = "cuda",
        freeze_xlsr: bool = False,
        assist_project_choice: int = 0,
        xlsr_selected_layers=None,
        xlsr_layer_fusion: str = "last",
        cqcc_n_coeffs: int = 20,
        cqcc_hop_length: int = 160,
        cqcc_n_bins: int = 672,
        cqcc_bins_per_octave: int = 96,
        cqcc_fmin: float = 15.625,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
        fusion_dim: int = 128,
        align_cqcc_to_ssl: bool = True,
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
            n_coeffs=cqcc_n_coeffs,
            fmin=cqcc_fmin,
        )
        self.fusion_dim = int(fusion_dim)
        self.align_cqcc_to_ssl = bool(align_cqcc_to_ssl)

        self.fusion = SSLToCQCCCrossAttention(
            ssl_dim=1024,
            cqcc_dim=self.cqcc.out_dim,
            out_dim=self.fusion_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )
        self.w2vaasist = AASIST(
            in_dim=self.fusion_dim,
            assist_project_choice=assist_project_choice,
        )

    def _align_cqcc_time(self, cqcc_feat: torch.Tensor, target_len: int) -> torch.Tensor:
        """Interpolate CQCC features from (B, T_cqcc, C) to (B, target_len, C)."""
        if cqcc_feat.size(1) == target_len:
            return cqcc_feat
        cqcc_feat = cqcc_feat.transpose(1, 2)
        cqcc_feat = F.interpolate(
            cqcc_feat,
            size=target_len,
            mode="linear",
            align_corners=False,
        )
        return cqcc_feat.transpose(1, 2)

    def forward(self, audio_data: torch.Tensor):
        ssl_feat = self.xlsr.extract_features(audio_data)
        cqcc_feat = self.cqcc(audio_data)

        if ssl_feat.dim() != 3:
            raise RuntimeError(f"Expected ssl_feat to be 3D (B, T, C), got {ssl_feat.shape}")

        if cqcc_feat.dim() != 3:
            raise RuntimeError(f"Expected cqcc_feat to be 3D (B, T, C), got {cqcc_feat.shape}")

        if ssl_feat.size(0) != cqcc_feat.size(0):
            raise RuntimeError(
                f"Batch size mismatch: ssl_feat batch={ssl_feat.size(0)}, "
                f"cqcc_feat batch={cqcc_feat.size(0)}"
            )

        if self.align_cqcc_to_ssl:
            cqcc_feat = self._align_cqcc_time(cqcc_feat, ssl_feat.size(1))
            if cqcc_feat.size(1) != ssl_feat.size(1):
                raise RuntimeError(
                    f"CQCC alignment failed: ssl_len={ssl_feat.size(1)}, "
                    f"cqcc_len={cqcc_feat.size(1)}"
                )

        fused_feat = self.fusion(ssl_feat, cqcc_feat)
        return self.w2vaasist(fused_feat)

    def train(self, mode: bool = True):
        super().train(mode)
        self.xlsr.eval()
        return self

    def eval(self):
        return self.train(False)
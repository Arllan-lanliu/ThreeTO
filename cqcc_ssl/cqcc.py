"""CQCC feature extraction utilities.

The training pipeline passes waveform tensors directly to the model.  To keep
the root data pipeline unchanged, CQCCs are extracted inside the model from the
same padded waveform batch used by XLSR.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CQCCExtractor(nn.Module):
    """Batch CQCC extractor based on CQT magnitude + log compression + DCT."""

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        n_bins: int = 84,
        bins_per_octave: int = 12,
        n_coeffs: int = 30,
        fmin: float = 15.625,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_bins = int(n_bins)
        self.bins_per_octave = int(bins_per_octave)
        self.n_coeffs = int(n_coeffs)
        self.fmin = float(fmin)
        self.eps = float(eps)
        self.out_dim = self.n_coeffs

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Return CQCCs as ``(B, T, n_coeffs)`` on ``audio.device``."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        device = audio.device
        dtype = audio.dtype
        wavs = audio.detach().float().cpu().numpy()

        feats = [self._extract_one(wav) for wav in wavs]
        max_t = max(feat.shape[0] for feat in feats)
        padded = np.zeros((len(feats), max_t, self.n_coeffs), dtype=np.float32)
        for i, feat in enumerate(feats):
            padded[i, : feat.shape[0], :] = feat
        return torch.as_tensor(padded, device=device, dtype=dtype)

    def _extract_one(self, wav: np.ndarray) -> np.ndarray:
        import librosa
        from scipy.fftpack import dct

        wav = np.asarray(wav, dtype=np.float32)
        cqt = librosa.cqt(
            wav,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            pad_mode="reflect",
        )
        log_power = np.log(np.abs(cqt) ** 2 + self.eps)
        coeff = dct(log_power, type=2, axis=0, norm="ortho")[: self.n_coeffs]
        coeff = coeff.T.astype(np.float32, copy=False)

        mean = coeff.mean(axis=0, keepdims=True)
        std = coeff.std(axis=0, keepdims=True)
        return (coeff - mean) / (std + 1e-5)


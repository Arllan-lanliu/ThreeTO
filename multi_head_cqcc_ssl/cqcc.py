# -*- coding: utf-8 -*-
"""CQCC feature extraction utilities for the multi_head_cqcc_ssl experiment.

The data pipeline still passes padded waveform tensors to the model. CQCC
features are extracted inside the model from that same waveform batch so the
rest of the training and inference code can remain unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CQCCExtractor(nn.Module):
    """Batch CQCC extractor: pre-emphasis, CQT/log-frequency power, DCT, deltas."""

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 160,
        n_bins: int = 672,
        bins_per_octave: int = 96,
        n_coeffs: int = 20,
        fmin: float = 15.625,
        eps: float = 1e-10,
        pre_emphasis: float = 0.97,
        use_deltas: bool = True,
        backend: str = "torch",
        torch_n_fft: int = 2048,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.n_bins = int(n_bins)
        self.bins_per_octave = int(bins_per_octave)
        self.n_coeffs = int(n_coeffs)
        self.fmin = float(fmin)
        self.eps = float(eps)
        self.pre_emphasis = float(pre_emphasis)
        self.use_deltas = bool(use_deltas)
        self.backend = str(backend).strip().lower()
        self.torch_n_fft = int(torch_n_fft)
        if self.backend not in {"librosa", "torch"}:
            raise ValueError(f"Unknown CQCC backend {backend!r}; use 'librosa' or 'torch'.")
        if self.torch_n_fft <= 0:
            raise ValueError(f"torch_n_fft must be positive, got {torch_n_fft!r}.")
        self.out_dim = self.n_coeffs * 3 if self.use_deltas else self.n_coeffs
        self.register_buffer("_window", torch.empty(0), persistent=False)
        self.register_buffer("_logfreq_fb", torch.empty(0), persistent=False)
        self.register_buffer("_dct_mat", torch.empty(0), persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Return CQCC-like features as ``(B, T, out_dim)`` on ``audio.device``."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if self.backend == "torch":
            return self._forward_torch(audio)

        device = audio.device
        dtype = audio.dtype
        wavs = audio.detach().float().cpu().numpy()
        feats = [self._extract_one(wav) for wav in wavs]
        max_t = max(feat.shape[0] for feat in feats)
        padded = np.zeros((len(feats), max_t, self.out_dim), dtype=np.float32)
        for i, feat in enumerate(feats):
            padded[i, : feat.shape[0], :] = feat
        return torch.as_tensor(padded, device=device, dtype=dtype)

    def _extract_one(self, wav: np.ndarray) -> np.ndarray:
        import librosa
        from scipy.fftpack import dct

        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = np.squeeze(wav)
        if wav.size <= 1:
            wav = np.pad(wav, (0, max(0, 2 - wav.size)))

        wav = np.append(wav[0], wav[1:] - self.pre_emphasis * wav[:-1])
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

        if self.use_deltas:
            delta = librosa.feature.delta(coeff, order=1, axis=0)
            delta2 = librosa.feature.delta(coeff, order=2, axis=0)
            feat = np.concatenate([coeff, delta, delta2], axis=1)
        else:
            feat = coeff

        mean = feat.mean(axis=0, keepdims=True)
        std = feat.std(axis=0, keepdims=True)
        return (feat - mean) / (std + 1e-5)

    def _forward_torch(self, audio: torch.Tensor) -> torch.Tensor:
        """GPU-friendly CQCC-like frontend with the same output contract."""
        x = audio.float()
        x = self._pre_emphasis_torch(x)
        window = self._hann_window(x.device, x.dtype)
        spec = torch.stft(
            x,
            n_fft=self.torch_n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        power = spec.abs().square().transpose(1, 2)
        fb = self._logfreq_filterbank(power.device, power.dtype, power.size(-1))
        log_power = torch.log(torch.matmul(power, fb.transpose(0, 1)) + self.eps)
        dct_mat = self._dct_matrix(log_power.device, log_power.dtype)
        coeff = torch.matmul(log_power, dct_mat.transpose(0, 1))

        if self.use_deltas:
            delta = self._delta_torch(coeff)
            delta2 = self._delta_torch(delta)
            feat = torch.cat([coeff, delta, delta2], dim=-1)
        else:
            feat = coeff

        mean = feat.mean(dim=1, keepdim=True)
        std = feat.std(dim=1, keepdim=True, unbiased=False)
        return ((feat - mean) / (std + 1e-5)).to(dtype=audio.dtype)

    def _pre_emphasis_torch(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.size(-1) <= 1:
            return audio
        first = audio[..., :1]
        rest = audio[..., 1:] - self.pre_emphasis * audio[..., :-1]
        return torch.cat([first, rest], dim=-1)

    def _hann_window(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._window.numel() != self.torch_n_fft
            or self._window.device != device
            or self._window.dtype != dtype
        ):
            self._window = torch.hann_window(
                self.torch_n_fft,
                periodic=True,
                device=device,
                dtype=dtype,
            )
        return self._window

    def _logfreq_filterbank(
        self,
        device: torch.device,
        dtype: torch.dtype,
        n_freq: int,
    ) -> torch.Tensor:
        if (
            self._logfreq_fb.shape == (self.n_bins, n_freq)
            and self._logfreq_fb.device == device
            and self._logfreq_fb.dtype == dtype
        ):
            return self._logfreq_fb

        freqs = torch.linspace(0, self.sample_rate / 2, n_freq, device=device, dtype=dtype)
        centers = self.fmin * (
            2.0 ** (torch.arange(self.n_bins, device=device, dtype=dtype) / self.bins_per_octave)
        )
        centers = centers.clamp(max=self.sample_rate / 2)
        lower = torch.cat([centers[:1] / (2.0 ** (1.0 / self.bins_per_octave)), centers[:-1]])
        upper = torch.cat([centers[1:], centers[-1:] * (2.0 ** (1.0 / self.bins_per_octave))])
        upper = upper.clamp(max=self.sample_rate / 2)

        left = (freqs.unsqueeze(0) - lower.unsqueeze(1)) / (
            centers.unsqueeze(1) - lower.unsqueeze(1)
        ).clamp_min(1e-6)
        right = (upper.unsqueeze(1) - freqs.unsqueeze(0)) / (
            upper.unsqueeze(1) - centers.unsqueeze(1)
        ).clamp_min(1e-6)
        fb = torch.minimum(left, right).clamp_min(0.0)
        fb = fb / fb.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self._logfreq_fb = fb
        return self._logfreq_fb

    def _dct_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._dct_mat.shape == (self.n_coeffs, self.n_bins)
            and self._dct_mat.device == device
            and self._dct_mat.dtype == dtype
        ):
            return self._dct_mat

        n = torch.arange(self.n_bins, device=device, dtype=dtype)
        k = torch.arange(self.n_coeffs, device=device, dtype=dtype).unsqueeze(1)
        mat = torch.cos(torch.pi / self.n_bins * (n + 0.5).unsqueeze(0) * k)
        mat[0] *= (1.0 / self.n_bins) ** 0.5
        if self.n_coeffs > 1:
            mat[1:] *= (2.0 / self.n_bins) ** 0.5
        self._dct_mat = mat
        return self._dct_mat

    @staticmethod
    def _delta_torch(feat: torch.Tensor) -> torch.Tensor:
        if feat.size(1) < 3:
            return torch.zeros_like(feat)
        x = feat.transpose(1, 2)
        x = F.pad(x, (1, 1), mode="replicate")
        return ((x[:, :, 2:] - x[:, :, :-2]) * 0.5).transpose(1, 2)

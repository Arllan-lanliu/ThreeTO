"""Minimal shape sanity checks for CQCC + XLSR fusion."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqcc_ssl.cqcc import CQCCExtractor
from cqcc_ssl.model import CQCCXLSRAASIST, SSLToCQCCCrossAttention


class _MockXLSR(nn.Module):
    """Return fixed-length SSL features without loading wav2vec2 weights."""

    def __init__(self, ssl_frames: int = 201, ssl_dim: int = 1024, freeze: bool = False, **_kwargs):
        super().__init__()
        self.ssl_frames = ssl_frames
        self.ssl_dim = ssl_dim
        self.freeze = freeze

    def extract_features(self, audio_data: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            audio_data.size(0),
            self.ssl_frames,
            self.ssl_dim,
            device=audio_data.device,
            dtype=audio_data.dtype,
        )


def _build_model(
    fusion_dim: int,
    align_cqcc_to_ssl: bool,
    device: torch.device,
) -> CQCCXLSRAASIST:
    with patch("cqcc_ssl.model.XLSR", _MockXLSR):
        model = CQCCXLSRAASIST(
            xlsr_model_dir="unused",
            device=str(device),
            fusion_dim=fusion_dim,
            align_cqcc_to_ssl=align_cqcc_to_ssl,
        )
    return model.to(device)


def test_cqcc_extractor_defaults():
    ext = CQCCExtractor()
    assert ext.hop_length == 160
    assert ext.n_bins == 672
    assert ext.bins_per_octave == 96
    assert ext.n_coeffs == 20
    assert ext.pre_emphasis == 0.97
    assert ext.use_deltas is True
    assert ext.out_dim == 60


def test_torch_cqcc_extractor_shape():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = CQCCExtractor(backend="torch", n_bins=96, torch_n_fft=512).to(device)
    audio = torch.randn(2, 16000, device=device)
    feat = ext(audio)
    assert feat.device == audio.device
    assert feat.shape[0] == 2
    assert feat.shape[2] == 60
    assert feat.shape[1] > 0


def test_align_cqcc_time():
    device = torch.device("cpu")
    model = _build_model(fusion_dim=1024, align_cqcc_to_ssl=False, device=device)
    cqcc = torch.randn(2, 404, 60, device=device)
    aligned = model._align_cqcc_time(cqcc, target_len=201)
    assert aligned.shape == (2, 201, 60)

    same = model._align_cqcc_time(aligned, target_len=201)
    assert same.shape == (2, 201, 60)


def test_fusion_block():
    fusion = SSLToCQCCCrossAttention(cqcc_dim=60, out_dim=1024)
    ssl_feat = torch.randn(2, 201, 1024)
    cqcc_feat = torch.randn(2, 404, 60)
    fused = fusion(ssl_feat, cqcc_feat)
    assert fused.shape == (2, 201, 1024)


def test_forward_default_and_aligned():
    device = torch.device("cpu")
    audio = torch.randn(2, 64600, device=device)

    for align in (False, True):
        model = _build_model(
            fusion_dim=1024,
            align_cqcc_to_ssl=align,
            device=device,
        )
        model.eval()

        ssl_feat = model.xlsr.extract_features(audio)
        cqcc_feat = model.cqcc(audio)
        assert ssl_feat.shape == (2, 201, 1024)
        assert cqcc_feat.shape[0] == 2
        assert cqcc_feat.shape[2] == 60
        assert cqcc_feat.shape[1] in (402, 403, 404)

        cqcc_for_fusion = (
            model._align_cqcc_time(cqcc_feat, ssl_feat.size(1))
            if align
            else cqcc_feat
        )
        if align:
            assert cqcc_for_fusion.shape == (2, 201, 60)

        fused_feat = model.fusion(ssl_feat, cqcc_for_fusion)
        assert fused_feat.shape == (2, 201, 1024)

        last_hidden, output = model(audio)
        assert last_hidden.ndim == 2
        assert output.shape == (2, 2)


def test_forward_paper_style_fusion_dim():
    device = torch.device("cpu")
    audio = torch.randn(2, 64600, device=device)
    model = _build_model(
        fusion_dim=128,
        align_cqcc_to_ssl=True,
        device=device,
    )
    model.eval()

    ssl_feat = model.xlsr.extract_features(audio)
    cqcc_feat = model.cqcc(audio)
    cqcc_feat = model._align_cqcc_time(cqcc_feat, ssl_feat.size(1))
    fused_feat = model.fusion(ssl_feat, cqcc_feat)
    assert fused_feat.shape == (2, 201, 128)

    last_hidden, output = model(audio)
    assert output.shape == (2, 2)


if __name__ == "__main__":
    test_cqcc_extractor_defaults()
    test_align_cqcc_time()
    test_fusion_block()
    test_forward_default_and_aligned()
    test_forward_paper_style_fusion_dim()
    print("All shape sanity checks passed.")

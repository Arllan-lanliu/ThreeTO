from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

from model.SSL import XLSR
from model.backbone.ASSIST import AASIST


@dataclass(frozen=True)
class HeadRange:
    name: str
    start: float
    end: float


def _parse_ranges(value) -> List[HeadRange]:
    """Parse head ranges from YAML/list/string into validated ratio ranges."""
    if value is None:
        items = [
            ("front", 0.00, 0.50),
            ("middle", 0.40, 0.90),
            ("tail", 0.80, 1.00),
        ]
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            items = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            items = []
            for part in text.split(";"):
                part = part.strip()
                if not part:
                    continue
                name, span = part.split(":", 1)
                start, end = span.split("-", 1)
                items.append((name.strip(), float(start), float(end)))
    else:
        items = value

    out: List[HeadRange] = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            name = str(item.get("name", f"head{i + 1}"))
            start = float(item["start"])
            end = float(item["end"])
        else:
            if len(item) == 2:
                name = f"head{i + 1}"
                start, end = item
            else:
                name, start, end = item
            name = str(name)
            start = float(start)
            end = float(end)
        if not (0.0 <= start < end <= 1.0):
            raise ValueError(f"Invalid head range {item!r}; expected 0 <= start < end <= 1.")
        out.append(HeadRange(name=name, start=start, end=end))
    return out


def _slice_by_ratio(features: torch.Tensor, start: float, end: float, min_frames: int) -> torch.Tensor:
    """Slice (B, T, D) by ratio while keeping enough frames for AASIST."""
    total_frames = int(features.size(1))
    lo = int(round(total_frames * start))
    hi = int(round(total_frames * end))
    lo = max(0, min(lo, total_frames - 1))
    hi = max(lo + 1, min(hi, total_frames))

    if hi - lo < min_frames:
        need = min(min_frames, total_frames)
        center = (lo + hi) // 2
        lo = max(0, center - need // 2)
        hi = min(total_frames, lo + need)
        lo = max(0, hi - need)
    return features[:, lo:hi, :]


class MultiHeadXLSRAASIST(nn.Module):
    """
    XLS-R frontend shared by one total AASIST head and several local AASIST heads.

    The model returns combined logits:

        total_logits + special_weight * mean(local_head_logits)

    All heads use AASIST with the default Linear projector (assist_project_choice=0).
    """

    def __init__(
        self,
        xlsr_model_dir: str,
        device: str = "cuda",
        selected_layers: Sequence[int] | None = None,
        layer_fusion: str = "cat_proj_v1",
        special_head_ranges=None,
        special_weight: float = 0.5,
        min_special_frames: int = 32,
    ):
        super().__init__()
        self.frontend = XLSR(
            model_dir=xlsr_model_dir,
            device=device,
            freeze=False,
            selected_layers=selected_layers,
            layer_fusion=layer_fusion,
        )
        self.total_head = AASIST(in_dim=1024, assist_project_choice=0)
        self.head_ranges = _parse_ranges(special_head_ranges)
        self.special_heads = nn.ModuleList(
            [AASIST(in_dim=1024, assist_project_choice=0) for _ in self.head_ranges]
        )
        self.special_weight = float(special_weight)
        self.min_special_frames = int(min_special_frames)

    def forward(self, audio_data, return_head_logits: bool = False):
        features = self.frontend.extract_features(audio_data)
        if isinstance(features, tuple):
            features = features[0]

        total_hidden, total_logits = self.total_head(features)
        special_logits = []
        special_hidden = []

        for head, span in zip(self.special_heads, self.head_ranges):
            local_feat = _slice_by_ratio(
                features,
                span.start,
                span.end,
                min_frames=self.min_special_frames,
            )
            hidden, logits = head(local_feat)
            special_hidden.append(hidden)
            special_logits.append(logits)

        if special_logits:
            special_mean = torch.stack(special_logits, dim=0).mean(dim=0)
            combined_logits = total_logits + self.special_weight * special_mean
        else:
            combined_logits = total_logits

        if not return_head_logits:
            return total_hidden, combined_logits

        return {
            "hidden": total_hidden,
            "logits": combined_logits,
            "total_logits": total_logits,
            "special_logits": special_logits,
            "special_hidden": special_hidden,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        return self


def build_multi_head_model(args) -> MultiHeadXLSRAASIST:
    selected_layers = getattr(args, "xlsr_selected_layers", None)
    if selected_layers is None:
        selected_layers = getattr(args, "selected_layers", None)
    layer_fusion = getattr(args, "xlsr_layer_fusion", None)
    if layer_fusion is None:
        layer_fusion = getattr(args, "layer_fusion", "cat_proj_v1")

    return MultiHeadXLSRAASIST(
        xlsr_model_dir=args.xlsr,
        device=str(getattr(args, "device", "cuda")),
        selected_layers=selected_layers,
        layer_fusion=layer_fusion,
        special_head_ranges=getattr(args, "special_head_ranges", None),
        special_weight=float(getattr(args, "special_weight", 0.5)),
        min_special_frames=int(getattr(args, "min_special_frames", 32)),
    )

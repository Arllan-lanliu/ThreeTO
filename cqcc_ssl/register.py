"""Register CQCC+XLSR fusion models with the root model registry."""

from __future__ import annotations

from model.model import register_model

from cqcc_ssl.model import CQCCXLSRAASIST


def _assist_project_choice(args) -> int:
    return int(getattr(args, "assist_project_choice", 0))


def _xlsr_kw(args) -> dict:
    return {
        "xlsr_selected_layers": getattr(args, "xlsr_selected_layers", None),
        "xlsr_layer_fusion": getattr(args, "xlsr_layer_fusion", "last"),
    }


def _cqcc_kw(args) -> dict:
    return {
        "cqcc_n_coeffs": int(getattr(args, "cqcc_n_coeffs", 20)),
        "cqcc_hop_length": int(getattr(args, "cqcc_hop_length", 160)),
        "cqcc_n_bins": int(getattr(args, "cqcc_n_bins", 672)),
        "cqcc_bins_per_octave": int(getattr(args, "cqcc_bins_per_octave", 96)),
        "cqcc_fmin": float(getattr(args, "cqcc_fmin", 15.625)),
        "fusion_heads": int(getattr(args, "cqcc_ssl_fusion_heads", 8)),
        "fusion_dropout": float(getattr(args, "cqcc_ssl_fusion_dropout", 0.1)),
        "fusion_dim": int(getattr(args, "cqcc_ssl_fusion_dim", 1024)),
        "align_cqcc_to_ssl": bool(getattr(args, "cqcc_ssl_align_cqcc_to_ssl", False)),
    }


@register_model("ft-cqccxlsraasist")
def _build_ft_cqcc_xlsr_aasist(args):
    return CQCCXLSRAASIST(
        xlsr_model_dir=args.xlsr,
        device=str(getattr(args, "device", "cuda")),
        freeze_xlsr=False,
        assist_project_choice=_assist_project_choice(args),
        **_xlsr_kw(args),
        **_cqcc_kw(args),
    )


@register_model("fr-cqccxlsraasist")
def _build_fr_cqcc_xlsr_aasist(args):
    return CQCCXLSRAASIST(
        xlsr_model_dir=args.xlsr,
        device=str(getattr(args, "device", "cuda")),
        freeze_xlsr=True,
        assist_project_choice=_assist_project_choice(args),
        **_xlsr_kw(args),
        **_cqcc_kw(args),
    )


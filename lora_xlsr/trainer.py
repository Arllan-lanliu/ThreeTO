from __future__ import annotations

import os

import torch
import torch.nn as nn

from model.model import build_model as build_base_model
from utils.optimizer import SAM

from lora_xlsr.modeling import LoRAXLSRAASIST


def _dev(args) -> str:
    return str(getattr(args, "device", "cuda"))


def _assist_project_choice(args) -> int:
    return int(getattr(args, "assist_project_choice", 0))


def _xlsr_frontend_kw(args) -> dict:
    sl = getattr(args, "xlsr_selected_layers", None)
    if sl is None:
        sl = getattr(args, "selected_layers", None)
    lf = getattr(args, "xlsr_layer_fusion", None)
    if lf is None:
        lf = getattr(args, "layer_fusion", "last")
    return {"selected_layers": sl, "layer_fusion": str(lf)}


def _lora_targets(args):
    targets = getattr(args, "lora_targets", ("q_proj", "k_proj", "v_proj", "out_proj"))
    if isinstance(targets, str):
        return tuple(t.strip() for t in targets.split(",") if t.strip())
    return tuple(targets)


def build_model(args):
    """Build the local LoRA variant, falling back to the project registry if off."""
    use_lora = bool(getattr(args, "use_lora", True))
    if use_lora and args.model in {"ft-w2v2aasist", "ft-w2v2assist_baseline"}:
        return LoRAXLSRAASIST(
            model_dir=args.xlsr,
            device=_dev(args),
            freeze=False,
            assist_project_choice=_assist_project_choice(args),
            use_lora=True,
            lora_r=int(getattr(args, "lora_r", 8)),
            lora_alpha=int(getattr(args, "lora_alpha", 16)),
            lora_targets=_lora_targets(args),
            lora_dropout=float(getattr(args, "lora_dropout", 0.1)),
            **_xlsr_frontend_kw(args),
        )
    return build_base_model(args)


def _split_lora_and_head_params(model):
    lora_params = []
    head_params = []
    lora_names = []
    head_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
            lora_names.append(name)
        else:
            head_params.append(param)
            head_names.append(name)
    return lora_params, head_params, lora_names, head_names


def _preview_names(title, names, limit=12):
    print(f"{title} parameter tensors: {len(names)}")
    for name in names[:limit]:
        print(f"  {name}")
    if len(names) > limit:
        print(f"  ... ({len(names) - limit} more)")


def build_model_and_optimizer(args):
    model = build_model(args).to(args.device)
    lora_params, head_params, lora_names, head_names = _split_lora_and_head_params(model)
    lora_count = sum(p.numel() for p in lora_params)
    head_count = sum(p.numel() for p in head_params)
    trainable_count = lora_count + head_count
    total_count = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable optimizer params: {trainable_count:,} / {total_count:,} "
        f"({100.0 * trainable_count / max(total_count, 1):.4f}%)"
    )
    print(f"LoRA params count: {lora_count:,}")
    print(f"Head params count: {head_count:,}")
    _preview_names("LoRA", lora_names)
    _preview_names("Head", head_names)

    if args.SAM or args.CSAM:
        optimizer = SAM(
            [
                {"params": lora_params, "lr": args.lr},
                {"params": head_params, "lr": getattr(args, "head_lr", 1e-5)},
            ],
            torch.optim.Adam,
            betas=(args.beta_1, args.beta_2),
            weight_decay=0.0005,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": args.lr},
                {"params": head_params, "lr": getattr(args, "head_lr", 1e-5)},
            ],
            betas=(args.beta_1, args.beta_2),
            eps=args.eps,
            weight_decay=1e-4,
        )

    _worse_sentinel = -float("inf") if args.save_best_by == "f1" else float("inf")
    _best_full_vals = {
        "loss": float("inf"),
        "eer": float("inf"),
        "f1": -float("inf"),
    }
    resume_info = dict(
        start_epoch=0,
        global_step=0,
        best_sample_val=_worse_sentinel,
        best_full_val=_worse_sentinel,
        best_full_vals=_best_full_vals,
        no_improve=0,
    )

    if args.continue_training:
        ckpt_path = os.path.join(args.out_fold, "checkpoint", "latest.pt")
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            resume_info.update(
                start_epoch=ckpt["epoch"] + 1,
                global_step=ckpt.get("global_step", 0),
                best_sample_val=ckpt.get("best_sample_val", _worse_sentinel),
                best_full_val=ckpt.get("best_full_val", _worse_sentinel),
                best_full_vals={**_best_full_vals, **ckpt.get("best_full_vals", {})},
                no_improve=ckpt.get("no_improve", 0),
            )
            print(f"Resumed from epoch {resume_info['start_epoch']}")
        else:
            print("Checkpoint not found, training from scratch.")

    class_weight = torch.FloatTensor(
        [4.0, 1.0] if args.train_task == "atadd-track1" else [3.5, 1.0]
    ).to(args.device)
    print(f"Class weight: {class_weight.tolist()}  |  save_best_by: {args.save_best_by}")

    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    return model, optimizer, criterion, resume_info

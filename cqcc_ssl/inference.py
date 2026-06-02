"""Inference entry point for CQCC + XLSR fusion experiments."""

from __future__ import annotations

import json
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cqcc_ssl.register  # noqa: F401 - registers local models
from scripts.inference import (
    _find_model_checkpoint,
    _init_args,
    build_model_for_inference,
    gen_binary_score,
    gen_score,
)


if __name__ == "__main__":
    args = _init_args()

    ckpt_path = _find_model_checkpoint(args.model_path)
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    print("Model:", args.model)
    model = build_model_for_inference(args)
    model.load_state_dict(checkpoint)
    model.eval()

    gen_score(model, args)
    print(f"Logit scores saved to: {args.score_file}")

    gen_binary_score(args.score_file, args.binary_score_file, args._decision_threshold)
    print(f"Binary predictions saved to: {args.binary_score_file}")

    meta_path = os.path.join(
        os.path.dirname(args.binary_score_file),
        f"{args.eval_task}_binary_threshold_meta.json",
    )
    meta = {
        "eval_threshold_mode": args.eval_threshold_mode,
        "decision_threshold": args._decision_threshold,
        "score_file": args.score_file,
        "binary_score_file": args.binary_score_file,
    }
    if args.eval_threshold_mode == "eer":
        meta["val_score_file"] = args.val_score_file
        meta["val_label_csv"] = args._val_label_csv_used
        meta["dev_eer"] = args._dev_eer_for_threshold
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    print(f"Threshold metadata: {meta_path}")

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.inference as base_inference  # noqa: E402
from lora_xlsr.runtime import attach_lora_args  # noqa: E402
from lora_xlsr.trainer import build_model  # noqa: E402

_BASE_INIT_ARGS = base_inference._init_args


def _model_config_from_argv(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_path", type=str, required=True)
    known, _ = parser.parse_known_args(argv)
    return os.path.join(known.model_path, "config.yaml")


def _init_args(argv=None):
    args = _BASE_INIT_ARGS(argv)
    return attach_lora_args(args, _model_config_from_argv(argv))


base_inference.build_model = build_model
base_inference._init_args = _init_args


if __name__ == "__main__":
    args = _init_args()
    ckpt_path = base_inference._find_model_checkpoint(args.model_path)
    checkpoint = base_inference.torch.load(ckpt_path, map_location=args.device)

    print("Model:", args.model)
    model = base_inference.build_model_for_inference(args)
    model.load_state_dict(checkpoint)
    model.eval()

    base_inference.gen_score(model, args)
    print(f"Logit scores saved to: {args.score_file}")

    base_inference.gen_binary_score(
        args.score_file,
        args.binary_score_file,
        args._decision_threshold,
    )
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
        base_inference.json.dump(meta, mf, indent=2)
    print(f"Threshold metadata: {meta_path}")

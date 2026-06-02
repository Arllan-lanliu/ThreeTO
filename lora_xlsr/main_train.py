from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main_train as base_train  # noqa: E402
from lora_xlsr.runtime import attach_lora_args, load_lora_config  # noqa: E402
from lora_xlsr.trainer import build_model_and_optimizer  # noqa: E402


def _cli_config_path():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    known, _ = parser.parse_known_args()
    if known.resume:
        return os.path.join(known.resume, "config.yaml")
    return known.config

def _save_lora_config(args, lora_cfg):
    path = os.path.join(args.out_fold, "config.yaml")
    if not os.path.isfile(path):
        return
    try:
        import yaml
    except ImportError:
        return
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw["lora"] = dict(lora_cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            raw,
            f,
            default_flow_style=False,
            allow_unicode=True,
            indent=2,
            sort_keys=False,
        )


def initParams():
    config_path = _cli_config_path()
    lora_cfg = load_lora_config(config_path)
    args = base_train.initParams()
    attach_lora_args(args, config_path)
    _save_lora_config(args, lora_cfg)
    print(
        "LoRA config: "
        f"use_lora={args.use_lora}, r={args.lora_r}, alpha={args.lora_alpha}, "
        f"targets={args.lora_targets}, dropout={args.lora_dropout}, "
        f"head_lr={args.head_lr}"
    )
    return args


base_train.build_model_and_optimizer = build_model_and_optimizer
train = base_train.train


if __name__ == "__main__":
    args = initParams()
    train(args)

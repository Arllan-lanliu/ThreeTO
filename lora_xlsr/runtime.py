from __future__ import annotations

import os


LORA_DEFAULTS = {
    "use_lora": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj"],
    "lora_dropout": 0.1,
    "head_lr": 1e-5,
}


def load_lora_config(path):
    data = dict(LORA_DEFAULTS)
    if not path or not os.path.isfile(path):
        return data
    try:
        import yaml
    except ImportError:
        return data
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw_lora = raw.get("lora", {})
    for key in LORA_DEFAULTS:
        if key in raw:
            data[key] = raw[key]
        if isinstance(raw_lora, dict) and key in raw_lora:
            data[key] = raw_lora[key]
    return data


def attach_lora_args(args, config_path):
    for key, value in load_lora_config(config_path).items():
        setattr(args, key, value)
    return args

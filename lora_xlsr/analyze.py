from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.analyze as base_analyze  # noqa: E402
from lora_xlsr.runtime import attach_lora_args  # noqa: E402
from lora_xlsr.trainer import build_model  # noqa: E402

_BASE_PARSE_ARGS = base_analyze.parse_args


def _model_config_from_argv(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_path", type=str, required=True)
    known, _ = parser.parse_known_args(argv)
    return os.path.join(known.model_path, "config.yaml")


def parse_args():
    args = _BASE_PARSE_ARGS()
    return attach_lora_args(args, _model_config_from_argv())


base_analyze.build_model = build_model
base_analyze.parse_args = parse_args


if __name__ == "__main__":
    base_analyze.main()

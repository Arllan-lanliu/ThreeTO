from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import atadd_eval_dataset
from multi_head_trunk.main_train import _inject_multi_head_args
from multi_head_trunk.model import build_multi_head_model
from utils.config import ATADDConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Multi-head XLSR-AASIST inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_task", choices=["atadd-track1", "atadd-track2"], default=None)
    parser.add_argument("--eval_audio", default=None)
    parser.add_argument("--score_file", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    temp, _ = parser.parse_known_args(argv)

    cfg_path = os.path.join(temp.model_path, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.yaml not found: {cfg_path}")
    cfg = ATADDConfig.from_yaml(cfg_path)
    train_args = cfg.to_namespace()
    args = parser.parse_args(argv)
    for k, v in vars(train_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args = _inject_multi_head_args(args, ["--config", cfg_path])

    if args.batch_size is None:
        args.batch_size = int(getattr(train_args, "batch_size", 24))
    if args.eval_task is None:
        args.eval_task = getattr(train_args, "train_task", "atadd-track1")
    if args.eval_audio is None:
        key = "atadd_t1_eval_audio" if args.eval_task == "atadd-track1" else "atadd_t2_eval_audio"
        args.eval_audio = getattr(train_args, key)
    if args.threshold is None:
        args.threshold = float(getattr(train_args, "score_threshold", 0.5))

    result_dir = os.path.join(args.model_path, "result")
    os.makedirs(result_dir, exist_ok=True)
    if args.score_file is None:
        args.score_file = os.path.join(result_dir, f"{args.eval_task}_logits_eval.csv")
    args.binary_score_file = os.path.join(result_dir, f"{args.eval_task}_binary_eval.csv")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def find_checkpoint(model_path: str) -> str:
    for rel in (
        "checkpoint_all_dev/best_loss.pt",
        "checkpoint_all_dev/best_f1.pt",
        "checkpoint/latest.pt",
    ):
        path = os.path.join(model_path, rel)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint found under {model_path}")


def load_state(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def main(argv=None):
    args = parse_args(argv)
    dataset = atadd_eval_dataset(args.eval_audio, audio_length=args.audio_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
    )

    model = build_multi_head_model(args).to(args.device)
    ckpt_path = find_checkpoint(args.model_path)
    model.load_state_dict(load_state(ckpt_path))
    model.eval()
    print(f"Checkpoint: {ckpt_path}")

    with open(args.score_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "score"])
        with torch.no_grad():
            for waveform, filenames in tqdm(loader, desc="scoring"):
                waveform = waveform.to(args.device, non_blocking=True)
                _, logits = model(waveform)
                scores = F.softmax(logits, dim=1)[:, 0].detach().cpu().numpy()
                for name, score in zip(filenames, scores):
                    writer.writerow([str(name).strip(), float(score)])

    with open(args.score_file, "r", encoding="utf-8-sig", newline="") as fin, \
            open(args.binary_score_file, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["name", "predict"])
        for row in reader:
            score = float(row["score"])
            writer.writerow([row["name"].strip(), "real" if score >= args.threshold else "fake"])

    meta_path = os.path.join(os.path.dirname(args.binary_score_file), f"{args.eval_task}_binary_threshold_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "threshold": args.threshold,
                "score_file": args.score_file,
                "binary_score_file": args.binary_score_file,
            },
            f,
            indent=2,
        )
    print(f"Scores: {args.score_file}")
    print(f"Binary: {args.binary_score_file}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import atadd_dataset
from multi_head_trunk.inference import find_checkpoint, load_state
from multi_head_trunk.main_train import _inject_multi_head_args
from multi_head_trunk.model import build_multi_head_model
from utils import metrics as em
from utils.config import ATADDConfig
from utils.helpers import parse_filter_types


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Dev-set analysis for multi-head XLSR-AASIST")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--eval_task", choices=["atadd-track1", "atadd-track2"], default=None)
    parser.add_argument("--eval_audio", default=None)
    parser.add_argument("--label_path", default=None)
    parser.add_argument("--score_suffix", default="_dev")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--metrics_only", action="store_true")
    args = parser.parse_args(argv)

    cfg_path = os.path.join(args.model_path, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.yaml not found: {cfg_path}")
    cfg = ATADDConfig.from_yaml(cfg_path)
    train_args = cfg.to_namespace()
    for k, v in vars(train_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args = _inject_multi_head_args(args, ["--config", cfg_path])

    if args.eval_task is None:
        args.eval_task = train_args.train_task
    if args.eval_audio is None:
        key = "atadd_t1_dev_audio" if args.eval_task == "atadd-track1" else "atadd_t2_dev_audio"
        args.eval_audio = getattr(train_args, key)
    if args.label_path is None:
        key = "atadd_t1_dev_label" if args.eval_task == "atadd-track1" else "atadd_t2_dev_label"
        args.label_path = getattr(train_args, key)
    if args.num_workers is None:
        args.num_workers = int(getattr(train_args, "num_workers", 4))

    args.filter_types_parsed = parse_filter_types(getattr(args, "filter_types", None))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args


def main(argv=None):
    args = parse_args(argv)
    result_dir = os.path.join(args.model_path, "analysis_dev", "result")
    os.makedirs(result_dir, exist_ok=True)
    score_path = os.path.join(result_dir, f"{args.eval_task}_logits{args.score_suffix}.csv")

    dataset = atadd_dataset(
        args.eval_audio,
        args.label_path,
        audio_length=args.audio_len,
        filter_types=args.filter_types_parsed,
        dev_subsample=False,
    )
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
    print(f"Dev audio: {args.eval_audio}")
    print(f"Dev labels: {args.label_path}")

    rows = []
    scores, labels, types = [], [], []
    generators = []
    with torch.no_grad():
        for waveform, filenames, batch_labels, batch_types, batch_generators in tqdm(loader, desc="dev-analysis"):
            waveform = waveform.to(args.device, non_blocking=True)
            _, logits = model(waveform)
            probs_real = F.softmax(logits, dim=1)[:, 0].detach().cpu().numpy()
            logits_np = logits.detach().cpu().numpy()
            labels_np = batch_labels.numpy()
            types_np = batch_types.numpy()
            for i, name in enumerate(filenames):
                score = float(probs_real[i])
                pred = "real" if score >= float(args.score_threshold) else "fake"
                label_name = "real" if int(labels_np[i]) == 0 else "fake"
                rows.append([
                    str(name).strip(),
                    score,
                    float(logits_np[i, 0]),
                    float(logits_np[i, 1]),
                    pred,
                    int(types_np[i]),
                    label_name,
                    str(batch_generators[i]),
                ])
                scores.append(score)
                labels.append(int(labels_np[i]))
                types.append(int(types_np[i]))
                generators.append(str(batch_generators[i]))

    with open(score_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "score", "logit_real", "logit_fake", "predict", "type_id", "label", "generator"])
        writer.writerows(rows)

    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    types_np = np.asarray(types, dtype=np.int64)
    preds_np = (scores_np < float(args.score_threshold)).astype(np.int64)
    eer, eer_thr = em.compute_eer(scores_np[labels_np == 0], scores_np[labels_np == 1])
    macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)

    print("\n===== Dev Performance =====")
    print(f"Loss checkpoint score file: {score_path}")
    print(f"EER={eer:.6f}  EER_thr={eer_thr:.6f}  Macro-F1@{float(args.score_threshold):.3f}={macro_f1:.6f}")
    print(classification_report(labels_np, preds_np, target_names=["real", "fake"], zero_division=0))

    type_names = {0: "speech", 1: "sound", 2: "singing", 3: "music"}
    print("\n===== Performance by Audio Type =====")
    for type_id in sorted(set(types)):
        mask = types_np == type_id
        if not np.any(mask):
            continue
        tl, ts, tp = labels_np[mask], scores_np[mask], preds_np[mask]
        t_eer = np.nan if len(np.unique(tl)) < 2 else em.compute_eer(ts[tl == 0], ts[tl == 1])[0]
        t_f1 = f1_score(tl, tp, average="macro", zero_division=0)
        print(f"{type_names.get(int(type_id), str(type_id))}: EER={t_eer:.6f} Macro-F1={t_f1:.6f} N={int(mask.sum())}")


if __name__ == "__main__":
    main()

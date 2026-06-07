from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import wandb
from utils import config, metrics as em
from utils.helpers import parse_filter_types, setup_seed
from utils.trainer import (
    adjust_learning_rate,
    args_for_wandb,
    build_dataloaders,
    build_full_dev_loader,
)

from multi_head_trunk.model import build_multi_head_model


torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_start_method("spawn", force=True)


def _cli_meta(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def _load_multi_head_yaml(path: str | None) -> Dict:
    if not path or not os.path.isfile(path):
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    mh = raw.get("multi_head", {}) or {}
    if not isinstance(mh, dict):
        raise ValueError("multi_head config must be a mapping.")
    return mh


def _inject_multi_head_args(args, argv=None):
    cli = _cli_meta(argv)
    cfg_path = cli.config
    if cli.resume is not None:
        cfg_path = os.path.join(cli.resume, "config.yaml")
    mh = _load_multi_head_yaml(cfg_path)
    args.special_weight = float(mh.get("special_weight", 0.5))
    args.special_loss_weight = float(mh.get("special_loss_weight", 0.3))
    args.special_head_ranges = mh.get(
        "special_head_ranges",
        [
            {"name": "front", "start": 0.00, "end": 0.50},
            {"name": "middle", "start": 0.40, "end": 0.90},
            {"name": "tail", "start": 0.80, "end": 1.00},
        ],
    )
    args.min_special_frames = int(mh.get("min_special_frames", 32))
    return args


def initParams(argv=None):
    args = config.initParams(argv)
    args = _inject_multi_head_args(args, argv)
    cfg = getattr(args, "_config", None)

    args.filter_types_parsed = parse_filter_types(args.filter_types)
    args.log_dir = os.path.join(args.out_fold, "logs")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(args.seed)

    ckpt_dir = os.path.join(args.out_fold, "checkpoint")
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")

    if args.continue_training and os.path.exists(latest_ckpt):
        pass
    else:
        if os.path.exists(args.out_fold):
            shutil.rmtree(args.out_fold)
        os.makedirs(args.log_dir)
        os.makedirs(ckpt_dir)
        os.makedirs(os.path.join(args.out_fold, "checkpoint_all_dev"))

        if cfg is not None:
            cfg.save_to_yaml(os.path.join(args.out_fold, "config.yaml"))
            _append_multi_head_config(os.path.join(args.out_fold, "config.yaml"), args)

        with open(os.path.join(args.log_dir, "train_loss.log"), "w") as f:
            f.write("step\tepoch\tbatch\ttrain_loss\tcombined_loss\tspecial_loss\n")
        with open(os.path.join(args.log_dir, "dev_loss.log"), "w") as f:
            f.write("step\ttag\tval_loss\tval_eer\tval_f1\t[per-type EER/F1]\t[per-generator F1]\n")
        with open(os.path.join(args.log_dir, "all_dev_loss.log"), "w") as f:
            f.write("step\ttag\tval_loss\tval_eer\tval_f1\t[per-type EER/F1]\t[per-generator F1]\n")

    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Device: {args.device}")
    print(f"Special heads: {args.special_head_ranges}")
    print(f"Special weight: {args.special_weight} | special loss weight: {args.special_loss_weight}")
    return args


def _append_multi_head_config(path: str, args) -> None:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        saved = yaml.safe_load(f) or {}
    saved["multi_head"] = {
        "special_weight": args.special_weight,
        "special_loss_weight": args.special_loss_weight,
        "special_head_ranges": args.special_head_ranges,
        "min_special_frames": args.min_special_frames,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(saved, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=False)


def build_model_optimizer_criterion(args):
    model = build_multi_head_model(args).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=0.0005,
    )
    class_weight = torch.FloatTensor([4.0, 1.0] if args.train_task == "atadd-track1" else [3.5, 1.0]).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    resume = {
        "start_epoch": 0,
        "global_step": 0,
        "best_sample_val": -float("inf") if args.save_best_by == "f1" else float("inf"),
        "best_full_val": -float("inf") if args.save_best_by == "f1" else float("inf"),
        "best_full_vals": {"loss": float("inf"), "eer": float("inf"), "f1": -float("inf")},
        "no_improve": 0,
    }
    if args.continue_training:
        ckpt_path = os.path.join(args.out_fold, "checkpoint", "latest.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            resume.update(
                start_epoch=ckpt.get("epoch", -1) + 1,
                global_step=ckpt.get("global_step", 0),
                best_sample_val=ckpt.get("best_sample_val", resume["best_sample_val"]),
                best_full_val=ckpt.get("best_full_val", resume["best_full_val"]),
                best_full_vals={**resume["best_full_vals"], **ckpt.get("best_full_vals", {})},
                no_improve=ckpt.get("no_improve", 0),
            )
            print(f"Resumed from {ckpt_path}")
    return model, optimizer, criterion, resume


def _classification_loss(outputs, labels, criterion, special_loss_weight: float):
    details = outputs if isinstance(outputs, dict) else None
    logits = details["logits"] if details else outputs
    combined_loss = criterion(logits, labels)
    special_loss = logits.new_tensor(0.0)
    if details and details["special_logits"] and special_loss_weight > 0:
        special_loss = torch.stack([criterion(x, labels) for x in details["special_logits"]]).mean()
    return combined_loss + special_loss_weight * special_loss, combined_loss.detach(), special_loss.detach(), logits


def train(args):
    model, optimizer, criterion, resume = build_model_optimizer_criterion(args)
    train_loader, val_loader = build_dataloaders(args)
    full_val_loader = build_full_dev_loader(args)

    start_epoch = resume["start_epoch"]
    best_sample_val = resume["best_sample_val"]
    best_full_val = resume["best_full_val"]
    best_full_vals = resume["best_full_vals"]
    no_improve = resume["no_improve"]

    def _metric_val(loss, eer, f1):
        return {"loss": loss, "eer": eer, "f1": f1}[args.save_best_by]

    def _is_better(metric, new, old):
        return new > old if metric == "f1" else new < old

    amp_enabled = False
    scaler = GradScaler(enabled=amp_enabled)
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            mode=args.wandb_mode,
            project=args.wandb_project,
            name=args.wandb_run_name or os.path.basename(os.path.normpath(args.out_fold.rstrip("/"))),
            config=args_for_wandb(args),
            dir=args.out_fold,
        )

    n_batches = len(train_loader)
    stop_training = False

    def _safe_metric_key(value):
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())

    def _jsonable_metrics(metrics):
        out = {}
        for key, values in metrics.items():
            out[str(key)] = {}
            for k, v in values.items():
                if isinstance(v, (float, np.floating)):
                    out[str(key)][k] = None if np.isnan(v) else float(v)
                elif isinstance(v, (int, np.integer)):
                    out[str(key)][k] = int(v)
                else:
                    out[str(key)][k] = v
        return out

    def _run_inference(loader):
        model.eval()
        loss_list, score_list, label_list, type_list, generator_list = [], [], [], [], []
        with torch.no_grad():
            for feat, _, labels, class_types, generators in tqdm(loader, leave=False, desc="eval"):
                feat = feat.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)
                with autocast(enabled=amp_enabled):
                    details = model(feat, return_head_logits=True)
                    logits = details["logits"]
                    loss = criterion(logits, labels)
                    score = F.softmax(logits, dim=1)[:, 0]
                loss_list.append(loss.item())
                score_list.append(score.detach())
                label_list.append(labels.detach())
                type_list.append(class_types.detach())
                generator_list.extend([str(g) for g in generators])

        val_loss = float(np.nanmean(loss_list))
        scores = torch.cat(score_list).cpu().numpy()
        labels_np = torch.cat(label_list).cpu().numpy()
        types = torch.cat(type_list).cpu().numpy()
        generators = np.asarray(generator_list, dtype=object)
        val_eer, eer_thr = em.compute_eer(scores[labels_np == 0], scores[labels_np == 1])
        thr = float(eer_thr if args.eval_threshold_mode == "eer" else args.score_threshold)
        preds = (scores < thr).astype(np.int64)
        val_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)

        type_metrics = {}
        for t in np.unique(types):
            mask = types == t
            tl, ts = labels_np[mask], scores[mask]
            tp = (ts < thr).astype(np.int64)
            type_metrics[t] = {
                "eer": np.nan if len(np.unique(tl)) < 2 else em.compute_eer(ts[tl == 0], ts[tl == 1])[0],
                "f1": f1_score(tl, tp, average="macro", zero_division=0),
            }

        generator_metrics = {}
        for generator in sorted(np.unique(generators), key=lambda x: str(x)):
            mask = generators == generator
            gl, gs = labels_np[mask], scores[mask]
            gp = (gs < thr).astype(np.int64)
            generator_metrics[str(generator)] = {
                "eer": np.nan if len(np.unique(gl)) < 2 else em.compute_eer(gs[gl == 0], gs[gl == 1])[0],
                "f1": f1_score(gl, gp, average="macro", zero_division=0),
                "support": int(mask.sum()),
                "real": int((gl == 0).sum()),
                "fake": int((gl == 1).sum()),
            }
        return val_loss, val_eer, val_f1, type_metrics, generator_metrics, thr

    def _log_metrics(filename, tag, step, val_loss, val_eer, val_f1, type_metrics, generator_metrics):
        with open(os.path.join(args.log_dir, filename), "a") as f:
            f.write(f"{step}\t{tag}\t{val_loss:.6f}\t{val_eer:.6f}\t{val_f1:.6f}")
            for t, m in type_metrics.items():
                f.write(f"\t{t}_EER:{m['eer']:.4f}\t{t}_F1:{m['f1']:.4f}")
            for generator, m in generator_metrics.items():
                key = _safe_metric_key(generator)
                f.write(f"\tGEN_{key}_F1:{m['f1']:.4f}\tGEN_{key}_N:{m['support']}")
            f.write("\n")

    def _save_latest(epoch, global_step):
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_sample_val": best_sample_val,
                "best_full_val": best_full_val,
                "best_full_vals": best_full_vals,
                "no_improve": no_improve,
            },
            os.path.join(args.out_fold, "checkpoint", "latest.pt"),
        )

    def do_sample_eval(epoch, global_step):
        nonlocal best_sample_val, no_improve
        tag = f"{epoch}.{global_step}"
        t0 = time.time()
        vals = _run_inference(val_loader)
        val_loss, val_eer, val_f1, type_metrics, generator_metrics, thr = vals
        _log_metrics("dev_loss.log", tag, global_step, val_loss, val_eer, val_f1, type_metrics, generator_metrics)
        print(f"\n[SampleEval @ {tag}] loss={val_loss:.4f} EER={val_eer:.4f} F1={val_f1:.4f} thr={thr:.4f} ({(time.time()-t0)/60:.1f} min)")
        cur = _metric_val(val_loss, val_eer, val_f1)
        if _is_better(args.save_best_by, cur, best_sample_val):
            best_sample_val = cur
            no_improve = 0
        else:
            no_improve += 1
        _save_latest(epoch, global_step)
        return args.patience > 0 and no_improve >= args.patience

    def do_full_eval(epoch, global_step):
        nonlocal best_full_val, best_full_vals
        tag = f"{epoch}.{global_step}"
        vals = _run_inference(full_val_loader)
        val_loss, val_eer, val_f1, type_metrics, generator_metrics, thr = vals
        _log_metrics("all_dev_loss.log", tag, global_step, val_loss, val_eer, val_f1, type_metrics, generator_metrics)
        print(f"\n[FullEval @ {tag}] loss={val_loss:.4f} EER={val_eer:.4f} F1={val_f1:.4f} thr={thr:.4f}")
        metric_values = {"loss": val_loss, "f1": val_f1}
        for metric_name, cur in metric_values.items():
            if not _is_better(metric_name, cur, best_full_vals[metric_name]):
                continue
            best_full_vals[metric_name] = cur
            path = os.path.join(args.out_fold, "checkpoint_all_dev", f"best_{metric_name}.pt")
            torch.save(model.state_dict(), path)
            if metric_name == args.save_best_by:
                best_full_val = cur
        _save_latest(epoch, global_step)

    for epoch in tqdm(range(start_epoch, args.num_epochs), desc="epochs"):
        t0 = time.time()
        model.train()
        train_losses = []
        adjust_learning_rate(args, args.lr, optimizer, epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        for i, (feat, _, labels, _, _) in enumerate(tqdm(train_loader, leave=False, desc=f"epoch {epoch}")):
            feat = feat.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(enabled=amp_enabled):
                details = model(feat, return_head_logits=True)
                loss, combined_loss, special_loss, _ = _classification_loss(
                    details, labels, criterion, args.special_loss_weight
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            gs = epoch * n_batches + i + 1
            with open(os.path.join(args.log_dir, "train_loss.log"), "a") as f:
                f.write(f"{gs}\t{epoch}\t{i}\t{loss.item():.6f}\t{combined_loss.item():.6f}\t{special_loss.item():.6f}\n")
            if use_wandb:
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/combined_loss": combined_loss.item(),
                        "train/special_loss": special_loss.item(),
                        "train/epoch": epoch,
                        "train/lr": current_lr,
                    },
                    step=gs,
                )

            if args.eval_steps > 0 and gs % args.eval_steps == 0:
                if gs < args.eval_warmup_steps:
                    print(f"[Warmup] skip sample eval at step {gs}")
                elif do_sample_eval(epoch, gs):
                    stop_training = True
                    break
                model.train()

            if args.full_eval_steps > 0 and gs % args.full_eval_steps == 0:
                if gs < args.eval_warmup_steps:
                    print(f"[Warmup] skip full eval at step {gs}")
                else:
                    do_full_eval(epoch, gs)
                model.train()

        print(f"Epoch {epoch} loss={np.mean(train_losses):.4f} time={(time.time()-t0)/60:.1f} min")
        if stop_training:
            break

        gs_end = (epoch + 1) * n_batches
        if gs_end < args.eval_warmup_steps:
            print(f"[Warmup] skip epoch-end eval at step {gs_end}")
        elif do_sample_eval(epoch, gs_end):
            break

    if use_wandb:
        wandb.finish()
    latest_path = os.path.join(args.out_fold, "checkpoint", "latest.pt")
    if os.path.exists(latest_path):
        os.remove(latest_path)
        print(f"Removed latest checkpoint after experiment: {latest_path}")
    return model


if __name__ == "__main__":
    parsed = initParams()
    train(parsed)

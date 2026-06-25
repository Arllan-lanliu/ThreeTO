# -*- coding: utf-8 -*-
"""Crop-aware datasets for trunk_vote.

This module deliberately lives under ``trunk_vote`` so the shared project code
remains read-only.  It reuses the original AT-ADD dataset augmentation helpers
but replaces the fixed "first audio_len samples" policy with trunk-vote crop
selection.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from data.RawBoost import process_Rawboost_feature
from data.dataset import atadd_dataset, torchaudio_load


DEFAULT_SR = 16000


def crop_starts(num_samples: int, crop_len: int, sr: int = DEFAULT_SR) -> List[int]:
    """Return deterministic dev/eval 4.04s crop starts with a 2s hop."""
    if num_samples <= crop_len:
        return [0]

    end = max(0, num_samples - crop_len)
    hop = int(round(2.0 * sr))
    starts = list(range(0, end + 1, hop))
    if starts[-1] != end:
        starts.append(end)
    return starts


def random_crop_start(num_samples: int, crop_len: int) -> int:
    """Sample one crop start uniformly for training."""
    if num_samples <= crop_len:
        return 0
    return random.randint(0, num_samples - crop_len)


def _to_1d_tensor(waveform) -> torch.Tensor:
    if isinstance(waveform, torch.Tensor):
        wav = waveform.detach().float()
    else:
        wav = torch.tensor(np.asarray(waveform), dtype=torch.float32)
    return wav.reshape(-1)


def crop_and_repeat_pad(waveform, start: int, crop_len: int, normalize: bool = True) -> torch.Tensor:
    """Slice one crop, repeat-pad if needed, then optionally normalize."""
    wav = _to_1d_tensor(waveform)
    if wav.numel() == 0:
        wav = torch.zeros(crop_len, dtype=torch.float32)
    start = min(max(0, int(start)), max(0, wav.numel() - 1))
    crop = wav[start : start + crop_len]
    if crop.numel() < crop_len:
        repeats = int(crop_len / max(crop.numel(), 1)) + 1
        crop = torch.tile(crop, (repeats,))[:crop_len]
    else:
        crop = crop[:crop_len]
    if normalize:
        crop = (crop - crop.mean()) / torch.sqrt(crop.var() + 1e-7)
    return crop


class atadd_random_crop_dataset(atadd_dataset):
    """Protocol dataset where each audio yields one random crop per access.

    Training keeps the original label/type for the audio. sound/music samples
    use one random crop; speech/singing samples keep the first crop. Protocol
    augmentation is applied after crop selection.
    """

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filename, class_type, label, generator = self.all_files[idx]
        filepath = os.path.join(self.path_to_audio, filename)

        waveform, sr = torchaudio_load(filepath)
        if class_type in {"sound", "music"}:
            start = random_crop_start(_to_1d_tensor(waveform).numel(), self.audio_length)
        else:
            start = 0
        waveform = crop_and_repeat_pad(waveform, start, self.audio_length, normalize=False)

        if self.aug_probs is not None:
            aug_prob = self.aug_probs.get(class_type, 0.0)
            if aug_prob > 0.0 and random.random() < aug_prob:
                wav_np = _to_1d_tensor(waveform).cpu().numpy()
                if class_type == "music":
                    if self._pitch_shift_aug is not None:
                        waveform = self._pitch_shift_aug.apply(wav_np)
                    elif self._spec_aug is not None:
                        waveform = self._spec_aug.apply(wav_np)
                elif class_type == "sound":
                    waveform = self._apply_sound_augmentation(wav_np, sr)
                else:
                    waveform = process_Rawboost_feature(wav_np, sr=sr, algo=5)

        waveform = crop_and_repeat_pad(waveform, 0, self.audio_length)

        if self.musanrir:
            waveform = self._apply_augmentation(waveform, waveform.size(0))

        return (
            waveform,
            filename,
            self.label[label],
            self.class_type[class_type],
            generator,
        )


class atadd_crop_dataset(atadd_dataset):
    """Protocol dataset where deterministic inference crops are items.

    Dev returns crop-level items; callers aggregate logits back to audio-level
    by filename before computing loss/F1/checkpoint metrics.
    """

    def __init__(self, *args, crop_sr: int = DEFAULT_SR, **kwargs):
        super().__init__(*args, **kwargs)
        self.crop_sr = int(crop_sr)
        self.crop_items: List[Tuple[int, int]] = []
        crop_counts = Counter()
        for base_idx, (filename, _class_type, _label, _generator) in enumerate(self.all_files):
            filepath = os.path.join(self.path_to_audio, filename)
            waveform, _sr = torchaudio_load(filepath)
            starts = crop_starts(_to_1d_tensor(waveform).numel(), self.audio_length, self.crop_sr)
            crop_counts[len(starts)] += 1
            self.crop_items.extend((base_idx, start) for start in starts)
        print(
            f"[trunk_vote] crop-expanded {len(self.all_files)} audios -> "
            f"{len(self.crop_items)} crops; crops/audio={dict(sorted(crop_counts.items()))}"
        )

    def __len__(self):
        return len(self.crop_items)

    def __getitem__(self, idx):
        base_idx, start = self.crop_items[idx]
        filename, class_type, label, generator = self.all_files[base_idx]
        filepath = os.path.join(self.path_to_audio, filename)

        waveform, sr = torchaudio_load(filepath)

        if self.aug_probs is not None:
            aug_prob = self.aug_probs.get(class_type, 0.0)
            if aug_prob > 0.0:
                import random

                if random.random() < aug_prob:
                    wav_np = _to_1d_tensor(waveform).cpu().numpy()
                    if class_type == "music":
                        if self._pitch_shift_aug is not None:
                            waveform = self._pitch_shift_aug.apply(wav_np)
                        elif self._spec_aug is not None:
                            waveform = self._spec_aug.apply(wav_np)
                    elif class_type == "sound":
                        waveform = self._apply_sound_augmentation(wav_np, sr)
                    else:
                        waveform = process_Rawboost_feature(wav_np, sr=sr, algo=5)

        waveform = crop_and_repeat_pad(waveform, start, self.audio_length)

        if self.musanrir:
            waveform = self._apply_augmentation(waveform, waveform.size(0))

        return (
            waveform,
            filename,
            self.label[label],
            self.class_type[class_type],
            generator,
        )


class atadd_eval_crop_dataset(Dataset):
    """Evaluation wav-folder dataset expanded to deterministic crops."""

    def __init__(
        self,
        path_to_audio: str,
        audio_length: int = 64600,
        exts: Sequence[str] = (".flac", ".wav"),
        crop_sr: int = DEFAULT_SR,
    ):
        super().__init__()
        self.path_to_audio = path_to_audio
        self.audio_length = int(audio_length)
        self.crop_sr = int(crop_sr)
        self.all_files = sorted(
            f for f in os.listdir(self.path_to_audio) if f.lower().endswith(tuple(exts))
        )
        self.crop_items: List[Tuple[int, int]] = []
        crop_counts = Counter()
        for base_idx, filename in enumerate(self.all_files):
            waveform, _sr = torchaudio_load(os.path.join(self.path_to_audio, filename))
            starts = crop_starts(_to_1d_tensor(waveform).numel(), self.audio_length, self.crop_sr)
            crop_counts[len(starts)] += 1
            self.crop_items.extend((base_idx, start) for start in starts)
        print(
            f"[trunk_vote] eval crop-expanded {len(self.all_files)} audios -> "
            f"{len(self.crop_items)} crops; crops/audio={dict(sorted(crop_counts.items()))}"
        )

    def __len__(self):
        return len(self.crop_items)

    def __getitem__(self, idx):
        base_idx, start = self.crop_items[idx]
        filename = self.all_files[base_idx]
        waveform, _sr = torchaudio_load(os.path.join(self.path_to_audio, filename))
        return crop_and_repeat_pad(waveform, start, self.audio_length), filename

    def collate_fn(self, samples):
        return default_collate(samples)

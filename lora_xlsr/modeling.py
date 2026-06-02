from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from model.backbone.ASSIST import AASIST
from model.layer_fusion import (
    IntermediateLayerFusion,
    normalize_layer_fusion,
    validate_layer_fusion_config,
)
from lora_xlsr.triton_compat import patch_triton_language


try:
    from peft import LoraConfig, get_peft_model
except ImportError as exc:  # pragma: no cover - exercised only when env misses peft
    LoraConfig = None
    get_peft_model = None
    _PEFT_IMPORT_ERROR = exc
else:
    _PEFT_IMPORT_ERROR = None


class LoRAXLSR(nn.Module):
    """XLSR frontend with optional PEFT LoRA adapters.

    This mirrors ``model.SSL.XLSR`` but keeps the implementation local to the
    LoRA experiment directory so the project-wide model files remain unchanged.
    """

    def __init__(
        self,
        model_dir,
        device="cuda",
        sampling_rate=16000,
        freeze=True,
        visual=False,
        return_hidden_states=False,
        selected_layers=None,
        layer_fusion="last",
        mhfa_compression_dim=None,
        mhfa_num_heads=8,
        mhfa_output_dim=None,
        mhfa_dropout=0.1,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_targets=("q_proj", "k_proj", "v_proj", "out_proj"),
        lora_dropout=0.1,
    ):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sampling_rate = sampling_rate
        self.return_hidden_states = return_hidden_states
        self.selected_layers = (
            tuple(int(i) for i in selected_layers) if selected_layers is not None else None
        )

        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_dir,
            do_normalize=False,
        )
        self.model = Wav2Vec2Model.from_pretrained(model_dir).to(self.device)
        self.model.config.output_hidden_states = True
        self.visual = visual

        self.use_lora = bool(use_lora)
        if self.use_lora:
            if get_peft_model is None:
                raise ImportError(
                    "PEFT is required for LoRA XLSR. Install it with: pip install peft"
                ) from _PEFT_IMPORT_ERROR
            lora_config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                target_modules=list(lora_targets),
                lora_dropout=float(lora_dropout),
                bias="none",
                task_type=None,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            self.freeze = False
            self.model.train()
        else:
            self.freeze = freeze
            if freeze:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                self.model.train()

        self.hidden_size = self.config.hidden_size
        lf = normalize_layer_fusion(layer_fusion)
        self.layer_fusion = lf
        validate_layer_fusion_config(lf, self.selected_layers, backend_name="XLSR")

        n_sel = len(self.selected_layers) if self.selected_layers else 0
        self.layer_fusion_mod = IntermediateLayerFusion(
            self.hidden_size,
            n_sel,
            lf,
            mhfa_compression_dim=mhfa_compression_dim,
            mhfa_num_heads=mhfa_num_heads,
            mhfa_output_dim=mhfa_output_dim,
            mhfa_dropout=mhfa_dropout,
        )

    def _fuse_hidden_states(self, outputs):
        return self.layer_fusion_mod.fuse(
            outputs.last_hidden_state,
            outputs.hidden_states,
            self.selected_layers,
            backend_name="XLSR",
        )

    def forward(self, audio_data, output_attentions=False):
        feat = self.processor(
            audio_data,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).input_values.to(self.device)
        feat = feat.squeeze(dim=0)

        if self.visual or output_attentions:
            outputs = self.model(
                feat,
                output_attentions=True,
                output_hidden_states=True,
            )
            attentions = outputs.attentions
            fused, hidden_states = self._fuse_hidden_states(outputs)
            if self.return_hidden_states:
                return fused, attentions, hidden_states
            return fused, attentions

        if self.freeze:
            with torch.no_grad():
                outputs = self.model(feat, output_hidden_states=True)
                fused, hidden_states = self._fuse_hidden_states(outputs)
        else:
            outputs = self.model(feat, output_hidden_states=True)
            fused, hidden_states = self._fuse_hidden_states(outputs)

        if self.return_hidden_states:
            return fused, hidden_states
        return fused

    def extract_features(self, audio_data):
        return self.forward(audio_data)


class LoRAXLSRAASIST(nn.Module):
    """ft-xlsr -> AASIST with PEFT LoRA applied to the XLSR frontend."""

    def __init__(
        self,
        model_dir,
        device="cuda",
        freeze=False,
        visual=False,
        assist_project_choice=0,
        selected_layers=None,
        layer_fusion="last",
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_targets=("q_proj", "k_proj", "v_proj", "out_proj"),
        lora_dropout=0.1,
    ):
        super().__init__()
        if int(assist_project_choice) == 2:
            patch_triton_language()
        self.wav2vec2 = LoRAXLSR(
            model_dir=model_dir,
            device=device,
            freeze=freeze,
            visual=visual,
            selected_layers=selected_layers,
            layer_fusion=layer_fusion,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_targets=lora_targets,
            lora_dropout=lora_dropout,
        )
        self.w2vaasist = AASIST(assist_project_choice=assist_project_choice)
        self.visual = visual

    def forward(self, audio_data, output_attentions=False):
        if output_attentions:
            features, attn = self.wav2vec2.forward(audio_data, output_attentions=True)
            last_hidden, output = self.w2vaasist(features)
            return last_hidden, output, attn
        if self.visual:
            features, attention_weights = self.wav2vec2.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
            return last_hidden, output, attention_weights
        features = self.wav2vec2.extract_features(audio_data)
        last_hidden, output = self.w2vaasist(features)
        return last_hidden, output

    def train(self, mode=True):
        super().train(mode)
        self.wav2vec2.train(mode)
        self.w2vaasist.train(mode)
        return self

    def eval(self):
        return self.train(False)

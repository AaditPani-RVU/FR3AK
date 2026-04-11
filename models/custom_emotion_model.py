from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


PLUTCHIK_EMOTIONS: Tuple[str, ...] = (
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
)


@dataclass
class EmotionModelOutput:
    logits: torch.Tensor
    confidence: Optional[torch.Tensor] = None
    cls_logits: Optional[torch.Tensor] = None


class EmotionAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int = 4, out_dim: int = 128) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads}).")

        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        num_heads, head_dim = self.n_heads, self.head_dim

        query = self.query.expand(batch_size, -1, -1)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            attention = attention.masked_fill(padding_mask, float("-inf"))

        attention = self.dropout(torch.softmax(attention, dim=-1))
        pooled = (attention @ value).squeeze(2).reshape(batch_size, hidden_size)
        projected = self.out_proj(pooled)
        return self.norm(projected)


class PlutchikEmotionModelV2(nn.Module):
    """Best-effort reconstruction of the FR3AK Plutchik checkpoint architecture."""

    def __init__(
        self,
        base_model_name: str,
        n_emotions: int = 8,
        out_dim: int = 128,
        head_hidden_dim: int = 32,
        aux_hidden_dim: int = 128,
        dropout: float = 0.1,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        encoder_config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_config(encoder_config)
        hidden_size = self.encoder.config.hidden_size

        self.emotion_blocks = nn.ModuleList(
            [EmotionAttentionBlock(hidden_size, n_heads=n_heads, out_dim=out_dim) for _ in range(n_emotions)]
        )
        self.emotion_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(out_dim, head_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden_dim, 1),
                )
                for _ in range(n_emotions)
            ]
        )
        self.temperature = nn.Parameter(torch.full((n_emotions,), 1.0))
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_size, aux_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, aux_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden_dim, n_emotions),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> EmotionModelOutput:
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = self.drop(encoder_output.last_hidden_state)
        cls_token = hidden_states[:, 0]

        logits = []
        for block, head in zip(self.emotion_blocks, self.emotion_heads):
            features = block(hidden_states, attention_mask)
            logits.append(head(features))

        logits_tensor = torch.cat(logits, dim=-1)
        temperature = torch.clamp(self.temperature, min=0.5, max=5.0)
        scores = torch.sigmoid(logits_tensor * temperature)

        confidence = self.conf_head(cls_token)
        cls_logits = self.cls_head(cls_token)
        return EmotionModelOutput(logits=scores, confidence=confidence.squeeze(-1), cls_logits=cls_logits)


def build_custom_emotion_model(
    state_dict: Dict[str, torch.Tensor],
    model_config: Dict[str, Any],
) -> PlutchikEmotionModelV2:
    base_model_name = _resolve_base_model_name(model_config)
    if not base_model_name:
        raise RuntimeError(
            "Unable to determine base HuggingFace model name from model_config.json. "
            "Provide one of: base_model, model_name, hf_model_name, or backbone."
        )

    n_emotions = _infer_num_emotions(state_dict) or len(PLUTCHIK_EMOTIONS)
    out_dim = _infer_linear_out_features(state_dict, "emotion_blocks.0.out_proj.weight", 128)
    head_hidden_dim = _infer_linear_out_features(state_dict, "emotion_heads.0.0.weight", 32)
    aux_hidden_dim = _infer_linear_out_features(state_dict, "conf_head.0.weight", 128)
    n_heads = _infer_attention_heads(state_dict, hidden_size=_infer_hidden_size(base_model_name, state_dict))

    model = PlutchikEmotionModelV2(
        base_model_name=base_model_name,
        n_emotions=n_emotions,
        out_dim=out_dim,
        head_hidden_dim=head_hidden_dim,
        aux_hidden_dim=aux_hidden_dim,
        n_heads=n_heads,
    )
    return model


def _resolve_base_model_name(model_config: Dict[str, Any]) -> Optional[str]:
    for key in ("base_model", "model_name", "hf_model_name", "backbone"):
        value = model_config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _infer_num_emotions(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    indices = set()
    for key in state_dict.keys():
        if key.startswith("emotion_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                indices.add(int(parts[1]))
        elif key.startswith("emotion_heads."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                indices.add(int(parts[1]))
    if indices:
        return max(indices) + 1
    return None


def _infer_linear_out_features(state_dict: Dict[str, torch.Tensor], key: str, default: int) -> int:
    tensor = state_dict.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
        return int(tensor.shape[0])
    return default


def _infer_hidden_size(base_model_name: str, state_dict: Dict[str, torch.Tensor]) -> int:
    candidate_keys = (
        "encoder.embeddings.word_embeddings.weight",
        "encoder.embeddings.word_embeddings.weight",
    )
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.shape[1])

    config = AutoConfig.from_pretrained(base_model_name)
    hidden_size = getattr(config, "hidden_size", None)
    if isinstance(hidden_size, int):
        return hidden_size
    raise RuntimeError("Unable to infer encoder hidden size for the custom emotion model.")


def _infer_attention_heads(state_dict: Dict[str, torch.Tensor], hidden_size: int) -> int:
    query = state_dict.get("emotion_blocks.0.query")
    if isinstance(query, torch.Tensor) and query.ndim == 3:
        if hidden_size % 4 == 0:
            return 4
    return 4
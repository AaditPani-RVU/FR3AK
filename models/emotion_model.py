from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import kagglehub
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


LOGGER = logging.getLogger(__name__)

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

KAGGLE_DATASET_REF = "bobhendriks/plutchik-model-v2"
LOCAL_MODEL_DIR_ENV = "FR3AK_EMOTION_MODEL_DIR"


@dataclass
class ModelArtifacts:
    root_dir: Path
    model_weights: Path
    model_config: Path
    tokenizer_dir: Path


class EmotionModel:
    """FR3AK emotion inference model backed by KaggleHub artifacts."""

    def __init__(self, device: Optional[str] = None, max_length: Optional[int] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.artifacts = self._resolve_artifacts()
        LOGGER.info("FR3AK emotion model directory: %s", self.artifacts.root_dir)
        LOGGER.debug("Emotion model path: %s", self.artifacts.model_weights)

        self.model_config = self._load_json(self.artifacts.model_config)
        self.expected_output_dim = self._extract_output_dim(self.model_config)
        if self.expected_output_dim is None:
            raise ValueError(
                "Unable to determine emotion output dimension from model_config.json. "
                "Add one of: output_dim, num_labels, n_labels, n_classes, num_classes, label_dim, "
                "or classification_head.<same keys>."
            )
        if self.expected_output_dim is not None and self.expected_output_dim != len(PLUTCHIK_EMOTIONS):
            raise ValueError(
                "Emotion model output dimension mismatch in model_config.json: "
                f"expected 8, found {self.expected_output_dim}."
            )
        LOGGER.debug("Detected output dimension from config: %s", self.expected_output_dim)

        config_max_length = self._extract_max_length_from_config(self.model_config)
        self.max_length = max_length if max_length is not None else (config_max_length or 256)

        self.tokenizer = self._load_tokenizer(self.artifacts.tokenizer_dir)
        tokenizer_max_length = self._extract_max_length_from_tokenizer(self.tokenizer)
        if max_length is None:
            if tokenizer_max_length is not None and config_max_length is not None:
                self.max_length = min(tokenizer_max_length, config_max_length)
            elif tokenizer_max_length is not None:
                self.max_length = tokenizer_max_length
            elif config_max_length is not None:
                self.max_length = config_max_length
            else:
                self.max_length = 256

        self._validate_optional_tokenizer_files(self.artifacts.tokenizer_dir)
        LOGGER.debug("Tokenizer path: %s", self.artifacts.tokenizer_dir)

        self.model = self._load_model(self.artifacts.model_weights, self.model_config)
        self.model.to(self.device)
        self.model.eval()

        inferred_output_dim = self._infer_runtime_output_dim()
        if inferred_output_dim != len(PLUTCHIK_EMOTIONS):
            raise ValueError(
                "Emotion model runtime output dimension mismatch: "
                f"expected 8, found {inferred_output_dim}."
            )
        LOGGER.debug("Detected runtime output dimension: %s", inferred_output_dim)

    def predict(self, text: str) -> Dict[str, float]:
        """Predict an 8D Plutchik probability vector for a message."""
        safe_text = text if text is not None else ""
        if not safe_text.strip():
            return {emotion: 0.0 for emotion in PLUTCHIK_EMOTIONS}

        encoded = self.tokenizer(
            safe_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)
            logits = self._extract_logits(output)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        if probs.numel() != len(PLUTCHIK_EMOTIONS):
            raise RuntimeError(
                "Emotion inference produced invalid probability size: "
                f"expected {len(PLUTCHIK_EMOTIONS)}, got {probs.numel()}."
            )

        probs = probs.detach().cpu()
        normalized = probs.tolist()
        return {
            emotion: float(score)
            for emotion, score in zip(PLUTCHIK_EMOTIONS, normalized)
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict 8D Plutchik probability vectors for multiple messages."""
        if not texts:
            return []

        safe_texts = [text if text is not None else "" for text in texts]
        non_empty_indices = [idx for idx, text in enumerate(safe_texts) if text.strip()]

        results: List[Dict[str, float]] = [
            {emotion: 0.0 for emotion in PLUTCHIK_EMOTIONS}
            for _ in safe_texts
        ]
        if not non_empty_indices:
            return results

        batch_texts = [safe_texts[idx] for idx in non_empty_indices]
        encoded = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)
            logits = self._extract_logits(output)
            probs_batch = torch.softmax(logits, dim=-1)

        if probs_batch.ndim == 1:
            probs_batch = probs_batch.unsqueeze(0)

        if probs_batch.shape[-1] != len(PLUTCHIK_EMOTIONS):
            raise RuntimeError(
                "Emotion batch inference produced invalid probability size: "
                f"expected {len(PLUTCHIK_EMOTIONS)}, got {probs_batch.shape[-1]}."
            )

        for batch_idx, record_idx in enumerate(non_empty_indices):
            row = probs_batch[batch_idx].detach().cpu().tolist()
            results[record_idx] = {
                emotion: float(score)
                for emotion, score in zip(PLUTCHIK_EMOTIONS, row)
            }

        return results

    def _resolve_artifacts(self) -> ModelArtifacts:
        root = self._resolve_model_root()

        model_weights = self._find_required_file(root, "best_model.pt")
        model_config = self._find_required_file(root, "model_config.json")

        tokenizer_json = self._find_required_file(root, "tokenizer.json")
        tokenizer_config = self._find_required_file(root, "tokenizer_config.json")
        spm_model = self._find_required_file(root, "spm.model")

        tokenizer_dir = self._select_common_parent(
            [tokenizer_json, tokenizer_config, spm_model]
        )

        return ModelArtifacts(
            root_dir=root,
            model_weights=model_weights,
            model_config=model_config,
            tokenizer_dir=tokenizer_dir,
        )

    def _resolve_model_root(self) -> Path:
        local_override = os.getenv(LOCAL_MODEL_DIR_ENV)
        if local_override:
            root = Path(local_override).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(
                    f"Environment variable {LOCAL_MODEL_DIR_ENV} points to missing path: {root}"
                )
            LOGGER.info("Using local emotion model directory from %s: %s", LOCAL_MODEL_DIR_ENV, root)
            return root

        # KaggleHub caches datasets locally, so repeated calls reuse the same cached path.
        downloaded = kagglehub.dataset_download(KAGGLE_DATASET_REF)
        root = Path(downloaded).resolve()
        LOGGER.info("Downloaded/resolved KaggleHub model directory: %s", root)
        return root

    @staticmethod
    def _find_required_file(root_dir: Path, filename: str) -> Path:
        candidates = list(root_dir.rglob(filename))
        if not candidates:
            raise FileNotFoundError(
                f"Required model artifact '{filename}' not found under: {root_dir}"
            )
        if len(candidates) > 1:
            # Prefer the shortest path depth to avoid picking nested stale copies.
            candidates.sort(key=lambda p: len(p.parts))
        return candidates[0]

    @staticmethod
    def _select_common_parent(paths: Sequence[Path]) -> Path:
        parents = [path.parent.resolve() for path in paths]
        first = parents[0]
        if all(parent == first for parent in parents[1:]):
            return first

        common = Path(os.path.commonpath([str(parent) for parent in parents]))
        if common.exists():
            return common
        return first

    @staticmethod
    def _validate_optional_tokenizer_files(tokenizer_dir: Path) -> None:
        optional_files = ("special_tokens_map.json", "added_tokens.json")
        for name in optional_files:
            candidate = tokenizer_dir / name
            if not candidate.exists():
                LOGGER.warning(
                    "Optional tokenizer artifact '%s' was not found in tokenizer directory: %s",
                    name,
                    tokenizer_dir,
                )

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path}, found {type(data).__name__}")
        return data

    @staticmethod
    def _extract_output_dim(config: Dict[str, Any]) -> Optional[int]:
        candidate_keys = (
            "output_dim",
            "num_labels",
            "n_labels",
            "n_classes",
            "num_classes",
            "label_dim",
        )
        for key in candidate_keys:
            value = config.get(key)
            if isinstance(value, int):
                return value

        classification_head = config.get("classification_head")
        if isinstance(classification_head, dict):
            for key in candidate_keys:
                value = classification_head.get(key)
                if isinstance(value, int):
                    return value

        return None

    @staticmethod
    def _extract_max_length_from_config(config: Dict[str, Any]) -> Optional[int]:
        candidate_keys = ("max_length", "model_max_length", "max_seq_length", "sequence_length")
        for key in candidate_keys:
            value = config.get(key)
            if isinstance(value, int) and value > 0:
                return value

        tokenizer_cfg = config.get("tokenizer")
        if isinstance(tokenizer_cfg, dict):
            for key in candidate_keys:
                value = tokenizer_cfg.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        return None

    @staticmethod
    def _extract_max_length_from_tokenizer(tokenizer: Any) -> Optional[int]:
        value = getattr(tokenizer, "model_max_length", None)
        if isinstance(value, int) and 0 < value <= 8192:
            return value
        return None

    def _load_tokenizer(self, tokenizer_dir: Path):
        try:
            return AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        except Exception as primary_error:
            try:
                return AutoTokenizer.from_pretrained(
                    str(tokenizer_dir),
                    local_files_only=True,
                    use_fast=False,
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    "Failed to load tokenizer from local artifacts. "
                    f"Primary error: {primary_error}; fallback error: {fallback_error}"
                ) from fallback_error

    def _load_model(self, model_weights_path: Path, model_config: Dict[str, Any]) -> nn.Module:
        payload = torch.load(model_weights_path, map_location="cpu")

        if isinstance(payload, nn.Module):
            return payload

        if isinstance(payload, dict):
            embedded_model = payload.get("model")
            if isinstance(embedded_model, nn.Module):
                return embedded_model

            state_dict = payload.get("state_dict") or payload.get("model_state_dict")
            if isinstance(state_dict, dict):
                return self._build_model_from_state_dict(state_dict, model_config)

        raise RuntimeError(
            "Unsupported checkpoint format in best_model.pt. Expected nn.Module or state_dict payload."
        )

    def _build_model_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        model_config: Dict[str, Any],
    ) -> nn.Module:
        base_model_name = self._resolve_base_model_name(model_config)
        if not base_model_name:
            raise RuntimeError(
                "Unable to determine base HuggingFace model name from model_config.json. "
                "Provide one of: base_model, model_name, hf_model_name, or backbone."
            )

        try:
            config = AutoConfig.from_pretrained(base_model_name)
        except Exception as error:
            raise RuntimeError(
                "Failed to load HuggingFace config for base model "
                f"'{base_model_name}'. Ensure internet/cache availability."
            ) from error

        config.num_labels = len(PLUTCHIK_EMOTIONS)
        model = AutoModelForSequenceClassification.from_config(config)

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_key = key[7:] if key.startswith("module.") else key
            cleaned_state_dict[cleaned_key] = value

        load_result = model.load_state_dict(cleaned_state_dict, strict=False)
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        if missing:
            LOGGER.warning("Model missing keys during load: %s", missing)
            if self._has_missing_critical_classifier_keys(missing):
                raise RuntimeError(
                    "Critical classifier layers are missing from checkpoint state_dict. "
                    "Cannot safely run emotion classification inference."
                )
        if unexpected:
            LOGGER.warning("Model unexpected keys during load: %s", unexpected)

        return model

    @staticmethod
    def _resolve_base_model_name(model_config: Dict[str, Any]) -> Optional[str]:
        for key in ("base_model", "model_name", "hf_model_name", "backbone"):
            value = model_config.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _has_missing_critical_classifier_keys(missing_keys: List[str]) -> bool:
        critical_markers = (
            "classifier",
            "score",
            "classification_head",
            "out_proj",
            "pre_classifier",
        )
        for key in missing_keys:
            lowered = key.lower()
            if any(marker in lowered for marker in critical_markers):
                return True
        return False

    def _infer_runtime_output_dim(self) -> int:
        encoded = self.tokenizer(
            "dimension check",
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=min(self.max_length, 32),
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
            logits = self._extract_logits(output)

        if logits.ndim == 1:
            return int(logits.shape[0])
        return int(logits.shape[-1])

    @staticmethod
    def _extract_logits(output: Any) -> torch.Tensor:
        if hasattr(output, "logits"):
            logits = output.logits
        elif isinstance(output, (tuple, list)) and output:
            logits = output[0]
        else:
            raise RuntimeError("Model forward output does not contain logits.")

        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Extracted logits are not a torch.Tensor.")
        return logits


def _demo_test() -> None:
    logging.basicConfig(level=logging.INFO)

    model = EmotionModel()
    samples = [
        "I appreciate your help, thank you.",
        "Sure, whatever you say.",
        "I am worried this will fail.",
        "",
    ]

    for text in samples:
        probs = model.predict(text)
        print(f"Input: {text!r}")
        print("Output:", probs)
        print("Sum:", sum(probs.values()))
        print("-" * 80)


if __name__ == "__main__":
    _demo_test()

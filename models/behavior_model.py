from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import torch
from torch import nn


warnings.filterwarnings("ignore")

KAGGLE_DATASET_REF = os.getenv("FR3AK_BEHAVIOR_DATASET_REF", "bobhendriks/incongruity-classfier")
LOCAL_MODEL_DIR_ENV = "FR3AK_BEHAVIOR_MODEL_DIR"

STAGE1_FILE = "stage1_genuine_vs_rest.json"
STAGE2_WEIGHTS_FILE = "stage2_sarcasm_vs_manip.pt"
STAGE2_CONFIG_FILE = "model2_config.json"

HIGH_CONFIDENCE_GENUINE_THRESHOLD = 0.98
SARCASM_THRESHOLD = 0.35
SARCASM_LEXICON_FILE = "sarcasm_lexicon.json"
STAGE2_MANIPULATIVE_INDEX = 1
STAGE2_MANIPULATIVE_THRESHOLD = 0.9
NEUTRAL_THRESHOLD = 0.5
NEUTRAL_CONFIDENCE_MULTIPLIER = 0.7


@dataclass
class BehaviorArtifacts:
    root_dir: Path
    stage1_file: Path
    stage2_weights_file: Path
    stage2_config_file: Path


class _LinearTextScorer:
    def __init__(self, config: Dict[str, Any], *, default_threshold: float = 0.5) -> None:
        self.threshold = float(
            config.get(
                "threshold",
                config.get("genuine_threshold", config.get("stage1_threshold", default_threshold)),
            )
        )
        self.schema_type = "unknown"
        self.n_features = 0

        self.vocabulary: Dict[str, int] = {}
        self.weights: List[float] = []
        self.bias = 0.0
        self.idf: Optional[List[float]] = None
        self.use_tfidf = False
        self.token_weight_map: Optional[Dict[str, float]] = None

        self.xgb_trees: Optional[List[Dict[str, Any]]] = None
        self.xgb_base_margin = 0.0
        self.xgb_objective = "binary:logistic"

        self._initialize_schema(config)

    def _initialize_schema(self, config: Dict[str, Any]) -> None:
        if self._looks_like_xgboost_schema(config):
            self._init_xgboost(config)
            return

        if self._try_init_linear_like(config):
            return

        raise RuntimeError(
            "Stage-1 model format is unsupported. Expected one of: XGBoost JSON gbtree, "
            "sklearn linear export, pipeline export, or token-weight dictionary."
        )

    @staticmethod
    def _looks_like_xgboost_schema(config: Dict[str, Any]) -> bool:
        learner = config.get("learner")
        if not isinstance(learner, dict):
            return False
        gradient_booster = learner.get("gradient_booster")
        if not isinstance(gradient_booster, dict):
            return False
        model = gradient_booster.get("model")
        if not isinstance(model, dict):
            return False
        trees = model.get("trees")
        return isinstance(trees, list) and bool(trees)

    def _init_xgboost(self, config: Dict[str, Any]) -> None:
        learner = config["learner"]
        gradient_booster = learner["gradient_booster"]
        model = gradient_booster["model"]
        trees = model.get("trees")
        if not isinstance(trees, list) or not trees:
            raise RuntimeError("Stage-1 XGBoost model has no trees.")

        learner_param = learner.get("learner_model_param", {})
        num_feature = learner_param.get("num_feature")
        self.n_features = int(num_feature) if str(num_feature).isdigit() else 46
        self.xgb_trees = trees

        base_score_raw = learner_param.get("base_score", "0.5")
        self.xgb_base_margin = self._xgb_base_score_to_margin(base_score_raw)

        objective = learner.get("objective", {})
        if isinstance(objective, dict):
            objective_name = objective.get("name")
            if isinstance(objective_name, str) and objective_name.strip():
                self.xgb_objective = objective_name.strip()

        self.schema_type = "xgboost_gbtree"

    @staticmethod
    def _xgb_base_score_to_margin(raw_value: Any) -> float:
        text = str(raw_value).strip().strip("[]")
        try:
            prob = float(text)
        except ValueError:
            prob = 0.5
        prob = max(min(prob, 1.0 - 1e-6), 1e-6)
        return math.log(prob / (1.0 - prob))

    def _try_init_linear_like(self, config: Dict[str, Any]) -> bool:
        vectorizer = self._find_nested_dict(config, ("vectorizer", "tfidf", "count_vectorizer"))
        classifier = self._find_nested_dict(config, ("classifier", "model", "logreg", "logistic_regression"))

        vocab = self._extract_vocabulary(config, vectorizer)
        weights = self._extract_weights(config, classifier)
        intercept = self._extract_intercept(config, classifier)

        if vocab and weights:
            self.vocabulary = vocab
            self.weights = weights
            self.bias = intercept
            self.idf = self._extract_idf(config, vectorizer, len(self.weights))
            self.use_tfidf = bool(self.idf is not None)
            self.n_features = len(self.weights)
            self.schema_type = "linear_vector"
            return True

        token_map = self._extract_token_weight_map(config)
        if token_map:
            self.token_weight_map = token_map
            self.bias = intercept
            self.n_features = len(token_map)
            self.schema_type = "token_weight_map"
            return True

        return False

    @staticmethod
    def _find_nested_dict(root: Dict[str, Any], keys: Sequence[str]) -> Optional[Dict[str, Any]]:
        stack = [root]
        while stack:
            current = stack.pop()
            for key in keys:
                maybe = current.get(key)
                if isinstance(maybe, dict):
                    return maybe
            for value in current.values():
                if isinstance(value, dict):
                    stack.append(value)
        return None

    @staticmethod
    def _extract_vocabulary(config: Dict[str, Any], vectorizer: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
        candidates: List[Any] = [
            config.get("vocabulary"),
            config.get("vocab"),
            config.get("token_to_idx"),
        ]
        if vectorizer:
            candidates.extend(
                [
                    vectorizer.get("vocabulary"),
                    vectorizer.get("vocab"),
                    vectorizer.get("token_to_idx"),
                ]
            )

        for candidate in candidates:
            if isinstance(candidate, dict) and candidate:
                parsed = {
                    token: int(index)
                    for token, index in candidate.items()
                    if isinstance(token, str) and isinstance(index, int) and index >= 0
                }
                if parsed:
                    return parsed

        feature_names = config.get("feature_names")
        if not isinstance(feature_names, list) and vectorizer:
            feature_names = vectorizer.get("feature_names") or vectorizer.get("feature_names_")
        if isinstance(feature_names, list) and feature_names:
            parsed = {
                token: idx
                for idx, token in enumerate(feature_names)
                if isinstance(token, str) and token
            }
            if parsed:
                return parsed
        return None

    @staticmethod
    def _extract_weights(config: Dict[str, Any], classifier: Optional[Dict[str, Any]]) -> Optional[List[float]]:
        candidates: List[Any] = [
            config.get("weights"),
            config.get("coef"),
            config.get("coefficients"),
            config.get("coefs"),
            config.get("coef_"),
        ]
        if classifier:
            candidates.extend(
                [
                    classifier.get("weights"),
                    classifier.get("coef"),
                    classifier.get("coefficients"),
                    classifier.get("coefs"),
                    classifier.get("coef_"),
                ]
            )

        for candidate in candidates:
            if isinstance(candidate, list) and candidate:
                if isinstance(candidate[0], list):
                    if not candidate[0]:
                        continue
                    return [float(value) for value in candidate[0]]
                return [float(value) for value in candidate]
        return None

    @staticmethod
    def _extract_intercept(config: Dict[str, Any], classifier: Optional[Dict[str, Any]]) -> float:
        candidates: List[Any] = [
            config.get("intercept"),
            config.get("bias"),
            config.get("intercept_"),
        ]
        if classifier:
            candidates.extend(
                [
                    classifier.get("intercept"),
                    classifier.get("bias"),
                    classifier.get("intercept_"),
                ]
            )

        for candidate in candidates:
            if isinstance(candidate, (int, float)):
                return float(candidate)
            if isinstance(candidate, list) and candidate:
                first = candidate[0]
                if isinstance(first, (int, float)):
                    return float(first)
        return 0.0

    @staticmethod
    def _extract_idf(
        config: Dict[str, Any],
        vectorizer: Optional[Dict[str, Any]],
        length: int,
    ) -> Optional[List[float]]:
        raw_idf = config.get("idf") or config.get("idf_vector")
        if raw_idf is None and vectorizer:
            raw_idf = vectorizer.get("idf") or vectorizer.get("idf_")
        if not isinstance(raw_idf, list):
            return None

        parsed = [float(v) for v in raw_idf]
        if len(parsed) < length:
            parsed.extend([1.0] * (length - len(parsed)))
        elif len(parsed) > length:
            parsed = parsed[:length]
        return parsed

    @staticmethod
    def _extract_token_weight_map(config: Dict[str, Any]) -> Optional[Dict[str, float]]:
        candidates = [
            config.get("token_weights"),
            config.get("weights_by_token"),
            config.get("weights"),
            config.get("coef"),
        ]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate:
                parsed = {
                    token: float(weight)
                    for token, weight in candidate.items()
                    if isinstance(token, str) and isinstance(weight, (int, float))
                }
                if parsed:
                    return parsed
        return None

    @staticmethod
    def _read_optional_idf(config: Dict[str, Any], length: int) -> Optional[List[float]]:
        raw_idf = config.get("idf") or config.get("idf_vector")
        if not isinstance(raw_idf, list):
            return None

        parsed = [float(v) for v in raw_idf]
        if len(parsed) < length:
            parsed.extend([1.0] * (length - len(parsed)))
        elif len(parsed) > length:
            parsed = parsed[:length]
        return parsed

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_']+", text.lower())

    def _xgboost_feature_vector(self, text: str) -> List[float]:
        stripped = text.strip()
        if not stripped:
            return [0.0] * self.n_features

        lowered = stripped.lower()
        tokens = self._tokenize(lowered)
        token_count = max(1, len(tokens))
        char_count = len(stripped)
        alpha_chars = sum(1 for ch in stripped if ch.isalpha())
        digit_chars = sum(1 for ch in stripped if ch.isdigit())
        upper_chars = sum(1 for ch in stripped if ch.isupper())
        punctuation_chars = sum(1 for ch in stripped if ch in "!?.,;:'\"-()")

        manipulation_cues = {
            "should", "must", "need", "have", "prove", "guilt", "blame", "owe", "deserve", "if"
        }
        sarcasm_cues = {
            "yeah", "right", "sure", "totally", "obviously", "great", "amazing", "wow"
        }
        genuine_cues = {
            "thanks", "thank", "sorry", "appreciate", "please", "understand", "help"
        }

        stats = [
            float(token_count),
            float(char_count),
            float(sum(len(token) for token in tokens) / token_count),
            float(upper_chars / max(1, alpha_chars)),
            float(digit_chars / max(1, char_count)),
            float(punctuation_chars / max(1, char_count)),
            float(stripped.count("!")),
            float(stripped.count("?")),
            float(stripped.count("...")),
            float(stripped.count(",")),
            float(stripped.count(".")),
            float(stripped.count(":")),
            float(stripped.count(";")),
            float(sum(1 for token in tokens if token in manipulation_cues) / token_count),
            float(sum(1 for token in tokens if token in sarcasm_cues) / token_count),
            float(sum(1 for token in tokens if token in genuine_cues) / token_count),
            float(sum(1 for token in tokens if token in {"i", "me", "my", "mine"}) / token_count),
            float(sum(1 for token in tokens if token in {"you", "your", "yours"}) / token_count),
            float(sum(1 for token in tokens if token in {"not", "no", "never", "nothing"}) / token_count),
            float(sum(1 for token in tokens if token in {"but", "however", "though", "yet"}) / token_count),
            float(sum(1 for token in tokens if token in {"really", "very", "so", "too", "extremely"}) / token_count),
            float(len(re.findall(r"[!?]{2,}", stripped))),
            float(len(re.findall(r"([a-zA-Z])\1{2,}", lowered))),
            float(1.0 if " /s" in lowered or "sarcasm" in lowered else 0.0),
            float(1.0 if "if you" in lowered else 0.0),
            float(1.0 if "yeah right" in lowered or "sure" in lowered else 0.0),
            float(1.0 if "thank" in lowered or "appreciate" in lowered else 0.0),
            float(1.0 if token_count > 20 else 0.0),
            float(1.0 if token_count < 4 else 0.0),
            float(sum(1 for token in tokens if token.endswith("ing")) / token_count),
            float(sum(1 for token in tokens if token.endswith("ly")) / token_count),
        ]

        remaining = self.n_features - len(stats)
        if remaining > 0:
            buckets = [0.0] * remaining
            for token in tokens:
                idx = hash(token) % remaining
                buckets[idx] += 1.0 / token_count
            stats.extend(buckets)

        if len(stats) < self.n_features:
            stats.extend([0.0] * (self.n_features - len(stats)))
        elif len(stats) > self.n_features:
            stats = stats[: self.n_features]
        return stats

    def _eval_xgboost_tree(self, tree: Dict[str, Any], features: Sequence[float]) -> float:
        left_children = tree.get("left_children", [])
        right_children = tree.get("right_children", [])
        split_indices = tree.get("split_indices", [])
        split_conditions = tree.get("split_conditions", [])
        base_weights = tree.get("base_weights", [])

        node = 0
        max_steps = max(1, len(left_children) + 2)
        for _ in range(max_steps):
            if node < 0 or node >= len(left_children):
                break

            left = int(left_children[node])
            right = int(right_children[node])
            if left == -1 and right == -1:
                if 0 <= node < len(base_weights):
                    return float(base_weights[node])
                return 0.0

            split_idx = int(split_indices[node]) if node < len(split_indices) else -1
            threshold = float(split_conditions[node]) if node < len(split_conditions) else 0.0
            value = features[split_idx] if 0 <= split_idx < len(features) else 0.0

            node = left if value < threshold else right

        if 0 <= node < len(base_weights):
            return float(base_weights[node])
        return 0.0

    def feature_vector(self, text: str) -> List[float]:
        if self.schema_type == "xgboost_gbtree":
            return self._xgboost_feature_vector(text)

        if self.schema_type == "token_weight_map":
            return []

        features = [0.0] * len(self.weights)
        tokens = self._tokenize(text)
        token_count = max(1, len(tokens))
        for token in tokens:
            idx = self.vocabulary.get(token)
            if idx is None or idx >= len(features):
                continue
            features[idx] += 1.0

        if self.use_tfidf:
            for idx, value in enumerate(features):
                if value > 0:
                    tf = value / token_count
                    idf = self.idf[idx] if self.idf is not None else 1.0
                    features[idx] = tf * idf
        return features

    def score(self, text: str) -> float:
        if self.schema_type == "xgboost_gbtree":
            if not self.xgb_trees:
                raise RuntimeError("Stage-1 XGBoost model is not initialized.")
            features = self._xgboost_feature_vector(text)
            margin = self.xgb_base_margin
            for tree in self.xgb_trees:
                margin += self._eval_xgboost_tree(tree, features)
            return _sigmoid(margin)

        if self.schema_type == "token_weight_map":
            if self.token_weight_map is None:
                raise RuntimeError("Stage-1 token-weight schema is not initialized.")
            tokens = self._tokenize(text)
            logit = self.bias
            for token in tokens:
                logit += self.token_weight_map.get(token, 0.0)
            return _sigmoid(logit)

        features = self.feature_vector(text)
        if len(features) != len(self.weights):
            raise RuntimeError(
                f"Stage-1 feature length mismatch: {len(features)} vs {len(self.weights)}."
            )

        logit = self.bias
        for index, value in enumerate(features):
            logit += value * self.weights[index]
        return _sigmoid(logit)


class _Stage2MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        if len(dims) < 2:
            raise ValueError("Stage-2 hidden_dims must define at least one hidden layer.")

        layers: List[nn.Module] = [
            nn.Linear(dims[0], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        for idx in range(1, len(dims) - 1):
            in_dim = dims[idx]
            out_dim = dims[idx + 1]

            # Match trained checkpoint layout: second hidden layer has BatchNorm,
            # later hidden layers are Linear->GELU->Dropout.
            if idx == 1:
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
            elif idx < len(dims) - 1:
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                ])
                if idx < len(dims) - 2:
                    layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BehaviorModel:
    """Two-stage behavior classifier: genuine-vs-rest, then sarcasm-vs-manipulative."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.artifacts = self._resolve_artifacts()

        self.stage1_config = self._load_json(self.artifacts.stage1_file)
        self.stage2_config = self._load_json(self.artifacts.stage2_config_file)

        self.stage1_model = self._load_stage1_model(self.stage1_config)
        self.stage2_feature_dim = self._resolve_stage2_feature_dim(self.stage2_config)
        self.stage2_model = self._load_stage2_model(self.artifacts.stage2_weights_file, self.stage2_config)
        self.stage2_model.to(self.device)
        self.stage2_model.eval()
        self.sarcasm_lexicon = self._load_sarcasm_lexicon()

    def predict(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {
                "label": "genuine",
                "confidence": 0.0,
                "stage1_score": 0.0,
                "stage2_score": 0.0,
                "is_sarcastic": False,
                "sarcasm_score": 0.0,
                "is_neutral": False,
                "neutral_score": 0.0,
            }

        stage1_score = self._run_stage1(text)
        sarcasm_score = self._detect_sarcasm(text)
        is_sarcastic = sarcasm_score >= SARCASM_THRESHOLD
        effective_genuine_threshold = max(
            self.stage1_model.threshold,
            HIGH_CONFIDENCE_GENUINE_THRESHOLD,
        )

        if stage1_score >= effective_genuine_threshold:
            neutral_score, is_neutral, neutral_reasons = self._detect_neutral(text, 0.0, sarcasm_score)
            if is_sarcastic:
                neutral_score = 0.0
                is_neutral = False
                neutral_reasons = []
            decision_reason = (
                "stage1_high_confidence_genuine"
                if not is_sarcastic
                else "sarcasm_override_applied_stage1_high_confidence"
            )
            result = {
                "label": "genuine",
                "confidence": float(stage1_score),
                "stage1_score": float(stage1_score),
                "stage2_score": 0.0,
                "is_sarcastic": bool(is_sarcastic),
                "sarcasm_score": float(sarcasm_score),
                "is_neutral": bool(is_neutral),
                "neutral_score": float(neutral_score),
            }
            if is_neutral:
                result["confidence"] = float(result["confidence"] * NEUTRAL_CONFIDENCE_MULTIPLIER)
            return result


        neutral_pre_score, neutral_pre_flag, neutral_pre_reasons = self._detect_neutral(
            text,
            0.0,
            sarcasm_score,
        )
        if not is_sarcastic and neutral_pre_flag:
            decision_reason = "neutral_pre_stage2_short_circuit"
            result = {
                "label": "genuine",
                "confidence": float(float(stage1_score) * NEUTRAL_CONFIDENCE_MULTIPLIER),
                "stage1_score": float(stage1_score),
                "stage2_score": 0.0,
                "is_sarcastic": False,
                "sarcasm_score": float(sarcasm_score),
                "is_neutral": True,
                "neutral_score": float(neutral_pre_score),
            }
            return result

        stage2_score = self._run_stage2(text)
        has_positive_intent, positive_intent_reasons = self._detect_positive_intent(text)

        neutral_score, is_neutral, neutral_reasons = self._detect_neutral(text, stage2_score, sarcasm_score)
        if is_sarcastic:
            neutral_score = 0.0
            is_neutral = False
            neutral_reasons = []

        if is_sarcastic:
            label = "genuine"
            decision_reason = "sarcasm_override_applied"
        elif has_positive_intent:
            label = "genuine"
            decision_reason = "positive_intent_override_applied"
        elif stage2_score > STAGE2_MANIPULATIVE_THRESHOLD:
            label = "manipulative"
            decision_reason = "stage2_above_manipulative_threshold"
        elif sarcasm_score < 0.5 and stage2_score < STAGE2_MANIPULATIVE_THRESHOLD:
            label = "genuine"
            decision_reason = "neutral_safety_low_signal"
        else:
            label = "genuine"
            decision_reason = "stage2_below_or_equal_manipulative_threshold"

        if label == "manipulative":
            confidence = stage2_score
        elif is_sarcastic:
            confidence = max(float(stage1_score), float(sarcasm_score))
        elif decision_reason in {"positive_intent_override_applied", "neutral_safety_low_signal"}:
            confidence = max(float(stage1_score), 1.0 - float(stage2_score))
        else:
            confidence = 1.0 - stage2_score
        result = {
            "label": label,
            "confidence": float(confidence),
            "stage1_score": float(stage1_score),
            "stage2_score": float(stage2_score),
            "is_sarcastic": bool(is_sarcastic),
            "sarcasm_score": float(sarcasm_score),
            "is_neutral": bool(is_neutral),
            "neutral_score": float(neutral_score),
        }
        if is_neutral:
            result["confidence"] = float(result["confidence"] * NEUTRAL_CONFIDENCE_MULTIPLIER)
        return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Optional[float]]]:
        if not texts:
            return []
        return [self.predict(text) for text in texts]

    def _resolve_artifacts(self) -> BehaviorArtifacts:
        root = self._resolve_model_root()

        stage1 = self._find_required_file(root, STAGE1_FILE)
        stage2 = self._find_required_file(root, STAGE2_WEIGHTS_FILE)
        model2_cfg = self._find_required_file(root, STAGE2_CONFIG_FILE)

        return BehaviorArtifacts(
            root_dir=root,
            stage1_file=stage1,
            stage2_weights_file=stage2,
            stage2_config_file=model2_cfg,
        )

    def _resolve_model_root(self) -> Path:
        local_override = os.getenv(LOCAL_MODEL_DIR_ENV)
        if local_override:
            root = Path(local_override).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(
                    f"{LOCAL_MODEL_DIR_ENV} points to missing path: {root}"
                )
            return root

        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data" / "incongruity-classifier"
        data_dir.mkdir(parents=True, exist_ok=True)

        required_paths = [
            data_dir / STAGE1_FILE,
            data_dir / STAGE2_WEIGHTS_FILE,
            data_dir / STAGE2_CONFIG_FILE,
        ]
        if all(path.exists() for path in required_paths):
            return data_dir.resolve()

        try:
            subprocess.run(
                [
                    "kaggle",
                    "datasets",
                    "download",
                    "-d",
                    KAGGLE_DATASET_REF,
                    "-p",
                    str(data_dir),
                    "--unzip",
                ],
                check=True,
                shell=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Failed to download behavior dataset via Kaggle CLI. "
                "Ensure Kaggle CLI is installed and credentials are set."
            ) from error

        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Behavior model download finished but required files are missing: "
                + ", ".join(missing)
            )

        return data_dir.resolve()

    @staticmethod
    def _find_required_file(root_dir: Path, filename: str) -> Path:
        candidates = list(root_dir.rglob(filename))
        if not candidates:
            raise FileNotFoundError(
                f"Required behavior artifact '{filename}' not found under: {root_dir}"
            )
        if len(candidates) > 1:
            candidates.sort(key=lambda path: len(path.parts))
        return candidates[0]

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object in {path}, found {type(parsed).__name__}")
        return parsed

    def _load_sarcasm_lexicon(self) -> Dict[str, set[str]]:
        repo_root = Path(__file__).resolve().parents[1]
        lexicon_path = repo_root / "data" / SARCASM_LEXICON_FILE
        if not lexicon_path.exists():
            return {
                "phrases": set(),
                "positive_words": set(),
                "negative_context_words": set(),
                "intensifiers": set(),
                "emoji": set(),
            }

        payload = self._load_json(lexicon_path)
        parsed: Dict[str, set[str]] = {}
        for key in ("phrases", "positive_words", "negative_context_words", "intensifiers", "emoji"):
            raw = payload.get(key, [])
            if isinstance(raw, list):
                parsed[key] = {
                    str(item).strip().lower()
                    for item in raw
                    if isinstance(item, str) and item.strip()
                }
            else:
                parsed[key] = set()

        return parsed

    @staticmethod
    def _load_stage1_model(config: Dict[str, Any]) -> _LinearTextScorer:
        model = _LinearTextScorer(config)
        return model

    def _load_stage2_model(self, weights_path: Path, config: Dict[str, Any]) -> nn.Module:
        model = self._build_stage2_model(config)
        payload = torch.load(weights_path, map_location="cpu")

        if isinstance(payload, nn.Module):
            return payload

        state_dict = self._extract_state_dict_payload(payload)
        load_result = model.load_state_dict(state_dict, strict=False)
        return model

    def _build_stage2_model(self, config: Dict[str, Any]) -> nn.Module:
        hidden_dims = self._resolve_hidden_dims(config)
        dropout = float(config.get("dropout", config.get("p_dropout", 0.1)))
        return _Stage2MLP(self.stage2_feature_dim, hidden_dims, dropout)

    def _run_stage1(self, text: str) -> float:
        return float(self.stage1_model.score(text))

    def _run_stage2(self, text: str) -> float:
        features = self._extract_stage2_features(text, self.stage2_config)
        tensor = torch.tensor([features], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.stage2_model(tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            if isinstance(logits, list):
                logits = logits[0]
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("Stage-2 model output is not a torch.Tensor.")

            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.shape[-1] == 2:
                probs = torch.softmax(logits, dim=-1)
                manip_prob = probs[0, STAGE2_MANIPULATIVE_INDEX].item()
                return float(manip_prob)

            if logits.shape[-1] == 1:
                return float(torch.sigmoid(logits).reshape(-1)[0].item())

            raise RuntimeError(
                f"Unexpected Stage-2 output shape: {tuple(logits.shape)}. "
                "Expected last dimension of 1 or 2."
            )

    def _detect_sarcasm(self, text: str) -> float:
        lowered = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9_']+", lowered)
        token_set = set(tokens)
        all_caps_words = re.findall(r"\b[A-Z]{2,}\b", text)

        phrases = self.sarcasm_lexicon.get("phrases", set())
        positive_words = self.sarcasm_lexicon.get("positive_words", set())
        negative_context = self.sarcasm_lexicon.get("negative_context_words", set())
        intensifiers = self.sarcasm_lexicon.get("intensifiers", set())
        emoji = self.sarcasm_lexicon.get("emoji", set())

        score = 0.0
        matched_rules: List[str] = []

        positive_count = sum(1 for token in tokens if token in positive_words)
        has_intensifier = bool(token_set.intersection(intensifiers))
        strong_sarcasm_phrases = {
            "yeah right",
            "as if",
            "sure thing",
            "just perfect",
            "how convenient",
            "what a surprise",
            "big surprise",
            "nice going",
            "oh great",
            "wow",
            "sure that makes sense",
            "sure that makes total sense",
            "sure, that makes total sense",
        }

        lexicon_phrase_hits = [phrase for phrase in phrases if phrase in lowered and phrase in strong_sarcasm_phrases]
        has_phrase_match = any(phrase in lowered for phrase in strong_sarcasm_phrases) or bool(lexicon_phrase_hits)
        has_contrast = bool(token_set.intersection(positive_words) and token_set.intersection(negative_context))

        expectation_violation_phrases = {
            "perfect timing",
            "just what i needed",
            "exactly what i needed",
            "exactly what i wanted",
            "couldn't be better",
            "nothing could go wrong",
            "as always",
            "of course",
            "naturally",
            "sure that makes sense",
            "sure, that makes sense",
            "sure that makes total sense",
            "sure, that makes total sense",
            "that makes sense",
        }
        has_expectation_violation = any(phrase in lowered for phrase in expectation_violation_phrases)

        love_how_patterns = (
            lowered.startswith("love how")
            or lowered.startswith("love when")
        )
        because_patterns = (
            "sure, because" in lowered
            or "yeah, because" in lowered
            or "sure because" in lowered
            or "yeah because" in lowered
            or "right because" in lowered
            or "right, because" in lowered
        )
        because_contradiction_clauses = {
            "that worked",
            "that works",
            "that's going to work",
            "that is going to work",
            "that will work",
            "that'll work",
            "that makes sense",
            "makes total sense",
        }
        has_because_contradiction = because_patterns and any(
            clause in lowered for clause in because_contradiction_clauses
        )

        thanks_for_patterns = (
            "thanks for" in lowered
            or "thank you for" in lowered
        )
        thanks_negative_phrases = {
            "ignoring me",
            "ignoring",
            "not replying",
            "late reply",
            "late response",
            "no response",
            "so fast",
            "so quickly",
            "for nothing",
            "making me wait",
        }
        thanks_negative_tokens = {
            "ignore",
            "ignoring",
            "ignored",
            "delay",
            "late",
            "slow",
            "waiting",
            "wait",
            "problem",
            "issue",
            "broken",
            "broke",
        }
        has_thanks_for_negative_outcome = (
            thanks_for_patterns
            and (
                any(phrase in lowered for phrase in thanks_negative_phrases)
                or bool(token_set.intersection(thanks_negative_tokens))
                or bool(token_set.intersection(negative_context))
            )
        )
        makes_sense_pattern = "that makes sense" in lowered
        explicit_sarcasm_phrase = (
            has_phrase_match
            or love_how_patterns
            or because_patterns
            or has_because_contradiction
            or has_thanks_for_negative_outcome
        )

        repetition_markers = {"again", "another", "as", "always", "keeps", "happening"}
        has_repetition_pattern = (
            ("again" in token_set)
            or ("another" in token_set)
            or ("as" in token_set and "always" in token_set)
            or ("keeps" in token_set and "happening" in token_set)
        )

        has_positive_repetition = positive_count > 0 and has_repetition_pattern

        neg_event_markers = {"another", "again", "delay", "problem", "issue", "broke", "broken"}
        has_positive_negative_event = positive_count > 0 and bool(token_set.intersection(neg_event_markers))

        triggered_gates: List[str] = []
        if has_expectation_violation:
            triggered_gates.append("expectation_violation")
        if has_repetition_pattern:
            triggered_gates.append("repetition")
        if explicit_sarcasm_phrase:
            triggered_gates.append("explicit_sarcasm_phrase")
        if has_because_contradiction:
            triggered_gates.append("because_contradiction")
        if has_thanks_for_negative_outcome:
            triggered_gates.append("thanks_for_negative_outcome")

        if not (has_expectation_violation or has_repetition_pattern or explicit_sarcasm_phrase):
            return 0.0


        if "yeah right" in lowered:
            score += 0.5
            matched_rules.append("boost:yeah_right:+0.5")
        if "wow" in token_set:
            score += 0.3
            matched_rules.append("boost:wow:+0.3")
        if "genius" in token_set:
            score += 0.4
            matched_rules.append("boost:genius:+0.4")
        if "amazing" in token_set:
            score += 0.3
            matched_rules.append("boost:amazing:+0.3")

        if has_phrase_match:
            score += 0.3
            matched_rules.append("phrase_match:+0.3")

        if any(mark in text for mark in emoji):
            score += 0.4
            matched_rules.append("emoji_match:+0.4")

        if has_contrast:
            score += 0.6
            matched_rules.append("positive_negative_context:+0.6")

        if has_expectation_violation:
            score += 0.4
            matched_rules.append("expectation_violation:+0.4")

        if has_positive_repetition:
            score += 0.4
            matched_rules.append("repetition_with_positive:+0.4")

        if has_positive_negative_event:
            score += 0.5
            matched_rules.append("positive_plus_negative_event:+0.5")

        if love_how_patterns:
            score += 0.5
            matched_rules.append("love_how_or_when:+0.5")

        if because_patterns:
            score += 0.5
            matched_rules.append("sure_or_yeah_because:+0.5")

        if has_because_contradiction:
            score += 0.5
            matched_rules.append("because_contradiction:+0.5")

        if has_thanks_for_negative_outcome:
            score += 0.5
            matched_rules.append("thanks_for_negative_outcome:+0.5")

        if makes_sense_pattern and because_patterns:
            score += 0.3
            matched_rules.append("that_makes_sense_with_because:+0.3")

        if intensifiers and positive_words:
            for idx, token in enumerate(tokens[:-1]):
                if token in intensifiers and tokens[idx + 1] in positive_words:
                    score += 0.3
                    matched_rules.append("intensifier_plus_positive_adjacent:+0.3")
                    break

        if positive_count >= 2:
            score += 0.4
            matched_rules.append("two_plus_positive_words:+0.4")

        if has_phrase_match and has_intensifier:
            score += 0.3
            matched_rules.append("phrase_plus_intensifier:+0.3")

        if positive_count > 0 and all_caps_words:
            score += 0.3
            matched_rules.append("positive_plus_all_caps:+0.3")

        if len(tokens) < 6 and positive_count > 0:
            score += 0.1
            matched_rules.append("short_sentence_positive:+0.1")

        if all_caps_words:
            score += 0.1
            matched_rules.append("all_caps:+0.1")

        if re.search(r"(!{2,}|\?{2,}|!\?|\?!)", text):
            score += 0.2
            matched_rules.append("repeated_punctuation:+0.2")

        final_score = float(max(0.0, min(1.0, score)))
        return final_score

    def _detect_neutral(self, text: str, stage2_score: float, sarcasm_score: float) -> Tuple[float, bool, List[str]]:
        if sarcasm_score > 0.0:
            return 0.0, False, ["sarcasm_present"]
        if stage2_score >= STAGE2_MANIPULATIVE_THRESHOLD:
            return 0.0, False, ["stage2_above_threshold"]

        lowered = text.strip().lower()
        tokens = re.findall(r"[a-zA-Z0-9_']+", lowered)
        token_count = len(tokens)

        score = 0.0
        reasons: List[str] = []

        exact_neutral_phrases = {
            "well that was something",
            "interesting choice",
            "good to know",
            "okay",
            "alright",
            "noted",
        }
        vague_terms = {
            "well",
            "something",
            "interesting",
            "okay",
            "alright",
            "noted",
            "fine",
        }
        generic_phrases = {
            "good to know",
            "interesting choice",
            "well that was something",
            "sounds good",
            "makes sense",
        }

        if lowered in exact_neutral_phrases:
            score += 0.7
            reasons.append("exact_neutral_phrase")

        if token_count <= 4:
            score += 0.25
            reasons.append("short_text")

        vague_hits = sum(1 for token in tokens if token in vague_terms)
        if vague_hits > 0:
            score += min(0.3, 0.15 * vague_hits)
            reasons.append("vague_terms")

        if any(phrase in lowered for phrase in generic_phrases):
            score += 0.25
            reasons.append("generic_phrase")

        neutral_score = float(max(0.0, min(1.0, score)))
        return neutral_score, neutral_score >= NEUTRAL_THRESHOLD, reasons

    @staticmethod
    def _detect_positive_intent(text: str) -> Tuple[bool, List[str]]:
        lowered = text.lower()
        reasons: List[str] = []

        strong_positive_markers = {
            "thanks for": "thanks_for",
            "appreciate": "appreciate",
            "proud of": "proud_of",
            "great effort": "great_effort",
        }
        for marker, reason in strong_positive_markers.items():
            if marker in lowered:
                reasons.append(reason)

        return bool(reasons), reasons

    def _extract_stage2_features(self, text: str, config: Dict[str, Any]) -> List[float]:
        feature_cols = config.get("feature_cols")
        if isinstance(feature_cols, list) and feature_cols:
            return self._extract_stage2_engineered_features(text, feature_cols)

        # Fallback if feature schema is unavailable in config.
        vocab = config.get("vocabulary") or config.get("vocab") or config.get("token_to_idx")
        stats_enabled = bool(config.get("use_stats", True))
        features: List[float] = []
        if isinstance(vocab, dict) and vocab:
            features.extend(self._extract_bow_features(text, vocab))
        if stats_enabled:
            features.extend(self._extract_stat_features(text))
        if not features:
            features = self._extract_stat_features(text)

        if len(features) != self.stage2_feature_dim:
            raise RuntimeError(
                f"Stage-2 fallback feature length mismatch: expected {self.stage2_feature_dim}, "
                f"got {len(features)}."
            )
        return features

    def _extract_stage2_engineered_features(self, text: str, feature_cols: List[Any]) -> List[float]:
        columns = [str(col) for col in feature_cols]
        tokens = re.findall(r"[a-zA-Z0-9_']+", text.lower())

        # Approximate Plutchik-style lexical cues to build the expected engineered schema.
        lexicon = {
            "joy": {"happy", "glad", "great", "good", "love", "thanks", "appreciate"},
            "trust": {"trust", "honest", "sure", "promise", "reliable", "faith"},
            "fear": {"fear", "afraid", "worry", "scared", "anxious", "terrified"},
            "surprise": {"wow", "unexpected", "sudden", "surprised", "shock"},
            "sadness": {"sad", "hurt", "upset", "down", "cry", "sorry"},
            "disgust": {"disgust", "gross", "awful", "nasty", "hate"},
            "anger": {"angry", "mad", "furious", "annoyed", "rage"},
            "anticipation": {"will", "soon", "expect", "plan", "hope", "maybe"},
        }
        emotions = ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")

        token_count = max(1, len(tokens))
        base = {
            emotion: float(sum(1 for token in tokens if token in lexicon[emotion])) / token_count
            for emotion in emotions
        }

        # Build pseudo temporal deltas using 4 chunks from the same message.
        chunks = self._split_tokens(tokens, 4)
        chunk_vecs: List[Dict[str, float]] = []
        for chunk in chunks:
            c_count = max(1, len(chunk))
            chunk_vecs.append(
                {
                    emotion: float(sum(1 for token in chunk if token in lexicon[emotion])) / c_count
                    for emotion in emotions
                }
            )

        dv1 = {emotion: chunk_vecs[1][emotion] - chunk_vecs[0][emotion] for emotion in emotions}
        dv2 = {emotion: chunk_vecs[2][emotion] - chunk_vecs[1][emotion] for emotion in emotions}
        dv3 = {emotion: chunk_vecs[3][emotion] - chunk_vecs[2][emotion] for emotion in emotions}

        tension = {
            "tension_joy_sadness": abs(base["joy"] - base["sadness"]),
            "tension_trust_disgust": abs(base["trust"] - base["disgust"]),
            "tension_fear_anger": abs(base["fear"] - base["anger"]),
            "tension_surprise_anticip": abs(base["surprise"] - base["anticipation"]),
        }
        tension_total = sum(tension.values())

        exclam = text.count("!")
        questions = text.count("?")
        negations = sum(1 for t in tokens if t in {"not", "no", "never", "nothing"})
        pressure = sum(1 for t in tokens if t in {"must", "should", "need", "have", "now"})
        hedging = sum(1 for t in tokens if t in {"maybe", "perhaps", "kinda", "sorta", "probably"})

        engineered: Dict[str, float] = {}
        engineered.update(base)
        for emotion in emotions:
            engineered[f"dv1_{emotion}"] = dv1[emotion]
            engineered[f"dv2_{emotion}"] = dv2[emotion]
            engineered[f"dv3_{emotion}"] = dv3[emotion]
        engineered.update(tension)

        engineered["drift"] = sum(abs(dv1[e]) + abs(dv2[e]) + abs(dv3[e]) for e in emotions) / len(emotions)
        engineered["actor_specificity"] = (
            sum(1 for t in tokens if t in {"you", "your", "yours"}) / token_count
        )
        engineered["spike"] = max((abs(v) for v in dv1.values()), default=0.0)
        engineered["tension_total"] = tension_total
        engineered["suppression"] = max(0.0, base["anger"] + base["sadness"] - base["joy"])
        engineered["incoherence"] = abs(base["trust"] - base["anger"]) + abs(base["joy"] - base["disgust"])
        engineered["lex_negation"] = float(negations) / token_count
        engineered["lex_pressure"] = float(pressure) / token_count
        engineered["lex_hedging"] = float(hedging) / token_count
        engineered["lex_punct"] = float(exclam + questions) / max(1, len(text))

        vector = [float(engineered.get(name, 0.0)) for name in columns]
        if len(vector) != self.stage2_feature_dim:
            raise RuntimeError(
                f"Stage-2 engineered feature length mismatch: expected {self.stage2_feature_dim}, "
                f"got {len(vector)}."
            )
        return vector

    @staticmethod
    def _split_tokens(tokens: List[str], n_parts: int) -> List[List[str]]:
        if n_parts <= 1:
            return [tokens]
        if not tokens:
            return [[] for _ in range(n_parts)]
        size = max(1, math.ceil(len(tokens) / n_parts))
        chunks: List[List[str]] = []
        for idx in range(n_parts):
            start = idx * size
            end = min(len(tokens), (idx + 1) * size)
            if start >= len(tokens):
                chunks.append([])
            else:
                chunks.append(tokens[start:end])
        while len(chunks) < n_parts:
            chunks.append([])
        return chunks

    @staticmethod
    def _extract_bow_features(text: str, vocabulary: Dict[str, Any]) -> List[float]:
        parsed_vocab: Dict[str, int] = {}
        for token, index in vocabulary.items():
            if isinstance(token, str) and isinstance(index, int):
                parsed_vocab[token] = index

        if not parsed_vocab:
            return []

        vector = [0.0] * (max(parsed_vocab.values()) + 1)
        tokens = re.findall(r"[a-zA-Z0-9_']+", text.lower())
        for token in tokens:
            idx = parsed_vocab.get(token)
            if idx is None or idx >= len(vector):
                continue
            vector[idx] += 1.0
        return vector

    @staticmethod
    def _extract_stat_features(text: str) -> List[float]:
        stripped = text.strip()
        if not stripped:
            return [0.0] * 8

        length = len(stripped)
        tokens = re.findall(r"[a-zA-Z0-9_']+", stripped)
        token_count = max(1, len(tokens))

        uppercase_chars = sum(1 for ch in stripped if ch.isupper())
        alpha_chars = sum(1 for ch in stripped if ch.isalpha())
        punctuation_chars = sum(1 for ch in stripped if ch in "!?.,;:-")

        exclamation_count = stripped.count("!")
        question_count = stripped.count("?")
        quote_count = stripped.count("\"") + stripped.count("'")
        ellipsis_count = stripped.count("...")

        uppercase_ratio = uppercase_chars / max(1, alpha_chars)
        punctuation_ratio = punctuation_chars / max(1, length)
        avg_token_length = sum(len(token) for token in tokens) / token_count

        return [
            float(token_count),
            float(length),
            float(exclamation_count),
            float(question_count),
            float(uppercase_ratio),
            float(punctuation_ratio),
            float(quote_count + ellipsis_count),
            float(avg_token_length),
        ]

    @staticmethod
    def _extract_state_dict_payload(payload: Any) -> Dict[str, torch.Tensor]:
        if isinstance(payload, dict) and payload and all(isinstance(k, str) for k in payload.keys()):
            if all(isinstance(v, (torch.Tensor, nn.Parameter)) for v in payload.values()):
                return {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in payload.items()
                }
            for key in ("state_dict", "model_state_dict"):
                candidate = payload.get(key)
                if isinstance(candidate, dict) and candidate and all(isinstance(k, str) for k in candidate.keys()):
                    return {
                        (k[7:] if k.startswith("module.") else k): v
                        for k, v in candidate.items()
                    }

        raise RuntimeError(
            "Unsupported stage-2 checkpoint format. Expected raw state_dict or wrapped state_dict/model_state_dict."
        )

    @staticmethod
    def _resolve_hidden_dims(config: Dict[str, Any]) -> List[int]:
        raw_hidden = config.get("hidden_dims") or config.get("mlp_hidden_dims") or config.get("layers")
        if isinstance(raw_hidden, list):
            parsed = [int(v) for v in raw_hidden if isinstance(v, int) and v > 0]
            if parsed:
                return parsed

        architecture = config.get("architecture")
        if isinstance(architecture, str) and "46" in architecture and "256" in architecture:
            return [256, 128, 64, 32]

        hidden_dim = config.get("hidden_dim")
        if isinstance(hidden_dim, int) and hidden_dim > 0:
            return [hidden_dim]

        return [64, 32]

    @staticmethod
    def _resolve_stage2_feature_dim(config: Dict[str, Any]) -> int:
        for key in ("input_dim", "feature_dim", "n_features"):
            value = config.get(key)
            if isinstance(value, int) and value > 0:
                return value

        vocab = config.get("vocabulary") or config.get("vocab") or config.get("token_to_idx")
        if isinstance(vocab, dict) and vocab:
            max_idx = max((idx for idx in vocab.values() if isinstance(idx, int)), default=-1)
            if max_idx >= 0:
                # +8 for statistical features.
                return max_idx + 1 + 8

        return 128


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)

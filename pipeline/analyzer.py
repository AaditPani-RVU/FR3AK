from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from models.behavior_model import BehaviorModel
from models.emotion_model import EmotionModel, PLUTCHIK_EMOTIONS


@dataclass
class UserState:
    message_count: int = 0
    emotion_sum: List[float] = field(default_factory=lambda: [0.0] * len(PLUTCHIK_EMOTIONS))
    first_emotion: Optional[List[float]] = None
    last_emotion: Optional[List[float]] = None
    sarcasm_count: int = 0
    manipulation_count: int = 0
    neutral_count: int = 0
    total_emotion_intensity: float = 0.0


class ConversationAnalyzer:
    """Sequential conversation analyzer for emotion and behavior signals."""

    def __init__(
        self,
        emotion_model: Optional[EmotionModel] = None,
        behavior_model: Optional[BehaviorModel] = None,
    ) -> None:
        self.emotion_model = emotion_model or EmotionModel()
        self.behavior_model = behavior_model or BehaviorModel()

    def analyze(self, records: Iterable[Any]) -> Dict[str, Any]:
        output_messages: List[Dict[str, Any]] = []
        per_user_state: Dict[str, UserState] = {}

        for index, record in enumerate(records):
            speaker, text, timestamp = self._extract_record_fields(record)
            if not speaker:
                speaker = f"unknown_{index + 1}"

            emotion_dict = self._safe_emotion_predict(text)
            behavior = self._safe_behavior_predict(text)

            emotion_vector = [
                float(emotion_dict.get(emotion, 0.0))
                for emotion in PLUTCHIK_EMOTIONS
            ]
            if len(emotion_vector) != len(PLUTCHIK_EMOTIONS):
                emotion_vector = [0.0] * len(PLUTCHIK_EMOTIONS)
            emotion_array = np.array(emotion_vector)
            raw_intensity = np.max(emotion_array) - np.mean(emotion_array)
            emotion_intensity = float(raw_intensity / (np.max(emotion_array) + 1e-6))

            message_result = {
                "speaker": speaker,
                "text": text,
                "index": int(index),
                "timestamp": timestamp if timestamp is not None else None,
                "emotion_vector": emotion_vector,
                "emotion_intensity": emotion_intensity,
                "label": str(behavior.get("label", "genuine")),
                "is_sarcastic": bool(behavior.get("is_sarcastic", False)),
                "is_neutral": bool(behavior.get("is_neutral", False)),
            }
            output_messages.append(message_result)

            state = per_user_state.setdefault(speaker, UserState())
            self._update_user_state(state, message_result)

        users_summary = {
            speaker: self._finalize_user_state(state)
            for speaker, state in per_user_state.items()
        }

        return {
            "users": users_summary,
            "messages": output_messages,
        }

    @staticmethod
    def _extract_record_fields(record: Any) -> Tuple[str, str, Any]:
        if isinstance(record, Mapping):
            speaker = str(record.get("speaker_id") or record.get("speaker") or "").strip()
            speaker = ConversationAnalyzer._normalize_speaker_identifier(speaker)
            message = str(record.get("cleaned_message") or record.get("message") or "").strip()
            timestamp = record.get("timestamp", None)
            return speaker, message, timestamp

        speaker = str(getattr(record, "speaker_id", "") or getattr(record, "speaker", "")).strip()
        speaker = ConversationAnalyzer._normalize_speaker_identifier(speaker)
        message = str(
            getattr(record, "cleaned_message", "")
            or getattr(record, "message", "")
            or ""
        ).strip()
        timestamp = getattr(record, "timestamp", None)
        return speaker, message, timestamp

    @staticmethod
    def _normalize_speaker_identifier(value: str) -> str:
        collapsed = " ".join(str(value).split()).strip()
        if not collapsed:
            return ""
        normalized = collapsed.lower()
        return normalized

    def _safe_emotion_predict(self, text: str) -> Dict[str, float]:
        safe_text = text if isinstance(text, str) else ""
        try:
            prediction = self.emotion_model.predict(safe_text)
            if isinstance(prediction, Mapping):
                return {str(k): float(v) for k, v in prediction.items()}
        except Exception:
            pass

        return {emotion: 0.0 for emotion in PLUTCHIK_EMOTIONS}

    def _safe_behavior_predict(self, text: str) -> Dict[str, Any]:
        safe_text = text if isinstance(text, str) else ""
        try:
            prediction = self.behavior_model.predict(safe_text)
            if isinstance(prediction, Mapping):
                return dict(prediction)
        except Exception:
            pass

        return {
            "label": "genuine",
            "is_sarcastic": False,
            "is_neutral": False,
        }

    @staticmethod
    def _update_user_state(state: UserState, message_result: Mapping[str, Any]) -> None:
        emotion_vector_raw = message_result.get("emotion_vector", [])
        emotion_vector = [
            float(value) if isinstance(value, (int, float)) else 0.0
            for value in emotion_vector_raw
        ]
        if len(emotion_vector) != len(PLUTCHIK_EMOTIONS):
            emotion_vector = [0.0] * len(PLUTCHIK_EMOTIONS)

        state.message_count += 1
        state.emotion_sum = [
            current + increment
            for current, increment in zip(state.emotion_sum, emotion_vector)
        ]
        state.total_emotion_intensity += float(message_result.get("emotion_intensity", 0.0) or 0.0)
        if state.first_emotion is None:
            state.first_emotion = list(emotion_vector)
        state.last_emotion = list(emotion_vector)

        if bool(message_result.get("is_sarcastic", False)):
            state.sarcasm_count += 1
        if str(message_result.get("label", "genuine")) == "manipulative":
            state.manipulation_count += 1
        if bool(message_result.get("is_neutral", False)):
            state.neutral_count += 1

    @staticmethod
    def _finalize_user_state(state: UserState) -> Dict[str, Any]:
        if state.message_count <= 0:
            return {
                "message_count": 0,
                "emotion_avg": [0.0] * len(PLUTCHIK_EMOTIONS),
                "sarcasm_frequency": 0.0,
                "manipulation_frequency": 0.0,
                "neutral_frequency": 0.0,
                "emotion_drift": [0.0] * len(PLUTCHIK_EMOTIONS),
            }

        emotion_avg = [value / state.message_count for value in state.emotion_sum]
        first = state.first_emotion or [0.0] * len(PLUTCHIK_EMOTIONS)
        last = state.last_emotion or [0.0] * len(PLUTCHIK_EMOTIONS)
        emotion_drift = [last_value - first_value for last_value, first_value in zip(last, first)]
        avg_emotion_intensity = state.total_emotion_intensity / state.message_count

        return {
            "message_count": state.message_count,
            "emotion_avg": emotion_avg,
            "sarcasm_frequency": state.sarcasm_count / state.message_count,
            "manipulation_frequency": state.manipulation_count / state.message_count,
            "neutral_frequency": state.neutral_count / state.message_count,
            "emotion_drift": emotion_drift,
            "total_emotion_intensity": float(state.total_emotion_intensity),
            "avg_emotion_intensity": float(avg_emotion_intensity),
        }


def analyze_conversation(
    records: Iterable[Any],
    emotion_model: Optional[EmotionModel] = None,
    behavior_model: Optional[BehaviorModel] = None,
) -> Dict[str, Any]:
    analyzer = ConversationAnalyzer(
        emotion_model=emotion_model,
        behavior_model=behavior_model,
    )
    return analyzer.analyze(records)

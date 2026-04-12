from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional

from models.emotion_model import PLUTCHIK_EMOTIONS


@dataclass(frozen=True)
class InsightProfile:
    dominant_emotion: str
    emotional_tone: str
    emotional_stability: str
    sarcasm_level: str
    manipulation_level: str
    neutral_level: str
    risk_flags: List[str]
    summary: str


class InsightEngine:
    """Deterministic interpreter for analyzer outputs."""

    def __init__(self, *, seed: Optional[int] = None, randomize: bool = False) -> None:
        self.seed = seed
        self.randomize = randomize
        self.summary_templates = [
            "{speaker} exhibits {tone} emotional signals with {stability}, showing {manipulation} manipulation tendencies and {sarcasm} sarcasm expression.",
            "{speaker} demonstrates {tone} emotional dynamics, characterized by {stability}, {sarcasm} sarcasm expression, and {manipulation} manipulation tendencies.",
            "{speaker} shows {tone} emotional cues with {stability}, alongside {manipulation} manipulation and {sarcasm} sarcasm expression.",
            "{speaker} presents {tone} emotional responses with {stability}, {manipulation} manipulation tendencies, and {sarcasm} sarcasm expression.",
            "{speaker} reflects {tone} emotional patterns through {stability}, including {manipulation} manipulation tendencies and {sarcasm} sarcasm expression.",
        ]

    def analyze(self, analyzer_output: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        users = analyzer_output.get("users", {}) if isinstance(analyzer_output, Mapping) else {}
        if not isinstance(users, Mapping):
            users = {}

        messages = analyzer_output.get("messages", []) if isinstance(analyzer_output, Mapping) else []
        speaker_messages: Dict[str, List[Mapping[str, Any]]] = {}
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                speaker = str(message.get("speaker", "")).strip()
                if not speaker:
                    continue
                speaker_messages.setdefault(speaker, []).append(message)

        return {
            str(speaker): self._build_user_insight(
                str(speaker),
                data,
                speaker_messages.get(str(speaker), []),
            )
            for speaker, data in users.items()
            if isinstance(speaker, str)
        }

    def _build_user_insight(
        self,
        speaker: str,
        data: Any,
        messages: List[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        user_data = data if isinstance(data, Mapping) else {}

        emotion_avg = self._coerce_vector(user_data.get("emotion_avg"), len(PLUTCHIK_EMOTIONS))
        emotion_drift = self._coerce_vector(user_data.get("emotion_drift"), len(PLUTCHIK_EMOTIONS))

        dominant_emotion = self._dominant_emotion(emotion_avg)
        emotional_stability = self._classify_stability(emotion_drift)
        sarcasm_frequency = self._safe_float(user_data.get("sarcasm_frequency", 0.0))
        manipulation_frequency = self._safe_float(user_data.get("manipulation_frequency", 0.0))
        neutral_frequency = self._safe_float(user_data.get("neutral_frequency", 0.0))
        avg_emotion_intensity = self._safe_float(user_data.get("avg_emotion_intensity", 0.0))
        message_count = int(self._safe_float(user_data.get("message_count", 0)))
        phrase_signals = self._message_phrase_signals(messages)

        sarcasm_level = self._classify_frequency(sarcasm_frequency, low=0.1, medium=0.3)
        neutral_level = self._classify_frequency(neutral_frequency, low=0.2, medium=0.5)
        emotional_tone_pre_override = self._classify_tone(dominant_emotion)
        manipulation_level = self._classify_manipulation_level(
            manipulation_frequency=manipulation_frequency,
            sarcasm_frequency=sarcasm_frequency,
            neutral_frequency=neutral_frequency,
            message_count=message_count,
            sarcasm_level=sarcasm_level,
            emotional_stability=emotional_stability,
            emotional_tone_pre_override=emotional_tone_pre_override,
            task_oriented_ratio=self._task_oriented_ratio(messages),
            has_coercive_phrases=phrase_signals["has_coercive"],
            has_frustration_signals=phrase_signals["has_frustration"],
        )
        emotional_tone = emotional_tone_pre_override
        emotional_intensity_level = self._classify_intensity(avg_emotion_intensity)

        # Behavioral signals take precedence over emotion-only tone assignment.
        if manipulation_level == "high":
            emotional_tone = "negative"
        elif manipulation_level == "medium":
            emotional_tone = "mixed"
        elif sarcasm_level == "high":
            emotional_tone = "mixed"

        risk_flags = self._build_risk_flags(
            manipulation_level=manipulation_level,
            sarcasm_level=sarcasm_level,
            emotional_stability=emotional_stability,
            emotional_tone=emotional_tone,
        )

        summary = self._build_summary(
            speaker=speaker,
            tone=emotional_tone,
            stability=emotional_stability,
            manipulation=manipulation_level,
            sarcasm=sarcasm_level,
            neutral=neutral_level,
            intensity=emotional_intensity_level,
        )

        profile = InsightProfile(
            dominant_emotion=dominant_emotion,
            emotional_tone=emotional_tone,
            emotional_stability=emotional_stability,
            sarcasm_level=sarcasm_level,
            manipulation_level=manipulation_level,
            neutral_level=neutral_level,
            risk_flags=risk_flags,
            summary=summary,
        )
        return {
            "dominant_emotion": profile.dominant_emotion,
            "emotional_tone": profile.emotional_tone,
            "emotional_stability": profile.emotional_stability,
            "sarcasm_level": profile.sarcasm_level,
            "manipulation_level": profile.manipulation_level,
            "neutral_level": profile.neutral_level,
            "emotional_intensity_level": emotional_intensity_level,
            "risk_flags": profile.risk_flags,
            "summary": profile.summary,
        }

    @staticmethod
    def _coerce_vector(values: Any, expected_length: int) -> List[float]:
        if not isinstance(values, list):
            return [0.0] * expected_length
        vector = [InsightEngine._safe_float(value) for value in values]
        if len(vector) < expected_length:
            vector.extend([0.0] * (expected_length - len(vector)))
        elif len(vector) > expected_length:
            vector = vector[:expected_length]
        return vector

    @staticmethod
    def _safe_float(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _dominant_emotion(emotion_avg: List[float]) -> str:
        if not emotion_avg:
            return "unknown"
        dominant_index = max(range(len(emotion_avg)), key=lambda idx: emotion_avg[idx])
        if dominant_index < len(PLUTCHIK_EMOTIONS):
            return PLUTCHIK_EMOTIONS[dominant_index]
        return "unknown"

    @staticmethod
    def _classify_stability(emotion_drift: List[float]) -> str:
        magnitude = math.sqrt(sum(value * value for value in emotion_drift))
        if magnitude < 0.1:
            return "stable"
        if magnitude <= 0.3:
            return "shifting"
        return "volatile"

    @staticmethod
    def _classify_frequency(value: float, *, low: float, medium: float) -> str:
        if value < low:
            return "low"
        if value <= medium:
            return "medium"
        return "high"

    @staticmethod
    def _classify_tone(dominant_emotion: str) -> str:
        if dominant_emotion in {"joy", "trust"}:
            return "positive"
        if dominant_emotion in {"sadness", "anger", "disgust", "fear"}:
            return "negative"
        if dominant_emotion in {"surprise", "anticipation"}:
            return "dynamic"
        return "mixed"

    @staticmethod
    def _classify_intensity(avg_emotion_intensity: float) -> str:
        # Thresholds assume analyzer outputs normalized dominance intensity in [0, 1]-like range.
        if avg_emotion_intensity > 0.9:
            return "high emotional intensity"
        if avg_emotion_intensity >= 0.7:
            return "moderate emotional intensity"
        return "low emotional intensity"

    def _classify_manipulation_level(
        self,
        *,
        manipulation_frequency: float,
        sarcasm_frequency: float,
        neutral_frequency: float,
        message_count: int,
        sarcasm_level: str,
        emotional_stability: str,
        emotional_tone_pre_override: str,
        task_oriented_ratio: float,
        has_coercive_phrases: bool,
        has_frustration_signals: bool,
    ) -> str:
        base_level = self._classify_frequency(manipulation_frequency, low=0.1, medium=0.3)

        # Keep high-confidence joint manipulation+sarcasm cases intact.
        if base_level == "high" and sarcasm_level == "high":
            return "high"

        # Minimum evidence: high manipulation needs enough samples.
        if message_count < 3:
            base_level = "low"

        # Consistency check to reduce false positives from isolated classifier spikes.
        if (
            manipulation_frequency > 0.3
            and sarcasm_frequency < 0.2
            and emotional_tone_pre_override == "positive"
            and base_level != "low"
        ):
            base_level = "medium"

        # Protect neutral-heavy users unless manipulation is very dominant.
        if neutral_frequency > 0.5 and manipulation_frequency < 0.5:
            base_level = "low"

        # Stabilizer detection: low sarcasm, moderate-or-lower manipulation signal,
        # and non-volatile drift should not be labeled manipulative.
        is_stabilizer = (
            sarcasm_frequency < 0.2
            and manipulation_frequency < 0.5
            and emotional_stability != "volatile"
        )
        if is_stabilizer:
            base_level = "low"

        # Stronger neutral protection for low-sarcasm users.
        if neutral_frequency > 0.3 and sarcasm_level == "low":
            base_level = "low"

        # Consistency guard for positive-tone, low-sarcasm medium cases.
        if (
            base_level == "medium"
            and sarcasm_level == "low"
            and emotional_tone_pre_override == "positive"
        ):
            base_level = "low"

        subtle_manipulation = (
            0.1 < manipulation_frequency <= 0.4
            and sarcasm_frequency < 0.3
            and message_count >= 3
        )
        protect_strong_non_manipulator = (
            (
                neutral_frequency > 0.4
                and sarcasm_frequency < 0.2
            )
            or (
                emotional_stability == "stable"
                and sarcasm_frequency < 0.2
                and manipulation_frequency <= 0.2
            )
        )

        if subtle_manipulation and not protect_strong_non_manipulator and base_level != "high":
            base_level = "medium"

        # Optional dampener: mostly short directive/task-oriented users with low
        # sarcasm should not be over-interpreted as manipulative.
        if (
            task_oriented_ratio >= 0.6
            and sarcasm_frequency < 0.2
            and base_level == "medium"
            and manipulation_frequency < 0.5
        ):
            base_level = "low"

        # Emotional frustration alone should not imply manipulation.
        if has_frustration_signals and not has_coercive_phrases and base_level != "high":
            base_level = "low"

        # Borderline low-sarcasm manipulation is downgraded unless coercive cues
        # are present; this keeps emotional but non-manipulative users low.
        if (
            sarcasm_level == "low"
            and manipulation_frequency <= 0.4
            and emotional_tone_pre_override != "negative"
            and not has_coercive_phrases
            and base_level != "high"
        ):
            base_level = "low"

        return base_level

    @staticmethod
    def _message_phrase_signals(messages: List[Mapping[str, Any]]) -> Dict[str, bool]:
        coercive_phrases = {
            "if you",
            "if we",
            "you should",
            "you must",
            "you need",
            "wouldnt be needed",
            "wouldn't be needed",
            "should have",
            "should've",
            "everyone is thinking",
            "thats what should",
            "that's what should",
        }
        frustration_phrases = {
            "im frustrated",
            "i'm frustrated",
            "getting frustrated",
            "this is annoying",
            "this is unfair",
            "bit unfair",
            "frustrating",
        }

        has_coercive = False
        has_frustration = False
        for message in messages:
            text = str(message.get("text", "") or "").strip().lower()
            normalized = text.replace("’", "'")
            normalized = normalized.replace("“", '"').replace("”", '"')
            compact = normalized.replace(" ", "")

            if any(phrase in normalized for phrase in coercive_phrases) or any(
                phrase.replace(" ", "") in compact for phrase in coercive_phrases
            ):
                has_coercive = True

            if any(phrase in normalized for phrase in frustration_phrases) or any(
                phrase.replace(" ", "") in compact for phrase in frustration_phrases
            ):
                has_frustration = True

        return {
            "has_coercive": has_coercive,
            "has_frustration": has_frustration,
        }

    @staticmethod
    def _task_oriented_ratio(messages: List[Mapping[str, Any]]) -> float:
        if not messages:
            return 0.0

        directive_terms = {
            "let",
            "lets",
            "please",
            "focus",
            "wrap",
            "finish",
            "review",
            "assign",
            "action",
            "need",
            "must",
            "should",
            "finalize",
            "coordinate",
        }

        task_like = 0
        for message in messages:
            text = str(message.get("text", "") or "").strip().lower()
            tokens = text.replace("'", "").split()
            if not tokens:
                continue

            short_message = len(tokens) <= 10
            has_directive = any(token in directive_terms for token in tokens)

            vector = message.get("emotion_vector", [])
            peak = 0.0
            if isinstance(vector, list) and vector:
                peak = max(float(v) for v in vector if isinstance(v, (int, float)))
            low_emotion_shape = peak < 0.23

            if short_message and has_directive and low_emotion_shape:
                task_like += 1

        return float(task_like) / float(len(messages))

    @staticmethod
    def _build_risk_flags(
        *,
        manipulation_level: str,
        sarcasm_level: str,
        emotional_stability: str,
        emotional_tone: str,
    ) -> List[str]:
        flags: List[str] = []
        if manipulation_level == "high":
            flags.append("high manipulation")
        if sarcasm_level == "high":
            flags.append("high sarcasm")
        if emotional_stability == "volatile":
            flags.append("emotional volatility")
        if emotional_tone == "negative":
            flags.append("persistent negative affect")
        return flags

    def _build_summary(
        self,
        *,
        speaker: str,
        tone: str,
        stability: str,
        manipulation: str,
        sarcasm: str,
        neutral: str,
        intensity: str,
    ) -> str:
        template = self._select_summary_template(speaker)
        speaker_display = speaker.capitalize() if speaker else speaker
        stability_phrase = f"{stability} patterns"
        if neutral == "high":
            stability_phrase = f"{stability_phrase} with predominantly neutral communication"
        elif neutral == "medium":
            stability_phrase = f"{stability_phrase} with occasional neutral responses"

        summary = template.format(
            speaker=speaker_display,
            tone=tone,
            stability=stability_phrase,
            manipulation=manipulation,
            sarcasm=sarcasm,
        )
        if intensity == "high emotional intensity":
            summary = f"{summary} The conversation also reflects {intensity}."
        elif intensity == "moderate emotional intensity":
            summary = f"{summary} The interaction reflects moderate emotional engagement."
        summary = " ".join(summary.split())
        if not summary.endswith("."):
            summary = f"{summary}."
        return summary

    def _select_summary_template(self, speaker: str) -> str:
        if not self.summary_templates:
            return "{speaker} shows {tone} emotional patterns with {stability}, alongside {manipulation} manipulation tendencies and {sarcasm} sarcasm usage."

        if self.randomize:
            seed_material = f"{speaker}|{self.seed if self.seed is not None else 'random'}"
            digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
            rng = random.Random(int(digest[:16], 16))
            return rng.choice(self.summary_templates)

        base = f"{speaker}|{self.seed if self.seed is not None else 'stable'}"
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(self.summary_templates)
        return self.summary_templates[index]


def analyze_insights(analyzer_output: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return InsightEngine().analyze(analyzer_output)

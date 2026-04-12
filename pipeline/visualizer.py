from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt

from models.emotion_model import PLUTCHIK_EMOTIONS

random.seed(42)


def _as_payload(
    analyzer_output: Mapping[str, Any],
    insights_output: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
    return {
        "analyzer": analyzer_output if isinstance(analyzer_output, Mapping) else {},
        "insights": insights_output if isinstance(insights_output, Mapping) else {},
    }


def _extract_messages(data: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    analyzer = data.get("analyzer", {}) if isinstance(data, Mapping) else {}
    if not isinstance(analyzer, Mapping):
        return []
    messages = analyzer.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [msg for msg in messages if isinstance(msg, Mapping)]


def _extract_users(data: Mapping[str, Any]) -> Mapping[str, Any]:
    analyzer = data.get("analyzer", {}) if isinstance(data, Mapping) else {}
    if not isinstance(analyzer, Mapping):
        return {}
    users = analyzer.get("users", {})
    if not isinstance(users, Mapping):
        return {}
    return users


def _group_messages_by_speaker(messages: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for message in messages:
        speaker = str(message.get("speaker", "unknown")).strip() or "unknown"
        grouped.setdefault(speaker, []).append(message)
    return grouped


def _message_index_and_intensity(message: Mapping[str, Any], fallback_index: int) -> Tuple[int, float]:
    index_raw = message.get("index", fallback_index)
    try:
        index = int(index_raw)
    except (TypeError, ValueError):
        index = int(fallback_index)

    intensity_raw = message.get("emotion_intensity")
    if isinstance(intensity_raw, (int, float)):
        return index, float(intensity_raw)

    vector_raw = message.get("emotion_vector", [])
    if isinstance(vector_raw, list):
        values = [float(v) for v in vector_raw if isinstance(v, (int, float))]
        if values:
            peak = max(values)
            mean_value = sum(values) / len(values)
            raw_intensity = peak - mean_value
            return index, float(raw_intensity / (peak + 1e-6))
        return index, 0.0

    return index, 0.0


def _message_index(message: Mapping[str, Any], fallback_index: int) -> int:
    index_raw = message.get("index", fallback_index)
    try:
        return int(index_raw)
    except (TypeError, ValueError):
        return int(fallback_index)


def _sorted_messages_by_index(messages: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    indexed = [(_message_index(msg, i), i, msg) for i, msg in enumerate(messages)]
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in indexed]


def plot_emotion_trend(data: Mapping[str, Any]) -> None:
    messages = _extract_messages(data)
    grouped = _group_messages_by_speaker(messages)

    plt.figure(figsize=(12, 5))
    for speaker, speaker_messages in grouped.items():
        sorted_messages = _sorted_messages_by_index(speaker_messages)
        xs: List[int] = []
        ys: List[float] = []
        for fallback_index, message in enumerate(sorted_messages):
            index, intensity = _message_index_and_intensity(message, fallback_index)
            xs.append(index)
            ys.append(float(intensity))
        plt.plot(xs, ys, marker="o", linewidth=2, label=speaker)

    plt.title("Emotional Intensity Over Time")
    plt.xlabel("Message Index")
    plt.ylabel("Emotion Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()


def plot_sarcasm_timeline(data: Mapping[str, Any]) -> None:
    messages = _extract_messages(data)
    grouped = _group_messages_by_speaker(messages)

    plt.figure(figsize=(12, 5))
    for user_index, (speaker, speaker_messages) in enumerate(grouped.items()):
        sorted_messages = _sorted_messages_by_index(speaker_messages)
        xs: List[int] = []
        ys: List[float] = []
        y_offset = user_index * 0.1
        for fallback_index, message in enumerate(sorted_messages):
            x_value = _message_index(message, fallback_index)
            y_value = 1.0 if bool(message.get("is_sarcastic", False)) else 0.0
            y_value = y_value + random.uniform(-0.05, 0.05) + y_offset
            xs.append(x_value)
            ys.append(float(y_value))
        plt.scatter(xs, ys, s=60, label=speaker)

    plt.title("Sarcasm Occurrence Timeline")
    plt.xlabel("Message Index")
    plt.ylabel("Sarcasm (0 = no, 1 = yes)")
    plt.yticks([0, 1])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()


def plot_manipulation_timeline(data: Mapping[str, Any]) -> None:
    messages = _extract_messages(data)
    grouped = _group_messages_by_speaker(messages)

    plt.figure(figsize=(12, 5))
    for user_index, (speaker, speaker_messages) in enumerate(grouped.items()):
        sorted_messages = _sorted_messages_by_index(speaker_messages)
        xs: List[int] = []
        ys: List[float] = []
        y_offset = user_index * 0.1
        for fallback_index, message in enumerate(sorted_messages):
            x_value = _message_index(message, fallback_index)
            is_manipulative = str(message.get("label", "genuine")).strip().lower() == "manipulative"
            y_value = 1.0 if is_manipulative else 0.0
            y_value = y_value + random.uniform(-0.05, 0.05) + y_offset
            xs.append(x_value)
            ys.append(float(y_value))
        plt.scatter(xs, ys, s=60, label=speaker)

    plt.title("Manipulation Detection Timeline")
    plt.xlabel("Message Index")
    plt.ylabel("Manipulation (0 = no, 1 = yes)")
    plt.yticks([0, 1])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()


def plot_user_comparison(data: Mapping[str, Any]) -> None:
    users = _extract_users(data)
    speakers = list(users.keys())
    if not speakers:
        plt.figure(figsize=(8, 4))
        plt.title("User Frequency Comparison")
        plt.xlabel("Users")
        plt.ylabel("Frequency")
        plt.tight_layout()
        return

    sarcasm_values = [float(users[speaker].get("sarcasm_frequency", 0.0)) for speaker in speakers]
    manipulation_values = [float(users[speaker].get("manipulation_frequency", 0.0)) for speaker in speakers]
    neutral_values = [float(users[speaker].get("neutral_frequency", 0.0)) for speaker in speakers]

    x_positions = list(range(len(speakers)))
    width = 0.25

    plt.figure(figsize=(12, 5))
    sarcasm_bars = plt.bar([x - width for x in x_positions], sarcasm_values, width=width, label="sarcasm_frequency")
    manipulation_bars = plt.bar(x_positions, manipulation_values, width=width, label="manipulation_frequency")
    neutral_bars = plt.bar([x + width for x in x_positions], neutral_values, width=width, label="neutral_frequency")

    plt.title("User Comparison: Sarcasm, Manipulation, Neutral")
    plt.xlabel("Users")
    plt.ylabel("Frequency")
    plt.xticks(x_positions, speakers, rotation=15)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bars in (sarcasm_bars, manipulation_bars, neutral_bars):
        for bar in bars:
            height = float(bar.get_height())
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.legend()
    plt.tight_layout()


def plot_plutchik_wheel(emotion_vector: List[float], title: str = "Emotion Profile") -> None:
    values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in emotion_vector]
    if len(values) < len(PLUTCHIK_EMOTIONS):
        values.extend([0.0] * (len(PLUTCHIK_EMOTIONS) - len(values)))
    elif len(values) > len(PLUTCHIK_EMOTIONS):
        values = values[: len(PLUTCHIK_EMOTIONS)]

    theta = [2.0 * math.pi * idx / len(PLUTCHIK_EMOTIONS) for idx in range(len(PLUTCHIK_EMOTIONS))]
    theta_closed = theta + [theta[0]]
    values_closed = values + [values[0]]

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(theta_closed, values_closed)
    ax.fill(theta_closed, values_closed, alpha=0.25)
    ax.set_xticks(theta)
    ax.set_xticklabels(list(PLUTCHIK_EMOTIONS))
    ax.tick_params(pad=10)
    ax.set_title(title)
    fig.tight_layout()


def build_visualization_data(
    analyzer_output: Mapping[str, Any],
    insights_output: Mapping[str, Any],
) -> Dict[str, Mapping[str, Any]]:
    return _as_payload(analyzer_output, insights_output)


def run_all_plots(data: Mapping[str, Any]) -> None:
    plot_emotion_trend(data)
    plot_sarcasm_timeline(data)
    plot_manipulation_timeline(data)
    plot_user_comparison(data)

    users = _extract_users(data)
    for speaker, user_data in users.items():
        if not isinstance(user_data, Mapping):
            continue
        emotion_avg = user_data.get("emotion_avg", [])
        if not isinstance(emotion_avg, list):
            emotion_avg = []
        plot_plutchik_wheel(emotion_avg, title=f"Plutchik Emotion Wheel - {speaker}")

    plt.show()

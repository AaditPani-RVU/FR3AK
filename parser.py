from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple


class Severity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass
class TimestampInfo:
    raw: str
    time_value: Optional[time] = None
    date_value: Optional[date] = None
    weekday: Optional[str] = None
    datetime_value: Optional[datetime] = None
    format_hint: Optional[str] = None


@dataclass
class ParseIssue:
    line_number: int
    severity: Severity
    code: str
    message: str
    raw_line: str


@dataclass
class SpeakerIdentity:
    original: str
    normalized: str
    speaker_id: str
    originals: List[str]


@dataclass
class ConversationRecord:
    line_number: int
    speaker: str
    normalized_speaker: str
    speaker_id: str
    timestamp: Optional[TimestampInfo]
    message: str
    raw_message: str
    cleaned_message: str
    has_timestamp_error: bool
    has_valid_timestamp: bool
    ordering_key: Tuple[int, Any]
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParseResult:
    records: List[ConversationRecord]
    issues: List[ParseIssue]
    speaker_mapping: Dict[str, SpeakerIdentity]
    speaker_id_mapping: Dict[str, SpeakerIdentity]

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self.records]


# Primary pattern for '{USER} at {TIME} : {message}'.
STRICT_LINE_RE = re.compile(
    r"^\s*(?P<speaker>.+?)\s+at\s+(?P<timestamp>.+?)\s*:\s*(?P<message>.*)\s*$"
)

# Lenient fallback to recover lines that are close to expected format.
LENIENT_LINE_RE = re.compile(
    r"^\s*(?P<speaker>.+?)\s+at\s*(?P<timestamp>.*?)\s*:\s*(?P<message>.*)\s*$"
)

CUSTOM_TIME_RE = re.compile(
    r"^(?P<clock>\d{1,2}:\d{2}:\d{2})(?:::(?P<weekday>[A-Za-z]{3,9}):(?P<day>\d{1,2}):(?P<month>\d{1,2}):(?P<year>\d{4}))?$"
)


def _normalize_speaker_name(raw_speaker: str) -> str:
    collapsed = " ".join(raw_speaker.strip().split())
    lowered = collapsed.lower()
    return re.sub(r"\s+", "_", lowered)


def _get_or_create_speaker_id(
    original_speaker: str,
    speaker_map: Dict[str, SpeakerIdentity],
    id_by_normalized: Dict[str, str],
) -> Tuple[str, str]:
    normalized = _normalize_speaker_name(original_speaker)

    if normalized in id_by_normalized:
        speaker_id = id_by_normalized[normalized]
    else:
        speaker_id = f"user_{len(id_by_normalized) + 1}"
        id_by_normalized[normalized] = speaker_id

    if normalized not in speaker_map:
        speaker_map[normalized] = SpeakerIdentity(
            original=original_speaker,
            normalized=normalized,
            speaker_id=speaker_id,
            originals=[original_speaker],
        )
    elif original_speaker not in speaker_map[normalized].originals:
        speaker_map[normalized].originals.append(original_speaker)

    return normalized, speaker_id


def _parse_timestamp(raw_timestamp: str) -> tuple[Optional[TimestampInfo], Optional[str]]:
    cleaned = raw_timestamp.strip()
    if not cleaned:
        return None, "Timestamp is missing."

    custom_match = CUSTOM_TIME_RE.match(cleaned)
    if custom_match:
        clock = custom_match.group("clock")
        weekday = custom_match.group("weekday")
        day_str = custom_match.group("day")
        month_str = custom_match.group("month")
        year_str = custom_match.group("year")

        try:
            parsed_time = datetime.strptime(clock, "%H:%M:%S").time()
        except ValueError:
            return None, f"Invalid time component in timestamp: '{cleaned}'."

        date_value: Optional[date] = None
        datetime_value: Optional[datetime] = datetime.combine(date(1900, 1, 1), parsed_time)

        if day_str and month_str and year_str:
            try:
                date_value = date(int(year_str), int(month_str), int(day_str))
                datetime_value = datetime.combine(date_value, parsed_time)
            except ValueError:
                return None, f"Invalid date component in timestamp: '{cleaned}'."

        return (
            TimestampInfo(
                raw=cleaned,
                time_value=parsed_time,
                date_value=date_value,
                weekday=weekday,
                datetime_value=datetime_value,
                format_hint="clock::weekday:dd:mm:yyyy" if weekday else "clock",
            ),
            None,
        )

    for fmt, hint in (
        ("%Y-%m-%d %H:%M:%S", "iso-datetime"),
        ("%d-%m-%Y %H:%M:%S", "dmy-datetime"),
        ("%H:%M:%S", "clock"),
    ):
        try:
            parsed = datetime.strptime(cleaned, fmt)
            if hint == "clock":
                return (
                    TimestampInfo(
                        raw=cleaned,
                        time_value=parsed.time(),
                        datetime_value=datetime.combine(date(1900, 1, 1), parsed.time()),
                        format_hint=hint,
                    ),
                    None,
                )
            return (
                TimestampInfo(
                    raw=cleaned,
                    time_value=parsed.time(),
                    date_value=parsed.date(),
                    datetime_value=parsed,
                    format_hint=hint,
                ),
                None,
            )
        except ValueError:
            continue

    return None, (
        "Timestamp could not be parsed. Supported formats include "
        "'HH:MM:SS::Day:DD:MM:YYYY', 'YYYY-MM-DD HH:MM:SS', and 'HH:MM:SS'."
    )


def parse_conversation(text: str, include_empty_lines: bool = False) -> ParseResult:
    records: List[ConversationRecord] = []
    issues: List[ParseIssue] = []
    speaker_mapping: Dict[str, SpeakerIdentity] = {}
    speaker_id_by_normalized: Dict[str, str] = {}
    speaker_id_mapping: Dict[str, SpeakerIdentity] = {}

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped and not include_empty_lines:
            continue

        strict_match = STRICT_LINE_RE.match(raw_line)
        match = strict_match if strict_match else LENIENT_LINE_RE.match(raw_line)

        if not match:
            records.append(
                ConversationRecord(
                    line_number=line_number,
                    speaker="",
                    normalized_speaker="",
                    speaker_id="",
                    timestamp=None,
                    message=stripped,
                    raw_message=stripped,
                    cleaned_message=stripped,
                    has_timestamp_error=True,
                    has_valid_timestamp=False,
                    ordering_key=(1, line_number),
                    is_valid=False,
                )
            )
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    severity=Severity.ERROR,
                    code="line_format_invalid",
                    message="Line does not match expected pattern '{USER} at {TIME} : {message}'.",
                    raw_line=raw_line,
                )
            )
            continue

        speaker = match.group("speaker").strip()
        normalized_speaker, speaker_id = _get_or_create_speaker_id(
            speaker,
            speaker_mapping,
            speaker_id_by_normalized,
        )
        raw_timestamp = match.group("timestamp").strip()
        raw_message = match.group("message")
        cleaned_message = raw_message.strip()

        timestamp_info, timestamp_error = _parse_timestamp(raw_timestamp)
        has_timestamp_error = timestamp_error is not None
        has_valid_timestamp = (not has_timestamp_error) and (timestamp_info is not None)

        line_has_error = False

        if not speaker:
            line_has_error = True
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    severity=Severity.ERROR,
                    code="speaker_missing",
                    message="Speaker is missing.",
                    raw_line=raw_line,
                )
            )

        if timestamp_error:
            line_has_error = True
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    severity=Severity.ERROR,
                    code="timestamp_invalid",
                    message=timestamp_error,
                    raw_line=raw_line,
                )
            )

        if not cleaned_message:
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    severity=Severity.WARNING,
                    code="message_empty",
                    message="Message content is empty.",
                    raw_line=raw_line,
                )
            )

        if timestamp_info and timestamp_info.datetime_value:
            ordering_key: Tuple[int, Any] = (0, timestamp_info.datetime_value)
        else:
            ordering_key = (1, line_number)

        message = cleaned_message

        if strict_match is None:
            issues.append(
                ParseIssue(
                    line_number=line_number,
                    severity=Severity.WARNING,
                    code="line_format_lenient_parse",
                    message="Line required lenient parsing due to inconsistent formatting.",
                    raw_line=raw_line,
                )
            )

        records.append(
            ConversationRecord(
                line_number=line_number,
                speaker=speaker,
                normalized_speaker=normalized_speaker,
                speaker_id=speaker_id,
                timestamp=timestamp_info,
                message=message,
                raw_message=raw_message,
                cleaned_message=cleaned_message,
                has_timestamp_error=has_timestamp_error,
                has_valid_timestamp=has_valid_timestamp,
                ordering_key=ordering_key,
                is_valid=not line_has_error,
            )
        )

    for identity in speaker_mapping.values():
        speaker_id_mapping[identity.speaker_id] = identity

    records.sort(key=lambda item: item.ordering_key)

    return ParseResult(
        records=records,
        issues=issues,
        speaker_mapping=speaker_mapping,
        speaker_id_mapping=speaker_id_mapping,
    )


def parse_conversation_lines(lines: List[str]) -> ParseResult:
    return parse_conversation("\n".join(lines))


def _demo_test() -> None:
    sample_lines = [
        "Alice at 12:30:01::Mon:12:03:2026 : I don't think that's what you meant",
        "Bob at 12:30:05::Mon:12:03:2026 : Oh yeah sure, totally",
        "Charlie at : Missing timestamp should be flagged",
        "Dana at 2026-03-12 12:32:00:No space before message separator but recoverable",
        "Completely malformed line with no separators",
        "Eve at 14:11:09 : ",
    ]

    result = parse_conversation_lines(sample_lines)

    print("=== Parsed Records ===")
    for record in result.records:
        print(record.to_dict())

    print("\n=== Issues ===")
    for issue in result.issues:
        print(asdict(issue))

    print("\n=== Speaker Mapping ===")
    for normalized, identity in result.speaker_mapping.items():
        print(normalized, asdict(identity))

    print("\n=== Speaker ID Mapping ===")
    for speaker_id, identity in result.speaker_id_mapping.items():
        print(speaker_id, asdict(identity))


if __name__ == "__main__":
    _demo_test()

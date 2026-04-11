from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Mapping

from pipeline.analyzer import analyze_conversation
from pipeline.insights import analyze_insights
from pipeline.visualizer import build_visualization_data, run_all_plots
from utils.parser import parse_conversation


LINE_WITH_HH_MM_RE = re.compile(r"^(?P<prefix>\s*.+?\s+at\s+)(?P<hhmm>\d{1,2}:\d{2})(?P<suffix>\s*:\s*.*)$")


def _record_to_pipeline_item(record: Any) -> Dict[str, Any]:
    timestamp_obj = getattr(record, "timestamp", None)
    timestamp_raw = getattr(timestamp_obj, "raw", None) if timestamp_obj is not None else None

    return {
        "speaker": str(getattr(record, "speaker", "") or getattr(record, "speaker_id", "")).strip(),
        "timestamp": timestamp_raw,
        "cleaned_message": str(getattr(record, "cleaned_message", "") or "").strip(),
    }


def _load_records_from_txt(txt_path: Path) -> List[Dict[str, Any]]:
    raw_text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
    normalized_lines: List[str] = []
    for raw_line in raw_text.splitlines():
        if not raw_line.strip():
            # Keep parser behavior predictable while ignoring empty lines.
            continue
        match = LINE_WITH_HH_MM_RE.match(raw_line)
        if match:
            normalized_lines.append(
                f"{match.group('prefix')}{match.group('hhmm')}:00{match.group('suffix')}"
            )
        else:
            normalized_lines.append(raw_line)

    text = "\n".join(normalized_lines)
    parse_result = parse_conversation(text)

    cleaned_records: List[Dict[str, Any]] = []
    for record in parse_result.records:
        pipeline_item = _record_to_pipeline_item(record)
        # Skip malformed lines and gracefully ignore empty messages.
        # Some parser timestamp formats can mark a record invalid even when
        # speaker/message extraction is successful, so we keep structurally
        # usable records for end-to-end pipeline validation.
        if not pipeline_item["speaker"]:
            continue
        if not pipeline_item["cleaned_message"]:
            continue

        cleaned_records.append(pipeline_item)

    return cleaned_records


def run_final_pipeline(conversation_file: Path) -> Dict[str, Any]:
    records = _load_records_from_txt(conversation_file)

    if not records:
        print("\n===== FINAL PIPELINE OUTPUT =====\n")
        print("No valid records were parsed from the input file.")
        return {
            "records": [],
            "analysis": {"users": {}, "messages": []},
            "insights": {},
            "visualization": {"analyzer": {"users": {}, "messages": []}, "insights": {}},
        }

    analysis_output = analyze_conversation(records)
    insights_output = analyze_insights(analysis_output)
    viz_data = build_visualization_data(analysis_output, insights_output)

    print("\n===== FINAL PIPELINE OUTPUT =====\n")
    for user, data in insights_output.items():
        if not isinstance(data, Mapping):
            continue
        print(f"{user}:")
        print("Summary:", data.get("summary", ""))
        print("Tone:", data.get("emotional_tone", ""))
        print("Sarcasm level:", data.get("sarcasm_level", ""))
        print("Manipulation level:", data.get("manipulation_level", ""))
        print("Neutral level:", data.get("neutral_level", ""))
        print("-" * 60)

    run_all_plots(viz_data)

    return {
        "records": records,
        "analysis": analysis_output,
        "insights": insights_output,
        "visualization": viz_data,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    conversation_file = repo_root / "tests" / "final_conversation.txt"
    run_final_pipeline(conversation_file)


if __name__ == "__main__":
    main()

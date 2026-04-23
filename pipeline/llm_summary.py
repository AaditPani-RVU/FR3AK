from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    """Return a cached OpenAI client, or None if no API key is configured."""
    global _client
    if _client is None:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            return None
        try:
            from openai import OpenAI
            _client = OpenAI(api_key=key)
        except ImportError:
            return None
    return _client


def generate_llm_summary(
    speaker: str,
    messages: List[Dict[str, Any]],
    insight_data: Dict[str, Any],
) -> Optional[str]:
    """Return a GPT-4o-mini summary for a single speaker, or None on failure/no key."""
    client = _get_client()
    if client is None:
        return None

    # Cap at 30 messages to keep token cost minimal
    msg_lines: List[str] = []
    for m in messages[:30]:
        flags: List[str] = []
        if m.get("is_sarcastic"):
            flags.append("sarcastic")
        if str(m.get("label", "genuine")).lower() == "manipulative":
            flags.append("manipulative")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        text = str(m.get("text", "") or "").strip()
        if text:
            msg_lines.append(f'- "{text}"{flag_str}')

    messages_block = "\n".join(msg_lines) if msg_lines else "(no messages)"

    user_prompt = (
        f"Participant: {speaker}\n\n"
        f"Messages:\n{messages_block}\n\n"
        f"Behavioral analytics:\n"
        f"- Dominant emotion: {insight_data.get('dominant_emotion', 'unknown')}\n"
        f"- Emotional tone: {insight_data.get('emotional_tone', 'mixed')}\n"
        f"- Emotional stability: {insight_data.get('emotional_stability', 'stable')}\n"
        f"- Sarcasm level: {insight_data.get('sarcasm_level', 'low')}\n"
        f"- Manipulation level: {insight_data.get('manipulation_level', 'low')}\n\n"
        f"Write a professional 4\u20135 sentence behavioral summary of this participant\u2019s "
        f"communication style. Be specific, reference patterns from the messages, and cover "
        f"their emotional tendencies, use of sarcasm or manipulation if present, and overall tone."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert conversation analyst. "
                        "Write concise, professional behavioral summaries based on "
                        "conversation data and computed analytics. "
                        "Be specific and reference actual patterns from the messages."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=0.4,
        )
        result = resp.choices[0].message.content
        return result.strip() if result else None
    except Exception:
        return None

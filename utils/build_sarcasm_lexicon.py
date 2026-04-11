from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger(__name__)


def _dedupe_sorted(items: List[str]) -> List[str]:
    cleaned = {item.strip().lower() for item in items if item and item.strip()}
    return sorted(cleaned)


def _normalize_phrase(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    cleaned = re.sub(r"[^a-z0-9\s'!?]", "", cleaned)
    return cleaned.strip()


def _valid_phrase(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    return 1 <= len(words) <= 10


def _coerce_text(record: object) -> Optional[str]:
    if isinstance(record, str):
        text = record.strip()
        return text if text else None
    return None


def _extract_ngrams(lines: Sequence[str], n_values: Sequence[int] = (2, 3)) -> List[str]:
    counter: Counter[str] = Counter()
    for line in lines:
        tokens = re.findall(r"[a-z0-9']+", line.lower())
        for n in n_values:
            if len(tokens) < n:
                continue
            for idx in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[idx : idx + n])
                if _valid_phrase(phrase):
                    counter[phrase] += 1

    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [phrase for phrase, count in ranked if count >= 2][:300]


def _extract_generalized_patterns(
    lines: Sequence[str],
    positive_words: Sequence[str],
    negative_context_words: Sequence[str],
    intensifiers: Sequence[str],
) -> List[str]:
    positive_set = set(positive_words)
    negative_set = set(negative_context_words)
    intensifier_set = set(intensifiers)

    counter: Counter[str] = Counter()
    contradiction_markers = {"but", "yet", "though", "however"}

    for line in lines:
        tokens = re.findall(r"[a-z0-9']+", line.lower())
        token_set = set(tokens)

        if token_set.intersection(positive_set) and token_set.intersection(negative_set):
            counter["positive negative context"] += 1

        if token_set.intersection(contradiction_markers) and token_set.intersection(positive_set):
            counter["positive but negative"] += 1

        if any(token in intensifier_set for token in tokens) and any(token in positive_set for token in tokens):
            counter["intensifier positive"] += 1

        # Exaggeration pattern from repeated punctuation and elongated words.
        if re.search(r"(!{2,}|\?{2,}|\.\.\.)", line):
            counter["exaggerated punctuation"] += 1
        if re.search(r"([a-z])\1{2,}", line.lower()):
            counter["elongated emphasis"] += 1

        # Extract reusable phrase chunks around key cues.
        for idx, token in enumerate(tokens):
            if token in {"yeah", "right", "wow", "sure", "great", "amazing", "genius"}:
                left = max(0, idx - 1)
                right = min(len(tokens), idx + 3)
                chunk = " ".join(tokens[left:right])
                if _valid_phrase(chunk):
                    counter[chunk] += 1

    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [phrase for phrase, count in ranked if count >= 2][:200]


def _load_local_sample_corpus(repo_root: Path) -> List[str]:
    candidates = [
        repo_root / "data" / "sarcasm_samples.txt",
        repo_root / "data" / "sarcasm_corpus.txt",
        repo_root / "data" / "sarcasm_examples.csv",
    ]
    lines: List[str] = []
    for path in candidates:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines.extend([line.strip() for line in raw if line.strip()])
    return lines


def _safe_load_dataset(*args, **kwargs):
    from datasets import load_dataset  # type: ignore
    return load_dataset(*args, **kwargs)


def _extract_lines_from_split(dataset_split, text_keys: Sequence[str], label_keys: Sequence[str]) -> List[str]:
    lines: List[str] = []
    for row in dataset_split:
        text_value = None
        for key in text_keys:
            maybe = row.get(key)
            if isinstance(maybe, str) and maybe.strip():
                text_value = maybe.strip()
                break
        if text_value is None:
            continue

        label_value = None
        for key in label_keys:
            if key in row:
                label_value = row.get(key)
                break

        if isinstance(label_value, int) and label_value != 1:
            continue
        if isinstance(label_value, str):
            lowered = label_value.strip().lower()
            if lowered and lowered not in {"1", "sarcastic", "sarcasm", "irony", "yes", "true"}:
                continue

        lines.append(text_value)
    return lines


def _try_load_tweet_eval_irony(max_rows: int) -> List[str]:
    try:
        ds = _safe_load_dataset("tweet_eval", "irony", split="train")
    except Exception:
        return []
    return _extract_lines_from_split(ds, text_keys=("text",), label_keys=("label",))[:max_rows]


def _try_load_sarcasm_headlines(max_rows: int) -> List[str]:
    candidates = [
        ("raquiba/Sarcasm_News_Headline", None),
        ("MidhunKanadan/sarcasm-detection", None),
    ]

    lines: List[str] = []
    for dataset_name, dataset_config in candidates:
        try:
            ds = _safe_load_dataset(dataset_name, dataset_config, split="train")
        except Exception:
            continue

        lines = _extract_lines_from_split(
            ds,
            text_keys=("headline", "text", "sentence", "title"),
            label_keys=("label", "is_sarcastic", "sarcasm", "class"),
        )
        if lines:
            break
    return lines[:max_rows]


def _try_load_mustard(max_rows: int) -> List[str]:
    candidates = [
        "tasksource/mustard",
        "jhamel/mustard",
        "Yaxin/MUSTARD",
    ]

    for dataset_name in candidates:
        try:
            ds = _safe_load_dataset(dataset_name, split="train")
        except Exception:
            continue

        lines = _extract_lines_from_split(
            ds,
            text_keys=("utterance", "text", "sentence", "response"),
            label_keys=("label", "sarcasm", "sarcastic", "is_sarcastic"),
        )
        if lines:
            return lines[:max_rows]
    return []


def _try_load_reddit_sarcasm(max_rows: int) -> List[str]:
    candidates = [
        "reddit_sarcasm",
        "mteb/reddit-sarcasm",
    ]
    for dataset_name in candidates:
        try:
            ds = _safe_load_dataset(dataset_name, split="train")
        except Exception:
            continue

        lines = _extract_lines_from_split(
            ds,
            text_keys=("text", "comment", "body"),
            label_keys=("label", "is_sarcastic", "sarcasm"),
        )
        if lines:
            return lines[:max_rows]
    return []


def _try_load_dataset_texts(max_rows_per_dataset: int = 2000) -> Tuple[Dict[str, List[str]], List[str]]:
    try:
        import datasets  # noqa: F401 # type: ignore
    except Exception:
        return {}, []

    outputs: Dict[str, List[str]] = {
        "tweet_eval_irony": _try_load_tweet_eval_irony(max_rows_per_dataset),
        "sarcasm_headlines": _try_load_sarcasm_headlines(max_rows_per_dataset),
        "mustard": _try_load_mustard(max_rows_per_dataset),
        "reddit_optional": _try_load_reddit_sarcasm(max_rows_per_dataset),
    }

    used = [name for name, lines in outputs.items() if lines]
    return outputs, used

    dataset_candidates = [
        ("tweet_eval", "irony", "text", "label"),
        ("ought/raft", "sarcasm", "Tweet", "Label"),
    ]

    collected: List[str] = []
    for name, config, text_col, label_col in dataset_candidates:
        try:
            dataset = load_dataset(name, config, split="train")
        except Exception:
            continue

        for row in dataset:
            if len(collected) >= max_rows:
                break
            text = row.get(text_col)
            label = row.get(label_col)
            if not isinstance(text, str):
                continue
            if isinstance(label, int) and label != 1:
                continue
            collected.append(text.strip())
        if len(collected) >= max_rows:
            break

    return collected


def _try_fetch_web_examples() -> List[str]:
    # Optional web enrichment done only in builder. Safe-fail to keep script independent.
    try:
        import requests  # type: ignore
    except Exception:
        return []

    urls = ["https://en.wikipedia.org/wiki/Sarcasm"]
    lines: List[str] = []
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                continue
            text = re.sub(r"\s+", " ", response.text)
            snippets = re.findall(r"[A-Za-z][A-Za-z\s,'!?-]{20,120}", text)
            lines.extend(snippets[:200])
        except Exception:
            continue
    return lines


def _collect_external_lines(repo_root: Path) -> tuple[Dict[str, List[str]], List[str]]:
    used_sources: List[str] = []
    data_by_source: Dict[str, List[str]] = {}

    local_lines = _load_local_sample_corpus(repo_root)
    if local_lines:
        used_sources.append("local_corpus")
        data_by_source["local_corpus"] = local_lines

    dataset_lines, dataset_sources = _try_load_dataset_texts()
    for name in dataset_sources:
        used_sources.append(name)
        data_by_source[name] = dataset_lines.get(name, [])

    web_lines = _try_fetch_web_examples()
    if web_lines:
        used_sources.append("web_text_pages")
        data_by_source["web_text_pages"] = web_lines

    return data_by_source, used_sources


def build_lexicon() -> Dict[str, List[str]]:
    base_phrases = [
        "yeah right",
        "sure thing",
        "as if",
        "what a surprise",
        "just perfect",
        "great job",
        "nice going",
        "love that for me",
        "how convenient",
        "big surprise",
        "totally fine",
        "good for you",
        "fantastic idea",
    ]

    positive_words = [
        "great",
        "amazing",
        "awesome",
        "brilliant",
        "perfect",
        "fantastic",
        "wonderful",
        "excellent",
        "nice",
        "genius",
        "lovely",
        "incredible",
    ]

    negative_context_words = [
        "fail",
        "failed",
        "broken",
        "late",
        "wrong",
        "mistake",
        "problem",
        "issue",
        "terrible",
        "awful",
        "bad",
        "hate",
        "annoying",
        "disaster",
        "stupid",
        "ridiculous",
    ]

    intensifiers = [
        "totally",
        "absolutely",
        "literally",
        "really",
        "so",
        "very",
        "super",
        "extremely",
        "completely",
    ]

    emoji = [
        "🙄",
        "😒",
        "😑",
        "😏",
        "🤦",
        "🤦‍♂️",
        "🤦‍♀️",
        "😂",
        "🤣",
        "😉",
    ]

    repo_root = Path(__file__).resolve().parents[1]

    # Programmatic expansion from seed templates.
    expanded_phrases: List[str] = list(base_phrases)
    for word in positive_words:
        expanded_phrases.extend(
            [
                f"oh {word}",
                f"so {word}",
                f"totally {word}",
                f"yeah, {word}",
            ]
        )

    for word in negative_context_words:
        expanded_phrases.extend(
            [
                f"love when this {word}",
                f"great, another {word}",
                f"just what i needed, {word}",
            ]
        )

    external_data, used_sources = _collect_external_lines(repo_root)

    dataset_phrase_counts: Dict[str, int] = {}
    extracted_by_source: List[str] = []
    for source_name, lines in external_data.items():
        ngrams = _extract_ngrams(lines) if lines else []
        patterns = _extract_generalized_patterns(
            lines,
            positive_words=positive_words,
            negative_context_words=negative_context_words,
            intensifiers=intensifiers,
        )
        combined = [*ngrams, *patterns]
        dataset_phrase_counts[source_name] = len(combined)
        extracted_by_source.extend(combined)

    merged_phrases: List[str] = []
    for phrase in [*expanded_phrases, *extracted_by_source]:
        normalized = _normalize_phrase(phrase)
        if _valid_phrase(normalized):
            merged_phrases.append(normalized)

    if not used_sources:
        used_sources = ["seed_fallback"]

    lexicon = {
        "phrases": _dedupe_sorted(merged_phrases),
        "positive_words": _dedupe_sorted(positive_words),
        "negative_context_words": _dedupe_sorted(negative_context_words),
        "intensifiers": _dedupe_sorted(intensifiers),
        "emoji": _dedupe_sorted(emoji),
        "meta_sources": _dedupe_sorted(used_sources),
        "meta_dataset_phrase_counts": dataset_phrase_counts,
    }
    return lexicon


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "data" / "sarcasm_lexicon.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lexicon = build_lexicon()
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(lexicon, handle, indent=2, ensure_ascii=False)

    print(f"Saved sarcasm lexicon to: {out_path}")
    print(f"sources={lexicon.get('meta_sources', [])}")
    print(f"dataset_phrase_counts={lexicon.get('meta_dataset_phrase_counts', {})}")
    print(f"phrases={len(lexicon['phrases'])}")
    print(f"positive_words={len(lexicon['positive_words'])}")
    print(f"negative_context_words={len(lexicon['negative_context_words'])}")
    print(f"intensifiers={len(lexicon['intensifiers'])}")
    print(f"emoji={len(lexicon['emoji'])}")


if __name__ == "__main__":
    main()

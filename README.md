# Functional Reasoning & Emotional Augmentation Kernel

Emotion • Behavior • Insight

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Server-009688?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Datasets-20BEFF?logo=kaggle&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Inference-EE4C2C?logo=pytorch&logoColor=white)

## Project Overview

A conversation intelligence system that analyses emotional and behavioral dynamics across multi-user chat transcripts. It combines sequential message parsing, Plutchik-based emotion modeling, behavior classification, insight generation, and LLM-powered summaries into a unified analytics workflow served through a FastAPI backend and a single-page web frontend.

Core capabilities:

- 8-dimensional Plutchik emotion modeling per message and per participant.
- Sarcasm detection from lexical, contextual, and rule-based signals.
- Manipulation detection through a staged behavioral classifier.
- Per-participant behavioral profiling with risk flags and stability analysis.
- LLM-generated summaries (GPT-4o-mini) grounded in model outputs, with template fallback.
- Interactive web dashboard: Plutchik wheel, emotion trend, sarcasm and manipulation timelines, full conversation viewer.

---

## Project Structure

```text
├── server.py                        # FastAPI server — primary entry point
├── app.py                           # Streamlit UI — legacy alternative
├── static/
│   └── index.html                   # Single-page web frontend (served by server.py)
├── models/
│   ├── emotion_model.py             # EmotionModel — Plutchik 8D inference
│   ├── behavior_model.py            # BehaviorModel — sarcasm & manipulation detection
│   └── custom_emotion_model.py      # PlutchikEmotionModelV2 architecture
├── pipeline/
│   ├── analyzer.py                  # ConversationAnalyzer — per-message & user aggregation
│   ├── insights.py                  # InsightEngine — profiling & behavioral interpretation
│   ├── visualizer.py                # Visualization payload builder
│   └── llm_summary.py               # GPT-4o-mini summary generation with template fallback
├── utils/
│   ├── parser.py                    # Conversation file parser (.txt / .json)
│   └── build_sarcasm_lexicon.py     # Sarcasm lexicon generator (utility, run once)
├── tests/
│   ├── analyzer_test.py
│   ├── emotion_test.py
│   ├── behavior_test.py
│   ├── insights_test.py
│   ├── visualizer_test.py
│   ├── final_pipeline_test.py
│   └── final_conversation.txt       # Sample conversation for testing
├── training/
│   ├── nb1_label_part1.ipynb
│   ├── nb2_label_part2.ipynb
│   ├── nb3_train_model.ipynb
│   ├── nb4_eval_export.ipynb
│   ├── nb5_generate_vlits.ipynb
│   ├── nb6_feature_extract.ipynb
│   └── nb7_train_model2.ipynb
├── data/
│   ├── plutchik-model-v2/           # Emotion model artifacts (auto-downloaded via Kaggle)
│   ├── incongruity-classifier/      # Behavior model artifacts (auto-downloaded via Kaggle)
│   └── sarcasm_lexicon.json         # Generated sarcasm phrase lexicon
├── requirements.txt
├── .env                             # Environment variables (not tracked)
└── .gitignore
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/AaditPani-RVU/FR3AK.git
cd FR3AK
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
# Required — Kaggle credentials for auto-downloading model artifacts
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key

# Required — dataset references used to resolve model artifacts
FR3AK_EMOTION_DATASET_REF=bobhendriks/plutchik-model-v2
FR3AK_BEHAVIOR_DATASET_REF=bobhendriks/incongruity-classfier

# Optional — enables LLM-powered summaries via GPT-4o-mini
OPENAI_API_KEY=
```

**Kaggle API token:** Go to kaggle.com → Account Settings → Create New API Token → extract `username` and `key` from `kaggle.json`.

If `OPENAI_API_KEY` is left blank, summaries fall back to deterministic templates automatically — the app runs fully without it.

---

## Running the App

### FastAPI Web Server (primary)

```bash
python server.py
```

Open `http://localhost:8000`. The server:
- Accepts `.txt` or `.json` conversation file uploads via `POST /api/analyze`
- Returns full analysis as JSON (records, analysis, insights, viz data)
- Serves the SPA frontend at `GET /`

### Streamlit Dashboard (legacy alternative)

```bash
streamlit run app.py
```

The Streamlit version provides the same pipeline with an interactive matplotlib-based dashboard. It does not use the LLM summary integration.

---

## Pipeline

### Stage 0 — Parsing

`utils/parser.py` handles `.txt` files with `speaker @ timestamp : message` format and `utils/parser.py` + JSON normalization handles `.json` files supporting multiple schema shapes (`records`, `messages`, or raw arrays). Output: flat list of `{speaker, cleaned_message, timestamp}` dicts.

### Stage 1 — ConversationAnalyzer (`pipeline/analyzer.py`)

Processes every message sequentially through the emotion model and behavior model. Computes per-message:

- `emotion_vector` — 8-element Plutchik float vector
- `emotion_intensity` — dominance score: `(max − mean) / (max + ε)`
- `label` — `"genuine"` or `"manipulative"`
- `is_sarcastic` — boolean
- `is_neutral` — boolean

Maintains running `UserState` per speaker and finalizes into per-user stats: `emotion_avg`, `emotion_drift` (last minus first emotion vector), `sarcasm_frequency`, `manipulation_frequency`, `neutral_frequency`, `avg_emotion_intensity`.

### Stage 2 — InsightEngine (`pipeline/insights.py`)

Converts raw frequencies into interpretable labels per participant:

| Field | Method |
|---|---|
| `dominant_emotion` | argmax of `emotion_avg` |
| `emotional_stability` | L2 norm of `emotion_drift` → stable / shifting / volatile |
| `emotional_tone` | derived from dominant emotion, overridden by manipulation/sarcasm level |
| `sarcasm_level` | frequency threshold → low / medium / high |
| `manipulation_level` | frequency threshold + 16 consistency checks (stabilizer detection, neutral protection, task-oriented dampening, frustration vs coercion) |
| `risk_flags` | active if any of: high manipulation, high sarcasm, emotional volatility, persistent negative affect |
| `summary` | GPT-4o-mini if key set, otherwise deterministic template |

### Stage 3 — LLM Summary (`pipeline/llm_summary.py`)

Called from `InsightEngine` per participant after all labels are finalized. Passes GPT-4o-mini:

- The participant's messages with inline `[sarcastic]` / `[manipulative]` flags from the behavior model
- Final computed fields: dominant emotion, tone, stability, sarcasm level, manipulation level

GPT narrates what the models detected — it does not perform any detection itself. Returns `None` on any failure; `InsightEngine` falls back to templates silently.

### Stage 4 — Visualization Payload (`pipeline/visualizer.py`)

`build_visualization_data()` packages analyzer + insights output into a single JSON-serializable dict. The FastAPI server returns this directly to the browser, where all charts (Chart.js) and the Plutchik wheel (SVG) are rendered client-side.

---

## Web Frontend (`static/index.html`)

Single-page app with three views:

1. **Upload** — drag-and-drop or file picker for `.txt`/`.json`, run pipeline button with animated progress
2. **Participants** — color-coded cards per speaker showing tone, sarcasm, manipulation tags and summary preview; "View Conversation" button opens full conversation log with per-message badges
3. **Analysis** — per-participant deep view:
   - Profile card with GPT summary, dominant emotion, stability, risk flags
   - Emotion intensity trend (Chart.js line)
   - Sarcasm detection timeline (Chart.js scatter)
   - Manipulation detection timeline (Chart.js scatter)
   - Plutchik emotion wheel (custom SVG with traditional emotion colors)

---

## Scoring Notes

Emotion intensity is computed from dominance, not probability mass:

```python
emotion_array = np.array(emotion_vector)
raw_intensity = np.max(emotion_array) - np.mean(emotion_array)
emotion_intensity = float(raw_intensity / (np.max(emotion_array) + 1e-6))
```

- High dominance of one emotion → intensity near 0.7–1.0
- Flat emotion distribution → intensity near 0.1–0.3

Used in `pipeline/analyzer.py` (per-message), `pipeline/insights.py` (user-level classification), and the emotion trend chart in the frontend.

---

## Testing

```bash
python tests/final_pipeline_test.py   # end-to-end integration
python tests/emotion_test.py          # emotion model loading and output shape
python tests/analyzer_test.py         # conversation analyzer aggregation
python tests/insights_test.py         # insight engine profiling consistency
python tests/visualizer_test.py       # visualization payload generation
```

---

## Model Training

The `training/` directory contains 7 notebooks covering the full model development lifecycle:

- Data labeling (nb1, nb2)
- Emotion model training and evaluation (nb3, nb4)
- VLITS dataset generation and feature extraction (nb5, nb6)
- Behavior/incongruity model training (nb7)

Models are pre-trained; notebooks are provided for reference and reproducibility.

---

## Sarcasm Lexicon

If you need to regenerate the sarcasm phrase lexicon:

```bash
python utils/build_sarcasm_lexicon.py
```

---

## Model & Data Sources

| Model | Kaggle Dataset |
|---|---|
| Emotion model (Plutchik 8D) | [plutchik-model-v2](https://www.kaggle.com/datasets/bobhendriks/plutchik-model-v2) |
| Behavior model (sarcasm/manipulation) | [incongruity-classfier](https://www.kaggle.com/datasets/bobhendriks/incongruity-classfier) |

Model artifacts are auto-downloaded at first run if not present locally in `data/`.

Training data sources include GoEmotions (Google), Kaggle emotion/sarcasm datasets, Twitter/Reddit sarcasm corpora, and MUSTARD.

---

## Libraries

| Library | Role |
|---|---|
| FastAPI + Uvicorn | Web server and API |
| PyTorch + Transformers | Emotion and behavior model inference |
| OpenAI SDK | GPT-4o-mini LLM summaries |
| Chart.js | Frontend timeline charts |
| NumPy | Numeric processing |
| Matplotlib | Plot functions (Streamlit version / tests) |
| Streamlit | Legacy interactive dashboard |
| python-dotenv | Environment variable loading |

---

## References

- [FastAPI](https://fastapi.tiangolo.com)
- [Kaggle](https://www.kaggle.com)
- [Plutchik Emotion Wheel](https://en.wikipedia.org/wiki/Robert_Plutchik)
- [OpenAI API](https://platform.openai.com/docs)
- [Streamlit](https://streamlit.io)

## Acknowledgements

Open-source AI and Python community, the Kaggle ecosystem for accessible dataset hosting, and researchers working on emotion-aware AI, conversational NLP, and interpretable behavioral analytics.

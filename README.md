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

### 1. Clone the Repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/AaditPani-RVU/FR3AK.git
cd FR3AK
```

This downloads the complete codebase, including pre-trained models, training notebooks, and the web frontend.

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

This installs FastAPI for the web server, PyTorch and Transformers for model inference, OpenAI SDK for LLM summaries, and other dependencies like NumPy, pandas, and scikit-learn.

### 3. Obtain Kaggle API Credentials

The system auto-downloads pre-trained model artifacts from Kaggle datasets. You need a Kaggle account and API key:

- Go to [kaggle.com](https://kaggle.com) and create an account if you don't have one.
- Navigate to Account Settings → Create New API Token.
- Download the `kaggle.json` file containing your `username` and `key`.

**Note:** Kaggle's API uses legacy authentication; ensure your account has API access enabled.

### 4. Configure Environment Variables

Create a `.env` file in the project root with your credentials:

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

- Replace `your_username` and `your_key` with values from your `kaggle.json`.
- The dataset references are fixed and should not be changed.
- If `OPENAI_API_KEY` is omitted, summaries use deterministic templates instead of GPT-4o-mini.

The app will automatically download model artifacts to `data/` on first run if not present.

### 5. Build Sarcasm Lexicon

Generate the sarcasm phrase lexicon used by the behavior model:

```bash
python utils/build_sarcasm_lexicon.py
```

This creates `data/sarcasm_lexicon.json` with curated sarcasm patterns. Run this once after setup; it takes a few minutes.

---

## Running the App

### Primary: FastAPI Web Server

Start the FastAPI server, which serves the single-page web frontend and handles analysis requests:

```bash
python server.py
```

- The server runs on `http://localhost:8000` by default.
- Upload `.txt` or `.json` conversation files via the web interface.
- The pipeline processes files sequentially: parsing → emotion modeling → behavior classification → insights → LLM summaries → visualization.
- Model artifacts are auto-downloaded on first run if missing.
- Use this for production deployment or full functionality.

### Fallback: Streamlit Dashboard

If the FastAPI server encounters issues (e.g., port conflicts or missing dependencies), use the Streamlit alternative:

```bash
streamlit run app.py
```

- Provides an interactive matplotlib-based dashboard.
- Same pipeline as FastAPI but without LLM summary integration or the custom web frontend.
- Suitable for quick testing or environments where FastAPI is not feasible.

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

| Field                 | Method                                                                                                                                   |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `dominant_emotion`    | argmax of `emotion_avg`                                                                                                                  |
| `emotional_stability` | L2 norm of `emotion_drift` → stable / shifting / volatile                                                                                |
| `emotional_tone`      | derived from dominant emotion, overridden by manipulation/sarcasm level                                                                  |
| `sarcasm_level`       | frequency threshold → low / medium / high                                                                                                |
| `manipulation_level`  | frequency threshold + 16 consistency checks (stabilizer detection, neutral protection, task-oriented dampening, frustration vs coercion) |
| `risk_flags`          | active if any of: high manipulation, high sarcasm, emotional volatility, persistent negative affect                                      |
| `summary`             | GPT-4o-mini if key set, otherwise deterministic template                                                                                 |

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

The `training/` directory contains 7 Jupyter notebooks that implement the complete model development pipeline. These notebooks were run collaboratively across multiple Kaggle sessions to train the emotion and behavior models from scratch. The process involves labeling large emotion datasets with an LLM, training a custom emotion model, generating emotion vectors for behavioral datasets, extracting anomaly and context features, and finally training a staged behavioral classifier.

Models are pre-trained and included in the repository; the notebooks are provided for transparency, reproducibility, and potential model updates.

### Step-by-Step Process Overview

1. **Data Labeling (nb1, nb2)**: Use Qwen2.5-3B-Instruct LLM to generate soft Plutchik emotion labels for GoEmotions dataset and incorporate SemEval continuous emotion intensities.
2. **Emotion Model Training (nb3)**: Train a custom DeBERTa-v3-base model with 8 separate EmotionAttentionBlocks for Plutchik 8D emotion prediction.
3. **Emotion Model Evaluation (nb4)**: Evaluate the trained emotion model on held-out data, perform qualitative analysis, and export model artifacts.
4. **VLITS Generation (nb5)**: Use the trained emotion model to generate Plutchik emotion vectors (V_lit) for MentalManip manipulation dataset and MUSTARD sarcasm dataset.
5. **Feature Extraction (nb6)**: Extract 46-dimensional features from V_lit vectors, including anomaly detection (spike, tension, suppression, incoherence), conversation context tracking (EMA deltas, drift, actor specificity), and lexical signals (negation, pressure, hedging, punctuation).
6. **Behavior Model Training (nb7)**: Train a two-stage IncongruityClassifier: XGBoost for Genuine vs Rest, MLP for Sarcasm vs Manipulation, using the extracted features.

### Detailed Notebook Breakdown

#### nb1_label_part1.ipynb — Plutchik Labeling: GoEmotions Part 1

- **Purpose**: Labels the first half (rows 0–21999) of the GoEmotions dataset with soft 8D Plutchik emotion scores using Qwen2.5-3B-Instruct LLM.
- **Setup**: Requires T4 GPU accelerator in Kaggle. Uses 4-bit quantized model for efficiency (~2GB VRAM).
- **Process**:
  - Load GoEmotions train split (first 22K sentences).
  - Define prompt template for JSON output of 9 keys: 8 emotions + confidence.
  - Implement robust parsing for LLM responses (handles single quotes, case insensitivity, partial JSON).
  - Batched inference with greedy decoding for consistent JSON output.
  - Checkpointing every 500 rows to resume on timeouts.
  - Quality filtering: keep only parses with confidence >= 0.2.
- **Output**: `plutchik_labeled_p1.csv` (shared as Kaggle dataset `plutchik-labels-p1`).
- **Runtime**: ~2 hours; Person 1 runs this.

#### nb2_label_part2.ipynb — Plutchik Labeling: GoEmotions Part 2 + SemEval EI-reg

- **Purpose**: Labels the second half of GoEmotions and incorporates SemEval EI-reg continuous emotion intensities for joy/sadness/fear/anger.
- **Setup**: Requires T4 GPU; same LLM and quantization as nb1.
- **Process**:
  - Label GoEmotions rows 22000+ with LLM (same prompt/parser as nb1).
  - Load SemEval EI-reg datasets (joy, sadness, fear, anger) — these provide continuous 0-1 intensities directly.
  - Map SemEval emotions to Plutchik axes (direct 1:1 mapping); set other axes to 0.
  - Combine LLM-labeled GoEmotions and SemEval data.
- **Outputs**: `plutchik_labeled_p2.csv` (LLM labels) and `semeval_continuous.csv` (SemEval intensities).
- **Runtime**: ~2 hours; Person 2 runs this.

#### nb3_train_model.ipynb — Plutchik Model v2: Training

- **Purpose**: Trains the PlutchikEmotionModelV2 architecture on the combined labeled data from nb1 and nb2.
- **Architecture**: DeBERTa-v3-base encoder + 8 separate EmotionAttentionBlocks (one per Plutchik axis) + per-emotion regression heads with temperature scaling.
- **Setup**: Requires T4 GPU; loads `plutchik-labels-p1` and `plutchik-labels-p2` as Kaggle inputs.
- **Process**:
  - Merge and deduplicate labeled datasets.
  - Curriculum learning: easy (easiest 33% samples, 4 epochs) → medium (easiest 66%, 4 epochs) → all (12 epochs total).
  - Layer-wise LR decay: encoder layers get lower LR than new attention blocks/heads.
  - Loss: weighted smoothed MSE, cosine similarity, rank loss (pairwise ordering), auxiliary classification.
  - Early stopping on validation loss; save best by Spearman correlation.
- **Output**: `plutchik_model_v2/` directory with `best_model.pt`, tokenizer, config, training history (shared as Kaggle dataset `plutchik-model-v2`).
- **Runtime**: ~4 hours; Person 3 runs this after nb1/nb2 complete.

#### nb4_eval_export.ipynb — Evaluation, Analysis & Final Export

- **Purpose**: Comprehensive evaluation of the trained Plutchik model, qualitative analysis, and artifact export.
- **Setup**: Loads `plutchik-model-v2` as Kaggle input; GPU optional.
- **Process**:
  - Load model and rebuild architecture.
  - Qualitative tests: 20+ sentences with predicted emotion bars.
  - Radar plots: multi-emotion examples visualized as polar charts.
  - Full test evaluation on GoEmotions test split: Spearman/ Pearson per emotion, vector cosine, dominant accuracy.
  - Score distribution analysis: check for compression near 0.
  - Training history plots: Spearman, cosine, dominant accuracy over epochs.
  - Error analysis: 10 worst predictions by cosine similarity.
- **Output**: Model artifacts confirmed; plots saved for reference.
- **Runtime**: ~30 minutes; Person 4 runs this.

#### nb5_generate_vlits.ipynb — Generate V_lit Vectors

- **Purpose**: Uses the trained Plutchik model to generate emotion vectors (V_lit) for MentalManip manipulation dataset and MUSTARD sarcasm dataset.
- **Setup**: Loads `plutchik-model-v2` as input; requires T4 GPU for inference on ~50K utterances.
- **Process**:
  - Load and parse MentalManip dialogues into turn-level rows (Person1: ..., Person2: ...).
  - Load sarcasm datasets: tweet_eval irony, Sarcasm_News_Headline, MUSTARD.
  - Unified labeling: MentalManip rows get manipulation labels/techniques; sarcasm rows get sarcasm label.
  - Batched inference: generate 8D Plutchik vectors for all texts.
- **Output**: `mentalmanip_vlits.csv` with dialogue_id, turn_index, speaker, text, is_manipulative, technique, vulnerability, source_dataset, and 8 emotion scores (shared as Kaggle dataset `mentalmanip-vlits`).
- **Runtime**: ~1 hour; Person C runs this.

#### nb6_feature_extract.ipynb — Feature Extraction: Anomaly + Context -> 42-dim Vectors

- **Purpose**: Transforms V_lit emotion vectors into 46-dimensional feature vectors for behavioral classification.
- **Setup**: Loads `mentalmanip-vlits` as input; CPU-only.
- **Process**:
  - Anomaly features: spike (dominance), tension (opposite pair products), suppression (flat emotions), incoherence (entropy).
  - Context tracking: EMA conversation context per dialogue, per-speaker context, deltas (dv1/dv2/dv3), drift, actor specificity.
  - Lexical features: negation density, pressure words density, hedging density, punctuation presence.
  - 4-class labeling: Genuine (0), Sarcasm (1), Manipulation (2), Ambiguous (3, remapped to 2).
- **Output**: `mentalmanip_features.csv` with 46 features + label (shared as Kaggle dataset `mentalmanip-features`).
- **Runtime**: ~10 minutes; Person B runs this.

#### nb7_train_model2.ipynb — Train Model 2: IncongruityClassifier

- **Purpose**: Trains the two-stage behavioral classifier on the extracted features.
- **Architecture**: Stage 1: XGBoost for Genuine vs Rest (binary); Stage 2: MLP (46→256→128→64→32→2) for Sarcasm vs Manipulation.
- **Setup**: Loads `mentalmanip-features` as input; GPU optional.
- **Process**:
  - Group-aware train/val/test split by dialogue_id to prevent leakage.
  - Class balancing with weights and oversampling.
  - Stage 1: XGBoost with early stopping; tune threshold for Genuine precision >= 0.40 while maximizing macro-F1.
  - Stage 2: MLP with Focal Loss, BatchNorm, curriculum epochs.
  - Final evaluation: confusion matrix, macro-F1 on test set.
- **Outputs**: `stage1_genuine_vs_rest.json` (XGBoost), `stage2_sarcasm_vs_manip.pt` (MLP), `model2_config.json` (shared as Kaggle dataset `incongruity-classifier`).
- **Runtime**: ~1 hour; Person D runs this.

---

## Model & Data Sources

| Model                                 | Kaggle Dataset                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------ |
| Emotion model (Plutchik 8D)           | [plutchik-model-v2](https://www.kaggle.com/datasets/bobhendriks/plutchik-model-v2)         |
| Behavior model (sarcasm/manipulation) | [incongruity-classfier](https://www.kaggle.com/datasets/bobhendriks/incongruity-classfier) |

Model artifacts are auto-downloaded at first run if not present locally in `data/`.

Training data sources include GoEmotions (Google), Kaggle emotion/sarcasm datasets, Twitter/Reddit sarcasm corpora, and MUSTARD.

---

## Libraries

| Library                | Role                                       |
| ---------------------- | ------------------------------------------ |
| FastAPI + Uvicorn      | Web server and API                         |
| PyTorch + Transformers | Emotion and behavior model inference       |
| OpenAI SDK             | GPT-4o-mini LLM summaries                  |
| Chart.js               | Frontend timeline charts                   |
| NumPy                  | Numeric processing                         |
| Matplotlib             | Plot functions (Streamlit version / tests) |
| Streamlit              | Legacy interactive dashboard               |
| python-dotenv          | Environment variable loading               |

---

## References

- [FastAPI](https://fastapi.tiangolo.com)
- [Kaggle](https://www.kaggle.com)
- [Plutchik Emotion Wheel](https://en.wikipedia.org/wiki/Robert_Plutchik)
- [OpenAI API](https://platform.openai.com/docs)
- [Streamlit](https://streamlit.io)

## Acknowledgements

Open-source AI and Python community, the Kaggle ecosystem for accessible dataset hosting, and researchers working on emotion-aware AI, conversational NLP, and interpretable behavioral analytics.

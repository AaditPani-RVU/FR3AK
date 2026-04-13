# Functional Reasoning & Emotional Augmentation Kernel

Emotion • Behavior • Insight

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Datasets-20BEFF?logo=kaggle&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)

## Project Overview

FR3AK is a conversation intelligence system designed to analyze emotional and behavioral dynamics across multi-user chat transcripts. It combines sequential message parsing, Plutchik-based emotion modeling, behavior classification, and insight generation into a unified analytics workflow.

Core capabilities include:

- Emotional analysis using an 8-dimensional Plutchik emotion representation.
- Sarcasm detection from lexical, contextual, and rule-based signals.
- Manipulation detection through staged behavioral classification.
- Visual analytics with emotional trend timelines, sarcasm/manipulation timelines, and Plutchik wheel signatures.

## Features

- Multi-user conversation parsing with speaker and timestamp normalization.
- Emotion tracking over time across each participant.
- Sarcasm and manipulation detection at message and user levels.
- Interactive Streamlit dashboard for profile and trajectory exploration.

## Scoring Notes

Emotion intensity in the analyzer is computed from emotion dominance, not probability mass sum.

Current analyzer formula:

```python
emotion_array = np.array(emotion_vector)
raw_intensity = np.max(emotion_array) - np.mean(emotion_array)
emotion_intensity = float(raw_intensity / (np.max(emotion_array) + 1e-6))
```

Interpretation:

- Higher dominance of one emotion gives higher intensity (typically near 0.7-1.0).
- Flatter emotion distributions give lower intensity (typically near 0.1-0.3).

Where this scoring is used:

- `pipeline/analyzer.py`: computes per-message `emotion_intensity` and aggregates user-level averages.
- `pipeline/visualizer.py`: uses `emotion_intensity` directly and applies the same dominance formula as a fallback if intensity is missing.
- `pipeline/insights.py`: maps `avg_emotion_intensity` into low/moderate/high intensity language.
- `app.py`: renders message-level intensity trends in the user dashboard.

## Project Structure

```text
FR3AK/
├── app.py
├── server.py
├── models/
│   ├── emotion_model.py
│   ├── behavior_model.py
│   └── custom_emotion_model.py
├── pipeline/
│   ├── analyzer.py
│   ├── insights.py
│   ├── llm_summary.py
│   └── visualizer.py
├── utils/
│   ├── parser.py
│   └── build_sarcasm_lexicon.py
├── tests/
│   ├── analyzer_test.py
│   ├── emotion_test.py
│   ├── behavior_test.py
│   ├── insights_test.py
│   ├── visualizer_test.py
│   ├── final_conversation.txt
│   └── final_pipeline_test.py
├── training/
│   ├── nb1_label_part1.ipynb
│   ├── nb2_label_part2.ipynb
│   ├── nb3_train_model.ipynb
│   ├── nb4_eval_export.ipynb
│   ├── nb5_generate_vlits.ipynb
│   ├── nb6_feature_extract.ipynb
│   └── nb7_train_model2.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

- `models/`: Emotion and behavior model loading, inference, and architecture logic.
- `pipeline/`: End-to-end analysis flow (analyze -> insight -> visualization payload -> LLM summary).
- `utils/`: Conversation parsing utilities and sarcasm lexicon builder.
- `tests/`: Smoke/integration scripts for pipeline components and full run validation.
- `training/`: Notebooks used to preprocess data and train models.
- `app.py`: Streamlit UI for interactive analysis and visual exploration.
- `server.py`: FastAPI server exposing the pipeline as a REST API (`POST /api/analyze`). # requires: pip install fastapi uvicorn

## Setup Instructions

### 1. Clone Repo

```bash
git clone https://github.com/AaditPani-RVU/FR3AK.git
cd FR3AK
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Create a `.env` file in the project root with only the required variables:

```env
FR3AK_BEHAVIOR_DATASET_REF=...
FR3AK_EMOTION_DATASET_REF=...

KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

Optionally, add an OpenAI key to enable LLM-generated behavioral summaries:

```env
OPENAI_API_KEY=your_openai_key
```

### Kaggle API Setup

1. Go to [Kaggle](https://www.kaggle.com/)
2. Navigate to Account Settings
3. Click "Create New API Token"
4. Download `kaggle.json`

From this file, extract:

- `username` -> `KAGGLE_USERNAME`
- `key` -> `KAGGLE_KEY`

Then place them in your `.env` file.

### Notes

- Dataset references are used to automatically fetch required data.
- No manual dataset setup is required if `.env` is configured correctly.
- The system is fully Kaggle-integrated.

## Model and Data Setup

FR3AK supports two model artifact flows:

- Runtime resolution via configured Kaggle dataset references.
- Local artifact usage after downloading model datasets from Kaggle.

Sarcasm lexicon construction:

```bash
python build_sarcasm_lexicon.py
```

Alternative from repository layout:

```bash
python utils/build_sarcasm_lexicon.py
```

Data/model sources used by the project include Kaggle-hosted model assets and dataset-backed artifacts configured in environment variables.

Primary model data:

- Behavior/sarcasm model:
  [incongruity-classfier](https://www.kaggle.com/datasets/bobhendriks/incongruity-classfier)

- Emotion model (Plutchik-based):
  [plutchik-model-v2](https://www.kaggle.com/datasets/bobhendriks/plutchik-model-v2)

## Model Training

The `training/` directory contains 7 notebooks used to train the models used in this project.

You can:

- Inspect how the models were trained
- Reproduce results
- Train your own models using these notebooks

This includes:

- Emotion model training (Plutchik-based)
- Behavior/sarcasm detection training

## Build Sarcasm Lexicon

Run this once before starting the pipeline. The lexicon is required by the sarcasm rule engine at runtime.

```bash
python utils/build_sarcasm_lexicon.py
```

## Run Pipeline

```bash
python tests/final_pipeline_test.py
```

This executes the full analysis path: input parsing, message-level emotion/behavior inference, user-level aggregation, comparative insights, and visualization payload generation.

## Run Streamlit App

```bash
streamlit run app.py
```

The dashboard provides:

- File upload for `.txt`/`.json` conversation data.
- User-level profile cards and behavioral summaries.
- Emotional trend, sarcasm timeline, and manipulation timeline plots.
- Plutchik emotional signature visualization per participant.

## Run API Server

```bash
python server.py
```

Starts a FastAPI server at `http://0.0.0.0:8000`. Exposes `POST /api/analyze` which accepts a `.txt` or `.json` file upload and returns the full analysis, insights, and visualization payload as JSON.

## Testing

The `tests/` folder contains component and pipeline validation scripts.

Run selected tests:

```bash
python tests/emotion_test.py
python tests/analyzer_test.py
python tests/insights_test.py
python tests/visualizer_test.py
```

What these validate:

- Emotion model loading and inference shape/output behavior.
- Conversation analyzer message and user aggregation logic.
- Insight generation consistency for multi-user interactions.
- Visualization flow and plot generation compatibility.

## Future Work

- Upgrade to stronger domain-tuned behavior models.
- Add real-time streaming conversation analysis.
- Expose FR3AK as a deployable API service.

## Data Sources

- Conversation input format is user-provided and supports `.txt` and `.json` conversation exports.
- Input records are parsed from speaker/message/timestamp structures through the project parsing pipeline.

This project is model-driven and does not rely on a fixed dataset at runtime. However, models may have been trained on publicly available datasets such as:

- GoEmotions (Google)
- Emotion datasets published on Kaggle and research repositories
- Sarcasm datasets (Twitter/Reddit based)
- Kaggle-hosted behavior/emotion datasets configured via environment variables

Notes by task area:

- Sarcasm detection: rule- and lexicon-augmented behavior inference, with optional lexicon generation from public text/data sources.
- Emotion analysis: transformer-based emotion modeling aligned to Plutchik-style dimensions.
- Manipulation detection: staged behavioral classifier pipeline using loaded model artifacts.

## Models & Libraries

- Transformers for tokenizer/model integration and model configuration loading.
- PyTorch as the primary inference backend for emotion and behavior modeling.
- Matplotlib for timeline and radial emotion visualizations.
- Streamlit for the interactive analysis dashboard and user workflow.
- NumPy for numeric processing in plotting and feature handling.

## References

- [Kaggle](https://www.kaggle.com)
- [Streamlit](https://streamlit.io)
- [Matplotlib](https://matplotlib.org)
- [Plutchik Emotion Wheel (theory reference)](https://en.wikipedia.org/wiki/Robert_Plutchik)
- Emotion AI and NLP sentiment modeling literature (general research inspiration)

## Acknowledgements

- The open-source AI and Python community for tools, libraries, and shared practices.
- The Kaggle ecosystem for accessible dataset hosting and model distribution.
- Researchers and builders working on emotion-aware AI, conversational NLP, and interpretable behavioral analytics.

---

## Architecture

### System Pipeline

The pipeline works in two stages:

```
Raw utterance
    → Model 1: encode into 8D Plutchik emotion vector (V_lit)
    → Context layer: track how the vector drifts over the conversation
    → Feature extraction: 46-dim vector per turn
    → Model 2: classify as Genuine / Manipulative
    → Rule engine: independently flag as Sarcastic (lexicon + patterns)
    → (Optional) LLM: generate per-speaker behavioral summary via GPT-4o-mini
```

The **Plutchik wheel** was chosen as the emotion representation because it is a structured, psychologically grounded model with 8 primary emotions and defined opposite pairs. Opposites matter: if someone simultaneously signals high joy and high sadness, that is a meaningful signal of emotional conflict — and exactly the kind of tension that feeds into manipulation detection.

---

### Model 1 — PlutchikModelV2 (Emotion Encoder)

Each utterance is encoded into an **8-dimensional soft score vector** (V_lit) over the Plutchik axes:

```
[joy, trust, fear, surprise, sadness, disgust, anger, anticipation]
```

All values are in `[0, 1]` and represent relative emotional loading, not a categorical label.

**Architecture**

```
Input text (max 128 tokens)
    → microsoft/deberta-v3-base tokenizer + encoder
    → last_hidden_state (B, L, H=768)
    → 8 × EmotionAttentionBlock  (one per Plutchik axis)
          learned query vector × encoder keys/values
          4-head attention → LayerNorm → Linear(H → 128)
    → 8 × emotion head: Linear(128→32) → GELU → Dropout(0.1) → Linear(32→1)
    → per-emotion temperature scaling: sigmoid(logit × clamp(T, 0.5, 5.0))
    → V_lit: 8D score vector
    (+ auxiliary confidence head and classification head trained jointly)
```

Each `EmotionAttentionBlock` attends over the full token sequence with its own learnable query, so each emotion axis focuses independently on the parts of the sentence most relevant to it.

**Training — NB1 & NB2: Plutchik Labeling**

GoEmotions (~43K Reddit comments) had no continuous Plutchik labels, so they were generated using **Qwen2.5-3B-Instruct** (4-bit NF4, T4 GPU, ~2 GB VRAM). Each sentence was prompted for a JSON object with soft float scores for all 8 emotions plus a confidence value. Greedy decoding was used for consistency. A robust regex parser extracted valid JSON; rows with parse failures or confidence `< 0.2` were dropped.

- NB1 labels rows 0–21,999 of GoEmotions
- NB2 labels rows 22,000–end of GoEmotions, then loads SemEval 2018 Task 1 EI-reg (joy/sadness/fear/anger, continuous 0–1 intensity scores — no LLM labeling needed, already continuous)

**Training — NB3: Model Training**

| Hyperparameter | Value |
|---|---|
| Base encoder | `microsoft/deberta-v3-base` |
| Precision | FP32 throughout (DeBERTa BF16 causes NaN in disentangled attention) |
| Optimizer | AdamW with per-layer LR decay |
| Encoder LR | `1e-5` |
| Attention block / head LR | `2e-5` |
| LR decay per encoder layer | `0.85` |
| LR schedule | Cosine with warmup |
| Total epochs | 12 (3-stage curriculum, 4 epochs each) |
| Effective batch size | 32 (batch 16 × grad accumulation 2) |
| Early stopping patience | 3 epochs |

Training used a **curriculum** that sorted examples by difficulty (variance of the emotion distribution) and exposed the model progressively:

| Stage | Data | Epochs |
|---|---|---|
| Easy | Easiest 33% | 4 |
| Medium | Easiest 66% | 4 |
| All | Full training set | 4 |

Loss function:

```python
emotion_loss = 0.35 * cosine_loss + 0.50 * rank_loss
total_loss   = emotion_loss + 0.05 * conf_loss + 0.20 * aux_cls_loss
```

Rank loss is weighted highest because the headline metric is Spearman correlation, which is rank-sensitive. The auxiliary classification head (remapped GoEmotions 27-label → 8 Plutchik axes) provides extra gradient signal during curriculum stages.

**Training — NB4: Evaluation**

Model 1 was evaluated on the GoEmotions test split using per-emotion Spearman ρ, mean vector cosine similarity, and dominant emotion accuracy.

---

### Model 2 — IncongruityClassifier (Manipulation Detector)

Model 2 classifies each utterance as **Genuine** or **Manipulative**. It is a two-stage classifier operating on a 46-dimensional feature vector built from V_lit plus contextual drift signals.

**Sarcasm is not predicted by this model.** It is detected separately by a rule engine (see below).

**Feature Vector (46 dimensions)**

| Group | Description | Dims |
|---|---|---|
| V_lit | Raw 8D Plutchik vector for this turn | 8 |
| dv1 | V_lit − conversation EMA 1 turn ago | 8 |
| dv2 | V_lit − conversation EMA 2 turns ago | 8 |
| dv3 | V_lit − conversation EMA 3 turns ago | 8 |
| Pair tensions | joy×sadness, trust×disgust, fear×anger, surprise×anticipation | 4 |
| Anomaly scores | drift, actor_specificity, spike, tension_total, suppression, incoherence | 6 |
| Lexical features | negation density, pressure-word density, hedging density, punct flag | 4 |

Context tracking uses a rolling EMA (α=0.3) per speaker. Delta features (dv1/dv2/dv3) capture how abruptly a speaker's emotional tone shifts — a large sudden shift is a key manipulation signal.

Anomaly scores:
- `spike`: max − mean of V_lit — single-emotion dominance
- `tension_total`: sum of opposite-pair co-activations (simultaneous high joy + high sadness, etc.)
- `suppression`: fires when all emotions are uniformly low (< 0.3)
- `incoherence`: normalized entropy of the emotion distribution
- `drift`: L2 distance between V_lit and the running conversation EMA
- `actor_specificity`: L2 distance between the speaker's personal EMA and the conversation EMA

Lexical features capture pragmatic signals invisible to emotion vectors: negation words (`not, never, don't` — sarcasm/gaslighting signal), pressure words (`everyone, always, supposed to` — manipulation signal), hedging words (`maybe, I think, sort of` — genuine speech signal), and exclamation/question marks.

**Architecture**

Stage 1 — `stage1_genuine_vs_rest.json`: an XGBoost or linear scorer that operates directly on raw text features (token statistics, character ratios, cue-word densities). High-confidence genuine utterances (`score ≥ 0.98`) are short-circuited before reaching Stage 2.

Stage 2 — `stage2_sarcasm_vs_manip.pt`: an MLP on the 46-dim feature vector:

```
46-dim input
    → Linear → BatchNorm → GELU → Dropout
    → Linear → BatchNorm → GELU → Dropout
    → Linear → GELU
    → Linear(→ 2)   [genuine, manipulative]
    → softmax → manipulative probability
```

If `manipulative_prob > 0.9` → label `manipulative`, otherwise `genuine`.

**Training — NB5: V_lit Generation**

Model 1 was run in batch inference over all behavior training examples to produce their V_lit vectors:
- MentalManip multi-turn dialogues (parsed turn-by-turn to preserve conversation structure; each `PersonN:` utterance becomes a separate row)
- MUSTARD sarcasm utterances
- tweet_eval irony examples
- Sarcasm News Headlines examples

**Training — NB6: Feature Extraction + Labeling**

Per-turn 46-dim features were extracted using `ConversationTracker`. Labels were assigned by annotator consensus and source:

| Condition | Label |
|---|---|
| Source is a sarcasm/irony dataset (MUSTARD, tweet_eval irony, headlines) | Sarcasm (1) |
| MentalManip, 0 annotator votes for manipulation | Genuine (0) |
| MentalManip, 1 annotator vote (disagreement) | Ambiguous (3) |
| MentalManip, ≥2 votes, technique is sarcasm/irony | Sarcasm (1) |
| MentalManip, ≥2 votes, known non-sarcasm technique | Manipulation (2) |
| MentalManip, ≥2 votes, technique unknown | Ambiguous (3) |

**Training — NB7: Model 2 Training**

The IncongruityClassifier was trained on the 46-dim feature matrix as a 3-class problem (Genuine / Sarcasm / Manipulation; Ambiguous rows excluded). Exported as `incongruity_classifier.pt` and hosted on Kaggle as `incongruity-classfier`.

---

### Sarcasm Detection — Rule Engine

Sarcasm is **not predicted by any trained model**. It is detected by `BehaviorModel._detect_sarcasm()`, a rule engine that computes a float score `[0, 1]` from the sarcasm lexicon and a set of hard-coded pattern rules. If the score exceeds `SARCASM_THRESHOLD = 0.35`, the message is flagged `is_sarcastic: True`.

The rule engine checks:
- **Lexicon phrase hits**: known sarcasm phrases (`yeah right`, `as if`, `how convenient`, `just perfect`, etc.)
- **Positive+negative contrast**: positive-valence word co-occurring with a negative-context word in the same utterance
- **Expectation violation phrases**: `perfect timing`, `of course`, `naturally`, `that makes sense`, etc.
- **"love how / love when" openers**
- **"sure/yeah because..." contradiction clauses**
- **Exaggerated punctuation** (`!!`, `??`, `...`)
- **Elongated words** (`sooooo`, `greaaaat`)
- **All-caps emphasis words**
- **Inline sarcasm markers** (` /s`)

The lexicon itself is generated by `utils/build_sarcasm_lexicon.py`, which seeds from hand-crafted base phrases, expands them with positive/negative/intensifier templates, and optionally enriches with n-grams extracted from sarcasm datasets (tweet_eval irony, MUSTARD, Sarcasm News Headlines, Reddit sarcasm).

The `is_sarcastic` flag is stored alongside `label` in every message result. Sarcasm does **not** override the manipulation label — it is an independent signal tracked separately in user-level aggregations and visualized on its own timeline.

---

### LLM Behavioral Summaries — GPT-4o-mini (Optional)

`pipeline/llm_summary.py` generates a professional 2–3 sentence behavioral summary for each speaker using **GPT-4o-mini**. This is entirely optional and activates only when `OPENAI_API_KEY` is present in `.env`.

The prompt includes:
- Up to 30 of the speaker's messages, each annotated with `[sarcastic]` or `[manipulative]` flags where applicable
- Computed analytics: dominant emotion, emotional tone, emotional stability, sarcasm level, manipulation level

The model is called with `temperature=0.4` and `max_tokens=200`. If the key is missing or the call fails, the function returns `None` and the pipeline continues without a summary.

---

### Datasets

| Dataset | Used for | Source |
|---|---|---|
| **GoEmotions** (Google) | LLM-labeling target for soft Plutchik scores (~43K Reddit comments) | `go_emotions` via HuggingFace |
| **SemEval 2018 Task 1 EI-reg** | Continuous emotion intensity for joy/sadness/fear/anger (0–1, no LLM needed) | `sem_eval_2018_task_1` via HuggingFace |
| **MentalManip** | Multi-turn dialogues with 3-annotator manipulation labels and technique annotations | `audreyeleven/MentalManip` via HuggingFace |
| **MUSTARD** | TV-show dialogue sarcasm (context-dependent) | `tasksource/mustard`, `jhamel/mustard`, or `Yaxin/MUSTARD` |
| **tweet_eval irony** | Conversational tweet-domain irony/sarcasm | `tweet_eval` (irony config) via HuggingFace |
| **Sarcasm News Headlines** | High-quality headline sarcasm labels | `raquiba/Sarcasm_News_Headline` or `MidhunKanadan/sarcasm-detection` |

GoEmotions provided the core training distribution for Model 1. SemEval EI-reg was included because it provides continuous intensity labels rather than categorical flags, better matching Model 1's soft regression objective. MUSTARD, tweet_eval irony, and Sarcasm Headlines were used in NB5/NB6 to generate V_lits for sarcasm-labeled rows in Model 2's training set, and also serve as the source corpus for the sarcasm lexicon builder.

---

### Deployed Artifacts

| Artifact | Kaggle Dataset | Loaded by |
|---|---|---|
| `best_model.pt` + tokenizer + `model_config.json` | `bobhendriks/plutchik-model-v2` | `models/emotion_model.py` |
| `stage1_genuine_vs_rest.json` + `stage2_sarcasm_vs_manip.pt` + `model2_config.json` | `bobhendriks/incongruity-classfier` | `models/behavior_model.py` |

At startup both model wrappers check for a local artifact directory first, then download via the Kaggle CLI using credentials from `.env`.

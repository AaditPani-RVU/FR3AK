# FR3AK — System Architecture

**Functional Reasoning & Emotional Augmentation Kernel**

Emotion • Behavior • Insight

---

## Overview

FR3AK is a conversation intelligence system that analyzes emotional and behavioral dynamics across multi-user chat transcripts. It works in two sequential stages: first encoding each utterance into a structured emotional representation, then using that representation — together with conversational context — to classify the communicative intent behind the message.

```
Raw utterance
    → Model 1: encode into 8D Plutchik emotion vector (V_lit)
    → Context layer: track how the vector drifts over the conversation
    → Feature extraction: 46-dim vector per turn
    → Model 2: classify as Genuine / Sarcasm / Manipulation
```

The **Plutchik wheel** was chosen as the emotion representation because it is a structured, psychologically grounded model with 8 primary emotions and defined opposite pairs. Opposites matter: if someone simultaneously signals high joy and high sadness, that is a meaningful signal of emotional conflict. This tension between opposite pairs is one of the primary signals feeding Model 2.

---

## Project Structure

```text
FR3AK/
├── app.py                        # Streamlit dashboard
├── models/
│   ├── emotion_model.py          # Model 1: Plutchik inference wrapper
│   ├── behavior_model.py         # Model 2: IncongruityClassifier wrapper
│   └── custom_emotion_model.py   # Architecture reconstruction for loading
├── pipeline/
│   ├── analyzer.py               # End-to-end message + user aggregation
│   ├── insights.py               # Comparative insight generation
│   └── visualizer.py             # Plot payloads (timelines, radar charts)
├── utils/
│   ├── parser.py                 # Conversation parsing (speaker/timestamp)
│   └── build_sarcasm_lexicon.py  # Lexicon builder for rule-based signals
├── tests/
│   ├── final_pipeline_test.py    # Full end-to-end smoke test
│   ├── emotion_test.py
│   ├── analyzer_test.py
│   ├── insights_test.py
│   └── visualizer_test.py
└── training/
    ├── nb1_label_part1.ipynb     # GoEmotions rows 0–21999 → Plutchik labels
    ├── nb2_label_part2.ipynb     # GoEmotions rows 22000+ + SemEval EI-reg
    ├── nb3_train_model.ipynb     # Train PlutchikModelV2 (Model 1)
    ├── nb4_eval_export.ipynb     # Evaluate and export Model 1
    ├── nb5_generate_vlits.ipynb  # Run Model 1 on behavior datasets → V_lits
    ├── nb6_feature_extract.ipynb # Context tracking + 46-dim feature vectors
    └── nb7_train_model2.ipynb    # Train IncongruityClassifier (Model 2)
```

---

## Stage 1 — Emotion Encoding (Model 1: PlutchikModelV2)

### What it does

Each utterance is passed through `PlutchikModelV2`, which outputs an **8-dimensional soft probability vector** (V_lit) over the Plutchik emotion axes:

```
[joy, trust, fear, surprise, sadness, disgust, anger, anticipation]
```

All values are in `[0, 1]`. The vector represents relative emotional loading, not a single categorical label.

### Architecture

```
Input text
    → DeBERTa-v3-base (microsoft/deberta-v3-base) tokenizer (max 128 tokens)
    → DeBERTa-v3-base encoder → last_hidden_state (B, L, H)
    → 8 × EmotionAttentionBlock (one per Plutchik axis)
        └── 4-head cross-attention: learned query × encoder keys/values
        └── LayerNorm + Linear projection → 128-dim per-emotion repr
    → 8 × emotion head: Linear(128→32) → GELU → Dropout(0.1) → Linear(32→1)
    → per-emotion temperature scaling: sigmoid(logit × clamp(T, 0.5, 5.0))
    → output: 8D score vector
    → auxiliary: CLS token → confidence head (sigmoid) + classification head (8-class)
```

Each `EmotionAttentionBlock` attends independently over the encoder's output sequence using a **learnable query vector**, allowing each emotion axis to focus on the parts of the sentence most relevant to it.

### Training Pipeline

Training used a **3-notebook curriculum**:

**NB1 & NB2 — Plutchik Labeling (data generation)**

Raw sentences from GoEmotions were fed to **Qwen2.5-3B-Instruct** (4-bit NF4 quantized, ~2 GB VRAM on T4) with a structured JSON prompt requesting soft scores for all 8 Plutchik dimensions plus a confidence value. Greedy decoding was used for consistency. A robust regex parser extracted valid JSON from model output. Rows with parse failures or confidence `< 0.2` were dropped.

- NB1: rows 0–21999 of GoEmotions train split
- NB2: rows 22000–end of GoEmotions, plus SemEval 2018 EI-reg (described below)

**NB3 — Model Training**

| Hyperparameter | Value |
|---|---|
| Base encoder | `microsoft/deberta-v3-base` |
| Precision | FP32 throughout (DeBERTa BF16 causes NaN in disentangled attention) |
| Optimizer | AdamW with layerwise LR decay |
| Encoder LR | `1e-5` |
| Head LR | `2e-5` |
| LR decay per encoder layer | `0.85` |
| LR schedule | Cosine with warmup |
| Epochs | 12 total (3-stage curriculum) |
| Effective batch size | 32 (batch 16 × gradient accumulation 2) |
| Early stopping patience | 3 epochs |

**Curriculum training** sorted training examples by difficulty (variance of the emotion distribution) and exposed the model progressively:

| Stage | Data | Epochs |
|---|---|---|
| Easy | Easiest 33% of examples | 4 |
| Medium | Easiest 66% of examples | 4 |
| All | Full training set | 4 |

**Loss function:**

```python
emotion_loss = (
    0.35 * cosine_loss(scores, labels)    # vector direction
  + 0.50 * rank_loss(scores, labels)      # pairwise rank ordering
)
total = emotion_loss + 0.05 * conf_loss + 0.20 * aux_cls_loss
```

Rank loss was weighted highest because Spearman correlation (the headline metric) depends on ordinal rank, not magnitude.

**NB4 — Evaluation**

Model 1 was evaluated on the GoEmotions test split using:
- Per-emotion Spearman correlation (ρ)
- Mean vector cosine similarity
- Dominant emotion accuracy

---

## Emotion Intensity Scoring

Emotion intensity is derived from **dominance**, not probability mass:

```python
raw_intensity = np.max(emotion_array) - np.mean(emotion_array)
emotion_intensity = float(raw_intensity / (np.max(emotion_array) + 1e-6))
```

- High dominance of a single emotion → intensity near 0.7–1.0
- Flat distribution across all emotions → intensity near 0.1–0.3

This is used in `pipeline/analyzer.py` (per-message and user-level averages), `pipeline/visualizer.py` (timeline plots), `pipeline/insights.py` (language mapping), and `app.py` (dashboard trend charts).

---

## Stage 2 — Behavior Classification (Model 2: IncongruityClassifier)

### What it does

Model 2 classifies each utterance as **Genuine**, **Sarcasm**, or **Manipulation** using a 46-dimensional feature vector built from the V_lit output of Model 1 plus contextual drift signals.

### Feature Vector (46 dimensions)

| Group | Features | Dims |
|---|---|---|
| Raw V_lit | 8 Plutchik emotion scores | 8 |
| Context delta (1 turn back) | V_lit − conversation EMA (1 step ago) | 8 |
| Context delta (2 turns back) | V_lit − conversation EMA (2 steps ago) | 8 |
| Context delta (3 turns back) | V_lit − conversation EMA (3 steps ago) | 8 |
| Pair tensions | joy×sadness, trust×disgust, fear×anger, surprise×anticipation | 4 |
| Anomaly scores | drift, actor_specificity, spike, tension_total, suppression, incoherence | 6 |
| Lexical features | negation density, pressure-word density, hedging density, exclamation/question mark | 4 |
| **Total** | | **46** |

**Context tracking** uses a rolling exponential moving average (EMA, α=0.3) of V_lit vectors per speaker across the conversation. Delta features capture how abruptly a speaker's emotional tone shifts relative to the running context — a large sudden shift is a sarcasm and manipulation signal.

**Anomaly scores:**
- `spike`: max − mean of V_lit — how dominant the top emotion is
- `tension_total`: sum of opposite-pair co-activations (e.g., high joy AND high sadness simultaneously)
- `suppression`: fires when all emotions are uniformly low (< 0.3), indicating emotional flatness
- `incoherence`: normalized entropy of the emotion distribution
- `drift`: L2 distance between V_lit and the running context EMA
- `actor_specificity`: L2 distance between the speaker's personal EMA and the conversation-level EMA

**Lexical features** capture pragmatic signals invisible to emotion vectors:
- Negation density: `not, never, don't, can't` etc. — sarcasm and gaslighting signal
- Pressure-word density: `everyone, always, supposed to, obviously` etc. — manipulation signal
- Hedging density: `maybe, perhaps, I think, sort of` etc. — genuine speech signal
- Punctuation: presence of `!` or `?` — emotional intensity signal

### Architecture

The IncongruityClassifier is a **two-stage neural classifier** trained on the 46-dim feature vectors. It is a small MLP (no GPU required at inference), exported as `incongruity_classifier.pt` and hosted on Kaggle as `incongruity-classfier`.

At inference, `behavior_model.py` implements a two-stage decision:

1. **Stage 1** (`stage1_genuine_vs_rest`): a linear/XGBoost scorer separates clearly genuine utterances from potentially incongruent ones. A high-confidence genuine threshold (`0.98`) fast-tracks obvious cases.
2. **Stage 2** (`stage2_sarcasm_vs_manip`): the remaining utterances are passed to the deeper MLP head, which distinguishes Sarcasm from Manipulation. A sarcasm lexicon (`sarcasm_lexicon.json`) and SARCASM_THRESHOLD (`0.35`) provide additional rule-based augmentation.

### Training Pipeline

**NB5 — V_lit Generation**

Model 1 was run in batch inference mode over all behavior training examples to produce their V_lit vectors:

- MentalManip multi-turn dialogues (parsed turn-by-turn to preserve conversation structure)
- MUSTARD sarcasm utterances
- tweet_eval irony examples
- Sarcasm News Headlines examples

All sarcasm examples were assigned `annotator_votes=3` (confirmed label). Multi-turn MentalManip dialogues were split into individual turn rows using `PersonN:` prefix parsing so that context-delta features in NB6 would be meaningful.

**NB6 — Feature Extraction**

Per-turn features were extracted using `ConversationTracker`, which maintains rolling per-speaker EMAs and a 3-step conversation history. Labels were assigned as follows:

| Condition | Label |
|---|---|
| Source is a sarcasm/irony dataset | 1 — Sarcasm |
| MentalManip, annotator votes = 0 | 0 — Genuine |
| MentalManip, annotator votes = 1 | 3 — Ambiguous |
| MentalManip, technique is sarcasm/irony/mocking | 1 — Sarcasm |
| MentalManip, votes ≥ 2, known technique | 2 — Manipulation |
| MentalManip, votes ≥ 2, technique unknown | 3 — Ambiguous |

**NB7 — Model 2 Training**

The IncongruityClassifier was trained on the 46-dim feature matrix as a 3-class classifier (Genuine / Sarcasm / Manipulation; Ambiguous examples excluded or down-weighted). Output is exported to Kaggle as `incongruity-classfier`.

---

## Datasets

| Dataset | Used for | Source |
|---|---|---|
| **GoEmotions** (Google) | LLM-labeling target for Plutchik soft labels | `go_emotions` via HuggingFace Datasets |
| **SemEval 2018 Task 1 EI-reg** | Continuous emotion intensity labels for joy/sadness/fear/anger (0–1 range, no LLM needed) | `sem_eval_2018_task_1` via HuggingFace Datasets |
| **MentalManip** | Multi-turn dialogues with 3-annotator manipulation labels and technique annotations | `audreyeleven/MentalManip` via HuggingFace |
| **MUSTARD** | TV-show dialogue sarcasm (context-dependent) | `Yaxin/MUSTARD` / `jhamel/mustard` / local CSV |
| **tweet_eval irony** | Conversational tweet-domain irony/sarcasm | `tweet_eval` (irony config) via HuggingFace |
| **Sarcasm News Headlines** | High-quality headline sarcasm labels | `raquiba/Sarcasm_News_Headline` or `MidhunKanadan/sarcasm-detection` |

GoEmotions provided the core training distribution for Model 1 (~43K Reddit comments with 27-label taxonomy). The 27 GoEmotions labels were remapped to the 8 Plutchik axes for evaluation (e.g., `admiration→trust`, `grief→sadness`, `nervousness→fear`). SemEval EI-reg was included because it provides **continuous intensity** labels rather than categorical flags, which better matches the soft-regression objective of Model 1.

---

## Deployed Model Artifacts (Kaggle)

| Artifact | Kaggle Dataset | Used by |
|---|---|---|
| `best_model.pt` + tokenizer + `model_config.json` | `bobhendriks/plutchik-model-v2` | `models/emotion_model.py` |
| `stage1_genuine_vs_rest.json` + `stage2_sarcasm_vs_manip.pt` + `model2_config.json` | `bobhendriks/incongruity-classfier` | `models/behavior_model.py` |

At startup, both model wrappers check for a local artifact directory, then fall back to downloading via the Kaggle CLI using credentials from `.env`:

```env
FR3AK_EMOTION_DATASET_REF=bobhendriks/plutchik-model-v2
FR3AK_BEHAVIOR_DATASET_REF=bobhendriks/incongruity-classfier
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

---

## Pipeline Runtime Flow

```
app.py / final_pipeline_test.py
    │
    ▼
utils/parser.py
    Parse .txt / .json conversation file
    → list of {speaker, timestamp, text} messages
    │
    ▼
pipeline/analyzer.py
    For each message:
        EmotionModel.predict(text) → V_lit (8D)
        emotion_intensity = dominance(V_lit)
        BehaviorModel.predict(V_lit, text, context) → {genuine, sarcasm, manipulation}
    Aggregate per user:
        avg_emotion_vector, avg_emotion_intensity
        sarcasm_count, manipulation_count
    │
    ▼
pipeline/insights.py
    Compare users across emotion profiles, intensity, behavioral patterns
    → natural-language insight strings
    │
    ▼
pipeline/visualizer.py
    Emotion trend timeline (per user, over turns)
    Sarcasm + manipulation timeline
    Plutchik radar chart (per user)
    → plot payloads consumed by app.py
```

---

## Features

- Multi-user conversation parsing with speaker and timestamp normalization.
- Emotion tracking over time across each participant.
- Sarcasm and manipulation detection at message and user levels.
- Interactive Streamlit dashboard for profile and trajectory exploration.

---

## Scoring Notes

Emotion intensity in the analyzer is computed from emotion dominance, not probability mass sum.

Current analyzer formula:

```python
emotion_array = np.array(emotion_vector)
raw_intensity = np.max(emotion_array) - np.mean(emotion_array)
emotion_intensity = float(raw_intensity / (np.max(emotion_array) + 1e-6))
```

Interpretation:

- Higher dominance of one emotion gives higher intensity (typically near 0.7–1.0).
- Flatter emotion distributions give lower intensity (typically near 0.1–0.3).

Where this scoring is used:

- `pipeline/analyzer.py`: computes per-message `emotion_intensity` and aggregates user-level averages.
- `pipeline/visualizer.py`: uses `emotion_intensity` directly and applies the same dominance formula as a fallback if intensity is missing.
- `pipeline/insights.py`: maps `avg_emotion_intensity` into low/moderate/high intensity language.
- `app.py`: renders message-level intensity trends in the user dashboard.

---

## Setup

### 1. Clone Repo

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
FR3AK_BEHAVIOR_DATASET_REF=bobhendriks/incongruity-classfier
FR3AK_EMOTION_DATASET_REF=bobhendriks/plutchik-model-v2

KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

Kaggle credentials are obtained from Account Settings → Create New API Token → `kaggle.json`.

### 4. Build Sarcasm Lexicon

```bash
python utils/build_sarcasm_lexicon.py
```

### 5. Run Pipeline

```bash
python tests/final_pipeline_test.py
```

### 6. Run Dashboard

```bash
streamlit run app.py
```

The dashboard accepts `.txt` or `.json` conversation uploads and renders per-user profile cards, emotional trend timelines, sarcasm/manipulation timelines, and Plutchik wheel signatures.

---

## Testing

```bash
python tests/emotion_test.py       # Model 1 output shape and inference
python tests/analyzer_test.py      # Message-level and user-level aggregation
python tests/insights_test.py      # Insight generation consistency
python tests/visualizer_test.py    # Plot generation and compatibility
```

---

## Libraries

| Library | Role |
|---|---|
| PyTorch | Inference backend for both models |
| Transformers (HuggingFace) | DeBERTa-v3-base tokenizer and encoder |
| NumPy | Numeric processing in feature extraction and plotting |
| Matplotlib | Timeline and radar (Plutchik wheel) visualizations |
| Streamlit | Interactive analysis dashboard |

---

## Future Work

- Upgrade to stronger domain-tuned behavior models.
- Add real-time streaming conversation analysis.
- Expose FR3AK as a deployable API service.

---

## References

- [Plutchik Emotion Wheel (theory)](https://en.wikipedia.org/wiki/Robert_Plutchik)
- [GoEmotions (Google)](https://huggingface.co/datasets/go_emotions)
- [SemEval 2018 Task 1 EI-reg](https://huggingface.co/datasets/sem_eval_2018_task_1)
- [MentalManip](https://huggingface.co/datasets/audreyeleven/MentalManip)
- [Kaggle](https://www.kaggle.com) — model artifact hosting
- [Streamlit](https://streamlit.io)
- [Matplotlib](https://matplotlib.org)

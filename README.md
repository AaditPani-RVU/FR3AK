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

## Project Structure

```text
FR3AK/
├── app.py
├── models/
│   ├── emotion_model.py
│   ├── behavior_model.py
│   └── custom_emotion_model.py
├── pipeline/
│   ├── analyzer.py
│   ├── insights.py
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
│   ├── nb3_train_model.ipy
│   ├── nb4_eval_export.ipy
│   ├── nb5_generate_vlits.ipy
│   ├── nb6_feature_extract.ipynb
│   └── nb7_train_model2.ipynb

├── requirements.txt
├── .gitignore
└── README.md
```

- `models/`: Emotion and behavior model loading, inference, and architecture logic.
- `pipeline/`: End-to-end analysis flow (analyze -> insight -> visualization payload).
- `utils/`: Conversation parsing utilities and sarcasm lexicon builder.
- `tests/`: Smoke/integration scripts for pipeline components and full run validation.
- `training`: Scripts used to preprocess data and train models.
- `app.py`: Streamlit UI for interactive analysis and visual exploration.

## Setup Instructions

### 1. Clone Repo

```bash
git clone <https://github.com/AaditPani-RVU/FR3AK.git>
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

### Kaggle API Setup

1. Go to https://www.kaggle.com/
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

## Model / Data Setup

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
	https://www.kaggle.com/datasets/bobhendriks/incongruity-classfier

- Emotion model (Plutchik-based):
	https://www.kaggle.com/datasets/bobhendriks/plutchik-model-v2

## Model Training

The `training/` directory contains 7 notebooks used to train the models used in this project.

You can:

- Inspect how the models were trained
- Reproduce results
- Train your own models using these notebooks

This includes:

- Emotion model training (Plutchik-based)
- Behavior/sarcasm detection training

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

- Kaggle: https://www.kaggle.com
- Streamlit: https://streamlit.io
- Matplotlib: https://matplotlib.org
- Plutchik Emotion Wheel (theory reference): https://en.wikipedia.org/wiki/Robert_Plutchik
- Emotion AI and NLP sentiment modeling literature (general research inspiration)

## Acknowledgements

- The open-source AI and Python community for tools, libraries, and shared practices.
- The Kaggle ecosystem for accessible dataset hosting and model distribution.
- Researchers and builders working on emotion-aware AI, conversational NLP, and interpretable behavioral analytics.

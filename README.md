# BBC News Category Predictor

A multi-class text classifier that reads a news article and predicts its category (business, entertainment, politics, sport, tech), trained from scratch on the BBC News dataset using a word-embedding + GRU model in PyTorch. Fourth portfolio project, and a natural extension of my [IMDB sentiment classifier](https://github.com/Aigor4/imdb-sentiment-classifier) — same GRU architecture, but multi-class (softmax) instead of binary (sigmoid), plus this one adds a Docker deployment.

## Overview

- **Input:** a news article (free text)
- **Output:** one of 5 categories — `business`, `entertainment`, `politics`, `sport`, `tech`
- **Data:** [BBC Full Text Document Classification](https://www.kaggle.com/datasets/alfathterry/bbc-full-text-document-classification) (Kaggle), 2225 articles (2126 after removing 99 duplicates), reasonably balanced across the 5 categories
- **Model:** word-level tokenizer with a custom vocabulary (18,474 words, min. frequency 2) → `nn.Embedding` → `nn.GRU` → `nn.Linear` → softmax (via `CrossEntropyLoss`)
- **Training:** from scratch, no pretrained embeddings, 20 epochs, Adam optimizer

## Project structure

```
bbc-news-classifier/
├── data/
│   └── bbc_data.csv        # raw dataset (2225 articles)
├── models/
│   ├── model.pt             # trained GRU weights
│   └── vocab.pkl            # vocabulary + max_len + category names
├── src/
│   ├── exploration.py       # shape, class balance, text-length stats
│   ├── etl.py                # load csv, clean text, label-encode, train/test split
│   ├── dataset.py            # builds the vocabulary, PyTorch Dataset class
│   ├── model.py              # the GRU model (nn.Module)
│   ├── train.py               # training loop, saves model.pt + vocab.pkl
│   ├── evaluate.py            # accuracy/precision/recall/F1 (macro) + confusion matrix
│   ├── predict.py             # CLI: enter an article, get a predicted category
│   └── app.py                 # Streamlit UI
├── tests/
│   └── test_etl.py            # unit tests for the ETL step
├── Dockerfile
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run everything from `src/` (the scripts use relative paths like `../models/`):

```bash
cd src
python train.py       # trains the GRU, saves models/model.pt + models/vocab.pkl
python evaluate.py     # prints accuracy/precision/recall/F1 (macro) + confusion matrix
python predict.py      # CLI: type an article, get the predicted category
```

Or launch the interactive app (also from `src/`):

```bash
cd src
streamlit run app.py
```

Run tests (from the project root):

```bash
pytest tests/
```

### Docker

```bash
docker build -t bbc-news-classifier .
docker run -p 8501:8501 bbc-news-classifier
```

Then open `localhost:8501`.

## Model performance

| Metric              | Score |
|---------------------|-------|
| Accuracy            | 0.472 |
| Precision (macro)   | 0.556 |
| Recall (macro)      | 0.471 |
| F1 (macro)          | 0.476 |

Confusion matrix (rows/cols in order: entertainment, business, sport, politics, tech):

```
[[37  2 25  9  1]
 [33 33 22 11  2]
 [19  1 62 18  1]
 [ 7  2 27 39  6]
 [ 1  1 11 26 30]]
```

See [Limitations](#limitations--possible-improvements) below for why this is well short of the IMDB project's accuracy, and why it isn't the point of this project.

## What was hardest

- **Overfitting on a small dataset.** With only ~1700 training articles and embeddings trained from scratch, the model memorizes the training set fast (train loss drops below 0.2 within 20 epochs) while test accuracy stalls around 50%. I tried `nn.Dropout` + `weight_decay` on the Adam optimizer to fight this — it didn't beat the un-regularized baseline at the same epoch count, and only matched it after doubling the epochs, so I reverted to the simpler baseline rather than add complexity that wasn't paying off.
- **Adapting my own IMDB project's code introduced copy-paste bugs.** Reusing the previous project's structure was faster, but I repeatedly left one line matching the *old* binary-classification pattern instead of updating it for multi-class — e.g. calling the model with the wrong variable name, forgetting to actually pass `average='macro'` into the sklearn metric calls after defining it, unpacking a `(label, prob)` tuple from `predict()` when this version only returns a single label.
- **Docker was new territory for me.** Getting a relative-path-dependent app (`../models/...`) to work inside a container meant understanding `WORKDIR`, image layers, and that only what's explicitly `COPY`'d is visible inside the build — plus discovering that my code's own architecture (see below) meant the container needed the raw dataset just to start up, not just the trained model.

I used AI (Claude) as a guide along the way — mainly to explain new concepts (Docker, regularization trade-offs) before I wrote the config/code myself, and to review/debug what I wrote, rather than to write the project for me.

## Limitations & possible improvements

- **~47-55% accuracy ceiling** (varies run to run — training isn't seeded). Expected for a from-scratch embedding on ~1700 training rows; a bigger dataset or pretrained embeddings (GloVe) would likely help a lot more than further tuning this model.
- **Regularization (dropout + weight decay) was tried and didn't help** at the same epoch budget — documented as a dead end rather than silently dropped.
- **The Docker image ships the full training dataset (`data/`), not just the trained model.** This is a shortcut, not a best practice: my code imports modules in a chain (`app.py` → `predict.py` → `model.py` → `dataset.py`) where `dataset.py` re-runs the ETL/vocabulary-building step as a side effect of being imported, so even pure inference code transitively needs the raw CSV. The "correct" fix would be to have `model.py` take `vocab_size`/`num_classes` as constructor arguments instead of computing them from a module-level import — I chose to ship the dataset instead of refactoring, since the dataset is small (~5MB) and the refactor touches several files. Worth revisiting if this project grows.
- No pretrained embeddings, no hyperparameter search, single train/test split (no cross-validation).

## Tech stack

Python, PyTorch, pandas, scikit-learn, Streamlit, pytest, Docker

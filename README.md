# CS444 — Information Retrieval Semester Project

**Course:** CS444 — Information Retrieval  
**Instructor:** Dr. Zoya  
**Algorithm Implementations:** Naive Bayes · Rocchio · k-Nearest Neighbors

---

## 📁 Project Structure

```
IR proj/
├── main.py                  # Entry point — run experiments here
├── download_datasets.py     # Auto-download all datasets
├── requirements.txt
├── src/
│   ├── preprocessing.py     # Text cleaning, TF-IDF vectorization pipeline
│   ├── naive_bayes.py       # Complement Naive Bayes classifier
│   ├── rocchio.py           # Rocchio centroid classifier (manual implementation)
│   ├── knn.py               # k-NN with cosine similarity
│   └── evaluate.py          # Metrics, comparison tables, plots
├── data/
│   └── raw/                 # Place downloaded datasets here
├── results/                 # Auto-generated: CSVs, plots, confusion matrices
└── notebooks/               # (Optional) Jupyter exploration notebooks
```

---

## ⚙️ Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Download Datasets

Run the auto-downloader to fetch all three datasets:

```bash
python download_datasets.py
```

This will download and extract into `data/raw/`:

| Dataset | Source | Docs |
|---|---|---|
| Health News in Twitter | UCI ML Repo | ~63,000 tweets across 16 health outlets |
| KOS Blog Posts | UCI Bag of Words | ~3,430 blog documents |
| Enron Emails | UCI Bag of Words | ~39,861 email documents |

> **Manual fallback:** If auto-download fails, visit the links below and place files in `data/raw/`:
> - https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter
> - https://archive.ics.uci.edu/ml/datasets/Bag+of+Words

---

## 🚀 Running the Project

### Run a single dataset

```bash
python main.py --dataset twitter     # Health News in Twitter
python main.py --dataset kos         # KOS Blog Posts (Bag of Words)
python main.py --dataset enron       # Enron Emails (Bag of Words)
```

### Run all datasets at once

```bash
python main.py --all
```

---

## 📊 Output

After running, the `results/` folder contains:

- **`<dataset>_comparison.csv`** — Table of Accuracy, Precision, Recall, F1 for all 3 classifiers
- **`<dataset>_comparison.png`** — Bar chart comparing all classifiers
- **`<dataset>_<Classifier>_cm.png`** — Confusion matrix heatmaps

---

## 🧠 Algorithms Implemented

### 1. Naive Bayes (`src/naive_bayes.py`)
- Uses **Complement Naive Bayes** (sklearn's `ComplementNB`)
- Better handles class imbalance than standard Multinomial NB
- Laplace smoothing with α = 1.0
- Formula: `P(c|d) ∝ P(c) × ∏ P(t|c)^tf(t,d)`

### 2. Rocchio (`src/rocchio.py`)
- **Manually implemented** (not from sklearn)
- Computes per-class centroid vectors from TF-IDF representations
- Classifies by cosine similarity to nearest centroid
- Supports optional negative feedback (β parameter)
- Formula: `μ(c) = (1/|Dc|) × Σ d∈Dc tfidf(d)`

### 3. k-Nearest Neighbors (`src/knn.py`)
- Uses sklearn's `KNeighborsClassifier` with `metric='cosine'`
- Distance-weighted voting (closer neighbors count more)
- k = 5 (tunable)
- Formula: `class(q) = majority_vote(top-k neighbors by cosine_sim)`

---

## 🔧 Preprocessing Pipeline

Located in `src/preprocessing.py`:

1. Lowercase + remove URLs, mentions, hashtags
2. Remove special characters and digits
3. Tokenization
4. Stopword removal (NLTK English stopwords)
5. Porter Stemming
6. TF-IDF Vectorization (unigrams + bigrams, max 20,000 features, sublinear TF)

---

## 📈 Evaluation Metrics

All results are reported with **weighted averages** across classes:

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / Total |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × (P × R) / (P + R) |

---

## 👥 Group Members

| Name | Roll No | 
|---|---|---|
| Saad Hanif Taj | 2022509 | 
| Ahmed Ali | 2022054 | 
| Aiza Azeem | 2022077 | 

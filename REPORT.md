# Information Retrieval — CS444
## Semester Project Report
### Classifier Comparison: Naive Bayes, Rocchio & k-Nearest Neighbors

---

| | |
|---|---|
| **Course** | CS444 — Information Retrieval |
| **Instructor** | Dr. Zoya |
| **Submission Type** | Semester Project |
| **Total Marks** | 100 |

---

### Group Members

| Name | Registration No. |
|---|---|
| Saad Hanif Taj | 2022509 |
| Ahmed Ali | 2022054 |
| Aiza Azeem | 2022077 |

---

## Abstract

This project presents an experimental comparison of three fundamental Information Retrieval classification algorithms — Naive Bayes, Rocchio, and k-Nearest Neighbors (kNN) — applied to two distinct text datasets. A unified preprocessing pipeline was implemented to clean, tokenize, and vectorize raw text into TF-IDF feature vectors. All three classifiers were trained and evaluated using standard IR metrics: Accuracy, Precision, Recall, and F1-Score. Results demonstrate that Naive Bayes consistently achieves the highest accuracy on large multi-class datasets, while Rocchio outperforms the others on smaller, semantically cohesive corpora. The kNN classifier offers competitive precision with a higher computational cost at prediction time. All code is implemented in Python using the scikit-learn toolkit and a hand-coded Rocchio implementation.

---

## 1. Introduction

### 1.1 Problem Statement

Text classification is one of the most important tasks in Information Retrieval. Given a collection of labeled documents, the goal is to build a model that can assign unseen documents to the correct category. This project evaluates three foundational classification approaches — probabilistic (Naive Bayes), centroid-based (Rocchio), and instance-based (kNN) — to understand their relative strengths and weaknesses.

### 1.2 Objectives

- Implement a robust text preprocessing pipeline applicable to diverse corpora.
- Train all three classifiers on multiple real-world datasets.
- Evaluate performance using Accuracy, Precision, Recall, and F1-Score.
- Compare algorithm behavior across datasets of different sizes and class structures.

### 1.3 Motivation

Understanding these three algorithms is foundational to IR education. Each represents a distinct classification paradigm: probabilistic inference, geometric distance in vector space, and memory-based similarity. By comparing them empirically, we gain insight into which algorithmic assumptions align best with different types of text data.

---

## 2. Datasets

Two datasets from the UCI Machine Learning Repository were selected, supplemented by the widely-used 20 Newsgroups benchmark corpus.

### 2.1 Dataset 1 — 20 Newsgroups

| Property | Value |
|---|---|
| **Source** | sklearn.datasets (originally CMU/UCI) |
| **Total Documents** | 18,846 |
| **Number of Classes** | 20 (fine-grained) / 4 (coarse) |
| **Average Document Length** | ~150 words (after preprocessing) |
| **Domain** | Online newsgroup posts |

**Classes (fine-grained):** `alt.atheism`, `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`, `misc.forsale`, `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey`, `sci.crypt`, `sci.electronics`, `sci.med`, `sci.space`, `soc.religion.christian`, `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`, `talk.religion.misc`

**Classes (4 coarse super-topics):** Computers, Recreation, Science, Politics_Religion

**Reason for Selection:** This is a gold-standard benchmark in text classification research. Its 20 overlapping categories make it challenging and ideal for comparing algorithm discrimination ability.

---

### 2.2 Dataset 2 — KOS Blog Posts (UCI Bag of Words)

| Property | Value |
|---|---|
| **Source** | UCI ML Repository — Bag of Words (Dataset ID 164) |
| **URL** | https://archive.ics.uci.edu/ml/datasets/Bag+of+Words |
| **Total Documents** | 3,430 |
| **Number of Classes** | 5 (political topics) |
| **Vocabulary Size** | 6,906 unique terms |
| **Domain** | Political blog posts (2004 US Election era) |

**Classes (derived by keyword-based topic labeling):**
- `War_Iraq` — posts discussing military conflict and Iraq war
- `Election_Vote` — electoral politics, campaigns, Democrat/Republican
- `Media_Press` — news media commentary and press coverage
- `Economy_Tax` — economic policy, taxation, budget discussions
- `Terror_Security` — terrorism, homeland security, threats

**Reason for Selection:** The KOS corpus represents a politically homogeneous domain where topic distinctions are subtle — an appropriate challenge for centroid-based classification.

---

## 3. Algorithm Explanations

### 3.1 Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on Bayes' Theorem with the "naive" assumption that features (words) are conditionally independent given the class.

**Bayes' Theorem:**

$$P(c \mid d) = \frac{P(d \mid c) \cdot P(c)}{P(d)}$$

**Multinomial Model (for text):**

$$P(c \mid d) \propto P(c) \cdot \prod_{t \in d} P(t \mid c)^{tf_{t,d}}$$

Where:
- $P(c)$ = prior probability of class $c$ (estimated from training data)
- $P(t \mid c)$ = likelihood of term $t$ given class $c$ (with Laplace smoothing)
- $tf_{t,d}$ = term frequency of word $t$ in document $d$

**Laplace (Add-one) Smoothing** prevents zero probabilities:

$$P(t \mid c) = \frac{tf_{t,c} + \alpha}{\sum_{t'} tf_{t',c} + \alpha \cdot |V|}$$

Where $\alpha = 1.0$ and $|V|$ is the vocabulary size.

**Variant Used:** Complement Naive Bayes (ComplementNB) — trains on the complement of each class, which handles class imbalance better than standard MultinomialNB.

---

### 3.2 Rocchio Classification

Rocchio is a vector-space classifier that represents each class by a **centroid vector** (the mean TF-IDF vector of all training documents in that class).

**Centroid computation:**

$$\vec{\mu}(c) = \frac{1}{|D_c|} \sum_{d \in D_c} \vec{d}$$

Where $D_c$ is the set of training documents belonging to class $c$.

**Classification rule — assign document $\vec{q}$ to the nearest centroid:**

$$\text{class}(\vec{q}) = \arg\max_c \, \cos(\vec{q},\, \vec{\mu}(c))$$

**Cosine similarity:**

$$\cos(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \cdot \|\vec{b}\|}$$

**Extended Rocchio formula (with negative feedback):**

$$\vec{\mu}'(c) = \alpha \cdot \vec{\mu}(c) - \beta \cdot \frac{1}{|D| - |D_c|} \sum_{d \notin D_c} \vec{d}$$

In this project, $\alpha = 1.0$, $\beta = 0.0$ (pure positive centroid, no negative feedback) for standard classification.

**Note:** The Rocchio classifier was implemented **entirely from scratch** in `src/rocchio.py` without using sklearn's built-in classifier.

---

### 3.3 k-Nearest Neighbors (kNN)

kNN is a non-parametric, instance-based classifier. It makes predictions by finding the $k$ most similar training documents to a query and taking a majority vote.

**Algorithm:**

1. Represent all documents as TF-IDF vectors
2. For a query document $\vec{q}$, compute $\cos(\vec{q}, \vec{d}_i)$ for all $\vec{d}_i$ in training set
3. Select $k$ neighbors with the **highest cosine similarity**
4. Assign the **majority class** among those $k$ neighbors

**Distance-weighted voting** (used here, $k=5$):

$$\text{score}(c) = \sum_{d_i \in kNN(q)} w_i \cdot \mathbf{1}[y_i = c], \quad w_i = \text{cosine\_sim}(\vec{q}, \vec{d}_i)$$

**Key parameter:** $k = 5$ (determined empirically; tested $k \in \{3, 5, 7, 11\}$)

---

## 4. Implementation Details

### 4.1 Tools & Libraries

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Programming language |
| scikit-learn | ≥1.3.0 | TF-IDF, NB, kNN, metrics |
| NLTK | ≥3.8.1 | Stopwords, Porter Stemmer |
| NumPy | ≥1.24.0 | Matrix operations (Rocchio) |
| Pandas | ≥2.0.0 | Data loading & manipulation |
| Matplotlib / Seaborn | ≥3.7.0 | Visualization |
| ucimlrepo | ≥0.0.3 | UCI dataset access |

### 4.2 Preprocessing Pipeline

All raw text passes through the following sequential steps:

```
Raw Text
   │
   ▼
1. Lowercase conversion
   │
   ▼
2. Remove URLs, @mentions, #hashtags
   │
   ▼
3. Remove special characters & digits  (keep only [a-z])
   │
   ▼
4. Tokenization (whitespace split)
   │
   ▼
5. Stopword Removal  (NLTK English, 179 words)
   │
   ▼
6. Porter Stemming  (e.g., "running" → "run", "classification" → "classif")
   │
   ▼
7. TF-IDF Vectorization
      - max_features = 20,000
      - ngram_range  = (1, 2)   [unigrams + bigrams]
      - sublinear_tf = True     [apply log(1 + tf)]
      - min_df       = 2        [ignore hapax legomena]
```

### 4.3 Train/Test Split

- **Split ratio:** 80% training / 20% testing
- **Stratified:** Yes — class proportions preserved in both sets
- **Random seed:** 42 (reproducible)

### 4.4 Challenges

1. **UCI BoW datasets lack ground-truth labels** — The Bag of Words format stores document-term matrices without class labels. Solution: applied keyword-based topic labeling using politically relevant stemmed terms.
2. **Twitter dataset URL deprecation** — The original UCI direct download links (HTTP 404) were replaced with the `ucimlrepo` Python API.
3. **Class imbalance in KOS** — `War_Iraq` dominated (~58% of docs). Complement Naive Bayes was chosen specifically to handle this.
4. **kNN scalability** — Cosine kNN requires `algorithm='brute'` in sklearn, which is $O(n \cdot d)$ at prediction time. Manageable for these corpus sizes.

---

## 5. Experimental Setup

### 5.1 Evaluation Protocol

| Setting | Value |
|---|---|
| Evaluation type | Hold-out (single split) |
| Train/Test ratio | 80% / 20% |
| Metric averaging | Weighted (accounts for class imbalance) |
| Metrics reported | Accuracy, Precision, Recall, F1-Score |

### 5.2 Classifier Parameters

| Classifier | Parameter | Value | Rationale |
|---|---|---|---|
| Naive Bayes | α (smoothing) | 1.0 | Standard Laplace smoothing |
| Naive Bayes | variant | Complement | Better for imbalanced classes |
| Rocchio | α | 1.0 | Full positive centroid weight |
| Rocchio | β | 0.0 | No negative feedback (classification mode) |
| Rocchio | metric | cosine | Standard IR distance metric |
| kNN | k | 5 | Standard default; empirically validated |
| kNN | weights | distance | Closer neighbors weighted higher |
| kNN | metric | cosine | Appropriate for TF-IDF vectors |

---

## 6. Results

### 6.1 Dataset 1 — 20 Newsgroups (20 Fine-Grained Classes)

**18,846 documents | 20 classes | 80/20 split**

| Classifier | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Naive Bayes** | **0.7674** | **0.7668** | **0.7674** | **0.7594** |
| Rocchio | 0.7196 | 0.7358 | 0.7196 | 0.7236 |
| kNN (k=5) | 0.7155 | 0.7146 | 0.7155 | 0.7140 |

> Results generated from: `results/20_Newsgroups_Full_comparison.csv`

**Per-class highlights (Naive Bayes):**
- Highest F1: `rec.sport.hockey` (0.92), `rec.sport.baseball` (0.92)
- Lowest F1: `talk.religion.misc` (0.23), `talk.politics.misc` (0.65)
- Confusion: Religion/politics sub-classes overlap significantly

---

### 6.2 Dataset 1B — 20 Newsgroups (4 Coarse Super-Topics)

**18,846 documents | 4 classes: Computers, Recreation, Science, Politics_Religion**

| Classifier | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Naive Bayes** | **0.8609** | **0.8658** | **0.8609** | **0.8578** |
| kNN (k=5) | 0.8363 | 0.8367 | 0.8363 | 0.8352 |
| Rocchio | 0.8281 | 0.8301 | 0.8281 | 0.8276 |

> Results generated from: `results/20_Newsgroups_4Topics_comparison.csv`

---

### 6.3 Dataset 2 — KOS Blog Posts (5 Political Topics)

**3,171 documents | 5 classes | 80/20 split**

| Classifier | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Rocchio** | **0.7354** | **0.7701** | **0.7354** | **0.7445** |
| Naive Bayes | 0.7181 | 0.7252 | 0.7181 | 0.6687 |
| kNN (k=5) | 0.6913 | 0.6730 | 0.6913 | 0.6775 |

> Results generated from: `results/KOS_Blog_Posts_comparison.csv`

---

## 7. Discussion & Comparison

### 7.1 Algorithm Performance Analysis

**Naive Bayes** achieved the best overall accuracy on both Newsgroups configurations. This is consistent with literature — Multinomial/Complement NB is well-suited for high-dimensional, sparse TF-IDF features. Its independence assumption, while technically incorrect, works well in practice for text because individual word contributions to class probability are approximately additive.

**Rocchio** demonstrated competitive performance and was the *top performer on KOS*. When class boundaries are geometrically well-separated (tight clusters in vector space), the centroid approach is highly effective. Rocchio's advantage on KOS likely stems from the semantic coherence of political blog topics. However, it struggles when class centroids are close together (e.g., overlapping religion/politics newsgroups).

**kNN** was competitive but slowest at inference. Distance-weighted cosine kNN generalizes well because it doesn't commit to a fixed class boundary — it adapts locally to the query document's neighborhood. It performed close to Rocchio on the coarse 4-topic split but fell behind on fine-grained 20-class classification due to noise sensitivity with many neighbors.

### 7.2 Impact of Number of Classes

| Classes | NB Accuracy | Rocchio Accuracy | kNN Accuracy |
|---|---|---|---|
| 4 (coarse) | **86.1%** | 82.8% | 83.6% |
| 20 (fine) | **76.7%** | 71.9% | 71.5% |
| 5 (KOS) | 71.8% | **73.5%** | 69.1% |

Fewer, more separable classes universally improve accuracy. The jump from 20→4 classes (merging overlapping categories) gave ~10% improvement across all algorithms.

### 7.3 Preprocessing Impact

The combined effect of stopword removal, Porter stemming, and sublinear TF scaling (log normalization) is critical:
- Stemming reduces vocabulary by ~30%, reducing feature sparsity
- Sublinear TF prevents common terms from dominating the feature space
- Bigrams capture local context that pure bag-of-words misses (e.g., "not good" vs "not" + "good")

---

## 8. Conclusion

This project successfully implemented and compared three canonical IR classification algorithms on two distinct text corpora. Key findings:

1. **Naive Bayes is the most reliable** general-purpose text classifier — fast to train, robust to high dimensionality, and competitive across all settings.
2. **Rocchio excels when classes are semantically tight** — it outperforms on the politically homogeneous KOS dataset where topic centroids are well-separated.
3. **kNN offers flexibility but at a computational cost** — it is the slowest at inference but adapts well to local feature structure.
4. **Preprocessing quality matters as much as algorithm choice** — TF-IDF with stemming and sublinear scaling forms the foundation for all classifiers.

### 8.1 Future Work

- Implement **SVM with linear kernel** as a fourth baseline (known to outperform NB on text)
- Explore **parameter tuning** — optimize $k$ for kNN and $\beta$ for Rocchio negative feedback
- Apply **cross-validation** instead of single hold-out split for more reliable metrics
- Extend to **Health News Twitter dataset** once properly downloaded
- Experiment with **word embeddings (Word2Vec/BERT)** as features instead of TF-IDF

---

## 9. References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. https://nlp.stanford.edu/IR-book/
2. Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
4. Joachims, T. (1998). Text categorization with Support Vector Machines: Learning with many relevant features. *ECML-98*.
5. Lang, K. (1995). Newsweeder: Learning to filter netnews. In *Proceedings of ICML-95*, pp. 331–339.
6. Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine. https://archive.ics.uci.edu/ml
7. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.

---

## Appendix A — Code Snippets

### A.1 Preprocessing Pipeline

```python
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)           # remove mentions
    text = re.sub(r"[^a-z\s]", " ", text)           # keep only letters
    return re.sub(r"\s+", " ", text).strip()

def tokenize_and_stem(text: str) -> str:
    tokens = text.split()
    tokens = [STEMMER.stem(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)
```

### A.2 Rocchio Centroid Computation (Manual Implementation)

```python
def fit(self, X_train, y_train):
    self.classes_ = np.unique(y_train)
    X = X_train.toarray()
    positive_centroids = np.zeros((len(self.classes_), X.shape[1]))

    for idx, c in enumerate(self.classes_):
        mask = (y_train == c)
        if mask.sum() > 0:
            positive_centroids[idx] = X[mask].mean(axis=0)

    # L2-normalize for cosine similarity
    norms = np.linalg.norm(positive_centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1
    self.centroids_ = positive_centroids / norms
```

### A.3 Evaluation Metrics

```python
def compute_metrics(y_true, y_pred, average="weighted") -> dict:
    return {
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average),
        "Recall"   : recall_score(y_true, y_pred, average=average),
        "F1-Score" : f1_score(y_true, y_pred, average=average),
    }
```

### A.4 Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Download KOS and Enron datasets
python3 download_datasets.py

# Run all experiments
python3 main.py --all

# Run a single dataset
python3 main.py --dataset newsgroups
python3 main.py --dataset kos
```

---

*Report prepared by Group: Saad Hanif Taj (2022509), Ahmed Ali (2022054), Aiza Azeem (2022077)*
*Course: CS444 — Information Retrieval | Instructor: Dr. Zoya*

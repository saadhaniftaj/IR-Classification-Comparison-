"""
main.py
-------
Entry point for the CS444 IR Semester Project.

Datasets (all built-in, zero downloads needed):
    --dataset newsgroups   20 Newsgroups — full 20 categories (18,846 docs)
    --dataset newsgroups4  20 Newsgroups — 4 coarse super-categories (easier comparison)
    --dataset kos          KOS Blog Posts — UCI BoW with KMeans topic labels
    --dataset twitter      Health News Twitter (manual download to data/raw/Health-Tweets/)
    --all                  Run newsgroups + newsgroups4 + kos (all auto-available)

Usage:
    python3 main.py --all
    python3 main.py --dataset newsgroups
    python3 main.py --dataset newsgroups4
    python3 main.py --dataset kos
"""

import os
import sys
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing  import (load_twitter_health, load_bag_of_words,
                             load_csv_generic, prepare_dataset, full_pipeline)
from naive_bayes    import NaiveBayesClassifier
from rocchio        import RocchioClassifier
from knn            import KNNClassifier
from evaluate       import (compute_metrics, full_report, compare_classifiers,
                             plot_comparison, plot_confusion_matrix)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data", "raw")


# ──────────────────────────────────────────────────────────────────────────────
# Core runner
# ──────────────────────────────────────────────────────────────────────────────

def run_dataset(dataset_name: str, df, classifiers: list):
    """Train each classifier, evaluate, save results and plots."""
    print(f"\n{'='*62}")
    print(f"  DATASET : {dataset_name}")
    print(f"  Samples : {len(df):,}   |   Classes : {df['label'].nunique()}")
    print(f"  Class distribution:")
    for lbl, cnt in df['label'].value_counts().items():
        print(f"           {str(lbl):<25} {cnt:>5} docs")
    print(f"{'='*62}")

    X_train, X_test, y_train, y_test, le, vectorizer = prepare_dataset(df)
    print(f"  Train : {X_train.shape[0]:,}  |  Test : {X_test.shape[0]:,}  "
          f"|  Features : {X_train.shape[1]:,}")

    results = []
    for clf in classifiers:
        print(f"\n  ── {clf.name}")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_t = time.time() - t0

        t1 = time.time()
        y_pred = clf.predict(X_test)
        pred_t = time.time() - t1

        m = compute_metrics(y_test, y_pred)
        m["Classifier"] = clf.name
        results.append(m)

        print(f"     Train {train_t:.2f}s | Predict {pred_t:.2f}s")
        print(f"     Accuracy : {m['Accuracy']:.4f}  Precision : {m['Precision']:.4f}  "
              f"Recall : {m['Recall']:.4f}  F1 : {m['F1-Score']:.4f}")
        print(full_report(y_test, y_pred, le))

        plot_confusion_matrix(y_test, y_pred, clf.name, dataset_name, le, RESULTS_DIR)

    ordered = [{"Classifier": r["Classifier"], "Accuracy": r["Accuracy"],
                 "Precision": r["Precision"], "Recall": r["Recall"],
                 "F1-Score":  r["F1-Score"]} for r in results]

    df_cmp = compare_classifiers(ordered, dataset_name, RESULTS_DIR)
    plot_comparison(df_cmp, dataset_name, RESULTS_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_newsgroups():
    """
    20 Newsgroups (full) — 20 categories, 18,846 documents.
    Built into sklearn — no download needed.
    """
    import pandas as pd
    from sklearn.datasets import fetch_20newsgroups
    print("  Fetching 20 Newsgroups (all 20 categories) from sklearn …")
    data = fetch_20newsgroups(subset="all",
                              remove=("headers", "footers", "quotes"),
                              random_state=42)
    df = pd.DataFrame({"text": data.data,
                       "label": [data.target_names[t] for t in data.target]})
    df["text"] = df["text"].apply(full_pipeline)
    df = df[df["text"].str.strip() != ""].dropna()
    return df


def load_newsgroups4():
    """
    20 Newsgroups with 4 coarse super-categories.
    Easier to classify — good for comparing algorithm sensitivity.
    """
    import pandas as pd
    from sklearn.datasets import fetch_20newsgroups

    # Map fine-grained labels → 4 coarse topics
    COARSE = {
        "comp.graphics"           : "Computers",
        "comp.os.ms-windows.misc" : "Computers",
        "comp.sys.ibm.pc.hardware": "Computers",
        "comp.sys.mac.hardware"   : "Computers",
        "comp.windows.x"          : "Computers",
        "rec.autos"               : "Recreation",
        "rec.motorcycles"         : "Recreation",
        "rec.sport.baseball"      : "Recreation",
        "rec.sport.hockey"        : "Recreation",
        "misc.forsale"            : "Recreation",
        "sci.crypt"               : "Science",
        "sci.electronics"         : "Science",
        "sci.med"                 : "Science",
        "sci.space"               : "Science",
        "alt.atheism"             : "Politics_Religion",
        "soc.religion.christian"  : "Politics_Religion",
        "talk.politics.guns"      : "Politics_Religion",
        "talk.politics.mideast"   : "Politics_Religion",
        "talk.politics.misc"      : "Politics_Religion",
        "talk.religion.misc"      : "Politics_Religion",
    }

    print("  Fetching 20 Newsgroups (4 coarse super-categories) from sklearn …")
    data = fetch_20newsgroups(subset="all",
                              remove=("headers", "footers", "quotes"),
                              random_state=42)
    labels = [COARSE[data.target_names[t]] for t in data.target]
    df = pd.DataFrame({"text": data.data, "label": labels})
    df["text"] = df["text"].apply(full_pipeline)
    df = df[df["text"].str.strip() != ""].dropna()
    return df


def load_kos():
    """
    KOS Blog Posts (UCI BoW) — labels derived via dominant term clusters.
    We use keyword matching on the vocab to assign 5 meaningful political topics.
    """
    import pandas as pd

    dw = os.path.join(DATA_DIR, "docword.kos.txt")
    vb = os.path.join(DATA_DIR, "vocab.kos.txt")

    if not (os.path.exists(dw) and os.path.exists(vb)):
        raise FileNotFoundError(
            "KOS dataset missing. Run: python3 download_datasets.py"
        )

    # Read vocab
    with open(vb) as f:
        vocab = [line.strip() for line in f]

    # Political topic keyword sets for labeling
    TOPIC_KEYWORDS = {
        "War_Iraq"      : {"war", "iraq", "soldier", "militari", "weapon", "bush", "saddam"},
        "Election_Vote" : {"elect", "vote", "democrat", "republican", "campaign", "senat", "presid"},
        "Terror_Security": {"terror", "secur", "attack", "al", "qaeda", "bomb", "threat"},
        "Media_Press"   : {"media", "press", "report", "news", "journal", "cnn", "fox"},
        "Economy_Tax"   : {"econom", "tax", "budget", "deficit", "spend", "job", "unemploy"},
    }

    doc_words  = {}
    doc_counts = {}  # doc_id → {word: count}
    with open(dw) as f:
        n_docs = int(f.readline())
        f.readline(); f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                did, wid, cnt = int(parts[0]), int(parts[1]), int(parts[2])
                word = vocab[wid - 1]
                doc_counts.setdefault(did, {})[word] = cnt

    rows = []
    for did, wc in doc_counts.items():
        # Score each topic by keyword overlap weighted by term count
        scores = {}
        for topic, kws in TOPIC_KEYWORDS.items():
            scores[topic] = sum(wc.get(k, 0) for k in kws)
        best = max(scores, key=scores.get)
        # Only label if at least one keyword matched
        label = best if scores[best] > 0 else "Other"
        text  = full_pipeline(" ".join(
            w for w, c in wc.items() for _ in range(min(c, 5))  # cap repeat to 5
        ))
        if text.strip():
            rows.append({"label": label, "text": text})

    df = pd.DataFrame(rows)
    # Drop "Other" if too large compared to named topics
    named = df[df["label"] != "Other"]
    return named if len(named) > 500 else df


def load_twitter():
    """Health News in Twitter — requires manual download."""
    import pandas as pd

    twitter_dir = os.path.join(DATA_DIR, "Health-Tweets")
    txt_files   = [f for f in os.listdir(twitter_dir) if f.endswith(".txt")] \
                  if os.path.isdir(twitter_dir) else []

    if not txt_files:
        raise FileNotFoundError(
            "Twitter dataset missing.\n"
            "Download from: https://www.kaggle.com/datasets/gauravduttakiit/health-news-in-twitter\n"
            f"Place .txt files in: {twitter_dir}"
        )

    frames = []
    for fname in txt_files:
        source = fname.replace(".txt", "")
        path   = os.path.join(twitter_dir, fname)
        try:
            df_part = pd.read_csv(path, sep="|", header=None,
                                  names=["id", "datetime", "text"],
                                  encoding="latin-1", on_bad_lines="skip")
            df_part["label"] = source
            frames.append(df_part[["label", "text"]])
        except Exception as e:
            print(f"  [WARN] {fname}: {e}")

    if not frames:
        raise ValueError("No Twitter files could be loaded.")

    df = pd.concat(frames, ignore_index=True).dropna(subset=["text"])
    df["text"] = df["text"].apply(full_pipeline)
    df = df[df["text"].str.strip() != ""]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Classifier factory
# ──────────────────────────────────────────────────────────────────────────────

def build_classifiers():
    return [
        NaiveBayesClassifier(alpha=1.0, variant="complement"),
        RocchioClassifier(alpha=1.0, beta=0.0, metric="cosine"),
        KNNClassifier(k=5, weights="distance"),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

DATASET_MAP = {
    "newsgroups" : ("20_Newsgroups_Full",    load_newsgroups),
    "newsgroups4": ("20_Newsgroups_4Topics", load_newsgroups4),
    "kos"        : ("KOS_Blog_Posts",        load_kos),
    "twitter"    : ("Health_News_Twitter",   load_twitter),
}


def main():
    parser = argparse.ArgumentParser(description="CS444 IR Project — Classifier Comparison")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset",
                       choices=list(DATASET_MAP.keys()),
                       help="Run a single dataset")
    group.add_argument("--all", action="store_true",
                       help="Run all auto-available datasets (newsgroups, newsgroups4, kos)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --all runs 3 properly-labeled datasets (no manual download required)
    if args.all:
        to_run = ["newsgroups", "newsgroups4", "kos"]
    else:
        to_run = [args.dataset]

    overall_start = time.time()
    for key in to_run:
        name, loader = DATASET_MAP[key]
        try:
            print(f"\nLoading {name} …")
            df = loader()
            run_dataset(name, df, build_classifiers())
        except FileNotFoundError as e:
            print(f"\n[SKIP] {name}: {e}\n")
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}\n")
            raise

    elapsed = time.time() - overall_start
    print(f"\n{'='*62}")
    print(f"  Finished in {elapsed:.1f}s")
    print(f"  Results saved to: results/")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()

"""
download_datasets.py
--------------------
Downloads all three datasets required for the CS444 IR project.

Usage:
    python3 download_datasets.py

Downloads:
    1. Health News in Twitter  (via ucimlrepo API  — dataset id=438)
    2. Bag of Words – KOS blog posts               (via direct URL)
    3. Bag of Words – Enron emails                 (via direct URL)
"""

import os
import gzip
import shutil
import urllib.request

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
TWITTER_DIR = os.path.join(DATA_DIR, "Health-Tweets")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TWITTER_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: str, label: str):
    if os.path.exists(dest):
        print(f"  [SKIP] {label} — already exists")
        return True
    print(f"  [↓]   {label} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1_048_576
        print(f"done ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"FAILED → {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def decompress_gz(gz_path: str, out_path: str):
    if os.path.exists(out_path):
        print(f"  [SKIP] {os.path.basename(out_path)} — already extracted")
        return
    print(f"  [x]   Extracting {os.path.basename(gz_path)} ...", end=" ", flush=True)
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    size_mb = os.path.getsize(out_path) / 1_048_576
    print(f"done ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Health News in Twitter  (ucimlrepo)
# ──────────────────────────────────────────────────────────────────────────────

def download_twitter():
    print("\n[1/3]  Health News in Twitter (ucimlrepo API)")

    # Check if already saved
    existing = [f for f in os.listdir(TWITTER_DIR) if f.endswith(".txt")]
    if existing:
        print(f"  [SKIP] {len(existing)} .txt files already present in Health-Tweets/")
        return

    try:
        from ucimlrepo import fetch_ucirepo
        print("  Fetching from UCI repository (id=438) …", end=" ", flush=True)
        ds = fetch_ucirepo(id=438)
        print("done")

        # The dataset returns a DataFrame; 'Channel' is the source/label column
        import pandas as pd
        X = ds.data.features   # columns may include: ChannelName, PublishDate, NewsHeadline
        y = ds.data.targets    # label/source column

        # Merge features + targets into one frame for easy handling
        df = pd.concat([X, y], axis=1)
        print(f"  Loaded {len(df)} rows | columns: {list(df.columns)}")

        # Save one .txt file per unique channel (pipe-delimited: id|datetime|text)
        # Detect the columns dynamically
        cols = [c.lower() for c in df.columns]

        # Map column positions
        def find_col(candidates):
            for c in candidates:
                for real in df.columns:
                    if c in real.lower():
                        return real
            return None

        label_col = find_col(["channel", "source", "label", "outlet"])
        text_col  = find_col(["headline", "text", "tweet", "news"])
        date_col  = find_col(["date", "time", "publish"])

        if label_col is None or text_col is None:
            # Fallback: dump everything into one file
            out = os.path.join(TWITTER_DIR, "all_health_tweets.txt")
            df.to_csv(out, sep="|", index=False, header=False)
            print(f"  Saved combined file → {out}")
        else:
            for channel, grp in df.groupby(label_col):
                safe = str(channel).replace("/", "_").replace(" ", "_")
                out  = os.path.join(TWITTER_DIR, f"{safe}.txt")
                grp[[label_col, date_col or label_col, text_col]].to_csv(
                    out, sep="|", index=False, header=False
                )
                print(f"  Saved {len(grp):>5} rows → {safe}.txt")

        print(f"  [✓] Twitter dataset saved to: {TWITTER_DIR}")

    except Exception as e:
        print(f"\n  [ERROR] ucimlrepo fetch failed: {e}")
        print("  Falling back to Kaggle mirror …")
        _download_twitter_kaggle_fallback()


def _download_twitter_kaggle_fallback():
    """
    Last-resort: point the user to the Kaggle mirror since UCI
    direct file hosting changed.
    """
    print("""
  ┌─────────────────────────────────────────────────────────────┐
  │  Manual download required for Health News in Twitter:       │
  │                                                             │
  │  1. Go to: https://www.kaggle.com/datasets/               │
  │            gauravduttakiit/health-news-in-twitter           │
  │  2. Download the ZIP and extract                            │
  │  3. Copy the .txt files into:                               │
  │       data/raw/Health-Tweets/                               │
  └─────────────────────────────────────────────────────────────┘
""")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Bag of Words – KOS Blog Posts
# ──────────────────────────────────────────────────────────────────────────────

def download_kos():
    print("\n[2/3]  Bag of Words — KOS Blog Posts")
    BASE = "https://archive.ics.uci.edu/static/public/164/bag+of+words.zip"

    # Try direct individual file URLs (new UCI static hosting)
    files = [
        ("https://archive.ics.uci.edu/static/public/164/data/docword.kos.txt.gz",
         "docword.kos.txt.gz", "docword.kos.txt"),
        ("https://archive.ics.uci.edu/static/public/164/data/vocab.kos.txt",
         "vocab.kos.txt", None),
    ]

    for url, fname, extracted in files:
        dest = os.path.join(DATA_DIR, fname)
        ok = download_file(url, dest, fname)
        if ok and extracted:
            decompress_gz(dest, os.path.join(DATA_DIR, extracted))


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Bag of Words – Enron Emails
# ──────────────────────────────────────────────────────────────────────────────

def download_enron():
    print("\n[3/3]  Bag of Words — Enron Emails")

    files = [
        ("https://archive.ics.uci.edu/static/public/164/data/docword.enron.txt.gz",
         "docword.enron.txt.gz", "docword.enron.txt"),
        ("https://archive.ics.uci.edu/static/public/164/data/vocab.enron.txt",
         "vocab.enron.txt", None),
    ]

    for url, fname, extracted in files:
        dest = os.path.join(DATA_DIR, fname)
        ok = download_file(url, dest, fname)
        if ok and extracted:
            decompress_gz(dest, os.path.join(DATA_DIR, extracted))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CS444 IR Project — Dataset Downloader")
    print("=" * 60)

    download_twitter()
    download_kos()
    download_enron()

    print("\n" + "=" * 60)
    print("  Verifying files …")
    expected = [
        ("data/raw/docword.kos.txt",   "KOS docword"),
        ("data/raw/vocab.kos.txt",     "KOS vocab"),
        ("data/raw/docword.enron.txt", "Enron docword"),
        ("data/raw/vocab.enron.txt",   "Enron vocab"),
    ]
    all_ok = True
    for rel_path, label in expected:
        full = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
        if os.path.exists(full):
            mb = os.path.getsize(full) / 1_048_576
            print(f"  [✓]  {label:<20} {mb:>7.1f} MB")
        else:
            print(f"  [✗]  {label:<20} MISSING")
            all_ok = False

    twitter_txts = [f for f in os.listdir(TWITTER_DIR) if f.endswith(".txt")]
    if twitter_txts:
        print(f"  [✓]  Twitter files       {len(twitter_txts):>7} .txt files")
    else:
        print("  [✗]  Twitter files        MISSING — see manual instructions above")
        all_ok = False

    print("=" * 60)
    if all_ok:
        print("  All datasets ready!  Run:  python3 main.py --all")
    else:
        print("  Some datasets missing. Check errors above.")
    print("=" * 60 + "\n")

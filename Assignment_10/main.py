"""Lab 10: Amazon sentiment, Word2Vec/GloVe/FastText, analogies, neural networks."""
from pathlib import Path
import re, string, zipfile, urllib.request, gzip, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EMB = ROOT / "embeddings"
RESULTS = ROOT / "results"
for d in (DATA, EMB, RESULTS):
    d.mkdir(exist_ok=True)

SEED = 42
SAMPLE_TRAIN = 10000
SAMPLE_TEST = 1000

# ---------- Data loading and preprocessing ----------
def find_csvs():
    csvs = list(DATA.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No CSV files in data/. Run python download_data.py first.")
    return csvs

def load_csvs():
    files = find_csvs()
    train_file = next((f for f in files if "train" in f.name.lower()), files[0])
    test_file = next((f for f in files if "test" in f.name.lower() and f != train_file), None)
    train = pd.read_csv(train_file, header=None)
    test = pd.read_csv(test_file, header=None) if test_file else None
    return train, test

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # Remove punctuation and numbers
    tokens = text.split()
    stop_words = set(ENGLISH_STOP_WORDS)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def prepare_data():
    train_df, test_df = load_csvs()
    # Dataset format: label, title, review
    train_labels = train_df.iloc[:, 0]
    train_reviews = train_df.iloc[:, 2] if train_df.shape[1] > 2 else train_df.iloc[:, 1]
    
    test_labels = test_df.iloc[:, 0]
    test_reviews = test_df.iloc[:, 2] if test_df.shape[1] > 2 else test_df.iloc[:, 1]

    # Preprocess reviews
    texts = train_reviews.astype(str).apply(preprocess).tolist()
    test_text = test_reviews.astype(str).apply(preprocess).tolist()

    y = train_labels.astype(int).values
    ytest = test_labels.astype(int).values
    
    # Ensure 0-indexed binary classes (0 and 1)
    if set(np.unique(y)).issubset({1, 2}):
        y = y - 1
        ytest = ytest - 1

    return texts, y, test_text, ytest

# ---------- Embedding download/loading ----------
def download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading", url)
    urllib.request.urlretrieve(url, dest)
    return dest

def load_word2vec():
    """Google News Word2Vec vectors (300d)."""
    from gensim.models import KeyedVectors
    print("Loading/download model: word2vec-google-news-300")
    
    # Check existing pre-saved model
    model_path = EMB / "word2vec" / "word2vec-google-news-300.model"
    if model_path.exists():
        return KeyedVectors.load(str(model_path), mmap="r")
    
    bin_path = EMB / "GoogleNews-vectors-negative300.bin.gz"
    if bin_path.exists():
        return KeyedVectors.load_word2vec_format(str(bin_path), binary=True)
        
    import gensim.downloader as api
    return api.load("word2vec-google-news-300")

def load_glove():
    """GloVe 50d vectors."""
    from gensim.models import KeyedVectors
    print("Loading/download model: glove-wiki-gigaword-50")
    
    txt = EMB / "glove" / "glove.6B.50d.txt"
    if not txt.exists():
        txt = EMB / "glove.6B.50d.txt"
        
    if txt.exists():
        try:
            return KeyedVectors.load_word2vec_format(str(txt), binary=False)
        except Exception:
            return KeyedVectors.load_word2vec_format(str(txt), binary=False, no_header=True)
            
    import gensim.downloader as api
    return api.load("glove-wiki-gigaword-50")

def load_fasttext():
    """FastText wiki-news vectors (300d)."""
    from gensim.models import KeyedVectors
    print("Loading/download model: fasttext-wiki-news-subwords-300")
    
    vec = EMB / "fasttext" / "wiki-news-300d-1M-subword.vec"
    if not vec.exists():
        vec = EMB / "wiki-news-300d-1M.vec"
        
    if vec.exists():
        return KeyedVectors.load_word2vec_format(str(vec), binary=False)
        
    import gensim.downloader as api
    return api.load("fasttext-wiki-news-subwords-300")

def get_word_vector(model, word):
    try:
        return model[word]
    except KeyError:
        return None

def sentence_vector(sentence, model, weighted=False, tfidf_scores=None):
    words = sentence if isinstance(sentence, list) else sentence.split()
    vectors, weights = [], []
    for w in words:
        v = get_word_vector(model, w)
        if v is not None:
            vectors.append(v)
            weights.append(tfidf_scores.get(w, 1.0) if weighted and tfidf_scores else 1.0)
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if weighted:
        weights = weights / (weights.sum() + 1e-12)
        return np.average(np.asarray(vectors), axis=0, weights=weights)
    return np.mean(vectors, axis=0)

# ---------- Q2 analogy evaluation ----------
def run_example_analogies(model, name):
    examples = [("good", "bad", "excellent"), ("delivery", "late", "fast")]
    rows = []
    for a, b, c in examples:
        try:
            # b - a + c, as requested in the lab prompt
            pred = model.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]
            rows.append({"Embedding": name, "Analogy": f"{b} - {a} + {c}", "Prediction": pred,
                         "Expected": "(check manually)", "Result": "See prediction"})
            print(f"{name}: {b} - {a} + {c} => {pred}")
        except Exception as e:
            print(f"Could not calculate {b} - {a} + {c}: {e}")
    return rows

def run_analogies(model, name):
    analogy_file = EMB / "questions-words.txt"
    if not analogy_file.exists():
        download("https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt", analogy_file)
    correct = total = 0
    # Evaluate a capped number of analogies to keep runtime reasonable.
    selected = []
    for line in analogy_file.read_text(errors="ignore").splitlines():
        if line.startswith(":"):
            section = line.lower()
            continue
        parts = line.strip().lower().split()
        if len(parts) == 4 and ("gram" not in section if 'section' in locals() else True):
            selected.append(parts)
        if len(selected) >= 1000:
            break
    for a, b, c, d in selected:
        try:
            pred = model.most_similar(positive=[b, c], negative=[a], topn=1)[0][0].lower()
            correct += int(pred == d)
            total += 1
        except Exception:
            continue
    acc = correct / total if total else 0
    print(f"{name} analogy accuracy: {acc:.4f} ({correct}/{total})")
    return acc

# ---------- Q1/Q3 model evaluation ----------
def train_eval_nn(Xtr, ytr, Xte, yte, name):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
    model = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(5):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"{name} epoch {epoch+1}/5 loss={total/len(loader):.4f}")
    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(device)).argmax(1).cpu().numpy()
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro"), pred

def main():
    texts, y, test_text, ytest = prepare_data()
    print(f"Samples: train={len(texts)}, test={len(test_text)}")
    print("Q1: Loading embedding models (Google Word2Vec, GloVe, FastText). Downloads are large.")
    loaders = {"Word2Vec": load_word2vec, "GloVe": load_glove, "FastText": load_fasttext}
    results, analogy_results, example_analogy_results = [], [], []
    for name, loader in loaders.items():
        try:
            print(f"\n===== {name} =====")
            model = loader()
            print("Dimension:", model.vector_size)
            
            # Q2 analogy examples
            example_analogy_results.extend(run_example_analogies(model, name))
            
            # Q3 simple mean and TF-IDF weighted mean
            for weighted in (False, True):
                label = "IDF Weighted Average" if weighted else "Simple Average"
                if weighted:
                    train_strings = [" ".join(t) if isinstance(t, list) else t for t in texts]
                    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
                    tfidf.fit(train_strings)
                    scores = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
                else:
                    scores = None
                Xtr_nn = np.stack([sentence_vector(s, model, weighted, scores) for s in texts])
                Xte_nn = np.stack([sentence_vector(s, model, weighted, scores) for s in test_text])
                
                if not weighted:
                    np.savez_compressed(RESULTS / f"{name.lower()}_sentence_vectors.npz",
                                        X_train=Xtr_nn, y_train=y, X_test=Xte_nn, y_test=ytest)
                                        
                acc, f1, pred = train_eval_nn(Xtr_nn, y, Xte_nn, ytest, f"{name}-{label}")
                results.append({"Embedding": name, "Representation": label, "Accuracy": acc, "MacroF1": f1})
                print(f"{name} | {label} | Accuracy={acc:.4f} | Macro-F1={f1:.4f}")
                
            # Q2 full analogy benchmark
            try:
                acc_analogy = run_analogies(model, name)
                analogy_results.append({"Embedding": name, "AnalogyAccuracy": acc_analogy})
            except Exception as e:
                print("Analogy evaluation skipped:", e)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR with {name}: {e}")
            print("Continuing to next embedding model.")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS / "q3_results.csv", index=False)
        print("\nQ3 RESULTS\n", df)
        try:
            ax = df.pivot(index="Embedding", columns="Representation", values="MacroF1").plot(kind="bar", figsize=(9, 5))
            ax.set_ylabel("Macro F1")
            ax.set_title("Embedding and Sentence Representation Comparison")
            plt.tight_layout()
            plt.savefig(RESULTS / "q3_macro_f1.png", dpi=200)
            plt.close()
        except Exception as e:
            print("Plotting error:", e)

    if analogy_results:
        adf = pd.DataFrame(analogy_results)
        adf.to_csv(RESULTS / "q2_analogy_results.csv", index=False)
        print("\nQ2 RESULTS\n", adf)

    if example_analogy_results:
        pd.DataFrame(example_analogy_results).to_csv(RESULTS / "q2_example_analogies.csv", index=False)
    print("Results saved to:", RESULTS)

if __name__ == "__main__":
    main()


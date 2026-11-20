# Assignment 10: Amazon Reviews with Pretrained Word Embeddings

The runner in `amazon_embeddings_assignment10.py` implements the three parts of
the assignment. It cleans reviews by lowercasing, removing punctuation and
digits, tokenizing alphabetic words, and removing scikit-learn's English
stopword list. It takes reproducible stratified samples of 10,000 training and
1,000 test rows by default. Review and label columns are inferred from common
names, or can be set explicitly.

## Inputs

Download the Amazon Reviews `train.csv` and `test.csv` from the Kaggle dataset.
The script supports the common headerless format (`label,title,review`) as
well as headered CSV files with label and review/text columns. Download the
Google News Word2Vec, GloVe, and FastText vectors separately. The vectors are
large, so this project does not download or redistribute them. Pass extracted
files to the script:

* Word2Vec: Gensim native model, or word2vec-format binary (`.bin`) or text.
* GloVe: Gensim native model or text vectors.
* FastText: extracted `wiki-news-300d-1M-subword.vec` text vectors or a Gensim
  FastText model. Do not pass the `.zip`; extract it first.

The loader detects GloVe's standard headerless text format and tries it with
`no_header=True` after the normal word2vec text loader fails.

## Run

Use 64-bit Python 3.13 on Windows. Gensim contains compiled C extensions, and
its current PyPI release has a Windows wheel for Python 3.13 but not 3.14; pip
therefore tries to compile it from source under Python 3.14 and fails. Python
3.13 is installed on this machine. From the `Assignment_10` folder, make a
separate virtual environment and install the pinned Gensim release:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the program with that environment's Python (also from `Assignment_10`):

```powershell
.\.venv\Scripts\python.exe amazon_embeddings_assignment10.py `
  --train path\to\train.csv --test path\to\test.csv `
  --word2vec path\to\GoogleNews-vectors-negative300.bin `
  --glove path\to\glove.6B.300d.txt `
  --fasttext path\to\wiki-news-300d-1M-subword.vec `
  --analogies path\to\questions-words.txt
```

Do not run `python amazon_embeddings_assignment10.py` unless `python` points
to the Python 3.13 virtual environment; the system Python 3.14 cannot import
Gensim.

Use `--word2vec-format text` for text vectors or `--word2vec-binary` for a
binary word2vec file with a nonstandard extension. Use `--*-format gensim` for
Gensim native model files. Set `--review-col` and `--label-col` for custom CSV
headers.

## Method and outputs

Q1 computes sentence vectors as the mean of in-vocabulary word vectors. The two
custom examples follow the arithmetic order shown (`a - b + c`). Q2 evaluates
the standard 3CosAdd benchmark operation (`b + c - a`) over all parseable rows
in `questions-words.txt`; OOV questions are excluded from accuracy and counted
in the output. Q3 fits a one-hidden-layer `MLPClassifier` for each embedding and
each representation: plain mean and TF-IDF-weighted mean. TF-IDF is fitted only
on the sampled training reviews. Macro-F1 is the primary comparison metric.

The output directory contains `macro_f1_results.csv`, `analogy_results.json`,
`run_info.json`, and one compressed `*_sentence_vectors.npz` per embedding.
Each vector archive contains average and TF-IDF-weighted train/test vectors and
their labels. The two prompt analogies are always evaluated; the full
`questions-words.txt` benchmark is run when `--analogies` is supplied. Results
depend on the precise vector files and their vocabulary coverage. For
meaningful comparisons, use 300-dimensional files for all three embeddings and
include the source files and sample seed when reporting results.

"""Train unigram through quadrigram Hindi language models with Laplace smoothing.

The program streams the Assignment 1 corpus: it never loads the 26 GB source or
the complete set of n-grams into memory.  Counts are stored in SQLite, so the
resulting database is also the trained model.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Assignment_1"))
from tokenizer import tokenize_paragraph  # noqa: E402

START = "<s>"
END = "</s>"
UNK = "<unk>"
SEPARATOR = "\x1f"


def sentences(source: Path) -> Iterator[list[str]]:
    """Yield the already-defined Assignment 1 tokens sentence by sentence."""
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for paragraph in handle:
            yield from tokenize_paragraph(paragraph)


def write_splits(source: Path, output: Path, train_size: int) -> None:
    """Create deterministic JSONL splits containing token lists."""
    output.mkdir(parents=True, exist_ok=True)
    targets = (("train.jsonl", train_size), ("development.jsonl", 1_000), ("test.jsonl", 1_000))
    iterator = sentences(source)
    for filename, required in targets:
        path = output / filename
        written = 0
        with path.open("w", encoding="utf-8") as handle:
            while written < required:
                try:
                    tokens = next(iterator)
                except StopIteration as error:
                    raise ValueError(f"Corpus has only {sum(x[1] for x in targets[:targets.index((filename, required))]) + written:,} sentences; at least {train_size + 2_000:,} are required.") from error
                handle.write(json.dumps(tokens, ensure_ascii=False) + "\n")
                written += 1
        print(f"Wrote {written:,} sentences to {path.name}")


def read_split(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def prepare_database(database: Path, vocabulary: Counter[str]) -> sqlite3.Connection:
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("CREATE TABLE vocabulary (token TEXT PRIMARY KEY, count INTEGER NOT NULL)")
    connection.execute("CREATE TABLE ngram_counts (n INTEGER NOT NULL, context TEXT NOT NULL, word TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY (n, context, word)) WITHOUT ROWID")
    connection.execute("CREATE TABLE context_counts (n INTEGER NOT NULL, context TEXT NOT NULL, count INTEGER NOT NULL, PRIMARY KEY (n, context)) WITHOUT ROWID")
    connection.executemany("INSERT INTO vocabulary VALUES (?, ?)", vocabulary.items())
    # Values predicted by a model are training words, sentence end, and <unk>.
    connection.execute("INSERT INTO metadata VALUES ('vocabulary_size', ?)", (str(len(vocabulary) + 2),))
    connection.execute("INSERT INTO metadata VALUES ('unknown_token', ?)", (UNK,))
    return connection


def add_counts(connection: sqlite3.Connection, ngrams: Counter[tuple[int, str, str]], contexts: Counter[tuple[int, str]]) -> None:
    connection.executemany(
        "INSERT INTO ngram_counts VALUES (?, ?, ?, ?) ON CONFLICT(n, context, word) DO UPDATE SET count=count+excluded.count",
        ((n, context, word, count) for (n, context, word), count in ngrams.items()),
    )
    connection.executemany(
        "INSERT INTO context_counts VALUES (?, ?, ?) ON CONFLICT(n, context) DO UPDATE SET count=count+excluded.count",
        ((n, context, count) for (n, context), count in contexts.items()),
    )


def train(train_path: Path, database: Path) -> None:
    vocabulary: Counter[str] = Counter()
    for tokens in read_split(train_path):
        vocabulary.update(tokens)
    connection = prepare_database(database, vocabulary)
    buffer_ngrams: Counter[tuple[int, str, str]] = Counter()
    buffer_contexts: Counter[tuple[int, str]] = Counter()
    sentence_count = 0
    for tokens in read_split(train_path):
        mapped = [token if token in vocabulary else UNK for token in tokens]
        for n in range(1, 5):
            padded = [START] * (n - 1) + mapped + [END]
            for index in range(n - 1, len(padded)):
                context = SEPARATOR.join(padded[index - n + 1:index])
                word = padded[index]
                buffer_ngrams[(n, context, word)] += 1
                buffer_contexts[(n, context)] += 1
        sentence_count += 1
        if sentence_count % 10_000 == 0:
            add_counts(connection, buffer_ngrams, buffer_contexts)
            connection.commit()
            buffer_ngrams.clear()
            buffer_contexts.clear()
            print(f"Counted {sentence_count:,} training sentences", flush=True)
    add_counts(connection, buffer_ngrams, buffer_contexts)
    connection.execute("INSERT INTO metadata VALUES ('training_sentences', ?)", (str(sentence_count),))
    connection.commit()
    connection.close()


class LaplaceLanguageModel:
    def __init__(self, database: Path):
        self.connection = sqlite3.connect(database)
        self.vocabulary_size = int(self.connection.execute("SELECT value FROM metadata WHERE key='vocabulary_size'").fetchone()[0])
        self.vocabulary = {row[0] for row in self.connection.execute("SELECT token FROM vocabulary")}

    def probability(self, word: str, context: Iterable[str] = ()) -> float:
        context = tuple(context)
        n = len(context) + 1
        if not 1 <= n <= 4:
            raise ValueError("Context must contain 0 to 3 tokens.")
        word = word if word in self.vocabulary or word == END else UNK
        normalized_context = tuple(token if token in self.vocabulary or token == START else UNK for token in context)
        context_key = SEPARATOR.join(normalized_context)
        row = self.connection.execute("SELECT count FROM ngram_counts WHERE n=? AND context=? AND word=?", (n, context_key, word)).fetchone()
        numerator = row[0] if row else 0
        row = self.connection.execute("SELECT count FROM context_counts WHERE n=? AND context=?", (n, context_key)).fetchone()
        denominator = row[0] if row else 0
        return (numerator + 1) / (denominator + self.vocabulary_size)

    def perplexity(self, split: Path, n: int) -> float:
        log_probability = 0.0
        predictions = 0
        for tokens in read_split(split):
            padded = [START] * (n - 1) + tokens + [END]
            for index in range(n - 1, len(padded)):
                log_probability += math.log(self.probability(padded[index], padded[index - n + 1:index]))
                predictions += 1
        return math.exp(-log_probability / predictions)

    def close(self) -> None:
        self.connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unigram, bigram, trigram and quadrigram Laplace language models.")
    parser.add_argument("--source", type=Path, default=ROOT / "Assignment_1" / "hi-1.txt")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--train-size", type=int, default=1_000_000)
    parser.add_argument("--rebuild-splits", action="store_true", help="Recreate the deterministic data splits.")
    args = parser.parse_args()
    if args.train_size < 1_000_000:
        parser.error("--train-size must be at least 1,000,000.")
    split_paths = [args.output / name for name in ("train.jsonl", "development.jsonl", "test.jsonl")]
    if args.rebuild_splits or not all(path.exists() for path in split_paths):
        write_splits(args.source, args.output, args.train_size)
    database = args.output / "laplace_language_models.sqlite3"
    train(split_paths[0], database)
    model = LaplaceLanguageModel(database)
    results = {
        split: {f"{n}-gram_perplexity": model.perplexity(path, n) for n in range(1, 5)}
        for split, path in (("development", split_paths[1]), ("test", split_paths[2]))
    }
    results["training_sentences"] = args.train_size
    results["development_sentences"] = 1_000
    results["test_sentences"] = 1_000
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    model.close()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

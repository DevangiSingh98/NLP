"""Evaluate Assignment 4's four n-gram models with add-K smoothing (K = 0.3)."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Assignment_4"))
from language_models import END, SEPARATOR, START, UNK, read_split  # noqa: E402


class AddKLanguageModel:
    """Read Assignment 4 counts and return add-K-smoothed probabilities."""

    def __init__(self, database: Path, k: float = 0.3) -> None:
        if k <= 0:
            raise ValueError("K must be positive.")
        self.connection = sqlite3.connect(database)
        self.k = k
        self.vocabulary_size = int(
            self.connection.execute("SELECT value FROM metadata WHERE key='vocabulary_size'").fetchone()[0]
        )
        self.vocabulary = {row[0] for row in self.connection.execute("SELECT token FROM vocabulary")}

    def probability(self, word: str, context: Iterable[str] = ()) -> float:
        """Return P_add-k(word | context) for unigram through quadrigram contexts."""
        context = tuple(context)
        n = len(context) + 1
        if not 1 <= n <= 4:
            raise ValueError("Context must contain 0 to 3 tokens.")
        word = word if word in self.vocabulary or word == END else UNK
        normalized_context = tuple(token if token in self.vocabulary or token == START else UNK for token in context)
        context_key = SEPARATOR.join(normalized_context)
        row = self.connection.execute(
            "SELECT count FROM ngram_counts WHERE n=? AND context=? AND word=?",
            (n, context_key, word),
        ).fetchone()
        ngram_count = row[0] if row else 0
        row = self.connection.execute(
            "SELECT count FROM context_counts WHERE n=? AND context=?", (n, context_key)
        ).fetchone()
        context_count = row[0] if row else 0
        return (ngram_count + self.k) / (context_count + self.k * self.vocabulary_size)

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
    parser = argparse.ArgumentParser(description="Evaluate Assignment 4 models with add-K smoothing.")
    parser.add_argument("--k", type=float, default=0.3, help="Add-K smoothing constant (default: 0.3).")
    parser.add_argument("--database", type=Path, default=ROOT / "Assignment_4" / "output" / "laplace_language_models.sqlite3")
    parser.add_argument("--splits", type=Path, default=ROOT / "Assignment_4" / "output")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "results.json")
    args = parser.parse_args()
    if not args.database.exists():
        parser.error(f"Assignment 4 model database not found: {args.database}")
    model = AddKLanguageModel(args.database, args.k)
    results: dict[str, object] = {
        "smoothing": "add-k",
        "k": args.k,
        "vocabulary_size": model.vocabulary_size,
    }
    for split in ("development", "test"):
        path = args.splits / f"{split}.jsonl"
        results[split] = {f"{n}-gram_perplexity": model.perplexity(path, n) for n in range(1, 5)}
    model.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

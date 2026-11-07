"""Evaluate the Assignment 4 n-gram counts with several smoothing methods.

The database is deliberately queried rather than copied into memory: the
Assignment 4 SQLite file remains the single trained model for every method.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Assignment_4"))
from language_models import END, SEPARATOR, START, UNK, read_split  # noqa: E402


class CountModel:
    """Common SQLite count access and perplexity evaluation."""

    def __init__(self, database: Path) -> None:
        self.connection = sqlite3.connect(database)
        self.vocabulary = {r[0] for r in self.connection.execute("SELECT token FROM vocabulary")}
        self.vocabulary_size = int(self.connection.execute(
            "SELECT value FROM metadata WHERE key='vocabulary_size'"
        ).fetchone()[0])
        self._distributions: dict[tuple[int, str], dict[str, int]] = {}

    def normalize(self, word: str, context: Iterable[str]) -> tuple[str, tuple[str, ...]]:
        word = word if word in self.vocabulary or word == END else UNK
        return word, tuple(x if x in self.vocabulary or x == START else UNK for x in context)

    def distribution(self, n: int, context: tuple[str, ...]) -> dict[str, int]:
        key = (n, SEPARATOR.join(context))
        if key not in self._distributions:
            self._distributions[key] = dict(self.connection.execute(
                "SELECT word, count FROM ngram_counts WHERE n=? AND context=?", key
            ))
        return self._distributions[key]

    def perplexity(self, split: Path, n: int) -> float:
        total_log_probability = 0.0
        predictions = 0
        for tokens in read_split(split):
            padded = [START] * (n - 1) + tokens + [END]
            for i in range(n - 1, len(padded)):
                probability = self.probability(padded[i], padded[i - n + 1:i])
                total_log_probability += math.log(max(probability, 1e-300))
                predictions += 1
        return math.exp(-total_log_probability / predictions)

    def close(self) -> None:
        self.connection.close()


class GoodTuringModel(CountModel):
    """Conditional Good--Turing smoothing with a per-context unseen reserve."""

    def __init__(self, database: Path, cutoff: int = 5) -> None:
        super().__init__(database)
        self.cutoff = cutoff
        self.count_of_counts: dict[int, Counter[int]] = {}
        for n in range(1, 5):
            self.count_of_counts[n] = Counter(dict(self.connection.execute(
                "SELECT count, COUNT(*) FROM ngram_counts WHERE n=? GROUP BY count", (n,)
            )))
        self._good_turing_parameters: dict[tuple[int, tuple[str, ...]], tuple[dict[str, float], float]] = {}

    def discounted_count(self, n: int, count: int) -> float:
        frequencies = self.count_of_counts[n]
        if 0 < count <= self.cutoff and frequencies[count + 1]:
            return min(float(count), (count + 1) * frequencies[count + 1] / frequencies[count])
        return float(count)

    def parameters(self, n: int, context: tuple[str, ...]) -> tuple[dict[str, float], float]:
        """Return seen-event probabilities and the probability of each unseen type."""
        key = (n, context)
        if key not in self._good_turing_parameters:
            counts = self.distribution(n, context)
            total = sum(counts.values())
            if not total:
                parameters = ({}, 1.0 / self.vocabulary_size)
            else:
                seen = {token: self.discounted_count(n, count) / total for token, count in counts.items()}
                unseen = max(0.0, 1.0 - sum(seen.values())) / max(1, self.vocabulary_size - len(counts))
                parameters = (seen, unseen)
            self._good_turing_parameters[key] = parameters
        return self._good_turing_parameters[key]

    def probability(self, word: str, context: Iterable[str] = ()) -> float:
        context = tuple(context)
        n = len(context) + 1
        word, context = self.normalize(word, context)
        seen, unseen = GoodTuringModel.parameters(self, n, context)
        return seen.get(word, unseen)


class InterpolatedModel(CountModel):
    """Linear interpolation of maximum-likelihood estimates at all lower orders.

    The unigram component is add-one smoothed so that an evaluation token mapped
    to ``<unk>`` retains non-zero probability.
    """

    # Weights run from highest order to unigram and sum to one.
    WEIGHTS = {2: (0.70, 0.30), 3: (0.60, 0.30, 0.10), 4: (0.50, 0.30, 0.15, 0.05)}

    def probability(self, word: str, context: Iterable[str] = ()) -> float:
        context = tuple(context)
        n = len(context) + 1
        if n == 1:
            raise ValueError("Interpolation is defined here for bigrams through quadgrams.")
        word, context = self.normalize(word, context)
        result = 0.0
        for order, weight in zip(range(n, 0, -1), self.WEIGHTS[n]):
            current_context = context[-(order - 1):] if order > 1 else ()
            counts = self.distribution(order, current_context)
            total = sum(counts.values())
            if order == 1:
                result += weight * (counts.get(word, 0) + 1) / (total + self.vocabulary_size)
            else:
                result += weight * (counts.get(word, 0) / total if total else 0.0)
        return result


class KatzBackoffModel(GoodTuringModel):
    """Katz backoff: discounted seen events, normalized lower-order backoff."""

    def __init__(self, database: Path, cutoff: int = 5) -> None:
        super().__init__(database, cutoff)
        # An unseen continuation of the same context shares the same alpha.
        # Caching it avoids recomputing the sum over every seen continuation.
        self._backoff_parameters: dict[tuple[int, tuple[str, ...]], tuple[dict[str, float], float]] = {}

    def parameters(self, n: int, context: tuple[str, ...]) -> tuple[dict[str, float], float]:
        key = (n, context)
        if key in self._backoff_parameters:
            return self._backoff_parameters[key]
        counts = self.distribution(n, context)
        total = sum(counts.values())
        if not total:
            parameters = ({}, 1.0)
        else:
            seen = {token: self.discounted_count(n, count) / total for token, count in counts.items()}
            lower_context = context[1:]
            lower_seen_mass = sum(self.probability(token, lower_context) for token in counts)
            alpha = max(0.0, 1.0 - sum(seen.values())) / max(1e-300, 1.0 - lower_seen_mass)
            parameters = (seen, alpha)
        self._backoff_parameters[key] = parameters
        return parameters

    def probability(self, word: str, context: Iterable[str] = ()) -> float:
        context = tuple(context)
        n = len(context) + 1
        word, context = self.normalize(word, context)
        if n == 1:
            return GoodTuringModel.probability(self, word, ())
        seen, alpha = self.parameters(n, context)
        if not seen:
            return self.probability(word, context[1:])
        if word in seen:
            return seen[word]
        return alpha * self.probability(word, context[1:])


class StupidBackoffModel(GoodTuringModel):
    """Brants et al.'s fixed-factor (0.4 by default) backoff."""

    def __init__(self, database: Path, alpha: float = 0.4) -> None:
        super().__init__(database)
        self.alpha = alpha

    def probability(self, word: str, context: Iterable[str] = ()) -> float:
        context = tuple(context)
        n = len(context) + 1
        word, context = self.normalize(word, context)
        if n == 1:
            return GoodTuringModel.probability(self, word, ())
        counts = self.distribution(n, context)
        total = sum(counts.values())
        if total and word in counts:
            return counts[word] / total
        return self.alpha * self.probability(word, context[1:])


class KneserNeyModel(CountModel):
    """Interpolated Kneser--Ney with absolute discounting."""

    def __init__(self, database: Path, discount: float = 0.75) -> None:
        super().__init__(database)
        if not 0 < discount < 1:
            raise ValueError("The Kneser--Ney discount must be between zero and one.")
        self.discount = discount
        self.continuations = dict(self.connection.execute(
            "SELECT word, COUNT(*) FROM ngram_counts WHERE n=2 GROUP BY word"
        ))
        self.continuation_total = sum(self.continuations.values())

    def probability(self, word: str, context: Iterable[str] = ()) -> float:
        context = tuple(context)
        n = len(context) + 1
        word, context = self.normalize(word, context)
        if n == 1:
            # Add one pseudo-continuation for every prediction token, including <unk>.
            return (self.continuations.get(word, 0) + 1) / (self.continuation_total + self.vocabulary_size)
        counts = self.distribution(n, context)
        total = sum(counts.values())
        if not total:
            return self.probability(word, context[1:])
        observed = max(counts.get(word, 0) - self.discount, 0.0) / total
        lambda_context = self.discount * len(counts) / total
        return observed + lambda_context * self.probability(word, context[1:])


def evaluate(model: CountModel, split: Path, orders: Iterable[int]) -> dict[str, float]:
    return {f"{n}-gram_perplexity": model.perplexity(split, n) for n in orders}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Assignment 4 counts using five smoothing techniques.")
    parser.add_argument("--database", type=Path, default=ROOT / "Assignment_4" / "output" / "laplace_language_models.sqlite3")
    parser.add_argument("--splits", type=Path, default=ROOT / "Assignment_4" / "output")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "output" / "smoothing_results.json")
    parser.add_argument("--good-turing-cutoff", type=int, default=5)
    parser.add_argument("--stupid-backoff-alpha", type=float, default=0.4)
    parser.add_argument("--kneser-ney-discount", type=float, default=0.75)
    args = parser.parse_args()
    if not args.database.exists() or args.database.stat().st_size == 0:
        parser.error(f"Trained Assignment 4 database is missing or empty: {args.database}")
    split_paths = {name: args.splits / f"{name}.jsonl" for name in ("development", "test")}
    if missing := [str(path) for path in split_paths.values() if not path.exists()]:
        parser.error("Missing split file(s): " + ", ".join(missing))
    factories = {
        "interpolated": (lambda: InterpolatedModel(args.database), range(2, 5)),
        "good_turing": (lambda: GoodTuringModel(args.database, args.good_turing_cutoff), range(1, 5)),
        "katz_backoff": (lambda: KatzBackoffModel(args.database, args.good_turing_cutoff), range(1, 5)),
        "stupid_backoff": (lambda: StupidBackoffModel(args.database, args.stupid_backoff_alpha), range(1, 5)),
        "kneser_ney": (lambda: KneserNeyModel(args.database, args.kneser_ney_discount), range(1, 5)),
    }
    results: dict[str, object] = {"parameters": {"good_turing_cutoff": args.good_turing_cutoff,
        "stupid_backoff_alpha": args.stupid_backoff_alpha, "kneser_ney_discount": args.kneser_ney_discount}}
    for name, (factory, orders) in factories.items():
        model = factory()
        results[name] = {split: evaluate(model, path, orders) for split, path in split_paths.items()}
        model.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

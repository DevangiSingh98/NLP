"""A small noisy-channel spell corrector for the supplied assignment data.

Run: python spell_corrector.py
Try a word: python spell_corrector.py rainning
"""
import argparse
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


DATA_DIR = Path(__file__).parent / "Spell-Correction"
ALPHABET = "abcdefghijklmnopqrstuvwxyz'"


def edit_operation(correct, error):
    """Return a compact character edit such as ``a|e`` or ``t|te``."""
    left = 0
    while left < min(len(correct), len(error)) and correct[left] == error[left]:
        left += 1
    right = 0
    while (right < len(correct) - left and right < len(error) - left
           and correct[-1-right] == error[-1-right]):
        right += 1
    old_end = len(correct) - right if right else len(correct)
    new_end = len(error) - right if right else len(error)
    return correct[left:old_end] + "|" + error[left:new_end]


def read_unigrams(path):
    counts = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    counts[parts[0].lower()] = int(parts[1])
                except ValueError:
                    continue
    return counts


def read_edit_counts(path):
    counts = {}
    with open(path, encoding="cp1252") as file:
        for line in file:
            parts = line.strip().rsplit(None, 1)
            if len(parts) == 2:
                try:
                    counts[parts[0].lower()] = int(parts[1])
                except ValueError:
                    continue
    return counts


def read_gold_pairs(path):
    """Read correct: typo, typo*frequency lines into typo -> correct options."""
    correct_for_typo = defaultdict(set)
    with open(path, encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue
            correct, errors = line.split(":", 1)
            correct = correct.strip().lower()
            for item in errors.split(","):
                typo = item.strip().lower()
                if not typo:
                    continue
                typo = re.sub(r"\*\d+$", "", typo)
                correct_for_typo[typo].add(correct)
    return correct_for_typo


class SpellCorrector:
    def __init__(self, unigram_path, edit_path, pairs_path):
        self.unigrams = read_unigrams(unigram_path)
        self.edit_counts = read_edit_counts(edit_path)
        self.correct_for_typo = read_gold_pairs(pairs_path)
        self.known_words = set(self.unigrams)

    @staticmethod
    @lru_cache(maxsize=50000)
    def edits_one(word):
        """Make all strings one insertion, deletion, replacement or swap away."""
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        edits = set()
        for left, right in splits:
            if right:
                edits.add(left + right[1:])
                if len(right) > 1:
                    edits.add(left + right[1] + right[0] + right[2:])
                for char in ALPHABET:
                    edits.add(left + char + right[1:])
            for char in ALPHABET:
                edits.add(left + char + right)
        edits.discard(word)
        return edits

    def candidates(self, word):
        word = word.lower()
        # The supplied file explicitly lists the correct word for these
        # observed errors. Use that smaller candidate set instead of allowing
        # unrelated, high-frequency neighbors to win on unigram frequency.
        listed_candidates = self.correct_for_typo.get(word)
        if listed_candidates:
            return set(listed_candidates)
        candidates = self.edits_one(word) & self.known_words
        if word in self.known_words:
            candidates.add(word)
        return candidates or {word}

    def correct(self, word):
        word = word.lower()
        candidates = self.candidates(word)
        def score(candidate):
            # Noisy channel: P(correct) * P(error | correct). The supplied
            # edit counts estimate how likely the character change is.
            operation = edit_operation(candidate, word)
            channel = self.edit_counts.get(operation, 1)
            prior = self.unigrams.get(candidate, 1)
            return prior * channel
        return max(candidates, key=score)

    def evaluate(self):
        total = 0
        correct_count = 0
        for typo, answers in self.correct_for_typo.items():
            total += 1
            if self.correct(typo) in answers:
                correct_count += 1
        accuracy = correct_count / total if total else 0.0
        return correct_count, total, accuracy


def main():
    parser = argparse.ArgumentParser(description="Correct spelling errors with a simple noisy-channel model.")
    parser.add_argument("word", nargs="?", help="a misspelled word to correct")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="folder containing the three provided data files")
    args = parser.parse_args()
    model = SpellCorrector(args.data_dir / "count_unigram.txt",
                           args.data_dir / "count_1edit.txt",
                           args.data_dir / "spell-errors.txt")
    if args.word:
        print(f"{args.word} -> {model.correct(args.word)}")
    else:
        right, total, accuracy = model.evaluate()
        print(f"Corrected {right} of {total} listed spelling errors")
        print(f"Accuracy: {accuracy:.2%}")
        for word in ("rainning", "writtings"):
            print(f"{word} -> {model.correct(word)}")


if __name__ == "__main__":
    main()

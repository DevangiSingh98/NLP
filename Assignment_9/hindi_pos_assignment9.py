#!/usr/bin/env python3
"""Compare MEMM, CRF, Naive Bayes, and Assignment 7 HMM POS taggers."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn_crfsuite
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Assignment_7.hindi_pos_tagger import (  # noqa: E402
    HMMTagger,
    NaiveBayesTagger,
    evaluate,
    read_data,
)


ALGORITHMS = (
    "Naive Bayes",
    "HMM forward Viterbi",
    "HMM backward Viterbi",
    "HMM forward-backward posterior decoding",
    "MEMM",
    "CRF",
)


def word_features(words, index):
    """Return the shared morphology, context, and length features."""
    word = words[index]
    features = {
        "current_word": word,
        "previous_word": words[index - 1] if index else "<BOS>",
        "next_word": words[index + 1] if index + 1 < len(words) else "<EOS>",
        "length_feature": "LESS" if len(word) <= 4 else "MORE",
    }
    for length in range(1, 5):
        features[f"prefix_{length}"] = (
            word[:length] if len(word) >= length else "NULL"
        )
        features[f"suffix_{length}"] = (
            word[-length:] if len(word) >= length else "NULL"
        )
    return features


def memm_features(words, index, previous_tag):
    features = word_features(words, index)
    features["previous_tag"] = previous_tag if previous_tag is not None else "<BOS_TAG>"
    return features


def sentence_feature_dicts(sentence):
    words = [word for word, _ in sentence]
    return [word_features(words, i) for i in range(len(words))]


class MEMMTagger:
    """A greedy MEMM using a maximum-entropy multiclass logistic classifier."""

    def __init__(self, max_iter=200):
        self.max_iter = max_iter
        self.vectorizer = DictVectorizer(sparse=True)
        self.classifier = LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            random_state=0,
        )

    def fit(self, sentences):
        feature_rows = []
        labels = []
        for sentence in sentences:
            words = [word for word, _ in sentence]
            previous_tag = None
            for index, (_, tag) in enumerate(sentence):
                feature_rows.append(memm_features(words, index, previous_tag))
                labels.append(tag)
                previous_tag = tag

        matrix = self.vectorizer.fit_transform(feature_rows)
        self.classifier.fit(matrix, labels)
        return self

    def predict_sentence(self, words):
        feature_rows = []
        previous_tag = None
        for index in range(len(words)):
            row = self.vectorizer.transform(
                [memm_features(words, index, previous_tag)]
            )
            previous_tag = self.classifier.predict(row)[0]
            feature_rows.append(previous_tag)
        return feature_rows

    def predict(self, sentences):
        return [
            self.predict_sentence([word for word, _ in sentence])
            for sentence in sentences
        ]


class CRFTagger:
    def __init__(self, max_iter=100):
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=max_iter,
            all_possible_transitions=True,
        )

    def fit(self, sentences):
        x_train = [sentence_feature_dicts(sentence) for sentence in sentences]
        y_train = [[tag for _, tag in sentence] for sentence in sentences]
        self.model.fit(x_train, y_train)
        return self

    def predict(self, sentences):
        x_data = [sentence_feature_dicts(sentence) for sentence in sentences]
        return self.model.predict(x_data)


def write_results(results, train, dev, test, output_dir, tags, timings):
    os.makedirs(output_dir, exist_ok=True)

    with open(
        os.path.join(output_dir, "summary_metrics.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["split", "algorithm", "accuracy", "macro_f1", "weighted_f1"])
        for split, result in results:
            writer.writerow(
                [
                    split,
                    result["algorithm"],
                    result["accuracy"],
                    result["macro_f1"],
                    result["weighted_f1"],
                ]
            )

    with open(
        os.path.join(output_dir, "all_per_tag_f1.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.writer(output)
        writer.writerow(["split", "algorithm", "tag", "f1"])
        for split, result in results:
            for tag in tags:
                writer.writerow(
                    [split, result["algorithm"], tag, result["per_tag_f1"].get(tag, 0)]
                )

    for split in ("dev", "test"):
        for result_split, result in results:
            if split != result_split:
                continue
            report_path = os.path.join(
                output_dir,
                f"{split}_{result['algorithm'].lower().replace(' ', '_').replace('-', '_')}_report.txt",
            )
            with open(report_path, "w", encoding="utf-8") as output:
                output.write(
                    classification_report(
                        result["ytrue"],
                        result["ypred"],
                        labels=result["labels"],
                        zero_division=0,
                        digits=4,
                    )
                )

    for split in ("dev", "test"):
        split_results = [result for name, result in results if name == split]
        x = np.arange(len(tags))
        width = 0.82 / max(len(split_results), 1)
        fig, axis = plt.subplots(figsize=(max(18, len(tags) * 0.55), 7))
        for index, result in enumerate(split_results):
            offsets = x + (index - (len(split_results) - 1) / 2) * width
            axis.bar(
                offsets,
                [result["per_tag_f1"].get(tag, 0) for tag in tags],
                width,
                label=result["algorithm"],
            )
        axis.set_xticks(x)
        axis.set_xticklabels(tags, rotation=55, ha="right")
        axis.set_ylim(0, 1.02)
        axis.set_ylabel("F1-score")
        axis.set_title(f"Hindi POS tagging: per-tag F1 on {split} split")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{split}_per_tag_f1.png"), dpi=180)
        plt.close(fig)

    metrics = ("accuracy", "macro_f1", "weighted_f1")
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    for axis, metric in zip(axes, metrics):
        dev_results = [result for split, result in results if split == "dev"]
        test_results = [result for split, result in results if split == "test"]
        positions = np.arange(len(dev_results))
        axis.bar(positions - 0.18, [r[metric] for r in dev_results], 0.36, label="Dev")
        axis.bar(positions + 0.18, [r[metric] for r in test_results], 0.36, label="Test")
        axis.set_xticks(positions)
        axis.set_xticklabels(
            ["NB", "HMM-F", "HMM-B", "HMM-FB", "MEMM", "CRF"],
            rotation=35,
            ha="right",
        )
        axis.set_ylim(0, 1.02)
        axis.set_title(metric.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.25)
        if axis is axes[0]:
            axis.legend()
    fig.suptitle("Overall POS tagging performance")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "overall_metrics.png"), dpi=180)
    plt.close(fig)

    with open(
        os.path.join(output_dir, "run_info.json"), "w", encoding="utf-8"
    ) as output:
        json.dump(
            {
                "train_sentences": len(train),
                "dev_sentences": len(dev),
                "test_sentences": len(test),
                "train_tokens": sum(map(len, train)),
                "dev_tokens": sum(map(len, dev)),
                "test_tokens": sum(map(len, test)),
                "tags": tags,
                "training_seconds": timings,
            },
            output,
            ensure_ascii=False,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "hindi_pos_results",
    )
    parser.add_argument("--memm-max-iter", type=int, default=200)
    parser.add_argument("--crf-max-iter", type=int, default=100)
    args = parser.parse_args()
    if args.memm_max_iter < 1 or args.crf_max_iter < 1:
        parser.error("iteration limits must be positive")

    paths = {
        split: args.data_dir / f"hindi-{split}-pos.txt"
        for split in ("train", "dev", "test")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error("Missing dataset file(s): " + ", ".join(missing))

    train, dev, test = [read_data(str(paths[name])) for name in ("train", "dev", "test")]
    if not train or not dev or not test:
        parser.error("Training, dev, and test datasets must all contain sentences")
    print(
        "Sentences / tokens:",
        {
            name: (len(data), sum(map(len, data)))
            for name, data in (("train", train), ("dev", dev), ("test", test))
        },
        flush=True,
    )

    nb = NaiveBayesTagger().fit(train)
    hmm = HMMTagger().fit(train)
    trained_models = {}
    timings = {}
    for name, model in (
        ("MEMM", MEMMTagger(args.memm_max_iter)),
        ("CRF", CRFTagger(args.crf_max_iter)),
    ):
        started = time.time()
        print(f"Training {name}...", flush=True)
        trained_models[name] = model.fit(train)
        timings[name] = round(time.time() - started, 2)
        print(f"Trained {name} in {timings[name]}s", flush=True)

    results = []
    tags = sorted(
        {tag for data in (train, dev, test) for sentence in data for _, tag in sentence}
    )
    for split_name, data in (("dev", dev), ("test", test)):
        print(f"Predicting {split_name}...", flush=True)
        predictions = {
            "Naive Bayes": nb.predict(data),
            "HMM forward Viterbi": hmm.predict(data, "viterbi"),
            "HMM backward Viterbi": hmm.predict(data, "backward_viterbi"),
            "HMM forward-backward posterior decoding": hmm.predict(
                data, "forward_backward"
            ),
            "MEMM": trained_models["MEMM"].predict(data),
            "CRF": trained_models["CRF"].predict(data),
        }
        for name in ALGORITHMS:
            result = evaluate(name, data, predictions[name], tags)
            results.append((split_name, result))
            print(
                f"{split_name:4} | {name:42} "
                f"accuracy={result['accuracy']:.4f} "
                f"macro-F1={result['macro_f1']:.4f} "
                f"weighted-F1={result['weighted_f1']:.4f}",
                flush=True,
            )

    write_results(results, train, dev, test, str(args.out_dir), tags, timings)
    print(f"Wrote results to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

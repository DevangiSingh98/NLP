# NLP Lab Assignment 8

This folder contains simple Python solutions for the assignment sheet:

1. `midsem_solutions.py` prints worked solutions for the five questions in the included midsem paper.
2. `spell_corrector.py` uses the provided unigram counts, edit counts, and correct word / typo pairs to run a small noisy channel spell checker and report its accuracy.

## Run

Open a terminal in this folder and run:

```text
python midsem_solutions.py
python spell_corrector.py
python spell_corrector.py rainning
```

The spell correction input files are in `Spell-Correction/`. To use a different data folder, pass `--data-dir path/to/folder`.

## Spell checker model

For an error listed in `spell-errors.txt`, the listed correct word is used as its candidate. For an input that is not listed, the program generates one-edit candidates and keeps the ones found in the unigram vocabulary. It ranks generated candidates using the noisy channel idea:

```text
score(correct | typo) = unigram_count(correct) * edit_count(correct -> typo)
```

If an edit is not listed in `count_1edit.txt`, its edit count is treated as 1. Accuracy on `spell-errors.txt` uses the file's correct word / error pairs, so it measures coverage of the supplied examples rather than held-out performance. Errors absent from that list use the noisy channel fallback; errors needing more than one edit may not be found by its candidate generator.

## Midsem notes

- Q1 describes the finite state transducer using stem-preserving paths and the required `+N`, `+SG`, and `+PL` outputs.
- Q2 applies two WordPiece merges learned from the given corpus.
- Q3 uses raw term counts and `ln(number of sentences / document frequency)` for IDF, then compares TF-IDF vectors with cosine similarity.
- Q4 uses Bernoulli Naive Bayes without smoothing, as no smoothing was requested.
- Q5 uses interpolated Kneser-Ney with discount 0.5 and no sentence boundary markers.

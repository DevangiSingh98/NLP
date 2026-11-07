# Hindi POS Tagger — Results

## Dataset

| Split | Sentences | Tokens |
|---|---:|---:|
| Train | 15,087 | 325,710 |
| Dev | 1,866 | 40,762 |
| Test | 1,910 | 41,415 |

## Overall results

| Split | Algorithm | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|
| dev | Naive Bayes | 0.9247 | 0.8331 | 0.9249 |
| dev | HMM forward Viterbi | 0.9304 | 0.9083 | 0.9300 |
| dev | HMM backward Viterbi | 0.9304 | 0.9083 | 0.9300 |
| dev | HMM forward-backward posterior decoding | 0.9304 | 0.9087 | 0.9299 |
| test | Naive Bayes | 0.9257 | 0.8444 | 0.9259 |
| test | HMM forward Viterbi | 0.9283 | 0.8685 | 0.9281 |
| test | HMM backward Viterbi | 0.9283 | 0.8685 | 0.9281 |
| test | HMM forward-backward posterior decoding | 0.9290 | 0.8639 | 0.9286 |

## Algorithms and implementation notes

- **Naive Bayes:** categorical Naive Bayes over the requested features: word length (`MORE` if length > 4, otherwise `LESS`), current word, previous word, next word, and previous tag. At training time the previous tag is gold; at prediction time it is the previously predicted tag, so decoding is left-to-right greedy. Additive smoothing uses α = 0.1.
- **HMM forward Viterbi:** standard left-to-right max-sum dynamic programming with start, transition, emission, and end probabilities estimated from the training split.
- **HMM backward Viterbi:** suffix max-sum dynamic programming, recovering the globally best sequence from the beginning. It should recover the same MAP sequence as standard Viterbi when both use the same model and end probabilities; matching results are expected, not a bug.
- **HMM forward-backward:** forward and backward log-probabilities are combined to obtain each token’s posterior tag distribution; each token receives its highest-posterior tag. This is posterior decoding (maximizes expected token accuracy), not necessarily the single highest-probability whole sequence.

## Interpretation

- On the test set, forward-backward posterior decoding has the highest token accuracy (0.9290).
- Standard and backward Viterbi tie on test macro-F1 (0.8685); the forward-backward decoder has macro-F1 0.8639. Naive Bayes has test macro-F1 0.8444.
- HMM methods perform better on macro-F1 than Naive Bayes on both dev and test, suggesting more balanced performance across the tag inventory. Weighted-F1 is strongly influenced by frequent tags.
- The two Viterbi variants match because they optimize the same sequence-level HMM objective; forward-backward posterior decoding is a different decision rule.
- Per-tag F1 is in `all_per_tag_f1.csv`; split-specific tables are `dev_per_tag_f1.csv` and `test_per_tag_f1.csv`. Classification reports include precision, recall, F1 and support for each tag.

## Outputs

- `hindi_pos_tagger.py`: complete runnable implementation.
- `summary_metrics.csv`: overall split/algorithm metrics.
- `all_per_tag_f1.csv`: per-tag F1 for every algorithm and split.
- `test_per_tag_f1.png`: grouped per-tag F1 comparison on test.
- `overall_metrics.png`: accuracy, macro-F1 and weighted-F1 on dev and test.
- `*_report.txt`: full classification reports.

## Re-run

```bash
python hindi_pos_tagger.py --data-dir /path/to/data --out-dir hindi_pos_results
```

Input files must be named `hindi-train-pos.txt`, `hindi-dev-pos.txt`, and `hindi-test-pos.txt`. Blank lines separate sentences; each nonblank line contains a word and its tag separated by whitespace.
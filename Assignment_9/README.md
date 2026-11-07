# Assignment 9: Hindi POS Tagging

This assignment compares a Maximum Entropy Markov Model (MEMM), a CRF, and the
Naive Bayes and HMM implementations from Assignment 7 using the train, dev, and
test files in this directory.

## Features

Both MEMM and CRF use prefixes and suffixes of lengths 1–4, previous/current/
next word, and the binary length feature (`LESS` for words of at most four
characters, otherwise `MORE`). A missing prefix or suffix length is represented
by `NULL`. Sentence boundaries use `<BOS>` and `<EOS>`.

The MEMM is a per-token maximum-entropy (multinomial logistic regression)
classifier. It is trained with gold previous tags and decoded left-to-right
using predicted previous tags. This follows the history-conditioned classifier
design in [arne-cl/nltk-maxent-pos-tagger](https://github.com/arne-cl/nltk-maxent-pos-tagger).
That repository's implementation is Python 2-era code and requires the external
MEGAM executable; this portable adaptation retains its MEMM training/decoding
approach while using scikit-learn's maximum-entropy classifier.

The CRF uses `sklearn-crfsuite` with L-BFGS and L1/L2 regularization, following
the [sklearn-crfsuite tutorial](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html).
The Assignment 7 Naive Bayes and HMM implementations are reused for baseline
comparisons, including the three HMM decoding strategies.

## Run

From the repository root:

```powershell
python -m pip install -r Assignment_9\requirements.txt
python Assignment_9\hindi_pos_assignment9.py
```

Or specify the dataset and output directories:

```powershell
python Assignment_9\hindi_pos_assignment9.py --data-dir Assignment_9 --out-dir Assignment_9\hindi_pos_results
```

The run writes `summary_metrics.csv`, `all_per_tag_f1.csv`, per-split
classification reports, `dev_per_tag_f1.png`, `test_per_tag_f1.png`,
`overall_metrics.png`, and `run_info.json` to the output directory. The default
maximum iteration limits are 200 for MEMM and 100 for CRF; both can be changed
with `--memm-max-iter` and `--crf-max-iter`.

## Results

The models were trained on 325,710 tokens and evaluated on 40,762 dev tokens
and 41,415 test tokens. Accuracy, macro-F1, and weighted-F1 are shown below:

| Model | Dev accuracy | Dev macro-F1 | Dev weighted-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes | 0.9247 | 0.8331 | 0.9249 | 0.9257 | 0.8444 | 0.9259 |
| HMM forward Viterbi | 0.9304 | 0.9083 | 0.9300 | 0.9283 | 0.8685 | 0.9281 |
| HMM backward Viterbi | 0.9304 | 0.9083 | 0.9300 | 0.9283 | 0.8685 | 0.9281 |
| HMM forward-backward posterior | 0.9304 | 0.9087 | 0.9299 | 0.9290 | 0.8639 | 0.9286 |
| MEMM | 0.9600 | 0.9595 | 0.9598 | 0.9618 | 0.9231 | 0.9617 |
| CRF | **0.9634** | **0.9649** | **0.9633** | **0.9645** | **0.9288** | **0.9644** |

CRF has the best overall dev and test results, followed by MEMM. Both
discriminative sequence taggers improve on the Assignment 7 Naive Bayes and
HMM baselines. Test macro-F1 is lower than dev macro-F1 for MEMM and CRF, so
their class-balanced gains are smaller on the test split than their accuracy
gains. `all_per_tag_f1.csv` and the dev/test per-tag charts provide the
per-class comparisons; `overall_metrics.png` compares aggregate metrics across
both splits.

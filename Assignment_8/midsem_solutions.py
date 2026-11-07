"""Simple worked solutions for the five NLP midsem questions.

Run with: python midsem_solutions.py
"""
from collections import Counter
from math import log, sqrt


def question_1():
    print("Q1. Noun plural FST")
    print("States: q0 (start), qS (singular output), qP (plural output)")
    print("Keep the noun stem on the arc; classify its ending before choosing the suffix.")
    print("State | Input condition          | Output       | Next state")
    print("q0    | any noun stem            | stem + +N    | qS")
    print("qS    | end of input             | +SG          | qS (final)")
    print("qS    | stem ends s/x/z/ch/sh    | stem + es    | qP")
    print("qS    | other stem               | stem + s     | qP")
    print("qP    | end of input             | +PL          | qP (final)")
    print("Example: book -> book+N+SG, book+s+N+PL; box -> box+N+SG, box+es+N+PL\n")


def question_2():
    corpus = ["low", "lower", "low", "lower", "lowest",
              "low", "lowest", "low", "lowest", "low", "new",
              "new", "newer", "newer", "new"]
    # WordPiece starts with characters plus an end-of-word marker.
    words = [list(word[:-1]) + [word[-1] + "</w>"] for word in corpus]
    counts = Counter(tuple(piece) for piece in words)
    learned_merges = []
    print("Q2. WordPiece: two merges")
    for step in range(2):
        pair_counts = Counter()
        for pieces, count in counts.items():
            for pair in zip(pieces, pieces[1:]):
                pair_counts[pair] += count
        pair, amount = pair_counts.most_common(1)[0]
        merged = pair[0] + pair[1]
        learned_merges.append(pair)
        updated = Counter()
        for pieces, count in counts.items():
            result = []
            i = 0
            while i < len(pieces):
                if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == pair:
                    result.append(merged)
                    i += 2
                else:
                    result.append(pieces[i])
                    i += 1
            updated[tuple(result)] += count
        counts = updated
        print(f"Merge {step + 1}: {pair} -> {merged} (count {amount})")
    def tokenize(word):
        pieces = list(word[:-1]) + [word[-1] + "</w>"]
        for left, right in learned_merges:
            pair_to_merge = left + right
            i = 0
            while i < len(pieces) - 1:
                if pieces[i] + pieces[i + 1] == pair_to_merge:
                    pieces[i:i + 2] = [pair_to_merge]
                    i += 1
                else:
                    i += 1
        return pieces
    print("Tokenization of newest:", tokenize("newest"))
    print("(Use the printed merges to see how the pieces were combined.)\n")


def question_3():
    sentences = [["cat", "dog", "animal"], ["cat", "cat", "pet"],
                 ["dog", "pet", "animal"], ["cat", "dog", "pet"]]
    documents = len(sentences)
    df = Counter(word for sentence in sentences for word in set(sentence))
    words = ["cat", "dog", "animal", "pet"]
    idf = {word: log(documents / df[word]) for word in words}
    vectors = []
    print("Q3. TF-IDF (raw TF, IDF = ln(N/df))")
    print("df:", dict(df))
    print("IDF:", {word: round(idf[word], 4) for word in words})
    for i, sentence in enumerate(sentences, 1):
        counts = Counter(sentence)
        vector = [counts[word] * idf[word] for word in words]
        vectors.append(vector)
        print(f"S{i} {sentence}: {[round(x, 4) for x in vector]}")
    similarities = {}
    for i in range(documents):
        for j in range(i + 1, documents):
            dot = sum(a * b for a, b in zip(vectors[i], vectors[j]))
            norm_i = sqrt(sum(a * a for a in vectors[i]))
            norm_j = sqrt(sum(b * b for b in vectors[j]))
            similarities[(i + 1, j + 1)] = dot / (norm_i * norm_j)
    print("Cosine similarities:", {k: round(v, 4) for k, v in similarities.items()})
    print("Closest pair:", max(similarities, key=similarities.get), "\n")


def question_4():
    spam = [(1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    not_spam = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    # Bernoulli NB without smoothing: calculate the probability of feature=0.
    prior_spam = len(spam) / (len(spam) + len(not_spam))
    prior_not_spam = len(not_spam) / (len(spam) + len(not_spam))
    likelihood_spam = (1 / 4) ** 3
    likelihood_not_spam = (2 / 3) ** 3
    print("Q4. Bernoulli Naive Bayes for E8=(0,0,0)")
    print("P(Spam)=4/7; P(NotSpam)=3/7")
    print("P(E8|Spam)=(1/4)(1/4)(1/4)=1/64")
    print("P(E8|NotSpam)=(2/3)(2/3)(2/3)=8/27")
    print("Scores:", prior_spam * likelihood_spam, "and", prior_not_spam * likelihood_not_spam)
    print("Prediction: NotSpam\n")


def question_5():
    sentences = ["I like NLP", "I like AI", "You like NLP", "You love AI", "We love NLP"]
    bigrams = Counter()
    preceding = {}
    for sentence in sentences:
        words = sentence.split()
        for left, right in zip(words, words[1:]):
            bigrams[(left, right)] += 1
            preceding.setdefault(right, set()).add(left)
    distinct_bigram_count = len(bigrams)
    continuation = {word: len(contexts) / distinct_bigram_count
                    for word, contexts in preceding.items()}
    def kn(previous, word, discount=0.5):
        previous_count = sum(count for (left, _), count in bigrams.items() if left == previous)
        count = bigrams[(previous, word)]
        distinct_followers = sum(1 for (left, _), n in bigrams.items()
                                 if left == previous and n > 0)
        return max(count - discount, 0) / previous_count + discount * distinct_followers / previous_count * continuation[word]
    print("Q5. Interpolated Kneser-Ney, discount=0.5")
    print("Distinct bigram types:", distinct_bigram_count)
    print("P_cont(AI)=2/8=0.25; P_cont(NLP)=2/8=0.25")
    print("P_KN(AI|love)=", round(kn("love", "AI"), 6), "= 0.375")
    print("P_KN(NLP|like)=", round(kn("like", "NLP"), 6), "= 0.583333\n")


if __name__ == "__main__":
    question_1()
    question_2()
    question_3()
    question_4()
    question_5()

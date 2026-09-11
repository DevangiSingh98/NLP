import math

def dynamic_programming_segment(text, word_counts, total_words):

    text = text.lower()

    if not text:
        return []

    # convert to lowercase
    word_counts = {
        word.lower(): count
        for word, count in word_counts.items()
        if count > 0
    }

    n = len(text)

    # max word length
    max_word_length = max(
        len(word) for word in word_counts
    )

    
    dp = [-float("inf")] * (n + 1)

    # starting index of the last word
    previous = [-1] * (n + 1)

    dp[0] = 0.0

    for i in range(1, n + 1):

        
        start_min = max(0, i - max_word_length)

        for start in range(start_min, i):

            word = text[start:i]

            if word not in word_counts:
                continue

            probability = word_counts[word] / total_words

            log_probability = math.log(probability)

            candidate_score = dp[start] + log_probability

            if candidate_score > dp[i]:
                dp[i] = candidate_score
                previous[i] = start

    # If no complete segmentation found, fall back to character-based segmentation
    if previous[n] == -1:
        return list(text)

    # Reconstruct segmentation
    words = []

    position = n

    while position > 0:

        start = previous[position]

        if start == -1:
            # Safety fallback
            words.append(text[position - 1])
            position -= 1
        else:
            words.append(text[start:position])
            position = start

    # reconstruct backwards
    words.reverse()

    return words
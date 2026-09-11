def greedy_segment(text, vocabulary):

    text = text.lower()
    vocabulary = {word.lower() for word in vocabulary}

    if not text:
        return []

    # Longest vocabulary word
    max_word_length = max(len(word) for word in vocabulary)

    result = []
    i = 0

    while i < len(text):

        longest_match = None

        # Try longest words first
        max_length = min(max_word_length, len(text) - i)

        for length in range(max_length, 0, -1):
            candidate = text[i:i + length]

            if candidate in vocabulary:
                longest_match = candidate
                break

        # If no matches, keep character
        if longest_match is None:
            longest_match = text[i]

        result.append(longest_match)
        i += len(longest_match)

    return result
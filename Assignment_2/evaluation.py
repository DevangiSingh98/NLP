def levenshtein_distance(actual, predicted):

    m = len(actual)
    n = len(predicted)

    # dp[i][j] = edit distance between actual[:i] and predicted[:j]

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Empty predicted sequence
    for i in range(m + 1):
        dp[i][0] = i

    # Empty actual sequence
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):

        for j in range(1, n + 1):

            if actual[i - 1] == predicted[j - 1]:
                cost = 0
            else:
                cost = 1

            deletion = dp[i - 1][j] + 1
            insertion = dp[i][j - 1] + 1
            substitution = dp[i - 1][j - 1] + cost

            dp[i][j] = min(
                deletion,
                insertion,
                substitution
            )

    return dp[m][n]


def evaluate(predictions, actuals):


    if len(predictions) != len(actuals):
        raise ValueError(
            "No. of predictions and actual answers must be equal"
        )

    total_cases = len(actuals)

    if total_cases == 0:
        return 0.0, 0.0

    correct = 0
    total_edit_distance = 0

    for predicted, actual in zip(predictions, actuals):

        if predicted == actual:
            correct += 1

        distance = levenshtein_distance(
            actual,
            predicted
        )

        total_edit_distance += distance

    accuracy = correct / total_cases

    average_edit_distance = (
        total_edit_distance / total_cases
    )

    return accuracy, average_edit_distance
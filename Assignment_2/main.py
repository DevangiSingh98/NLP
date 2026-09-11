import json

from greedy import greedy_segment
from dynamic_programming import dynamic_programming_segment
from evaluation import evaluate


DATASET_FILE = "text_segmentation_dataset.json"


def extract_test_case(case):
    
    if isinstance(case, dict):

        # Possible names for input text
        text_keys = [
            "text",
            "input",
            "string",
            "text_input",
            "sentence",
            "word_string"
        ]

        # Possible names for expected segmentation
        answer_keys = [
            "expected",
            "answer",
            "target",
            "segmentation",
            "words",
            "correct",
            "output",
            "ground_truth"
        ]

        text = None
        expected = None

        for key in text_keys:
            if key in case:
                text = case[key]
                break

        for key in answer_keys:
            if key in case:
                expected = case[key]
                break

        if text is None:
            raise ValueError(
                f"Could not find input text in test case:\n{case}"
            )

        if expected is None:
            raise ValueError(
                f"Could not find expected segmentation in test case:\n{case}"
            )

        return normalize_text(text), normalize_expected(expected)

    # If test case is a list/tuple
    if isinstance(case, list) or isinstance(case, tuple):

        if len(case) >= 2:
            return (
                normalize_text(case[0]),
                normalize_expected(case[1])
            )

    raise ValueError(
        f"Unsupported test case format:\n{case}"
    )


def normalize_text(text):
    
    # Convert input text to the form expected by algorithms.

    if isinstance(text, list):
        text = "".join(str(x) for x in text)

    text = str(text)

    return text.replace(" ", "").replace("\t", "").lower()


def normalize_expected(expected):
    
    # Convert expected segmentation into a list of words.


    if isinstance(expected, list):
        return [str(word).lower() for word in expected]

    if isinstance(expected, tuple):
        return [str(word).lower() for word in expected]

    expected = str(expected).strip().lower()

    # If expected answer contains spaces, split into individual words
    
    if " " in expected:
        return expected.split()


    if "|" in expected:
        return [
            word.strip()
            for word in expected.split("|")
            if word.strip()
        ]

    # If no separator, treat as one word.
    return [expected]


def print_example(text, expected, greedy_result, dp_result):
    
    print("\nExample")
    print("-" * 60)

    print("Input:")
    print(text)

    print("\nExpected:")
    print(" ".join(expected))

    print("\nGreedy:")
    print(" ".join(greedy_result))

    print("\nDynamic Programming:")
    print(" ".join(dp_result))


def main():


    with open(DATASET_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("dataset loaded successfully")
    print()

 
    print("Available keys:")
    print(data.keys())

    metadata = data.get("metadata", {})

    word_counts = data.get("word_counts", {})
    test_cases = data.get("test_cases", [])

    print()
    print("Vocabulary Size:", len(word_counts))

    total_words = metadata.get(
        "total_corpus_words",
        metadata.get(
            "total_words",
            sum(word_counts.values())
        )
    )

    print("Total Corpus Words:", total_words)

    if "the" in word_counts:
        print("Frequency of 'the':", word_counts["the"])

    print("Number of Test Cases:", len(test_cases))



    actuals = []
    texts = []

    for case in test_cases:

        try:

            text, expected = extract_test_case(case)

            texts.append(text)
            actuals.append(expected)

        except ValueError as error:

            print("\nERROR:")
            print(error)

            return


    # RUN GREEDY


    vocabulary = set(word_counts.keys())

    greedy_predictions = []

    for text in texts:

        prediction = greedy_segment(
            text,
            vocabulary
        )

        greedy_predictions.append(prediction)

    print("Greedy segmentation completed")

    
    # RUN DYNAMIC PROGRAMMING
    

    dp_predictions = []

    for text in texts:

        prediction = dynamic_programming_segment(
            text,
            word_counts,
            total_words
        )

        dp_predictions.append(prediction)

    print("Dynamic Programming segmentation completed")

    
    # EVALUATION
    

    greedy_accuracy, greedy_edit_distance = evaluate(
        greedy_predictions,
        actuals
    )

    dp_accuracy, dp_edit_distance = evaluate(
        dp_predictions,
        actuals
    )

   

    print("\n")
    print("TEXT SEGMENTATION RESULTS")
    print("_" * 70)

    print("\nGreedy Based Approach")
    print("-" * 70)

    print(
        f"Accuracy: {greedy_accuracy * 100:.2f}%"
    )

    print(
        f"Average Edit Distance: "
        f"{greedy_edit_distance:.4f}"
    )

    print("\nDynamic Programming Approach")
    print("-" * 70)

    print(
        f"Accuracy: {dp_accuracy * 100:.2f}%"
    )

    print(
        f"Average Edit Distance: "
        f"{dp_edit_distance:.4f}"
    )

    # Compare

    print("\n")
    print("COMPARISON")
    print("_" * 70)

    print(
        f"{'Method':<25}"
        f"{'Accuracy':<20}"
        f"{'Avg Edit Distance':<20}"
    )

    print("-" * 70)

    print(
        f"{'Greedy':<25}"
        f"{greedy_accuracy * 100:.2f}%"
        f"{'':<15}"
        f"{greedy_edit_distance:.4f}"
    )

    print(
        f"{'Dynamic Programming':<25}"
        f"{dp_accuracy * 100:.2f}%"
        f"{'':<15}"
        f"{dp_edit_distance:.4f}"
    )


    print("\n")
    
    print("sample predictions: \n")

    number_of_examples = min(5, len(texts))

    for i in range(number_of_examples):

        print_example(
            texts[i],
            actuals[i],
            greedy_predictions[i],
            dp_predictions[i]
        )

if __name__ == "__main__":
    main()
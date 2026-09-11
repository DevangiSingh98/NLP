import string
from pathlib import Path


ALPHABET = set(string.ascii_lowercase)
TRANSITIONS = {("start", letter): "word" for letter in ALPHABET}
TRANSITIONS.update({("word", letter): "word" for letter in ALPHABET})
FINAL_STATES = {"word"}


def recognize(text):
    state = "start"
    for character in text:
        state = TRANSITIONS.get((state, character))
        if state is None:
            return "Not Accepted"
    return "Accepted" if state in FINAL_STATES else "Not Accepted"


def recognize_file(input_path, output_path):
    words = Path(input_path).read_text(encoding="utf-8").splitlines()
    lines = [f"{word} = {recognize(word)}" for word in words if word.strip()]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        recognize_file(sys.argv[1], sys.argv[2])
    else:
        print(recognize(input("Word: ")))

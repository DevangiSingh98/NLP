from pathlib import Path


FST = {
    "q0": {"ies": ("qpl_y", "y+N+PL"), "es": ("qpl_e", "+N+PL"), "s": ("qpl_s", "+N+PL"), "": ("qsg", "+N+SG")},
    "qpl_y": {"condition": "stem is not empty"},
    "qpl_e": {"condition": "stem ends in s, z, x, ch, or sh"},
    "qpl_s": {"condition": "stem does not end in s, z, x, ch, sh, or y"},
    "qsg": {"condition": "word is lowercase letters"},
}


def analyze(word):
    word = word.strip()
    if not word or not word.isascii() or not word.islower() or not word.isalpha():
        return "Invalid Word"

    if word.endswith("ies") and len(word) > 3:
        return f"{word[:-3]}y+N+PL"
    if word.endswith("es"):
        root = word[:-2]
        if root.endswith(("s", "z", "x", "ch", "sh")):
            return f"{root}+N+PL"
    if word.endswith("s") and len(word) > 1:
        root = word[:-1]
        if not root.endswith(("s", "z", "x", "ch", "sh", "y")):
            return f"{root}+N+PL"
        return "Invalid Word"
    return f"{word}+N+SG"


def analyze_file(input_path, output_path):
    nouns = Path(input_path).read_text(encoding="utf-8").splitlines()
    lines = [f"{noun} = {analyze(noun)}" for noun in nouns if noun.strip()]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        analyze_file(sys.argv[1], sys.argv[2])
    else:
        print(analyze(input("Noun: ")))

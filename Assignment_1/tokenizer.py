from __future__ import annotations

import re
from typing import Iterator

TOKEN_RE = re.compile(
    r"(?:"
    r"https?://[^\s<>\"'`{}|\\^\[\]]+"
    r"|www\.[^\s<>\"'`{}|\\^\[\]]+"
    r"|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
    r"|(?:\d{1,4}[/-]){2}\d{1,4}"  
    r"|\d+(?:[.,]\d+)+"              
    r"|\d+"
    r"|[^\W\d_]+(?:['’][^\W\d_]+)*" # Unicode words + Devanagari
    r"|[^\s]"
    r")",
    re.UNICODE,
)


SENTENCE_BREAK_RE = re.compile(r"[.!?।॥]+(?:[\"'”’»）\]]+)?(?=\s|$)")
TRAILING_URL_PUNCTUATION = ".,;:!?)]}»”’"


def sentence_tokenize(paragraph: str) -> list[str]:
   
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    if not paragraph:
        return []
    sentences: list[str] = []
    start = 0
    for match in SENTENCE_BREAK_RE.finditer(paragraph):
        candidate = paragraph[start:match.end()].strip()
        if candidate:
            sentences.append(candidate)
        start = match.end()
    remainder = paragraph[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def word_tokenize(sentence: str) -> list[str]:
    # Return tokens, preserving URLs, e-mails, dates etc
    matches = TOKEN_RE.findall(sentence)
    tokens: list[str] = []
    
    for match in matches:
        if not re.match(r"(?:https?://|www\.)", match, re.I):
            tokens.append(match)
            continue
        url = match
        suffix = ""
        while url and url[-1] in TRAILING_URL_PUNCTUATION:
            suffix = url[-1] + suffix
            url = url[:-1]
        if url:
            tokens.extend([url, *suffix])
        else:
            tokens.append(match)
    return tokens


def tokenize_paragraph(paragraph: str) -> Iterator[list[str]]:
    for sentence in sentence_tokenize(paragraph):
        tokens = word_tokenize(sentence)
        if tokens:
            yield tokens

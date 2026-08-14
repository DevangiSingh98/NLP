"""Stream a corpus, tokenize it, write Parquet sentence shards and statistics."""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from corpus_stats import CorpusStatistics
from tokenizer import tokenize_paragraph


def text_documents(path: Path) -> Iterator[tuple[str, str]]:
    """Yield each non-empty input line without loading the corpus into RAM."""
    with path.open("r", encoding="utf-8", errors="replace") as corpus:
        for index, line in enumerate(corpus):
            if line.strip():
                yield str(index), line


def oscar_documents(language: str, split: str) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset

    for index, row in enumerate(load_dataset("oscar-corpus/OSCAR-2301", language, split=split, streaming=True)):
        if text := row.get("text", ""):
            yield str(index), text


def chunks(items: Iterable[dict[str, object]], size: int) -> Iterator[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def process(documents: Iterable[tuple[str, str]], output: Path, chunk_size: int, limit: int | None) -> dict[str, int | float]:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose another path or delete it first.")
    output.mkdir(parents=True)
    stats = CorpusStatistics(output / "vocabulary.sqlite3")

    def records() -> Iterator[dict[str, object]]:
        for document_count, (document_id, paragraph) in enumerate(documents):
            if limit is not None and document_count >= limit:
                return
            for sentence_number, tokens in enumerate(tokenize_paragraph(paragraph)):
                stats.add(tokens)
                yield {"document_id": document_id, "sentence_number": sentence_number,
                       "tokenized_sentence": " ".join(tokens), "tokens": tokens}

    schema = pa.schema([("document_id", pa.string()), ("sentence_number", pa.int32()),
                        ("tokenized_sentence", pa.string()), ("tokens", pa.list_(pa.string()))])
    for shard, batch in enumerate(chunks(records(), chunk_size)):
        pq.write_table(pa.Table.from_pylist(batch, schema=schema), output / f"sentences-{shard:05d}.parquet", compression="zstd")
        stats.connection.commit()
        print(f"Wrote shard {shard:05d}: {len(batch):,} sentences", flush=True)

    result = stats.close()
    (output / "statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for temporary in (output / "vocabulary.sqlite3", output / "vocabulary.sqlite3-wal", output / "vocabulary.sqlite3-shm"):
        if temporary.exists():
            temporary.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("text", "oscar"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Local text file (required for --source text)")
    parser.add_argument("--language", default="hi", help="OSCAR language configuration, e.g. hi")
    parser.add_argument("--split", default="train")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-documents", type=int, help="Optional cap useful for a smoke test")
    args = parser.parse_args()
    if args.source == "text":
        if args.input is None:
            parser.error("--input is required with --source text")
        documents = text_documents(args.input)
    else:
        documents = oscar_documents(args.language, args.split)
    print(json.dumps(process(documents, args.output, args.chunk_size, args.max_documents), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

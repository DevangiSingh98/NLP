from __future__ import annotations

import sqlite3
from pathlib import Path


class CorpusStatistics:
    

    def __init__(self, database_path: Path) -> None:
        self.sentences = self.tokens = self.characters = 0
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS vocabulary (token TEXT PRIMARY KEY)")

    def add(self, tokens: list[str]) -> None:
        self.sentences += 1
        self.tokens += len(tokens)
        self.characters += sum(len(token) for token in tokens)
        self.connection.executemany(
            "INSERT OR IGNORE INTO vocabulary(token) VALUES (?)", ((token,) for token in tokens)
        )

    def close(self) -> dict[str, int | float]:
        self.connection.commit()
        unique = self.connection.execute("SELECT COUNT(*) FROM vocabulary").fetchone()[0]
        self.connection.close()
        return {
            "total_sentences": self.sentences,
            "total_words": self.tokens,
            "total_characters": self.characters,
            "average_sentence_length": self.tokens / self.sentences if self.sentences else 0.0,
            "average_word_length": self.characters / self.tokens if self.tokens else 0.0,
            "unique_tokens": unique,
            "type_token_ratio": unique / self.tokens if self.tokens else 0.0,
        }

from __future__ import annotations

import csv
import shlex
from pathlib import Path
from typing import Any, Iterator

from veriscope_training.datasets.loaders import open_text


def parse_arff(path: str | Path) -> tuple[list[str], Iterator[dict[str, Any]]]:
    attributes: list[str] = []
    data_lines: list[str] = []
    in_data = False

    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            lowered = line.lower()
            if not in_data:
                if lowered.startswith("@attribute"):
                    lexer = shlex.shlex(line, posix=True)
                    lexer.whitespace_split = True
                    tokens = list(lexer)
                    if len(tokens) >= 3:
                        attributes.append(tokens[1])
                    continue
                if lowered.startswith("@data"):
                    in_data = True
                continue
            data_lines.append(raw_line.rstrip("\n"))

    def row_iterator() -> Iterator[dict[str, Any]]:
        reader = csv.reader(data_lines)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("{"):
                raise ValueError(f"Sparse ARFF rows are not supported in {path}.")
            padded = list(row) + [""] * max(0, len(attributes) - len(row))
            yield dict(zip(attributes, padded[: len(attributes)]))

    return attributes, row_iterator()

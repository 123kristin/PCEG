"""Question-level data pipeline for PCEG.

The mask construction mirrors pyKT: adjacent valid responses form ``masks``;
the shifted ``selectmasks`` field forms ``smasks`` for loss and evaluation.
"""

import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


def _parse_ints(value) -> list[int]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(float(token)) for token in text.split(",") if token.strip()]


def _parse_concepts(value, max_concepts: int) -> list[list[int]]:
    groups = []
    for group in str(value).strip().split(","):
        if group == "-1":
            current = []
        else:
            current = [int(float(token)) for token in group.split("_") if token.strip()]
        groups.append((current + [-1] * max_concepts)[:max_concepts])
    return groups


def load_consistency_scores(path: str | Path | None) -> dict[str, float]:
    """Load item scores, preferring the mapped ID used by sequence CSV files."""
    if path is None:
        return {}
    scores: dict[str, float] = {}
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = record.get("int_qid")
            if item_id is None:
                item_id = record.get("mapped_id")
            if item_id is None:
                item_id = record.get("q_id")
            if item_id is None:
                item_id = record.get("question_id")
            if item_id is None:
                raise ValueError(f"missing item ID in {path}:{line_number}")
            scores[str(item_id).strip()] = float(record.get("consistency_score", 0.0))
    return scores


def build_popularity_map(
    sequence_csv: str | Path, training_folds: set[int] | list[int]
) -> dict[str, float]:
    """Compute log-count popularity using only the selected training folds."""
    frame = pd.read_csv(sequence_csv)
    if "questions" not in frame.columns:
        raise ValueError(f"{sequence_csv} has no 'questions' column")
    if "fold" in frame.columns:
        frame = frame[frame["fold"].isin(set(training_folds))]
    counts = Counter(
        question
        for sequence in frame["questions"]
        for question in _parse_ints(sequence)
        if question >= 0
    )
    logged = {str(item): math.log1p(count) for item, count in counts.items()}
    maximum = max(logged.values(), default=0.0)
    if maximum <= 1e-6:
        return logged
    return {item: value / maximum for item, value in logged.items()}


class PCEGDataset(Dataset):
    def __init__(
        self,
        sequence_csv: str | Path,
        semantic_file: str | Path,
        consistency_file: str | Path | None,
        *,
        folds: set[int] | list[int] | None = None,
        popularity_map: dict[str, float] | None = None,
        max_sequence_length: int = 200,
        max_concepts: int = 4,
    ):
        self.sequence_csv = Path(sequence_csv)
        self.semantic_matrix = torch.load(
            semantic_file, map_location="cpu", weights_only=True
        ).float()
        if self.semantic_matrix.ndim != 2:
            raise ValueError("semantic feature tensor must have shape [num_questions, dim]")
        self.semantic_matrix = torch.nan_to_num(self.semantic_matrix, nan=0.0)
        self.consistency = load_consistency_scores(consistency_file)
        self.popularity = {
            str(key).strip(): float(value)
            for key, value in (popularity_map or {}).items()
        }
        self.max_sequence_length = max_sequence_length
        self.max_concepts = max_concepts

        frame = pd.read_csv(self.sequence_csv)
        required = {"questions", "concepts", "responses", "selectmasks"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{self.sequence_csv} is missing columns: {sorted(missing)}")
        if folds is not None:
            if "fold" not in frame.columns:
                raise ValueError("fold filtering requested, but the CSV has no fold column")
            frame = frame[frame["fold"].isin(set(folds))]
        self.rows = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        questions = _parse_ints(row["questions"])
        concepts = _parse_concepts(row["concepts"], self.max_concepts)
        responses = _parse_ints(row["responses"])
        selectmasks = _parse_ints(row["selectmasks"])
        lengths = {len(questions), len(concepts), len(responses), len(selectmasks)}
        if len(lengths) != 1:
            raise ValueError(f"sequence length mismatch at row {index}")
        questions = questions[: self.max_sequence_length]
        concepts = concepts[: self.max_sequence_length]
        responses = responses[: self.max_sequence_length]
        selectmasks = selectmasks[: self.max_sequence_length]
        if len(questions) < 2:
            raise ValueError(f"row {index} has fewer than two interactions")

        raw_questions = torch.tensor(questions, dtype=torch.long)
        raw_responses = torch.tensor(responses, dtype=torch.long)
        valid_questions = raw_questions.ge(0) & raw_questions.lt(self.semantic_matrix.size(0))
        safe_questions = raw_questions.masked_fill(~valid_questions, 0)
        semantics = self.semantic_matrix[safe_questions].clone()
        semantics[~valid_questions] = 0.0
        return {
            "questions": safe_questions,
            "concepts": torch.tensor(concepts, dtype=torch.long),
            "responses": raw_responses.clamp(min=0),
            "response_valid": raw_responses.ne(-1),
            "selectmasks": torch.tensor(selectmasks, dtype=torch.long),
            "semantic_vectors": semantics,
            "consistency": torch.tensor(
                [self.consistency.get(str(item), 0.0) for item in questions],
                dtype=torch.float32,
            ),
            "popularity": torch.tensor(
                [self.popularity.get(str(item), 0.0) for item in questions],
                dtype=torch.float32,
            ),
            "length": len(questions),
            "row_index": index,
            "student_id": int(float(row.get("uid", index))),
        }

    @staticmethod
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        target_length = max(item["length"] for item in batch) - 1

        def pad_1d(tensor, value=0):
            return torch.nn.functional.pad(
                tensor, (0, target_length - tensor.size(0)), value=value
            )

        def pad_2d(tensor, value=0):
            output = tensor.new_full((target_length, tensor.size(1)), value)
            output[: tensor.size(0)] = tensor
            return output

        output = {
            "questions": [],
            "concepts": [],
            "responses": [],
            "target_questions": [],
            "target_responses": [],
            "semantic_vectors": [],
            "consistency": [],
            "popularity": [],
            "masks": [],
            "smasks": [],
            "row_index": [],
            "student_id": [],
        }
        for item in batch:
            output["questions"].append(pad_1d(item["questions"][:-1]))
            output["concepts"].append(pad_2d(item["concepts"][:-1], value=-1))
            output["responses"].append(pad_1d(item["responses"][:-1]))
            output["target_questions"].append(pad_1d(item["questions"][1:]))
            output["target_responses"].append(pad_1d(item["responses"][1:]))
            output["semantic_vectors"].append(pad_2d(item["semantic_vectors"][:-1]))
            output["consistency"].append(pad_1d(item["consistency"][:-1]))
            output["popularity"].append(pad_1d(item["popularity"][:-1]))
            adjacent_valid = item["response_valid"][:-1] & item["response_valid"][1:]
            output["masks"].append(pad_1d(adjacent_valid, value=False))
            output["smasks"].append(
                pad_1d(item["selectmasks"][1:].ne(-1), value=False)
            )
            output["row_index"].append(torch.tensor(item["row_index"]))
            output["student_id"].append(torch.tensor(item["student_id"]))
        return {key: torch.stack(value) for key, value in output.items()}

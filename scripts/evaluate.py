#!/usr/bin/env python3
"""Evaluate a PCEG checkpoint and optionally export gate traces."""

import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from pceg.data import PCEGDataset
from pceg.engine import forward_batch, move_batch
from pceg.model import PCEG, PCEGConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-csv", required=True)
    parser.add_argument("--semantic-file", required=True)
    parser.add_argument("--consistency-file", required=True)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    metadata = checkpoint["metadata"]
    config = PCEGConfig(**metadata["model_config"])
    model = PCEG(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    shared = metadata["paper_config"]
    max_concepts = {
        "DBE-KT22": 4,
        "XES3G5M": 6,
        "NIPS_task34": 2,
    }[metadata["dataset"]]
    dataset = PCEGDataset(
        args.sequence_csv,
        args.semantic_file,
        args.consistency_file,
        folds={args.fold},
        popularity_map=metadata["popularity_map"],
        max_sequence_length=int(shared["max_sequence_length"]),
        max_concepts=max_concepts,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PCEGDataset.collate,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    targets = []
    probabilities = []

    with predictions_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "dataset",
                "encoder",
                "assignment",
                "seed",
                "row_index",
                "student_id",
                "step",
                "question_id",
                "target",
                "probability",
                "consistency",
                "popularity",
                "static_gate",
                "energy",
                "alpha",
            ],
        )
        writer.writeheader()
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                predictions, trace = forward_batch(model, batch, return_trace=True)
                selected = batch["smasks"].bool()
                for row, step in selected.nonzero(as_tuple=False).tolist():
                    target = int(batch["target_responses"][row, step])
                    probability = float(predictions[row, step])
                    targets.append(target)
                    probabilities.append(probability)
                    writer.writerow(
                        {
                            "dataset": metadata["dataset"],
                            "encoder": metadata["encoder"],
                            "assignment": metadata["valid_fold"],
                            "seed": metadata["seed"],
                            "row_index": int(batch["row_index"][row]),
                            "student_id": int(batch["student_id"][row]),
                            "step": step,
                            "question_id": int(batch["target_questions"][row, step]),
                            "target": target,
                            "probability": probability,
                            "consistency": float(batch["consistency"][row, step]),
                            "popularity": float(batch["popularity"][row, step]),
                            "static_gate": float(trace["static_gate"][row, step, 0]),
                            "energy": float(trace["energy"][row, step, 0]),
                            "alpha": float(trace["alpha"][row, step, 0]),
                        }
                    )

    metrics = {
        "dataset": metadata["dataset"],
        "encoder": metadata["encoder"],
        "assignment": metadata["valid_fold"],
        "seed": metadata["seed"],
        "auc": float(roc_auc_score(targets, probabilities)),
        "accuracy": float(accuracy_score(targets, [p >= 0.5 for p in probabilities])),
        "count": len(targets),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()

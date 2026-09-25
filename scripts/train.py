#!/usr/bin/env python3
"""Train one PCEG train-validation assignment."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pceg.data import PCEGDataset, build_popularity_map
from pceg.engine import evaluate, save_checkpoint, set_seed, train_epoch
from pceg.model import PCEG, PCEGConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["DBE-KT22", "XES3G5M", "NIPS_task34"])
    parser.add_argument("--encoder", required=True, choices=["qwen3b", "qwen7b", "gemma3_4b"])
    parser.add_argument("--sequence-csv", required=True)
    parser.add_argument("--semantic-file", required=True)
    parser.add_argument("--consistency-file", required=True)
    parser.add_argument("--valid-fold", type=int, required=True, choices=range(5))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-config", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Override the paper seed for crossed-seed runs")
    return parser.parse_args()


def main():
    args = parse_args()
    paper = json.loads(Path(args.paper_config).read_text(encoding="utf-8"))
    shared = paper["shared"]
    dataset_config = paper["datasets"][args.dataset]
    encoder_config = paper["encoders"][args.encoder]
    setting = paper["settings"][f"{args.dataset}/{args.encoder}"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(shared["seed"] if args.seed is None else args.seed)
    set_seed(seed)
    training_folds = set(range(5)) - {args.valid_fold}
    popularity = build_popularity_map(args.sequence_csv, training_folds)
    (output_dir / "popularity.json").write_text(
        json.dumps(popularity, indent=2, sort_keys=True), encoding="utf-8"
    )

    dataset_kwargs = dict(
        sequence_csv=args.sequence_csv,
        semantic_file=args.semantic_file,
        consistency_file=args.consistency_file,
        popularity_map=popularity,
        max_sequence_length=int(shared["max_sequence_length"]),
        max_concepts=int(dataset_config["max_concepts"]),
    )
    train_data = PCEGDataset(**dataset_kwargs, folds=training_folds)
    valid_data = PCEGDataset(**dataset_kwargs, folds={args.valid_fold})
    loader_kwargs = dict(
        batch_size=int(shared["batch_size"]),
        collate_fn=PCEGDataset.collate,
        num_workers=args.num_workers,
    )
    train_loader = DataLoader(
        train_data,
        shuffle=bool(shared["shuffle_training_sequences"]),
        **loader_kwargs,
    )
    valid_loader = DataLoader(valid_data, shuffle=False, **loader_kwargs)

    model_config = PCEGConfig(
        num_questions=int(dataset_config["num_questions"]),
        num_concepts=int(dataset_config["num_concepts"]),
        semantic_dim=int(encoder_config["semantic_dim"]),
        hidden_dim=int(shared["hidden_dim"]),
        dropout=float(shared["dropout"]),
        norm_radius=float(setting["norm_radius"]),
        gate_bias=float(setting["gate_bias"]),
        prior_lambda=float(setting["prior_lambda"]),
    )
    device = torch.device(args.device)
    model = PCEG(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(setting["learning_rate"]),
        betas=tuple(shared["adam_betas"]),
        eps=float(shared["adam_epsilon"]),
        weight_decay=float(shared["weight_decay"]),
    )
    metadata = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "valid_fold": args.valid_fold,
        "seed": seed,
        "training_folds": sorted(training_folds),
        "model_config": model_config.to_dict(),
        "paper_config": shared,
        "setting": setting,
        "popularity_map": popularity,
    }
    (output_dir / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    best_auc = float("-inf")
    best_epoch = -1
    history_path = output_dir / "history.jsonl"
    for epoch in range(1, int(shared["max_epochs"]) + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, valid_loader, device)
        record = {"epoch": epoch, "train_loss": loss, **metrics.to_dict()}
        with history_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        if metrics.auc > best_auc + float(shared["checkpoint_min_delta"]):
            best_auc = metrics.auc
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, metadata)
        if epoch - best_epoch >= int(shared["early_stopping_patience"]):
            break

    summary = {"best_epoch": best_epoch, "best_valid_auc": best_auc}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

"""Training and evaluation helpers shared by the public command-line tools."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def forward_batch(model, batch: dict, *, return_trace: bool = False):
    return model(
        batch["questions"],
        batch["concepts"],
        batch["responses"],
        batch["target_questions"],
        batch["semantic_vectors"],
        batch["consistency"],
        batch["popularity"],
        return_trace=return_trace,
    )


def masked_loss(predictions: torch.Tensor, batch: dict) -> torch.Tensor:
    selected = batch["smasks"].bool()
    if not selected.any():
        raise ValueError("batch has no selected targets")
    return F.binary_cross_entropy(
        predictions[selected].double(), batch["target_responses"][selected].double()
    )


@dataclass(frozen=True)
class Metrics:
    auc: float
    accuracy: float
    count: int

    def to_dict(self) -> dict:
        return {"auc": self.auc, "accuracy": self.accuracy, "count": self.count}


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Metrics:
    model.eval()
    targets = []
    predictions = []
    for batch in loader:
        batch = move_batch(batch, device)
        batch_predictions = forward_batch(model, batch)
        selected = batch["smasks"].bool()
        targets.extend(batch["target_responses"][selected].cpu().tolist())
        predictions.extend(batch_predictions[selected].cpu().tolist())
    if len(set(targets)) < 2:
        raise ValueError("AUC requires both response classes in the selected targets")
    return Metrics(
        auc=float(roc_auc_score(targets, predictions)),
        accuracy=float(accuracy_score(targets, np.asarray(predictions) >= 0.5)),
        count=len(targets),
    )


def train_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        predictions = forward_batch(model, batch)
        loss = masked_loss(predictions, batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("training loader is empty")
    return float(np.mean(losses))


def save_checkpoint(path: str | Path, model, metadata: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, destination)


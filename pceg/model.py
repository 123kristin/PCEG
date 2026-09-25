"""PCEG model used in the camera-ready experiments."""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PCEGConfig:
    num_questions: int
    num_concepts: int
    semantic_dim: int
    hidden_dim: int = 200
    dropout: float = 0.7
    norm_radius: float = 5.0
    gate_bias: float = 0.5
    prior_lambda: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


class PCEG(nn.Module):
    """Prior-Calibrated Energy Gating for question-level KT.

    Inputs use the one-step-shifted pyKT protocol. ``questions``, ``concepts``,
    ``responses`` and ``semantic_vectors`` describe observed interactions;
    ``target_questions`` identifies the next questions to score.
    """

    def __init__(self, config: PCEGConfig):
        super().__init__()
        if min(config.num_questions, config.num_concepts, config.semantic_dim) <= 0:
            raise ValueError("num_questions, num_concepts and semantic_dim must be positive")
        self.config = config
        dim = config.hidden_dim

        self.question_embedding = nn.Embedding(config.num_questions, dim)
        self.concept_embedding = nn.Embedding(config.num_concepts, dim)
        self.semantic_projection = nn.Sequential(
            nn.Linear(config.semantic_dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
        )
        self.prior_mlp = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.energy_mlp = nn.Sequential(
            nn.Linear(dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.recurrent_cell = nn.LSTMCell(dim * 4, dim)
        self.output_dropout = nn.Dropout(config.dropout)
        self.output_layer = nn.Linear(dim, config.num_questions)

    def _mean_concept_embedding(self, concepts: torch.Tensor) -> torch.Tensor:
        valid = concepts.ne(-1)
        safe = concepts.masked_fill(~valid, 0)
        embeddings = self.concept_embedding(safe) * valid.unsqueeze(-1)
        denominator = valid.sum(dim=-1, keepdim=True).clamp(min=1)
        return embeddings.sum(dim=-2) / denominator

    def _target_probability(
        self, hidden: torch.Tensor, target_questions: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.output_dropout(hidden)
        weight = self.output_layer.weight[target_questions.long()]
        bias = self.output_layer.bias[target_questions.long()]
        return torch.sigmoid((hidden * weight).sum(dim=-1) + bias)

    @staticmethod
    def _interaction_input(
        fused_question: torch.Tensor,
        concept: torch.Tensor,
        response: torch.Tensor,
    ) -> torch.Tensor:
        response = response.unsqueeze(-1).float()
        return torch.cat(
            [
                response * fused_question,
                response * concept,
                (1.0 - response) * fused_question,
                (1.0 - response) * concept,
            ],
            dim=-1,
        )

    def initial_state(self, batch_size: int, device=None):
        device = device or self.question_embedding.weight.device
        hidden = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        return hidden, torch.zeros_like(hidden)

    def step(
        self,
        question: torch.Tensor,
        concepts: torch.Tensor,
        response: torch.Tensor,
        target_question: torch.Tensor,
        semantic_vector: torch.Tensor,
        consistency: torch.Tensor,
        popularity: torch.Tensor,
        state=None,
    ):
        """Run one stateful online interaction for latency or deployment."""
        device = self.question_embedding.weight.device
        question = question.to(device)
        concepts = concepts.to(device)
        response = response.to(device)
        target_question = target_question.to(device)
        semantic_vector = semantic_vector.to(device).float()
        consistency = consistency.to(device).float().reshape(-1, 1)
        popularity = popularity.to(device).float().reshape(-1, 1)
        if state is None:
            state = self.initial_state(question.size(0), device)
        hidden, cell = state

        radius = self.config.norm_radius
        question_vector = F.normalize(
            self.question_embedding(question), p=2, dim=-1
        ) * radius
        semantic_vector = F.normalize(
            self.semantic_projection(semantic_vector), p=2, dim=-1
        ) * radius
        concept_vector = self._mean_concept_embedding(concepts)
        prior = torch.cat([consistency, popularity], dim=-1)
        static_gate = torch.sigmoid(
            self.prior_mlp(prior) + self.config.gate_bias * consistency
        )
        semantic_step = static_gate * semantic_vector
        energy = self.energy_mlp(
            torch.cat([hidden, question_vector, semantic_step], dim=-1)
        )
        alpha = torch.sigmoid(
            -energy + self.config.prior_lambda * (2.0 * static_gate - 1.0)
        )
        fused = alpha * semantic_step + (1.0 - alpha) * question_vector
        recurrent_input = self._interaction_input(fused, concept_vector, response)
        hidden, cell = self.recurrent_cell(recurrent_input, (hidden, cell))
        prediction = self._target_probability(hidden, target_question)
        return prediction, (hidden, cell), {
            "static_gate": static_gate,
            "energy": energy,
            "alpha": alpha,
        }

    def forward(
        self,
        questions: torch.Tensor,
        concepts: torch.Tensor,
        responses: torch.Tensor,
        target_questions: torch.Tensor,
        semantic_vectors: torch.Tensor,
        consistency: torch.Tensor,
        popularity: torch.Tensor,
        *,
        return_trace: bool = False,
    ):
        if questions.ndim != 2 or concepts.ndim != 3:
            raise ValueError("questions must be [B,T] and concepts must be [B,T,C]")
        if questions.shape != responses.shape or questions.shape != target_questions.shape:
            raise ValueError("question, response and shifted-question shapes must match")
        if semantic_vectors.shape[:2] != questions.shape:
            raise ValueError("semantic_vectors must align with [B,T]")
        if consistency.shape != questions.shape or popularity.shape != questions.shape:
            raise ValueError("consistency and popularity must align with [B,T]")

        device = self.question_embedding.weight.device
        questions = questions.to(device)
        concepts = concepts.to(device)
        responses = responses.to(device)
        target_questions = target_questions.to(device)
        semantic_vectors = semantic_vectors.to(device).float()
        consistency = consistency.to(device).float().unsqueeze(-1)
        popularity = popularity.to(device).float().unsqueeze(-1)

        radius = self.config.norm_radius
        question_vectors = F.normalize(
            self.question_embedding(questions), p=2, dim=-1
        ) * radius
        semantic_vectors = F.normalize(
            self.semantic_projection(semantic_vectors), p=2, dim=-1
        ) * radius
        concept_vectors = self._mean_concept_embedding(concepts)

        prior_inputs = torch.cat([consistency, popularity], dim=-1)
        prior_logits = self.prior_mlp(prior_inputs) + self.config.gate_bias * consistency
        static_gate = torch.sigmoid(prior_logits)
        weighted_semantics = static_gate * semantic_vectors

        batch_size, sequence_length = questions.shape
        hidden = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        cell = torch.zeros_like(hidden)
        predictions = []
        alphas = []
        energies = []

        for step in range(sequence_length):
            semantic_step = weighted_semantics[:, step]
            energy_inputs = torch.cat(
                [hidden, question_vectors[:, step], semantic_step], dim=-1
            )
            energy = self.energy_mlp(energy_inputs)
            alpha = torch.sigmoid(
                -energy
                + self.config.prior_lambda * (2.0 * static_gate[:, step] - 1.0)
            )
            fused = alpha * semantic_step + (1.0 - alpha) * question_vectors[:, step]
            recurrent_input = self._interaction_input(
                fused, concept_vectors[:, step], responses[:, step]
            )
            hidden, cell = self.recurrent_cell(recurrent_input, (hidden, cell))
            predictions.append(
                self._target_probability(hidden, target_questions[:, step])
            )
            if return_trace:
                alphas.append(alpha)
                energies.append(energy)

        prediction_tensor = torch.stack(predictions, dim=1)
        if not return_trace:
            return prediction_tensor
        trace = {
            "static_gate": static_gate,
            "alpha": torch.stack(alphas, dim=1),
            "energy": torch.stack(energies, dim=1),
        }
        return prediction_tensor, trace

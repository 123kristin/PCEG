import unittest

import torch

from pceg.engine import masked_loss
from pceg.model import PCEG, PCEGConfig


class PCEGModelTest(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(7)
        return {
            "questions": torch.tensor([[0, 1, 2], [1, 2, 3]]),
            "concepts": torch.tensor([[[0, -1], [1, 2], [2, -1]], [[1, -1], [2, -1], [0, 1]]]),
            "responses": torch.tensor([[1, 0, 1], [0, 1, 0]]),
            "target_questions": torch.tensor([[1, 2, 3], [2, 3, 4]]),
            "semantic_vectors": torch.randn(2, 3, 4),
            "consistency": torch.tensor([[0.2, 0.5, 0.9], [0.3, 0.6, 0.8]]),
            "popularity": torch.tensor([[0.1, 0.4, 0.7], [0.2, 0.5, 0.9]]),
        }

    def config(self):
        return PCEGConfig(
            num_questions=5,
            num_concepts=3,
            semantic_dim=4,
            hidden_dim=6,
            dropout=0.0,
            norm_radius=5.0,
            gate_bias=0.5,
            prior_lambda=2.0,
        )

    def test_pceg_trace_obeys_prior_calibrated_energy_equation(self):
        model = PCEG(self.config()).eval()
        predictions, trace = model(**self.inputs(), return_trace=True)
        expected = torch.sigmoid(
            -trace["energy"] + 2.0 * (2.0 * trace["static_gate"] - 1.0)
        )
        self.assertEqual(tuple(predictions.shape), (2, 3))
        self.assertTrue(torch.allclose(trace["alpha"], expected, atol=1e-7))

    def test_masked_loss_backpropagates(self):
        model = PCEG(self.config())
        predictions = model(**self.inputs())
        batch = {
            "smasks": torch.tensor([[True, True, False], [True, False, True]]),
            "target_responses": torch.tensor([[0, 1, 0], [1, 0, 1]]),
        }
        loss = masked_loss(predictions, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.energy_mlp[0].weight.grad)

    def test_stateful_steps_match_batched_forward(self):
        model = PCEG(self.config()).eval()
        inputs = self.inputs()
        with torch.no_grad():
            expected, expected_trace = model(**inputs, return_trace=True)
            state = None
            predictions = []
            traces = {"static_gate": [], "energy": [], "alpha": []}
            for step in range(inputs["questions"].size(1)):
                prediction, state, trace = model.step(
                    inputs["questions"][:, step],
                    inputs["concepts"][:, step],
                    inputs["responses"][:, step],
                    inputs["target_questions"][:, step],
                    inputs["semantic_vectors"][:, step],
                    inputs["consistency"][:, step],
                    inputs["popularity"][:, step],
                    state,
                )
                predictions.append(prediction)
                for key in traces:
                    traces[key].append(trace[key])
        self.assertTrue(torch.allclose(torch.stack(predictions, dim=1), expected, atol=1e-7))
        for key in traces:
            self.assertTrue(
                torch.allclose(torch.stack(traces[key], dim=1), expected_trace[key], atol=1e-7)
            )


if __name__ == "__main__":
    unittest.main()

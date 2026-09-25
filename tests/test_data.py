import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from pceg.data import PCEGDataset, build_popularity_map, load_consistency_scores


class PCEGDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = self.root / "sequences.csv"
        pd.DataFrame(
            [
                {
                    "fold": 0,
                    "uid": 10,
                    "questions": "0,1,2,-1",
                    "concepts": "0,1_2,2,-1",
                    "responses": "1,0,1,-1",
                    "selectmasks": "-1,1,1,-1",
                },
                {
                    "fold": 1,
                    "uid": 11,
                    "questions": "1,2,0",
                    "concepts": "1,2,0",
                    "responses": "0,1,0",
                    "selectmasks": "-1,1,1",
                },
            ]
        ).to_csv(self.csv, index=False)
        self.features = self.root / "features.pt"
        torch.save(torch.arange(12, dtype=torch.float32).reshape(3, 4), self.features)
        self.results = self.root / "results.jsonl"
        records = [
            {"question_id": "99", "int_qid": 0, "consistency_score": 0.7},
            {"question_id": "0", "int_qid": 1, "consistency_score": 0.8},
            {"question_id": "1", "int_qid": 2, "consistency_score": 0.9},
        ]
        self.results.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_consistency_prefers_mapped_item_id(self):
        scores = load_consistency_scores(self.results)
        self.assertEqual(scores, {"0": 0.7, "1": 0.8, "2": 0.9})

    def test_masks_match_question_level_pykt_protocol(self):
        dataset = PCEGDataset(
            self.csv,
            self.features,
            self.results,
            folds={0},
            popularity_map={"0": 0.1, "1": 0.2, "2": 0.3},
            max_concepts=2,
        )
        batch = PCEGDataset.collate([dataset[0]])
        self.assertEqual(batch["masks"].tolist(), [[True, True, False]])
        self.assertEqual(batch["smasks"].tolist(), [[True, True, False]])
        self.assertTrue(
            torch.allclose(batch["consistency"], torch.tensor([[0.7, 0.8, 0.9]]))
        )
        self.assertEqual(batch["student_id"].tolist(), [10])

    def test_popularity_uses_training_folds_only(self):
        popularity = build_popularity_map(self.csv, {0})
        self.assertEqual(set(popularity), {"0", "1", "2"})
        self.assertAlmostEqual(popularity["0"], 1.0)
        self.assertAlmostEqual(popularity["1"], 1.0)


if __name__ == "__main__":
    unittest.main()


import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts import evaluate
from scripts import train_flip_delta_force


class ScriptHelperTests(unittest.TestCase):
    def test_standardizers_are_computed_in_one_loader_pass(self):
        class CountingLoader:
            def __init__(self, batches):
                self.batches = batches
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return iter(self.batches)

        loader = CountingLoader(
            [
                {
                    "action": torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
                    "target": torch.tensor([[[2.0]], [[4.0]]]),
                },
                {
                    "action": torch.tensor([[[5.0, 6.0]]]),
                    "target": torch.tensor([[[6.0]]]),
                },
            ]
        )
        stats = train_flip_delta_force.fit_standardizers(
            loader, ["action", "target"]
        )
        self.assertEqual(loader.iterations, 1)
        torch.testing.assert_close(stats["action"]["mean"], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(stats["target"]["mean"], torch.tensor([4.0]))

    def test_evaluation_uses_batches(self):
        class Dataset:
            def __init__(self, *_args, **_kwargs):
                pass

            def __len__(self):
                return 5

            def __getitem__(self, index):
                value = float(index)
                return {
                    "frame": torch.full((1, 2, 2), value),
                    "force": torch.tensor([value]),
                }

        class Model(torch.nn.Module):
            input_mode = "edge"

            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, frame):
                self.calls += 1
                return frame.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)

        model = Model()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            evaluate, "ForceDataset", Dataset
        ):
            result = evaluate.evaluate_episode(
                model,
                "EP000001",
                ["Fz"],
                0.0,
                torch.device("cpu"),
                tmp,
                batch_size=4,
                workers=0,
            )
            self.assertTrue((Path(tmp) / "EP000001" / "force_curve.png").is_file())
        self.assertEqual(model.calls, 2)
        np.testing.assert_allclose(result["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()

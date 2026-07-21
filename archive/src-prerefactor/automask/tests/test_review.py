from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import review


class ReviewCoreTests(unittest.TestCase):
    def test_validate_mask_contract(self):
        image = np.zeros((3, 4))
        result = review.validate_mask(np.ones((3, 4), dtype=np.uint8), image)
        self.assertEqual(result.dtype, np.bool_)
        with self.assertRaises(ValueError):
            review.validate_mask(np.zeros((4, 3), dtype=bool), image)
        with self.assertRaises(TypeError):
            review.validate_mask(np.full((3, 4), 2), image)

    def test_int16_minimum_display_is_finite(self):
        image = np.array([[-32768, -1, 0, 1, 32767]], dtype=np.int16)
        self.assertTrue(np.isfinite(review.display_image(image)).all())

    def test_latest_append_only_verdict_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            labels = Path(temporary)
            verdicts = labels / \"verdicts.csv\"
            with mock.patch.object(review, \"LABELS\", labels), mock.patch.object(review, \"VERDICTS\", verdicts):
                base = {field: \"x\" for field in review.FIELDS}
                review.append_verdict({**base, \"image_id\": \"shot\", \"verdict\": \"yes\"})
                review.append_verdict({**base, \"image_id\": \"shot\", \"verdict\": \"no\"})
                self.assertEqual(review.read_latest_verdicts(), {\"shot\": \"no\"})
                with verdicts.open(newline=\"\") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 2)


if __name__ == \"__main__\":
    unittest.main()
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from src.utils.npy import _safe_load_npy


class SafeNpyTests(unittest.TestCase):
    def test_safe_load_npy_falls_back_for_object_wrapped_array(self) -> None:
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.npy")
            np.save(path, np.array(expected, dtype=object), allow_pickle=True)

            with self.assertRaises(ValueError):
                np.load(path, mmap_mode="r")

            loaded = _safe_load_npy(path, mmap_mode="r")

        self.assertEqual(loaded.shape, expected.shape)
        np.testing.assert_allclose(loaded.astype(np.float32), expected)


if __name__ == "__main__":
    unittest.main()

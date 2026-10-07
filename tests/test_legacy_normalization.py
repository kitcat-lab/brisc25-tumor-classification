"""Regression test for cached Normalization state in nested legacy H5 models."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from model_loading import restore_normalization_state


class LegacyNormalizationTest(unittest.TestCase):
    def test_nested_loaded_transform_matches_original(self):
        layer = tf.keras.layers.Normalization()
        layer.adapt(np.array([[1., 4.], [3., 8.], [5., 12.]], dtype='float32'))
        inner_input = tf.keras.Input((2,))
        inner = tf.keras.Model(inner_input, layer(inner_input))
        outer_input = tf.keras.Input((2,))
        model = tf.keras.Model(outer_input, inner(outer_input))
        x = np.array([[2., 5.]], dtype='float32')
        expected = model(x).numpy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'nested.h5'; model.save(path)
            loaded = tf.keras.models.load_model(path, compile=False)
            before = [w.numpy().copy() for w in loaded.weights]
            restored = restore_normalization_state(loaded)
            self.assertEqual(len(restored), 1)
            np.testing.assert_allclose(loaded(x).numpy(), expected, rtol=1e-6)
            for a, b in zip(before, loaded.weights):
                np.testing.assert_array_equal(a, b.numpy())


if __name__ == '__main__':
    unittest.main()

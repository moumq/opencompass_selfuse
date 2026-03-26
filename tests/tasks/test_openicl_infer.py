"""Unit tests for OpenICLInferTask."""

import unittest

from mmengine.config import ConfigDict

from opencompass.tasks.openicl_infer import OpenICLInferTask


class TestOpenICLInferTask(unittest.TestCase):

    def test_set_default_value(self):
        task = OpenICLInferTask.__new__(OpenICLInferTask)
        cfg = ConfigDict({})
        task._set_default_value(cfg, 'key', 'value')
        self.assertEqual(cfg['key'], 'value')

    def test_set_default_value_skips_none(self):
        task = OpenICLInferTask.__new__(OpenICLInferTask)
        cfg = ConfigDict({})
        task._set_default_value(cfg, 'batch_size', None)
        self.assertNotIn('batch_size', cfg)


if __name__ == '__main__':
    unittest.main()

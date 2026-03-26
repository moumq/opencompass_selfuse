"""Tests for examples/eval_flexible.py."""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / 'examples' / 'eval_flexible.py'


def _build_stub_modules():
    mmengine_mod = types.ModuleType('mmengine')
    mmengine_mod.__path__ = []
    mmengine_config_mod = types.ModuleType('mmengine.config')

    class FakeConfig(dict):

        @classmethod
        def fromfile(cls, path):
            basename = Path(path).stem
            if basename == 'aime2025_gen':
                return cls(
                    {'aime2025_datasets': [{
                        'abbr': 'aime2025'
                    }]})
            if basename == 'aime2025_llmjudge_gen_5e9f4f':
                return cls(
                    {'aime2025_datasets': [{
                        'abbr': 'aime2025-llmjudge'
                    }]})
            if basename == 'demo_gsm8k_chat_gen':
                return cls(
                    {'demo_gsm8k_chat_datasets': [{
                        'abbr': 'demo-gsm8k'
                    }]})
            if basename == 'gpt_4o_2024_05_13':
                return cls({'models': [{'abbr': 'gpt-4o'}]})
            return cls({})

    mmengine_config_mod.Config = FakeConfig
    mmengine_mod.config = mmengine_config_mod

    opencompass_mod = types.ModuleType('opencompass')
    opencompass_mod.__path__ = []

    partitioners_mod = types.ModuleType('opencompass.partitioners')
    partitioners_mod.NaivePartitioner = type('NaivePartitioner', (), {})
    partitioners_mod.NumWorkerPartitioner = type('NumWorkerPartitioner', (), {})

    runners_mod = types.ModuleType('opencompass.runners')
    runners_mod.LocalRunner = type('LocalRunner', (), {})

    tasks_mod = types.ModuleType('opencompass.tasks')
    tasks_mod.OpenICLEvalWatchTask = type('OpenICLEvalWatchTask', (), {})
    tasks_mod.OpenICLInferTask = type('OpenICLInferTask', (), {})

    utils_mod = types.ModuleType('opencompass.utils')
    utils_mod.match_files = lambda *args, **kwargs: []

    utils_run_mod = types.ModuleType('opencompass.utils.run')

    def fake_match_cfg_file(_cfg_dirs, names):
        resolved = []
        known = {
            'aime2025_gen',
            'aime2025_llmjudge_gen_5e9f4f',
            'demo_gsm8k_chat_gen',
            'gpt_4o_2024_05_13',
        }
        for name in names:
            if name not in known:
                raise ValueError(f'No config found for {name}')
            resolved.append((name, f'/virtual/{name}.py'))
        return resolved

    utils_run_mod.match_cfg_file = fake_match_cfg_file
    utils_mod.run = utils_run_mod

    return {
        'mmengine': mmengine_mod,
        'mmengine.config': mmengine_config_mod,
        'opencompass': opencompass_mod,
        'opencompass.partitioners': partitioners_mod,
        'opencompass.runners': runners_mod,
        'opencompass.tasks': tasks_mod,
        'opencompass.utils': utils_mod,
        'opencompass.utils.run': utils_run_mod,
    }


def _load_eval_flexible_module(config):
    with tempfile.NamedTemporaryFile(
            'w', suffix='.json', encoding='utf-8',
            delete=False) as tmp_file:
        json.dump(config, tmp_file)
        tmp_path = tmp_file.name

    old_cfg = os.environ.get('FLEX_EVAL_CONFIG')
    old_keep = os.environ.get('FLEX_EVAL_KEEP_INTERNALS')
    os.environ['FLEX_EVAL_CONFIG'] = tmp_path
    os.environ['FLEX_EVAL_KEEP_INTERNALS'] = '1'
    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(sys.modules, _build_stub_modules(), clear=False))
            spec = importlib.util.spec_from_file_location(
                f'eval_flexible_{uuid.uuid4().hex}', MODULE_PATH)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    finally:
        if old_cfg is None:
            os.environ.pop('FLEX_EVAL_CONFIG', None)
        else:
            os.environ['FLEX_EVAL_CONFIG'] = old_cfg
        if old_keep is None:
            os.environ.pop('FLEX_EVAL_KEEP_INTERNALS', None)
        else:
            os.environ['FLEX_EVAL_KEEP_INTERNALS'] = old_keep
        os.unlink(tmp_path)


class TestEvalFlexible(unittest.TestCase):
    """Regression tests for flexible eval schema handling."""

    def test_new_schema_inline_keye_defaults_to_ckpt(self):
        config = {
            'models': [{
                'inline': {
                    'class': 'KeyeFastAPI',
                    'name': 'istep0000200_2nd_autothink',
                    'api_base': 'http://127.0.0.1:15553/v1',
                    'max_tokens': 2048,
                }
            }],
            'datasets': [{
                'ref': 'aime2025_gen'
            }],
        }

        module = _load_eval_flexible_module(config)

        self.assertEqual(len(module.models), 1)
        self.assertEqual(module.models[0]['type'], 'KeyeChat')
        self.assertEqual(module.models[0]['abbr'],
                         'istep0000200_2nd_autothink')
        self.assertEqual(module.models[0]['ckpt'],
                         'istep0000200_2nd_autothink')
        self.assertTrue(module.datasets)

    def test_legacy_model_data_schema_is_normalized(self):
        config = {
            'model': {
                'demo_autothink': {
                    'class': 'KeyeFastAPI',
                    'max_tokens': 2048,
                    'temperature': 0,
                    'img_detail': 'high',
                    'retry': 20,
                    'timeout': 180,
                    'verbose': True,
                    'api_base': 'http://127.0.0.1:15553/v1',
                }
            },
            'data': {
                'aime2025': {
                    'class': 'AIME2025',
                    'dataset': 'aime2025_gen',
                }
            },
        }

        module = _load_eval_flexible_module(config)

        self.assertEqual(module.FLEX_JOB['models'][0]['inline']['class'],
                         'KeyeChat')
        self.assertEqual(module.FLEX_JOB['datasets'][0]['ref'], 'aime2025_gen')
        self.assertEqual(module.models[0]['type'], 'KeyeChat')
        self.assertEqual(module.models[0]['ckpt'], 'demo_autothink')
        self.assertEqual(module.models[0]['abbr'], 'demo_autothink')
        self.assertTrue(module.datasets)

    def test_legacy_keye_names_schema_is_normalized(self):
        config = {
            'model': {
                'class_name': 'keye',
                'names': ['istep0000200_2nd_autothink'],
                'common_params': {
                    'api_base': 'http://127.0.0.1:15553/v1',
                    'max_tokens': 2048,
                },
                'per_model_params': {
                    'istep0000200_2nd_autothink': {
                        'autothink': True
                    }
                }
            },
            'data': ['aime2025_gen'],
        }

        module = _load_eval_flexible_module(config)

        self.assertEqual(module.models[0]['type'], 'KeyeChat')
        self.assertEqual(module.models[0]['ckpt'],
                         'istep0000200_2nd_autothink')
        self.assertTrue(module.models[0]['autothink'])
        self.assertTrue(module.datasets)

    def test_human_friendly_dataset_name_auto_maps_to_gen_ref(self):
        config = {
            'model': {
                'keye2.5': {
                    'class': 'KeyeFastAPI',
                    'api_base': 'http://127.0.0.1:8000/v1',
                }
            },
            'data': {
                'AIME2025': {
                    'class': 'AIME',
                    'dataset': 'AIME2025',
                }
            },
        }

        module = _load_eval_flexible_module(config)

        self.assertEqual(module.FLEX_JOB['datasets'][0]['ref'], 'AIME2025')
        self.assertEqual(module.FLEX_JOB['datasets'][0]['aliases'],
                         ['AIME2025', 'AIME'])
        self.assertTrue(module.datasets)
        self.assertEqual(module.datasets[0]['abbr'], 'aime2025')

    def test_exact_dataset_ref_overrides_family_default_mapping(self):
        config = {
            'model': {
                'keye2.5': {
                    'class': 'KeyeFastAPI',
                    'api_base': 'http://127.0.0.1:8000/v1',
                }
            },
            'data': {
                'AIME2025': {
                    'class': 'AIME',
                    'dataset': 'aime2025_llmjudge_gen_5e9f4f',
                }
            },
        }

        module = _load_eval_flexible_module(config)

        self.assertEqual(module.FLEX_JOB['datasets'][0]['ref'],
                         'aime2025_llmjudge_gen_5e9f4f')
        self.assertEqual(module.datasets[0]['abbr'], 'aime2025-llmjudge')

    def test_default_runtime_uses_concurrent_infer_task(self):
        config = {
            'models': [{
                'inline': {
                    'class': 'KeyeFastAPI',
                    'name': 'demo_model',
                    'api_base': 'http://127.0.0.1:15553/v1',
                }
            }],
            'datasets': [{
                'ref': 'aime2025_gen'
            }],
        }

        module = _load_eval_flexible_module(config)

        self.assertEqual(module.infer['runner']['task']['type'],
                         'OpenICLInferConcurrentTask')
        self.assertEqual(module.infer['runner']['task']['poll_interval'],
                         module.eval['runner']['task']['watch_interval'])
        self.assertEqual(module.infer['runner']['task']['log_interval'],
                         module.eval['runner']['task']['log_interval'])

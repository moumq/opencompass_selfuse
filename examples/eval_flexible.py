"""General flexible evaluation entry for OpenCompass.

Usage:
    cp examples/eval_flexible.sample.json /tmp/eval_job.json
    FLEX_EVAL_CONFIG=/path/to/eval_job.json python run.py examples/eval_flexible.py --dry-run
    FLEX_EVAL_CONFIG=/path/to/eval_job.json python run.py examples/eval_flexible.py --mode infer
    FLEX_EVAL_CONFIG=/path/to/eval_job.json python run.py examples/eval_flexible.py
"""

import json
import os
from copy import deepcopy
from typing import Dict, List, Tuple

from mmengine.config import Config

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalWatchTask, OpenICLInferTask
from opencompass.utils import match_files
from opencompass.utils.run import match_cfg_file

MODEL_CLASS_ALIASES = {
    'keyeapi': 'KeyeChat',
    'keyefastapi': 'KeyeChat',
    'keyechat': 'KeyeChat',
}

DEFAULT_SCHEDULER = dict(
    infer_strategy='split',
    infer_num_worker=8,
    infer_num_split=8,
    infer_max_num_workers=8,
    eval_n=999999,
    eval_max_num_workers=1,
    watch_interval=3.0,
    heartbeat_timeout=300.0,
    log_interval=30.0,
)

DEFAULT_JOB = dict(
    models=[
        dict(ref='gpt_4o_2024_05_13'),
    ],
    datasets=[
        dict(ref='demo_gsm8k_chat_gen'),
    ],
    scheduler=DEFAULT_SCHEDULER,
    work_dir='./outputs/flexible_eval',
)


def _get_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dedup_files(files: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    deduped = []
    for name, path in files:
        norm = os.path.realpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append((name, path))
    return deduped


def _resolve_cfg_files(cfg_dirs: List[str], ref_name: str,
                       kind: str) -> List[Tuple[str, str]]:
    # Relative path under config roots, e.g. "openai/gpt_4o_2024_05_13"
    if os.sep in ref_name or '/' in ref_name:
        path_like = ref_name.replace('/', os.sep)
        for cfg_dir in cfg_dirs:
            for candidate in (path_like, f'{path_like}.py'):
                full_path = os.path.join(cfg_dir, candidate)
                if os.path.isfile(full_path):
                    basename = os.path.basename(full_path)
                    if basename.endswith('.py'):
                        basename = basename[:-3]
                    return [(basename, full_path)]

    # Explicit path has the highest priority.
    if os.path.isfile(ref_name):
        basename = os.path.basename(ref_name)
        if basename.endswith('.py'):
            basename = basename[:-3]
        return [(basename, ref_name)]

    try:
        return match_cfg_file(cfg_dirs, [ref_name])
    except Exception as exact_err:
        pattern = ref_name if ref_name.endswith('.py') else f'{ref_name}.py'
        fuzzy_files = []
        for cfg_dir in cfg_dirs:
            fuzzy_files.extend(match_files(cfg_dir, pattern, fuzzy=True))
        fuzzy_files = _dedup_files(fuzzy_files)

        if len(fuzzy_files) == 1:
            return fuzzy_files
        if len(fuzzy_files) > 1:
            resolved = '\n'.join([f'- {path}' for _, path in fuzzy_files])
            raise ValueError(
                f'Ambiguous {kind} ref "{ref_name}". Matched files:\n'
                f'{resolved}\nPlease use an exact file name.'
            )
        raise exact_err


def _ensure_entry(entry: Dict, key_name: str, index: int):
    if not isinstance(entry, dict):
        raise TypeError(f'{key_name}[{index}] must be a dict.')
    present_keys = [k for k in ('ref', 'inline') if k in entry]
    if len(present_keys) != 1:
        raise ValueError(
            f'{key_name}[{index}] must contain exactly one of "ref" or '
            f'"inline". Got keys: {list(entry.keys())}'
        )


def _normalize_model_class(class_name: str) -> str:
    if not isinstance(class_name, str):
        raise TypeError(f'inline model "class" must be a string, got: '
                        f'{type(class_name)}')
    return MODEL_CLASS_ALIASES.get(class_name.lower(), class_name)


def _load_job_cfg():
    cfg_path = os.environ.get('FLEX_EVAL_CONFIG', '').strip()
    if not cfg_path:
        return deepcopy(DEFAULT_JOB)

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    if 'model' in cfg or 'data' in cfg:
        raise ValueError(
            'Legacy Keye schema is no longer supported. '
            'Please use top-level "models" and "datasets" lists.'
        )
    return cfg


def _load_models(model_entries: List[Dict]) -> List[Dict]:
    if not model_entries:
        raise ValueError('"models" must not be empty.')

    repo_root = _get_repo_root()
    model_cfg_dirs = [
        os.path.join(repo_root, 'opencompass', 'configs', 'models'),
    ]

    models = []
    for idx, entry in enumerate(model_entries):
        _ensure_entry(entry, 'models', idx)

        if 'ref' in entry:
            ref_name = entry['ref']
            matched_cfgs = _resolve_cfg_files(model_cfg_dirs, ref_name, 'model')
            for _, cfg_path in matched_cfgs:
                cfg = Config.fromfile(cfg_path)
                if 'models' not in cfg:
                    raise ValueError(
                        f'Model config "{cfg_path}" does not contain '
                        '"models" field.'
                    )
                models.extend(deepcopy(cfg['models']))
        else:
            inline_cfg = deepcopy(entry['inline'])
            if not isinstance(inline_cfg, dict):
                raise TypeError(f'models[{idx}].inline must be a dict.')

            cls = inline_cfg.pop('class', None)
            if cls is None:
                raise ValueError(
                    f'models[{idx}].inline must contain "class" field.')
            inline_cfg['type'] = _normalize_model_class(cls)
            # Support user-friendly name in config and make output path
            # deterministic as "<name>/<dataset>.json".
            if 'name' in inline_cfg and 'abbr' not in inline_cfg:
                inline_cfg['abbr'] = inline_cfg.pop('name')
            else:
                inline_cfg.pop('name', None)
                inline_cfg.setdefault('abbr', f'inline-model-{idx}')
            models.append(inline_cfg)

    if not models:
        raise ValueError('No model is resolved from "models".')
    return models


def _load_datasets(dataset_entries: List[Dict]) -> List[Dict]:
    if not dataset_entries:
        raise ValueError('"datasets" must not be empty.')

    repo_root = _get_repo_root()
    dataset_cfg_dirs = [
        os.path.join(repo_root, 'opencompass', 'configs', 'datasets'),
        os.path.join(repo_root, 'opencompass', 'configs',
                     'dataset_collections'),
    ]

    datasets = []
    for idx, entry in enumerate(dataset_entries):
        _ensure_entry(entry, 'datasets', idx)
        if 'inline' in entry:
            raise ValueError(
                'datasets[].inline is unsupported in this interface. '
                'Please use datasets[].ref.'
            )

        ref_name = entry['ref']
        if '/' in ref_name:
            dataset_name, suffix = ref_name.split('/', 1)
        else:
            dataset_name, suffix = ref_name, '_datasets'

        matched_cfgs = _resolve_cfg_files(dataset_cfg_dirs, dataset_name,
                                          'dataset')
        for _, cfg_path in matched_cfgs:
            cfg = Config.fromfile(cfg_path)
            matched = False
            for key in cfg.keys():
                if key.endswith(suffix):
                    datasets.extend(deepcopy(cfg[key]))
                    matched = True
            if not matched:
                raise ValueError(
                    f'Dataset config "{cfg_path}" has no field ending with '
                    f'"{suffix}".'
                )

    if not datasets:
        raise ValueError('No dataset is resolved from "datasets".')
    return datasets


def _build_runtime_cfg(job_cfg: Dict):
    scheduler = deepcopy(DEFAULT_SCHEDULER)
    scheduler.update(deepcopy(job_cfg.get('scheduler', {})))

    infer_cfg = dict(
        partitioner=dict(
            type=NumWorkerPartitioner,
            strategy=scheduler['infer_strategy'],
            num_worker=scheduler['infer_num_worker'],
            num_split=scheduler['infer_num_split'],
        ),
        runner=dict(
            type=LocalRunner,
            max_num_workers=scheduler['infer_max_num_workers'],
            task=dict(type=OpenICLInferTask),
        ),
    )

    eval_cfg = dict(
        partitioner=dict(type=NaivePartitioner, n=scheduler['eval_n']),
        runner=dict(
            type=LocalRunner,
            max_num_workers=scheduler['eval_max_num_workers'],
            task=dict(
                type=OpenICLEvalWatchTask,
                watch_interval=scheduler['watch_interval'],
                heartbeat_timeout=scheduler['heartbeat_timeout'],
                log_interval=scheduler['log_interval'],
            ),
        ),
    )
    return infer_cfg, eval_cfg


FLEX_JOB = _load_job_cfg()
models = _load_models(FLEX_JOB.get('models', []))
datasets = _load_datasets(FLEX_JOB.get('datasets', []))
infer, eval = _build_runtime_cfg(FLEX_JOB)
work_dir = FLEX_JOB.get(
    'work_dir',
    os.environ.get('FLEX_EVAL_WORK_DIR', './outputs/flexible_eval'))

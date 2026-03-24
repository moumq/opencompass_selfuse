"""Flexible one-file evaluation config for KeyeChat.

Usage:
    python run.py examples/eval_keye_flexible.py --dry-run
    KEYE_JOB_CONFIG=/path/to/your_job.json python run.py examples/eval_keye_flexible.py
"""

import json
import os
from copy import deepcopy

from mmengine.config import Config

from opencompass.models import KeyeChat
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalWatchTask, OpenICLInferTask
from opencompass.utils import match_files
from opencompass.utils.run import match_cfg_file

# Edit this block only.
DEFAULT_KEYE_JOB = dict(
    model=dict(
        class_name='keye',  # informative field for human readability
        names=[
            'istep0000200_2nd_autothink',
        ],
        common_params=dict(
            key='EMPTY',
            api_base='http://127.0.0.1:15553/v1',
            query_per_second=1,
            batch_size=1,
            temperature=0,
            max_tokens=10240,
            max_seq_len=32768,
            retry=60,
            timeout=180,
            verbose=True,
            img_detail='high',
        ),
        per_model_params={
            # 'istep0000200_2nd_autothink': dict(autothink=True),
        },
    ),
    datasets=[
        'aime2025_gen',
        'aime2026_gen',
        'gpqa_gen',
        'mmlu_pro_gen',
    ],
    # High-throughput defaults:
    # - split inference tasks to keep queue full
    # - watch eval so finished datasets are scored immediately
    scheduler=dict(
        infer_strategy='split',
        infer_num_worker=8,
        infer_num_split=8,
        infer_max_num_workers=8,
        eval_n=999999,
        eval_max_num_workers=1,
        watch_interval=3.0,
        heartbeat_timeout=300.0,
        log_interval=30.0,
    ),
)


def _normalize_job_cfg(raw_cfg):
    """Normalize job config into internal schema.

    Internal schema:
    {
      "model": {
        "names": [...],
        "common_params": {...},
        "per_model_params": {...}
      },
      "datasets": [...]
    }
    """
    cfg = deepcopy(raw_cfg)

    # Accept historical key name "data" as alias of "datasets".
    if 'datasets' not in cfg and 'data' in cfg:
        data_cfg = cfg['data']
        dataset_names = []
        if isinstance(data_cfg, dict):
            for _, one in data_cfg.items():
                if isinstance(one, dict) and 'dataset' in one:
                    dataset_names.append(one['dataset'])
        elif isinstance(data_cfg, list):
            dataset_names = list(data_cfg)
        cfg['datasets'] = dataset_names

    # Accept user-provided model map:
    # "model": {"model_name": {"class": "...", ...}, ...}
    model_cfg = cfg.get('model', {})
    if isinstance(model_cfg, dict) and 'names' not in model_cfg:
        names = []
        per_model_params = {}
        for model_name, params in model_cfg.items():
            if not isinstance(params, dict):
                continue
            names.append(model_name)
            one_params = deepcopy(params)
            one_params.pop('class', None)
            per_model_params[model_name] = one_params
        cfg['model'] = dict(
            class_name='keye',
            names=names,
            common_params={},
            per_model_params=per_model_params,
        )

    return cfg


def _load_keye_job():
    cfg_path = os.environ.get('KEYE_JOB_CONFIG', '').strip()
    if not cfg_path:
        return deepcopy(DEFAULT_KEYE_JOB)
    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw_cfg = json.load(f)
    return _normalize_job_cfg(raw_cfg)


def _build_models(job_cfg):
    model_cfg = deepcopy(job_cfg['model'])
    names = model_cfg.get('names', [])
    if not names:
        raise ValueError('KEYE_JOB["model"]["names"] must not be empty.')

    common_params = deepcopy(model_cfg.get('common_params', {}))
    per_model_params = deepcopy(model_cfg.get('per_model_params', {}))

    models = []
    for name in names:
        model = dict(
            type=KeyeChat,
            ckpt=name,
            abbr=f'keye-{name.replace("_", "-")}',
            **common_params,
        )
        model.update(per_model_params.get(name, {}))
        models.append(model)
    return models


def _build_runtime_cfg(job_cfg):
    scheduler = deepcopy(job_cfg.get('scheduler', {}))

    infer_strategy = scheduler.get('infer_strategy', 'split')
    infer_num_worker = scheduler.get('infer_num_worker', 8)
    infer_num_split = scheduler.get('infer_num_split', 8)
    infer_max_num_workers = scheduler.get('infer_max_num_workers', 8)

    eval_n = scheduler.get('eval_n', 999999)
    eval_max_num_workers = scheduler.get('eval_max_num_workers', 1)
    watch_interval = scheduler.get('watch_interval', 3.0)
    heartbeat_timeout = scheduler.get('heartbeat_timeout', 300.0)
    log_interval = scheduler.get('log_interval', 30.0)

    infer_cfg = dict(
        partitioner=dict(
            type=NumWorkerPartitioner,
            strategy=infer_strategy,
            num_worker=infer_num_worker,
            num_split=infer_num_split,
        ),
        runner=dict(
            type=LocalRunner,
            max_num_workers=infer_max_num_workers,
            task=dict(type=OpenICLInferTask),
        ),
    )

    eval_cfg = dict(
        partitioner=dict(type=NaivePartitioner, n=eval_n),
        runner=dict(
            type=LocalRunner,
            max_num_workers=eval_max_num_workers,
            task=dict(
                type=OpenICLEvalWatchTask,
                watch_interval=watch_interval,
                heartbeat_timeout=heartbeat_timeout,
                log_interval=log_interval,
            ),
        ),
    )

    return infer_cfg, eval_cfg


def _load_datasets(dataset_names):
    if not dataset_names:
        raise ValueError('KEYE_JOB["datasets"] must not be empty.')

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_root = os.path.join(repo_root, 'opencompass', 'configs')
    datasets_dir = [
        os.path.join(cfg_root, 'datasets'),
        os.path.join(cfg_root, 'dataset_collections'),
    ]

    datasets = []
    for raw_name in dataset_names:
        if '/' in raw_name:
            dataset_name, dataset_key_suffix = raw_name.split('/', 1)
        else:
            dataset_name, dataset_key_suffix = raw_name, '_datasets'

        try:
            matched_cfgs = match_cfg_file(datasets_dir, [dataset_name])
        except ValueError:
            matched_cfgs = []
            # Fuzzy fallback for user-friendly names.
            pattern = dataset_name if dataset_name.endswith(
                '.py') else f'{dataset_name}.py'
            for ds_dir in datasets_dir:
                matched_cfgs.extend(match_files(ds_dir, pattern, fuzzy=True))
            if not matched_cfgs:
                raise

        for _, cfg_path in matched_cfgs:
            cfg = Config.fromfile(cfg_path)
            for key in cfg.keys():
                if key.endswith(dataset_key_suffix):
                    datasets += cfg[key]

    if not datasets:
        raise ValueError(
            f'No datasets resolved from: {dataset_names}. '
            'Please check dataset names and suffixes.'
        )
    return datasets


KEYE_JOB = _load_keye_job()
models = _build_models(KEYE_JOB)
datasets = _load_datasets(KEYE_JOB['datasets'])
infer, eval = _build_runtime_cfg(KEYE_JOB)
work_dir = os.environ.get('KEYE_WORK_DIR', './outputs/keye_flexible')

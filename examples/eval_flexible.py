from __future__ import annotations

"""General flexible evaluation entry for OpenCompass.

Usage:
    cp examples/eval_flexible.sample.json /tmp/eval_job.json
    FLEX_EVAL_CONFIG=/path/to/eval_job.json python run.py examples/eval_flexible.py --dry-run
    FLEX_EVAL_CONFIG=/path/to/eval_job.json python run.py examples/eval_flexible.py --mode infer
    FLEX_EVAL_CONFIG=/path/to/eval_job.json python run.py examples/eval_flexible.py

Both schemas below are supported:

New schema:
    {
      "models": [{"ref": "gpt_4o_2024_05_13"}],
      "datasets": [{"ref": "aime2025_gen"}]
    }

Legacy convenient schema for local API models:
    {
      "model": {
        "istep0000200_2nd_autothink": {
          "class": "KeyeFastAPI",
          "api_base": "http://127.0.0.1:15553/v1"
        }
      },
      "data": {
        "aime2025": {
          "dataset": "aime2025_gen"
        }
      }
    }
"""

import json
import os
import re
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache

from mmengine.config import Config

from opencompass.utils import match_files
from opencompass.utils.run import match_cfg_file

MODEL_CLASS_ALIASES = {
    'keye': 'KeyeChat',
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

LEGACY_DATASET_KEYS = ('dataset', 'ref', 'name')


def _get_repo_root() -> str:
    current_file = globals().get('__file__')
    if not current_file:
        current_file = os.path.join(os.getcwd(), 'examples', 'eval_flexible.py')
    return os.path.dirname(os.path.dirname(os.path.abspath(current_file)))


def _dedup_files(files: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    deduped = []
    for name, path in files:
        norm = os.path.realpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append((name, path))
    return deduped


def _dedup_strings(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _normalize_name(name: str) -> str:
    normalized = re.sub(r'[^0-9a-zA-Z]+', '_', name).strip('_')
    return normalized.lower()


def _compact_name(name: str) -> str:
    return re.sub(r'[^0-9a-zA-Z]+', '', name).lower()


def _split_ref_suffix(ref_name: str) -> tuple[str, str]:
    if '/' in ref_name:
        return tuple(ref_name.split('/', 1))
    return ref_name, ''


def _dataset_family_aliases(name: str) -> list[str]:
    aliases = [name, name.lower(), _normalize_name(name), _compact_name(name)]
    return _dedup_strings(aliases)

@lru_cache(maxsize=1)
def _build_dataset_family_catalog() -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    repo_root = _get_repo_root()
    index_dir = os.path.join(repo_root, 'opencompass', 'configs',
                             'flexible_eval')
    default_refs_path = os.path.join(index_dir, 'benchmark_default_refs.json')
    variant_refs_path = os.path.join(index_dir, 'benchmark_variant_refs.json')
    if not os.path.isfile(default_refs_path) or not os.path.isfile(
            variant_refs_path):
        raise FileNotFoundError(
            'Flexible dataset mapping files are missing. Expected:\n'
            f'- {default_refs_path}\n'
            f'- {variant_refs_path}'
        )

    with open(default_refs_path, 'r', encoding='utf-8') as f:
        default_refs = json.load(f)
    with open(variant_refs_path, 'r', encoding='utf-8') as f:
        variant_refs = json.load(f)

    alias_to_default_refs: dict[str, set[str]] = defaultdict(set)
    family_to_variants: dict[str, tuple[str, ...]] = {}
    for family, default_ref in default_refs.items():
        variants = tuple(variant_refs.get(family, []))
        family_to_variants[family] = variants
        for alias in _dataset_family_aliases(family):
            alias_to_default_refs[alias].add(default_ref)

    alias_to_default = {}
    for alias, default_refs in alias_to_default_refs.items():
        if len(default_refs) == 1:
            alias_to_default[alias] = next(iter(default_refs))

    # Runtime source of truth lives inside the OpenCompass repo itself:
    # - benchmark_default_refs.json: benchmark family -> default exact ref
    # - benchmark_variant_refs.json: benchmark family -> all exact refs
    return alias_to_default, family_to_variants


def _resolve_cfg_files(cfg_dirs: list[str],
                       ref_name: str,
                       kind: str,
                       allow_fuzzy: bool = True) -> list[tuple[str, str]]:
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
        if not allow_fuzzy:
            raise
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


def _ensure_entry(entry: dict, key_name: str, index: int):
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

    return _normalize_job_cfg(cfg)


def _normalize_legacy_model_map(model_cfg: dict) -> list[dict]:
    if not isinstance(model_cfg, dict):
        raise TypeError('"model" must be a dict in legacy schema.')

    model_entries = []
    for model_name, params in model_cfg.items():
        if not isinstance(params, dict):
            raise TypeError(
                f'Legacy model "{model_name}" must map to a dict.')

        inline_cfg = deepcopy(params)
        inline_cfg.setdefault('name', model_name)
        cls = _normalize_model_class(inline_cfg.get('class', 'KeyeChat'))
        inline_cfg['class'] = cls

        if 'path' not in inline_cfg and 'ckpt' not in inline_cfg:
            if cls == 'KeyeChat':
                inline_cfg['ckpt'] = model_name
            else:
                inline_cfg['path'] = model_name

        model_entries.append(dict(inline=inline_cfg))

    return model_entries


def _normalize_legacy_keye_model_cfg(model_cfg: dict) -> list[dict]:
    if not isinstance(model_cfg, dict):
        raise TypeError('"model" must be a dict in legacy schema.')

    names = deepcopy(model_cfg.get('names', []))
    if not isinstance(names, list):
        raise TypeError('"model.names" must be a list.')
    if not names:
        raise ValueError('"model.names" must not be empty.')

    common_params = deepcopy(model_cfg.get('common_params', {}))
    if not isinstance(common_params, dict):
        raise TypeError('"model.common_params" must be a dict.')

    per_model_params = deepcopy(model_cfg.get('per_model_params', {}))
    if not isinstance(per_model_params, dict):
        raise TypeError('"model.per_model_params" must be a dict.')

    class_name = _normalize_model_class(
        model_cfg.get('class')
        or model_cfg.get('class_name')
        or 'KeyeChat')

    model_entries = []
    for model_name in names:
        if not isinstance(model_name, str):
            raise TypeError('Every value in "model.names" must be a string.')

        inline_cfg = deepcopy(common_params)
        per_model_cfg = deepcopy(per_model_params.get(model_name, {}))
        if not isinstance(per_model_cfg, dict):
            raise TypeError(
                f'"model.per_model_params[{model_name}]" must be a dict.')
        inline_cfg.update(per_model_cfg)
        inline_cfg['class'] = _normalize_model_class(
            inline_cfg.get('class', class_name))
        inline_cfg.setdefault('name', model_name)
        if 'path' not in inline_cfg and 'ckpt' not in inline_cfg:
            if inline_cfg['class'] == 'KeyeChat':
                inline_cfg['ckpt'] = model_name
            else:
                inline_cfg['path'] = model_name
        model_entries.append(dict(inline=inline_cfg))

    return model_entries


def _normalize_legacy_dataset_entries(dataset_cfg) -> list[dict]:
    if isinstance(dataset_cfg, list):
        entries = []
        for idx, one in enumerate(dataset_cfg):
            if isinstance(one, str):
                entries.append(dict(ref=one, aliases=[]))
                continue
            if isinstance(one, dict):
                dataset_ref = None
                for key in LEGACY_DATASET_KEYS:
                    value = one.get(key)
                    if isinstance(value, str) and value:
                        dataset_ref = value
                        break
                if dataset_ref is None:
                    raise ValueError(
                        f'Legacy data[{idx}] must contain one of '
                        f'{LEGACY_DATASET_KEYS}.')
                aliases = []
                for key in ('class', 'name'):
                    value = one.get(key)
                    if isinstance(value, str) and value:
                        aliases.append(value)
                entries.append(dict(ref=dataset_ref, aliases=aliases))
                continue
            raise TypeError(
                f'Legacy data[{idx}] must be a string or dict, got {type(one)}'
            )
        return entries

    if not isinstance(dataset_cfg, dict):
        raise TypeError('"data" must be a dict or list in legacy schema.')

    entries = []
    for dataset_name, params in dataset_cfg.items():
        if isinstance(params, str):
            entries.append(dict(ref=params, aliases=[dataset_name]))
            continue

        if not isinstance(params, dict):
            raise TypeError(
                f'Legacy data "{dataset_name}" must map to a dict or string.')

        dataset_ref = None
        for key in LEGACY_DATASET_KEYS:
            value = params.get(key)
            if isinstance(value, str) and value:
                dataset_ref = value
                break
        if dataset_ref is None:
            dataset_ref = dataset_name
        aliases = [dataset_name]
        class_name = params.get('class')
        if isinstance(class_name, str) and class_name:
            aliases.append(class_name)
        entries.append(dict(ref=dataset_ref, aliases=_dedup_strings(aliases)))

    return entries


def _normalize_job_cfg(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise TypeError('Evaluation config must be a JSON object.')

    normalized = deepcopy(cfg)
    has_new_schema = 'models' in normalized or 'datasets' in normalized
    has_legacy_schema = 'model' in normalized or 'data' in normalized

    if has_new_schema and has_legacy_schema:
        raise ValueError(
            'Please use either top-level "models"/"datasets" or '
            '"model"/"data", not both.'
        )

    if has_legacy_schema:
        if 'model' in normalized:
            model_cfg = normalized.pop('model')
            if isinstance(model_cfg, dict) and 'names' in model_cfg:
                normalized['models'] = _normalize_legacy_keye_model_cfg(
                    model_cfg)
            else:
                normalized['models'] = _normalize_legacy_model_map(model_cfg)
        if 'data' in normalized:
            normalized['datasets'] = _normalize_legacy_dataset_entries(
                normalized.pop('data'))

    return normalized


def _load_models(model_entries: list[dict]) -> list[dict]:
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
            if 'max_out_len' not in inline_cfg and 'max_tokens' in inline_cfg:
                inline_cfg['max_out_len'] = inline_cfg['max_tokens']
            # Support user-friendly name in config and make output path
            # deterministic as "<name>/<dataset>.json".
            if 'name' in inline_cfg and 'abbr' not in inline_cfg:
                inline_cfg['abbr'] = inline_cfg.pop('name')
            else:
                inline_cfg.pop('name', None)
                inline_cfg.setdefault('abbr', f'inline-model-{idx}')

            if 'path' not in inline_cfg and 'ckpt' not in inline_cfg:
                if inline_cfg['type'] == 'KeyeChat':
                    inline_cfg['ckpt'] = inline_cfg['abbr']
                else:
                    inline_cfg['path'] = inline_cfg['abbr']
            models.append(inline_cfg)

    if not models:
        raise ValueError('No model is resolved from "models".')
    return models


def _load_dataset_cfgs_from_ref(dataset_cfg_dirs: list[str], ref_name: str) -> list[dict]:
    if '/' in ref_name:
        dataset_name, suffix = ref_name.split('/', 1)
    else:
        dataset_name, suffix = ref_name, '_datasets'

    matched_cfgs = _resolve_cfg_files(
        dataset_cfg_dirs, dataset_name, 'dataset', allow_fuzzy=False)
    datasets = []
    matched_suffix = False
    for _, cfg_path in matched_cfgs:
        cfg = Config.fromfile(cfg_path)
        for key in cfg.keys():
            if key.endswith(suffix):
                datasets.extend(deepcopy(cfg[key]))
                matched_suffix = True
    if not matched_suffix:
        raise ValueError(
            f'Dataset ref "{ref_name}" is found, but no config key ends with '
            f'"{suffix}".')
    return datasets


def _resolve_default_dataset_ref(raw_ref: str, alias_to_default: dict[str, str]) -> str | None:
    base_name, explicit_suffix = _split_ref_suffix(raw_ref)
    for alias in _dataset_family_aliases(base_name):
        default_ref = alias_to_default.get(alias)
        if default_ref is None:
            continue
        if explicit_suffix:
            return f'{default_ref}/{explicit_suffix}'
        return default_ref
    return None


def _load_datasets(dataset_entries: list[dict]) -> list[dict]:
    if not dataset_entries:
        raise ValueError('"datasets" must not be empty.')

    repo_root = _get_repo_root()
    dataset_cfg_dirs = [
        os.path.join(repo_root, 'opencompass', 'configs', 'datasets'),
        os.path.join(repo_root, 'opencompass', 'configs',
                     'dataset_collections'),
    ]

    datasets = []
    alias_to_default, family_to_variants = _build_dataset_family_catalog()
    for idx, entry in enumerate(dataset_entries):
        _ensure_entry(entry, 'datasets', idx)
        if 'inline' in entry:
            raise ValueError(
                'datasets[].inline is unsupported in this interface. '
                'Please use datasets[].ref.'
            )

        aliases = entry.get('aliases', [])
        if aliases and not isinstance(aliases, list):
            raise TypeError(f'datasets[{idx}].aliases must be a list.')

        raw_refs = _dedup_strings(
            [raw_ref for raw_ref in [entry['ref']] + list(aliases)
             if isinstance(raw_ref, str)])

        last_exact_error = None
        resolved = False
        for raw_ref in raw_refs:
            try:
                datasets.extend(_load_dataset_cfgs_from_ref(dataset_cfg_dirs,
                                                            raw_ref))
                resolved = True
                break
            except Exception as err:
                last_exact_error = err

        if resolved:
            continue

        default_candidates = _dedup_strings([
            default_ref for default_ref in (
                _resolve_default_dataset_ref(raw_ref, alias_to_default)
                for raw_ref in raw_refs) if default_ref is not None
        ])

        last_default_error = None
        for default_ref in default_candidates:
            try:
                datasets.extend(_load_dataset_cfgs_from_ref(dataset_cfg_dirs,
                                                            default_ref))
                resolved = True
                break
            except Exception as err:
                last_default_error = err

        if resolved:
            continue

        family_hints = _dedup_strings([
            raw_ref for raw_ref in raw_refs
            if _resolve_default_dataset_ref(raw_ref, alias_to_default)
        ])
        if family_hints:
            mapped_refs = ', '.join(
                [f'{family} -> {_resolve_default_dataset_ref(family, alias_to_default)}'
                 for family in family_hints])
            raise ValueError(
                f'Unable to resolve dataset entry {entry["ref"]!r}. Exact ref '
                f'takes precedence; current family defaults are: {mapped_refs}. '
                f'If you need a different recipe, please write the exact ref '
                f'from dataset_refs.txt. Last error: {last_default_error or last_exact_error}'
            )

        available_families = ', '.join(sorted(family_to_variants.keys())[:20])
        raise ValueError(
            f'Unable to resolve dataset entry {entry["ref"]!r}. It is neither '
            f'an exact dataset ref nor a known benchmark family. '
            f'Example benchmark families: {available_families}. '
            f'Last error: {last_exact_error}'
        )

    if not datasets:
        raise ValueError('No dataset is resolved from "datasets".')
    return datasets


def _build_runtime_cfg(job_cfg: dict):
    scheduler = deepcopy(DEFAULT_SCHEDULER)
    scheduler.update(deepcopy(job_cfg.get('scheduler', {})))

    infer_cfg = dict(
        partitioner=dict(
            type='NumWorkerPartitioner',
            strategy=scheduler['infer_strategy'],
            num_worker=scheduler['infer_num_worker'],
            num_split=scheduler['infer_num_split'],
        ),
        runner=dict(
            type='LocalRunner',
            max_num_workers=scheduler['infer_max_num_workers'],
            # Use the concurrent API infer task so multiple datasets can make
            # progress at the same time instead of waiting for strict
            # dataset-by-dataset completion.
            task=dict(
                type='OpenICLInferConcurrentTask',
                poll_interval=scheduler['watch_interval'],
                log_interval=scheduler['log_interval'],
            ),
        ),
    )

    eval_cfg = dict(
        partitioner=dict(type='NaivePartitioner', n=scheduler['eval_n']),
        runner=dict(
            type='LocalRunner',
            max_num_workers=scheduler['eval_max_num_workers'],
            task=dict(
                type='OpenICLEvalWatchTask',
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

if os.environ.get('FLEX_EVAL_KEEP_INTERNALS') != '1':
    for _internal_name in [
            'FLEX_JOB',
            'MODEL_CLASS_ALIASES',
            'DEFAULT_SCHEDULER',
            'DEFAULT_JOB',
            'LEGACY_DATASET_KEYS',
            '_get_repo_root',
            '_dedup_files',
            '_dedup_strings',
            '_normalize_name',
            '_compact_name',
            '_split_ref_suffix',
            '_dataset_family_aliases',
            '_build_dataset_family_catalog',
            '_resolve_cfg_files',
            '_ensure_entry',
            '_normalize_model_class',
            '_load_job_cfg',
            '_iter_legacy_models',
            '_normalize_legacy_job_cfg',
            '_resolve_model_cfgs_from_ref',
            '_load_models',
            '_load_dataset_cfgs_from_ref',
            '_resolve_default_dataset_ref',
            '_load_datasets',
            '_build_runtime_cfg',
            'json',
            'os',
            're',
            'defaultdict',
            'deepcopy',
            'lru_cache',
            'Config',
            'match_files',
            'match_cfg_file',
    ]:
        globals().pop(_internal_name, None)
    del _internal_name

for _unused_name in [
        'Config', 'DEFAULT_JOB', 'DEFAULT_SCHEDULER', 'LEGACY_DATASET_KEYS',
        'MODEL_CLASS_ALIASES', 'annotations'
]:
    globals().pop(_unused_name, None)

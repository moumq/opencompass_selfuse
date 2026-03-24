import argparse
import ast
import datetime as dt
import json
import os
from typing import Dict, List

try:
    from mmengine.config import Config  # type: ignore
except Exception:  # pragma: no cover
    Config = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export reusable model/dataset config catalog for '
        'eval_flexible.py')
    parser.add_argument(
        '--repo-root',
        default='.',
        help='OpenCompass repo root. Defaults to current directory.')
    parser.add_argument(
        '--output',
        default='',
        help='Output path. If not provided, print JSON to stdout.')
    parser.add_argument(
        '--format',
        choices=['json', 'markdown'],
        default='json',
        help='Output format.')
    return parser.parse_args()


def _iter_py_files(root: str) -> List[str]:
    files = []
    for cur_root, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            if filename == '__init__.py':
                continue
            files.append(os.path.join(cur_root, filename))
    return sorted(files)


def _to_rel(path: str, repo_root: str) -> str:
    return os.path.relpath(path, repo_root).replace(os.sep, '/')


def _path_ref(path: str, base_root: str) -> str:
    rel = os.path.relpath(path, base_root).replace(os.sep, '/')
    if rel.endswith('.py'):
        rel = rel[:-3]
    return rel


def _extract_top_level_names(py_path: str) -> List[str]:
    with open(py_path, 'r', encoding='utf-8') as f:
        src = f.read()
    try:
        tree = ast.parse(src, filename=py_path)
    except SyntaxError:
        return []

    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.append(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.append(alias.asname or alias.name.split('.')[-1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.asname or alias.name.split('.')[-1])
    return names


def _extract_cfg_keys(py_path: str) -> List[str]:
    if Config is None:
        return []
    try:
        cfg = Config.fromfile(py_path)
        return list(cfg.keys())
    except Exception:
        return []


def _extract_model_file_valid(py_path: str) -> bool:
    cfg_keys = _extract_cfg_keys(py_path)
    if cfg_keys:
        return 'models' in cfg_keys
    return 'models' in _extract_top_level_names(py_path)


def _extract_dataset_suffix_keys(py_path: str) -> List[str]:
    cfg_keys = _extract_cfg_keys(py_path)
    if cfg_keys:
        return sorted({k for k in cfg_keys if k.endswith('_datasets')})
    names = _extract_top_level_names(py_path)
    return sorted({name for name in names if name.endswith('_datasets')})


def _collect_models(repo_root: str) -> List[Dict]:
    model_root = os.path.join(repo_root, 'opencompass', 'configs', 'models')
    files = _iter_py_files(model_root)
    entries = []
    for path in files:
        if not _extract_model_file_valid(path):
            continue
        basename = os.path.basename(path)[:-3]
        entries.append(
            dict(
                ref=basename,
                path_ref=_path_ref(path, model_root),
                config_path=_to_rel(path, repo_root),
            ))
    return entries


def _collect_datasets(repo_root: str) -> List[Dict]:
    dataset_roots = [
        os.path.join(repo_root, 'opencompass', 'configs', 'datasets'),
        os.path.join(repo_root, 'opencompass', 'configs',
                     'dataset_collections'),
    ]
    entries = []

    for root in dataset_roots:
        files = _iter_py_files(root)
        for path in files:
            basename = os.path.basename(path)[:-3]
            suffix_keys = _extract_dataset_suffix_keys(path)

            # Default suffix ref used by eval_flexible.py
            entries.append(
                dict(
                    ref=basename,
                    suffix='_datasets',
                    path_ref=_path_ref(path, root),
                    config_path=_to_rel(path, repo_root),
                ))

            for key in suffix_keys:
                entries.append(
                    dict(
                        ref=f'{basename}/{key}',
                        suffix=key,
                        path_ref=_path_ref(path, root),
                        config_path=_to_rel(path, repo_root),
                    ))
    return entries


def _render_markdown(catalog: Dict) -> str:
    lines = []
    lines.append('# OpenCompass Flexible Config Catalog')
    lines.append('')
    lines.append(f'- Generated at: {catalog["generated_at"]}')
    lines.append(f'- Model refs: {len(catalog["models"])}')
    lines.append(f'- Dataset refs: {len(catalog["datasets"])}')
    lines.append('')
    lines.append('## Model Refs')
    lines.append('')
    lines.append('| ref | path_ref | config_path |')
    lines.append('|---|---|---|')
    for item in catalog['models']:
        lines.append(
            f'| {item["ref"]} | {item["path_ref"]} | {item["config_path"]} |'
        )
    lines.append('')
    lines.append('## Dataset Refs')
    lines.append('')
    lines.append('| ref | suffix | path_ref | config_path |')
    lines.append('|---|---|---|---|')
    for item in catalog['datasets']:
        lines.append(
            f'| {item["ref"]} | {item["suffix"]} | '
            f'{item["path_ref"]} | {item["config_path"]} |')
    lines.append('')
    return '\n'.join(lines)


def main():
    args = parse_args()
    repo_root = os.path.abspath(args.repo_root)
    catalog = dict(
        generated_at=dt.datetime.now().isoformat(timespec='seconds'),
        repo_root=repo_root,
        models=_collect_models(repo_root),
        datasets=_collect_datasets(repo_root),
    )

    if args.format == 'json':
        payload = json.dumps(catalog, ensure_ascii=False, indent=2)
    else:
        payload = _render_markdown(catalog)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(payload)
        print(f'Catalog written to: {args.output}')
    else:
        print(payload)


if __name__ == '__main__':
    main()

import os

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils.logging import get_logger

from .base import BaseDataset

_LOCAL_CACHE_SUBDIR = 'data/hle'
_LOCAL_FILENAME = 'hle.jsonl'


def _get_local_cache_path() -> str | None:
    """Return the local JSONL cache path if COMPASS_DATA_CACHE is set."""
    cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
    if not cache_dir:
        return None
    return os.path.join(cache_dir, _LOCAL_CACHE_SUBDIR, _LOCAL_FILENAME)


@LOAD_DATASET.register_module()
class HLEDataset(BaseDataset):

    @staticmethod
    def load(path: str, category: str | None = None):
        logger = get_logger()
        local_path = _get_local_cache_path()

        # --- Try local cache first ---
        if local_path and os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            logger.info(f'Loading HLE from local cache: {local_path}')
            ds = Dataset.from_json(local_path)
        else:
            # --- Download from HuggingFace Hub ---
            logger.info(
                f'Local cache not found, downloading HLE '
                f'from HuggingFace Hub: {path}')
            dataset = load_dataset(path)
            ds = dataset['test']

            # Filter out image-based questions BEFORE caching
            # (image column contains PIL objects that can't be serialized)
            ds = ds.filter(lambda x: x['image'] == '')
            # Drop the image column to avoid serialization issues
            if 'image' in ds.column_names:
                ds = ds.remove_columns(['image'])

            # Save to local cache for future runs
            if local_path and len(ds) > 0:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                try:
                    ds.to_json(local_path, force_ascii=True)
                except Exception:
                    import json
                    with open(local_path, 'w', encoding='utf-8') as f:
                        for row in ds:
                            f.write(json.dumps(
                                {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                                 for k, v in row.items()},
                                ensure_ascii=False) + '\n')
                logger.info(f'Saved HLE cache to: {local_path}')

        # Filter text-only (for cached data that may still have image column)
        if 'image' in ds.column_names:
            ds = ds.filter(lambda x: x['image'] == '')
        if category:
            ds = ds.filter(lambda x: x['category'] == category)
        ds = ds.rename_column('question', 'problem')

        return DatasetDict(train=ds, test=ds)

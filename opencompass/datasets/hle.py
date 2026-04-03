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
        if local_path and os.path.isfile(local_path):
            logger.info(f'Loading HLE from local cache: {local_path}')
            ds = Dataset.from_json(local_path)
        else:
            # --- Download from HuggingFace Hub ---
            logger.info(
                f'Local cache not found, downloading HLE '
                f'from HuggingFace Hub: {path}')
            dataset = load_dataset(path)
            ds = dataset['test']

            # Save to local cache for future runs
            if local_path:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                ds.to_json(local_path)
                logger.info(f'Saved HLE cache to: {local_path}')

        ds = ds.filter(lambda x: x['image'] == '')
        if category:
            ds = ds.filter(lambda x: x['category'] == category)
        ds = ds.rename_column('question', 'problem')

        return DatasetDict(train=ds, test=ds)

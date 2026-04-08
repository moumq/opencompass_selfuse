import os

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils.logging import get_logger

from .base import BaseDataset

_LOCAL_CACHE_SUBDIR = 'data/hle_verified'
_LOCAL_FILENAME = 'hle_verified.jsonl'


def _get_local_cache_path() -> str | None:
    """Return the local JSONL cache path if COMPASS_DATA_CACHE is set."""
    cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
    if not cache_dir:
        return None
    return os.path.join(cache_dir, _LOCAL_CACHE_SUBDIR, _LOCAL_FILENAME)


@LOAD_DATASET.register_module()
class HLEVerifiedDataset(BaseDataset):
    """Loads the HLE-Verified dataset (skylenage/HLE-Verified).

    Data loading priority:
      1. Local JSONL cache at ``$COMPASS_DATA_CACHE/data/hle_verified/``
      2. HuggingFace Hub (downloads & caches locally for next time)

    By default uses Gold + Revision subsets (verified correct questions).
    Set ``subset`` to control which verified classes to include:
      - 'gold'      : only Gold subset (668 items, unmodified confirmed correct)
      - 'revision'  : only Revision subset (1143 items, corrected & re-verified)
      - 'gold+revision' : both (default, 1811 items)
      - 'all'       : all including Uncertain (2500 items)
    """

    @staticmethod
    def load(path: str = 'skylenage/HLE-Verified',
             subset: str = 'gold+revision',
             category: str | None = None):
        logger = get_logger()
        local_path = _get_local_cache_path()

        # --- Try local cache first ---
        if local_path and os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            logger.info(f'Loading HLE-Verified from local cache: {local_path}')
            ds = Dataset.from_json(local_path)
        else:
            # --- Download from HuggingFace Hub ---
            logger.info(
                f'Local cache not found, downloading HLE-Verified '
                f'from HuggingFace Hub: {path}')
            dataset = load_dataset(path)
            split_key = list(dataset.keys())[0]
            ds = dataset[split_key]

            # Save to local cache for future runs
            if local_path:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                ds.to_json(local_path)
                logger.info(f'Saved HLE-Verified cache to: {local_path}')

        # Filter by Verified_Classes
        subset_lower = subset.lower()
        if subset_lower != 'all':
            allowed = set()
            if 'gold' in subset_lower:
                allowed.add('Gold subset')
            if 'revision' in subset_lower:
                allowed.add('Revision subset')
            if 'uncertain' in subset_lower:
                allowed.add('Uncertain subset')
            if allowed:
                ds = ds.filter(lambda x: x['Verified_Classes'] in allowed)

        # Filter out image-based questions (keep text-only)
        if 'image' in ds.column_names:
            ds = ds.filter(lambda x: x['image'] == '')

        # Optional category filter
        if category:
            ds = ds.filter(lambda x: x['category'] == category)

        # Rename to match HLE convention
        ds = ds.rename_column('question', 'problem')

        return DatasetDict(train=ds, test=ds)

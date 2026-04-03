import os

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils.logging import get_logger

from .base import BaseDataset

_LOCAL_CACHE_SUBDIR = 'data/simpleqa_verified'
_LOCAL_FILENAME = 'simpleqa_verified.jsonl'


def _get_local_cache_path() -> str | None:
    """Return the local JSONL cache path if COMPASS_DATA_CACHE is set."""
    cache_dir = os.environ.get('COMPASS_DATA_CACHE', '')
    if not cache_dir:
        return None
    return os.path.join(cache_dir, _LOCAL_CACHE_SUBDIR, _LOCAL_FILENAME)


@LOAD_DATASET.register_module()
class SimpleQAVerifiedDataset(BaseDataset):
    """Loads SimpleQA Verified (google/simpleqa-verified).

    Data loading priority:
      1. Local JSONL cache at ``$COMPASS_DATA_CACHE/data/simpleqa_verified/``
      2. HuggingFace Hub (downloads & caches locally for next time)

    1000 verified, deduplicated, topic-balanced questions from Google DeepMind.
    Schema: problem, answer, topic, answer_type, etc.
    """

    @staticmethod
    def load(path: str = 'google/simpleqa-verified', **kwargs):
        logger = get_logger()
        local_path = _get_local_cache_path()

        if local_path and os.path.isfile(local_path):
            logger.info(
                f'Loading SimpleQA-Verified from local cache: {local_path}')
            dataset = Dataset.from_json(local_path)
        else:
            logger.info(
                f'Local cache not found, downloading SimpleQA-Verified '
                f'from HuggingFace Hub: {path}')
            dataset = load_dataset(path, split='eval')

            if local_path:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                dataset.to_json(local_path)
                logger.info(
                    f'Saved SimpleQA-Verified cache to: {local_path}')

        return DatasetDict(train=dataset, test=dataset)

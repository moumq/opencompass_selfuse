from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SimpleQAVerifiedDataset(BaseDataset):
    """Loads SimpleQA Verified (google/simpleqa-verified).

    1000 verified, deduplicated, topic-balanced questions from Google DeepMind.
    Schema: problem, answer, topic, answer_type, etc.
    """

    @staticmethod
    def load(path: str = 'google/simpleqa-verified', **kwargs):
        dataset = load_dataset(path, split='eval')
        # The dataset already uses 'problem' and 'answer' columns,
        # matching the SimpleQA convention.
        return DatasetDict(train=dataset, test=dataset)

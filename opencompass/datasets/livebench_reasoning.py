from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class LiveBenchReasoningDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'livebench/reasoning', split: str = 'test', **kwargs):
        hf_dataset = load_dataset(path, split=split)

        rows = []
        for row in hf_dataset:
            turns = row.get('turns', [])
            question = turns[0] if isinstance(turns, list) and len(turns) > 0 else ''
            rows.append({
                'question': str(question),
                'answer': str(row.get('ground_truth', '')),
                'question_id': str(row.get('question_id', '')),
                'category': str(row.get('category', '')),
                'task': str(row.get('task', '')),
                'level': row.get('level', None),
            })

        dataset = Dataset.from_list(rows)
        return DatasetDict(train=dataset, test=dataset)

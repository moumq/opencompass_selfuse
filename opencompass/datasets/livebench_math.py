from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset

DEFAULT_LIVEBENCH_RELEASE_OPTION = '2024-11-25'


def _match_release(row: dict, release_version: str) -> bool:
    if not release_version:
        return True
    candidate_fields = [
        'livebench_release_date',
        'release_date',
        'release',
        'version',
    ]
    for field in candidate_fields:
        value = str(row.get(field, '')).strip()
        if value:
            return value == release_version
    return True


@LOAD_DATASET.register_module()
class LiveBenchMathDataset(BaseDataset):

    @staticmethod
    def load(path: str = 'livebench/math',
             split: str = 'test',
             release_version: str = DEFAULT_LIVEBENCH_RELEASE_OPTION,
             **kwargs):
        hf_dataset = load_dataset(path, split=split)

        rows = []
        for row in hf_dataset:
            if not _match_release(row, release_version):
                continue
            turns = row.get('turns', [])
            question = turns[0] if isinstance(turns, list) and len(turns) > 0 else ''
            sample = dict(row)
            sample['question'] = str(question)
            sample['answer'] = str(row.get('ground_truth', ''))
            sample['question_id'] = str(row.get('question_id', ''))
            sample['livebench_release_date'] = str(
                row.get('livebench_release_date',
                        row.get('release_date', release_version)))
            sample['livebench_release_option'] = release_version
            rows.append(sample)

        dataset = Dataset.from_list(rows)
        return DatasetDict(train=dataset, test=dataset)

import os

from datasets import Dataset, DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class IMOAnswerBenchDataset(BaseDataset):

    @staticmethod
    def load(
        path: str =
        'https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench_v2.csv',
        split: str = 'train',
        question_column: str = 'Problem',
        answer_column: str = 'Short Answer',
        id_column: str = 'Problem ID',
        **kwargs,
    ):
        if path.startswith('http://') or path.startswith('https://'):
            csv_path = path
        else:
            local_path = get_data_path(path, local_mode=True)
            csv_path = (os.path.join(local_path, 'answerbench_v2.csv')
                        if os.path.isdir(local_path) else local_path)

        hf_dataset = load_dataset('csv', data_files=csv_path, split=split)

        rows = []
        for row in hf_dataset:
            question = str(row.get(question_column, '')).strip()
            answer = str(row.get(answer_column, '')).strip()
            rows.append({
                'question': question,
                'answer': answer,
                'problem_id': str(row.get(id_column, '')),
                'category': str(row.get('Category', '')),
                'subcategory': str(row.get('Subcategory', '')),
                'source': str(row.get('Source', '')),
            })

        dataset = Dataset.from_list(rows)
        return DatasetDict(train=dataset, test=dataset)

from datasets import DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class HLEVerifiedDataset(BaseDataset):
    """Loads the HLE-Verified dataset (skylenage/HLE-Verified).

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
        dataset = load_dataset(path)
        # The dataset has a single 'train' split containing all subsets
        split_key = list(dataset.keys())[0]
        ds = dataset[split_key]

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

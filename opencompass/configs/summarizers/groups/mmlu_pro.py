categories = ['math', 'physics', 'chemistry', 'law', 'engineering', 'other', 'economics', 'health', 'psychology', 'business', 'biology', 'philosophy', 'computer science', 'history']

# Sample counts per category in the MMLU-Pro test set (TIGER-Lab/MMLU-Pro),
# used for sample-weighted (micro) averaging instead of naive subcategory averaging.
_mmlu_pro_sample_counts = {
    'biology': 717,
    'business': 789,
    'chemistry': 1132,
    'computer_science': 410,
    'economics': 844,
    'engineering': 969,
    'health': 818,
    'history': 381,
    'law': 1101,
    'math': 1351,
    'other': 924,
    'philosophy': 499,
    'physics': 1299,
    'psychology': 798,
}

mmlu_pro_summary_groups = [
    {
        'name': 'mmlu_pro',
        'subsets': ['mmlu_pro_' + c.replace(' ', '_') for c in categories],
        'weights': {'mmlu_pro_' + k: v for k, v in _mmlu_pro_sample_counts.items()},
    },
]

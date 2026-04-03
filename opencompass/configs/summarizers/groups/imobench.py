IMOBENCH_NUM_RUNS = 8

imobench_summary_groups = [
    {
        'name': 'IMO-AnswerBench-v2',
        'subsets': [f'IMO-AnswerBench-v2-run{i}' for i in range(IMOBENCH_NUM_RUNS)],
    },
]

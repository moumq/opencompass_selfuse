from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    LiveBenchEvaluator,
    LiveBenchMathDataset,
    LiveBenchReasoningDataset,
)

LIVEBENCH_RELEASE_OPTION = '2024-11-25'

livebench_reader_cfg = dict(input_columns=['question'], output_column='answer')

livebench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{question}',
            )
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=4096),
)

livebench_eval_cfg = dict(
    evaluator=dict(type=LiveBenchEvaluator),
    pred_role='BOT',
)

livebench_datasets = [
    dict(
        abbr=f'LiveBench-Reasoning-{LIVEBENCH_RELEASE_OPTION}',
        type=LiveBenchReasoningDataset,
        path='livebench/reasoning',
        split='test',
        release_version=LIVEBENCH_RELEASE_OPTION,
        reader_cfg=livebench_reader_cfg,
        infer_cfg=livebench_infer_cfg,
        eval_cfg=livebench_eval_cfg,
        mode='singlescore',
    ),
    dict(
        abbr=f'LiveBench-Math-{LIVEBENCH_RELEASE_OPTION}',
        type=LiveBenchMathDataset,
        path='livebench/math',
        split='test',
        release_version=LIVEBENCH_RELEASE_OPTION,
        reader_cfg=livebench_reader_cfg,
        infer_cfg=livebench_infer_cfg,
        eval_cfg=livebench_eval_cfg,
        mode='singlescore',
    ),
]

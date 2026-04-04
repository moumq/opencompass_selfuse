from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import CustomDataset
from opencompass.evaluator import MATHVerifyEvaluator

hmmt_nov_reader_cfg = dict(input_columns=['problem'], output_column='answer')

hmmt_nov_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\nPlease put your final answer within \\boxed{}.',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Rule-based evaluation using math_verify (sympy symbolic comparison),
# matching MathArena's official grading approach.
hmmt_nov_eval_cfg = dict(
    evaluator=dict(type=MATHVerifyEvaluator),
)

hmmt_nov_datasets = [
    dict(
        type=CustomDataset,
        abbr='hmmt_nov',
        path='data/hmmt_nov/hmmt_nov.jsonl',
        local_mode=True,
        reader_cfg=hmmt_nov_reader_cfg,
        infer_cfg=hmmt_nov_infer_cfg,
        eval_cfg=hmmt_nov_eval_cfg,
        n=1,
    )
]

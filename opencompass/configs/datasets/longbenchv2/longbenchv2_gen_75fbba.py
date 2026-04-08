from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LongBenchv2Dataset, LongBenchv2Evaluator
from opencompass.utils.text_postprocessors import last_option_postprocess

LongBenchv2_reader_cfg = dict(
    input_columns=['context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'difficulty', 'length'],
    output_column='answer',
)

# Official LongBench v2 0-shot prompt (no CoT) from
# https://github.com/THUDM/LongBench/blob/main/prompts/0shot.txt
LongBenchv2_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\nWhat is the correct answer to this question: {question}\nChoices:\n(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\n\nFormat your response as follows: "The correct answer is (insert answer here)".',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBenchv2_eval_cfg = dict(
    evaluator=dict(type=LongBenchv2Evaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=last_option_postprocess, options='ABCD')
)

LongBenchv2_datasets = [
    dict(
        type=LongBenchv2Dataset,
        abbr='LongBenchv2',
        path='opencompass/longbenchv2',
        reader_cfg=LongBenchv2_reader_cfg,
        infer_cfg=LongBenchv2_infer_cfg,
        eval_cfg=LongBenchv2_eval_cfg,
    )
]

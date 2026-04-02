from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import (
    IMOAnswerBenchDataset,
    generic_llmjudge_postprocess,
    math_postprocess_v2,
)

ANSWERBENCH_V1_PATH = (
    'https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench.csv'
)
ANSWERBENCH_V2_PATH = (
    'https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench_v2.csv'
)

imobench_reader_cfg = dict(input_columns=['question'], output_column='answer')

imobench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.',
            )
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8192),
)

GRADER_TEMPLATE = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given.
2. Because the candidate's answer may be different from the standard answer in form, please judge semantic equivalence.
3. If the prediction is given with \\boxed{}, ignore \\boxed{} and only judge answer consistency.

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
Just return "A" or "B", with no extra text.

<Original Question Begin>:
{question}
<Original Question End>

<Gold Target Begin>:
{answer}
<Gold Target End>

<Predicted Answer Begin>:
{prediction}
<Predicted End>
""".strip()


def make_imobench_eval_cfg(path: str):
    return dict(
        evaluator=dict(
            type=GenericLLMEvaluator,
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin=[
                        dict(
                            role='SYSTEM',
                            fallback_role='HUMAN',
                            prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                        )
                    ],
                    round=[dict(role='HUMAN', prompt=GRADER_TEMPLATE)],
                ),
            ),
            dataset_cfg=dict(
                type=IMOAnswerBenchDataset,
                path=path,
                reader_cfg=imobench_reader_cfg,
            ),
            judge_cfg=dict(),
            dict_postprocessor=dict(type=generic_llmjudge_postprocess),
        ),
        pred_postprocessor=dict(type=math_postprocess_v2),
        pred_role='BOT',
    )


imobench_datasets = [
    dict(
        abbr='IMO-AnswerBench-v1',
        type=IMOAnswerBenchDataset,
        path=ANSWERBENCH_V1_PATH,
        split='train',
        reader_cfg=imobench_reader_cfg,
        infer_cfg=imobench_infer_cfg,
        eval_cfg=make_imobench_eval_cfg(ANSWERBENCH_V1_PATH),
        mode='singlescore',
    ),
    dict(
        abbr='IMO-AnswerBench-v2',
        type=IMOAnswerBenchDataset,
        path=ANSWERBENCH_V2_PATH,
        split='train',
        reader_cfg=imobench_reader_cfg,
        infer_cfg=imobench_infer_cfg,
        eval_cfg=make_imobench_eval_cfg(ANSWERBENCH_V2_PATH),
        mode='singlescore',
    ),
]

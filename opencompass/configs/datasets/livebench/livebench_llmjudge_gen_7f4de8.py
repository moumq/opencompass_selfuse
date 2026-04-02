from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import (
    LiveBenchMathDataset,
    LiveBenchReasoningDataset,
    generic_llmjudge_postprocess,
)

livebench_reader_cfg = dict(input_columns=['question'], output_column='answer')

livebench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Question: {question}\nPlease reason step by step, and put your final answer clearly at the end.',
            )
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=8192),
)

GRADER_TEMPLATE = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given.
2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
3. Some answers may be expressed in different ways (symbolic or textual). As long as the meaning is equivalent, consider it correct.
4. If the prediction is given with \\boxed{}, ignore \\boxed{} and only judge answer consistency.

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
Just return the letters "A" or "B", with no text around it.

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

livebench_eval_cfg = dict(
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
        judge_cfg=dict(),
        dict_postprocessor=dict(type=generic_llmjudge_postprocess),
    ),
    pred_role='BOT',
)

livebench_datasets = [
    dict(
        abbr='LiveBench-Reasoning',
        type=LiveBenchReasoningDataset,
        path='livebench/reasoning',
        split='test',
        reader_cfg=livebench_reader_cfg,
        infer_cfg=livebench_infer_cfg,
        eval_cfg=dict(
            **livebench_eval_cfg,
            evaluator=dict(
                **livebench_eval_cfg['evaluator'],
                dataset_cfg=dict(
                    type=LiveBenchReasoningDataset,
                    path='livebench/reasoning',
                    split='test',
                    reader_cfg=livebench_reader_cfg,
                ),
            ),
        ),
        mode='singlescore',
    ),
    dict(
        abbr='LiveBench-Math',
        type=LiveBenchMathDataset,
        path='livebench/math',
        split='test',
        reader_cfg=livebench_reader_cfg,
        infer_cfg=livebench_infer_cfg,
        eval_cfg=dict(
            **livebench_eval_cfg,
            evaluator=dict(
                **livebench_eval_cfg['evaluator'],
                dataset_cfg=dict(
                    type=LiveBenchMathDataset,
                    path='livebench/math',
                    split='test',
                    reader_cfg=livebench_reader_cfg,
                ),
            ),
        ),
        mode='singlescore',
    ),
]

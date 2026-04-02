from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.datasets import (
    IMOAnswerBenchDataset,
    generic_llmjudge_postprocess,
    math_postprocess_v2,
)

ANSWERBENCH_V2_PATH = (
    'https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/answerbench_v2.csv'
)
IMOBENCH_NUM_RUNS = 8

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
# System Role: Deterministic Mathematical Autograder

You are a precise, automated grading system. Your sole function is to determine if the final answer provided in the Model Solution is mathematically equivalent to the Golden Answer. You must not grade the reasoning or steps, only the final result.

# 1. Grading Guidelines (Equivalence Rules)
Equivalence is mandatory for a correct grade. You must rigorously verify if the answers represent the exact same mathematical value or expression, even if the format differs.
- Algebraic Equivalence: e.g., n(n+1)/2 is equivalent to n^2/2 + n/2.
- Numerical Equivalence: e.g., 1/2 is equivalent to 0.5; sqrt(2)/2 is equivalent to 1/sqrt(2).
- Set/List Equivalence: Unless specified as an ordered tuple/vector, the order of elements does not matter.
- Partial Credit: No partial credit is allowed. If the answer is incomplete or partially incorrect, it is incorrect.
- No Answers: If no clear, unambiguous final answer can be extracted, the solution must be graded as incorrect.

# 2. Output Protocol (Strict Compliance Required)
You must execute the task using a two-part structure.

Part 1: Analysis
You MUST perform your analysis within <thinking></thinking> tags and follow these steps:
1. Golden Answer: State the Golden Answer.
2. Extracted Model Answer: State the extracted answer. If none found, state "No clear final answer found."
3. Equivalence Analysis: Compare the two answers using the grading guidelines.
4. Conclusion: State the final determination ("Correct" or "Incorrect").

Part 2: Final Grade
Immediately following the closing </thinking> tag, output ONLY one of:
- \\boxed{Correct}
- \\boxed{Incorrect}

Do not add any text outside the <thinking> tags or the final \\boxed{} output.

Problem: {question}
Model Solution: {prediction}
Golden Answer: {answer}
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
                            prompt='You are a precise, automated grading system for mathematical answers.',
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
            dict_postprocessor=dict(
                type=generic_llmjudge_postprocess,
                true_tag='Correct',
                false_tag='Incorrect',
            ),
        ),
        pred_postprocessor=dict(type=math_postprocess_v2),
        pred_role='BOT',
    )


imobench_datasets = [
    dict(
        abbr=f'IMO-AnswerBench-v2-run{idx}',
        type=IMOAnswerBenchDataset,
        path=ANSWERBENCH_V2_PATH,
        split='train',
        reader_cfg=imobench_reader_cfg,
        infer_cfg=imobench_infer_cfg,
        eval_cfg=make_imobench_eval_cfg(ANSWERBENCH_V2_PATH),
        mode='singlescore',
    )
    for idx in range(IMOBENCH_NUM_RUNS)
]

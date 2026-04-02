import re

from opencompass.registry import DICT_POSTPROCESSORS
from opencompass.utils import get_logger


def get_final_results(judged_answers,
                      references,
                      origial_responses,
                      metric_name='accuracy',
                      true_tag: str = 'A',
                      false_tag: str = 'B'):
    count = 0
    is_correct_count = 0
    is_incorrect_count = 0
    is_not_attempted_count = 0
    attempted_judge_count = 0
    details = []
    for i, j, k in zip(judged_answers, references, origial_responses):
        if i in [true_tag, false_tag]:
            attempted_judge_count += 1
        grade_letter = i
        detail = {
            'pred': k,
            'ref': j,
            'origin_grade_response': i,
            'grade_letter': grade_letter,
            'correct': False,
        }
        count += 1
        if grade_letter == true_tag:
            is_correct_count += 1
            detail['correct'] = True
        elif grade_letter == false_tag:
            is_incorrect_count += 1
        else:
            is_not_attempted_count += 1
        details.append(detail)

    is_correct = is_correct_count / count
    is_incorrect = is_incorrect_count / count
    is_given_attempted = is_correct + is_incorrect
    accuracy_given_attempted = (is_correct / is_given_attempted
                                if is_given_attempted > 0 else 0)
    attempted_judge_ratio = attempted_judge_count / count

    f1 = (2 * accuracy_given_attempted * is_correct /
          (accuracy_given_attempted + is_correct) if
          (accuracy_given_attempted + is_correct) > 0 else 0)
    result = {
        metric_name: is_correct * 100,
        f'{metric_name}_given_attempted': accuracy_given_attempted * 100,
        'f1': f1,
        'attempted_ratio': attempted_judge_ratio * 100,
        'correct_count': is_correct_count,
        'incorrect_count': is_incorrect_count,
        'not_attempted_count': is_not_attempted_count,
        'details': details,
    }
    return result


def _normalize_judge_tag(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def _generic_llmjudge_postprocess(judgement: str,
                                  true_tag: str = 'A',
                                  false_tag: str = 'B'):
    judgement = str(judgement or '').strip()
    true_tag_norm = _normalize_judge_tag(true_tag)
    false_tag_norm = _normalize_judge_tag(false_tag)

    boxed_pattern = re.compile(
        rf'\\boxed\s*\{{\s*({re.escape(true_tag_norm)}|{re.escape(false_tag_norm)})\s*\}}',
        flags=re.IGNORECASE,
    )
    match = boxed_pattern.search(judgement)
    if match:
        tag = _normalize_judge_tag(match.group(1))
        if tag.lower() == true_tag_norm.lower():
            return true_tag
        if tag.lower() == false_tag_norm.lower():
            return false_tag

    normalized = _normalize_judge_tag(judgement)
    exact_pattern = re.compile(
        rf'^(?:({re.escape(true_tag_norm)})|({re.escape(false_tag_norm)}))$',
        flags=re.IGNORECASE,
    )
    match = exact_pattern.match(normalized)
    if match:
        if match.group(1):
            return true_tag
        if match.group(2):
            return false_tag

    line_pattern = re.compile(
        rf'(?mi)^\s*({re.escape(true_tag_norm)}|{re.escape(false_tag_norm)})\s*$')
    match = line_pattern.search(judgement)
    if match:
        tag = _normalize_judge_tag(match.group(1))
        if tag.lower() == true_tag_norm.lower():
            return true_tag
        if tag.lower() == false_tag_norm.lower():
            return false_tag

    token_pattern = re.compile(
        rf'(?<![A-Za-z])({re.escape(true_tag_norm)}|{re.escape(false_tag_norm)})(?![A-Za-z])',
        flags=re.IGNORECASE,
    )
    match = token_pattern.search(judgement)
    if match:
        tag = _normalize_judge_tag(match.group(1))
        if tag.lower() == true_tag_norm.lower():
            return true_tag
        if tag.lower() == false_tag_norm.lower():
            return false_tag

    return 'unknown'


@DICT_POSTPROCESSORS.register_module()
def generic_llmjudge_postprocess(
    output: dict,
    output_path: str,
    true_tag: str = 'A',
    false_tag: str = 'B',
) -> dict:

    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocess(
            v['prediction'], true_tag, false_tag)
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            try:
                references.append(v['gold'])

            except KeyError:
                get_logger().warning(
                    f'No gold answer for {k}, use empty string as reference!')
                references.append('')
    results = get_final_results(judged_answers,
                                references,
                                origial_responses,
                                true_tag=true_tag,
                                false_tag=false_tag)
    results['details'] = output
    return results


def generic_llmjudge_academic_postprocess(
    output: dict,
    output_path: str,
    metric_name: str = 'accuracy',
) -> dict:
    judged_answers = []
    origial_responses = []
    references = []
    for k, v in output.items():
        origial_responses.append(v['prediction'])
        processed_judge = _generic_llmjudge_postprocess(v['prediction'])
        if processed_judge is not None:
            judged_answers.append(processed_judge)
            references.append(v['gold'])
    results = get_final_results(judged_answers, references, origial_responses,
                                metric_name)
    results['details'] = output
    # For academic summarizer
    results.pop('f1', None)
    return results

import ast
import re
import unicodedata
import warnings
from multiprocessing import Process, Queue

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


def _strip_think_tags(text: str) -> str:
    text = text or ''
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _last_boxed_only_string(string: str):
    idx = string.rfind('\\boxed')
    if '\\boxed ' in string:
        return '\\boxed ' + string.split('\\boxed ')[-1].split('$')[0]
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1].replace('$', '').replace('fbox', 'boxed')


def _remove_boxed(s: str):
    if '\\boxed ' in s:
        left = '\\boxed '
        if s[:len(left)] == left:
            return s[len(left):]
    left = '\\boxed{'
    if s[:len(left)] == left and s[-1] == '}':
        return s[len(left):-1]
    return s


def _solution_matches(text: str):
    matches = re.findall(r'<solution>(.*?)</solution>', text, re.IGNORECASE | re.DOTALL)
    if not matches:
        matches = re.findall(r'</solution>(.*?)</solution>', text, re.IGNORECASE | re.DOTALL)
    return matches


def _parse_ground_truth_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return value


def _web_of_lies_v2(ground_truth: str, llm_answer: str, debug: bool = False) -> int:
    score = 0
    parsed_answer = None
    solution_matches = _solution_matches(llm_answer)
    if solution_matches:
        parsed_answer = solution_matches[-1]

    bold_words = re.findall(r'\*\*(.*?)\*\*', llm_answer)
    if parsed_answer is None and bold_words:
        bold_words = [
            word.lower().strip().replace(',', '').replace('.', '')[0:max(len(word), 3)]
            for match in bold_words for word in match.split()
        ]
        parsed = []
        i = len(bold_words) - 1
        while i >= 0 and len(parsed) < 3:
            if bold_words[i] in ['yes', 'no', 'unknown']:
                parsed = [bold_words[i]] + parsed
            i -= 1
        parsed_answer = ', '.join(parsed) if parsed else None

    if parsed_answer is None or parsed_answer.strip() == '':
        normalized = llm_answer.replace('\\\\boxed{\\\\textbf{', '\\\\boxed{')
        normalized = normalized.replace('\\\\fbox{', '\\\\boxed{')
        normalized = normalized.replace('\\textbf{', '\\boxed{')
        last_boxed = _last_boxed_only_string(normalized)
        if last_boxed:
            parsed_answer = _remove_boxed(last_boxed)

    if parsed_answer is None:
        final_comb = None
        final_comb_index = -1
        for comb in [('yes', 'yes', 'yes'), ('yes', 'yes', 'no'), ('yes', 'yes', 'unknown'),
                     ('yes', 'no', 'yes'), ('yes', 'no', 'no'), ('yes', 'no', 'unknown'),
                     ('yes', 'unknown', 'yes'), ('yes', 'unknown', 'no'), ('yes', 'unknown', 'unknown'),
                     ('no', 'yes', 'yes'), ('no', 'yes', 'no'), ('no', 'yes', 'unknown'),
                     ('no', 'no', 'yes'), ('no', 'no', 'no'), ('no', 'no', 'unknown'),
                     ('no', 'unknown', 'yes'), ('no', 'unknown', 'no'), ('no', 'unknown', 'unknown'),
                     ('unknown', 'yes', 'yes'), ('unknown', 'yes', 'no'), ('unknown', 'yes', 'unknown'),
                     ('unknown', 'no', 'yes'), ('unknown', 'no', 'no'), ('unknown', 'no', 'unknown'),
                     ('unknown', 'unknown', 'yes'), ('unknown', 'unknown', 'no'), ('unknown', 'unknown', 'unknown')]:
            index = llm_answer.lower().find(', '.join(comb))
            if index != -1 and index > final_comb_index:
                final_comb = comb
                final_comb_index = index
        if final_comb is not None:
            parsed_answer = ', '.join(final_comb)

    if parsed_answer and parsed_answer == str(ground_truth).lower():
        score = 1
    if parsed_answer and parsed_answer.count('yes') + parsed_answer.count('no') + parsed_answer.count('unknown') == 3 and str(ground_truth).lower() in parsed_answer:
        score = 1

    if debug and score == 0:
        print('INCORRECT')
        print('GROUND TRUTH', ground_truth)
        print('PARSED ANSWER', parsed_answer)
        print('END OF OUTPUT', llm_answer[-50:])
    return score


def _web_of_lies_v3(ground_truth: str, llm_answer: str, debug: bool = False) -> float:
    parsed_answer = None
    solution_matches = _solution_matches(llm_answer)
    if solution_matches:
        parsed_answer = solution_matches[-1]

    if parsed_answer is None or parsed_answer.strip() == '':
        normalized = llm_answer.replace('\\\\boxed{\\\\textbf{', '\\\\boxed{')
        normalized = normalized.replace('\\\\fbox{', '\\\\boxed{')
        normalized = normalized.replace('\\textbf{', '\\boxed{')
        last_boxed = _last_boxed_only_string(normalized)
        if last_boxed:
            parsed_answer = _remove_boxed(last_boxed)
            parsed_answer = parsed_answer.replace('\\text{', '').replace('}', '')

    if parsed_answer is None:
        final_comb = None
        final_comb_index = -1
        for comb in [('yes', 'yes', 'yes'), ('yes', 'yes', 'no'), ('yes', 'yes', 'unknown'),
                     ('yes', 'no', 'yes'), ('yes', 'no', 'no'), ('yes', 'no', 'unknown'),
                     ('yes', 'unknown', 'yes'), ('yes', 'unknown', 'no'), ('yes', 'unknown', 'unknown'),
                     ('no', 'yes', 'yes'), ('no', 'yes', 'no'), ('no', 'yes', 'unknown'),
                     ('no', 'no', 'yes'), ('no', 'no', 'no'), ('no', 'no', 'unknown'),
                     ('no', 'unknown', 'yes'), ('no', 'unknown', 'no'), ('no', 'unknown', 'unknown'),
                     ('unknown', 'yes', 'yes'), ('unknown', 'yes', 'no'), ('unknown', 'yes', 'unknown'),
                     ('unknown', 'no', 'yes'), ('unknown', 'no', 'no'), ('unknown', 'no', 'unknown'),
                     ('unknown', 'unknown', 'yes'), ('unknown', 'unknown', 'no'), ('unknown', 'unknown', 'unknown')]:
            index = llm_answer.lower().find(', '.join(comb))
            if index != -1 and index > final_comb_index:
                final_comb = comb
                final_comb_index = index
        if final_comb is not None:
            parsed_answer = ', '.join(final_comb)

    if parsed_answer is None:
        return 0.0

    parsed_answer_list = [item.strip() for item in parsed_answer.split(',')]
    ground_truth_list = [item.strip() for item in str(ground_truth).split(',')]
    num_correct = 0
    total = len(ground_truth_list)
    for i in range(total):
        if i >= len(parsed_answer_list):
            break
        if parsed_answer_list[i] == ground_truth_list[i]:
            num_correct += 1
    score = ((num_correct == total) + num_correct / total) / 2
    if debug and score < 1:
        print('INCORRECT', ground_truth, parsed_answer)
    return score


def _house_traversal(ground_truth: str, llm_answer: str, debug: bool = False) -> int:
    bold_words = re.findall(r'(\*{2,})(.*?)\1', llm_answer.lower())
    if not bold_words:
        return 0
    last_bold = bold_words[-1][1]
    ground_truth_names = str(ground_truth).lower().split(' ')
    if len(bold_words) >= len(ground_truth_names):
        if all(name in bold_words[-1 - i][1] for i, name in enumerate(ground_truth_names[::-1])):
            return 1
    score = 1
    last_index = -1
    for name in ground_truth_names:
        index = last_bold.find(name)
        if index == -1 or index <= last_index:
            score = 0
            break
        last_index = index
    if debug and score == 0:
        print('INCORRECT', ground_truth, llm_answer)
    return score


def _spatial(ground_truth: str, llm_answer: str, debug: bool = False) -> int:
    if llm_answer == ground_truth:
        return 1
    word_to_number = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }
    bold_words = re.findall(r'\*\*([^\*]+)\*\*', llm_answer)
    words_to_check = []
    for i in range(3):
        if bold_words and len(bold_words) > i:
            words_to_check.append(bold_words[-i - 1].strip().lower())
    score = 0
    for word in words_to_check:
        if word == str(ground_truth).strip().lower():
            score = 1
        if word in word_to_number and word_to_number[word] == str(ground_truth).strip().lower():
            score = 1
        for answer in ['tetrahedra', 'tetrahedron', 'triangle', 'square']:
            if str(ground_truth).strip().lower() == answer and answer in word and len(word) < (2 * len(answer) + 5):
                score = 1
    last_boxed = None
    parsed_answer = None
    if score == 0:
        normalized = llm_answer.replace('\\\\fbox{', '\\\\boxed{')
        last_boxed = _last_boxed_only_string(normalized)
        if last_boxed:
            parsed_answer = _remove_boxed(last_boxed)
            parsed_answer = parsed_answer.replace('\\textbf{', '').replace('\\mathbf{', '').replace('\\text{', '').replace('}', '')
            if parsed_answer == str(ground_truth):
                score = 1
    if debug and score == 0:
        print('INCORRECT', ground_truth, parsed_answer, last_boxed)
    return score


def _logic_with_navigation(ground_truth, llm_answer: str, debug: bool = False) -> int:
    ground_truth = _parse_ground_truth_list(ground_truth)
    parsed_answer = None
    solution_matches = _solution_matches(llm_answer)
    if solution_matches:
        solution_text = solution_matches[-1].strip()
        coord_match = re.search(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', solution_text)
        if coord_match:
            parsed_answer = [int(coord_match.group(1)), int(coord_match.group(2))]
        elif re.search(r'(-?\d+)\s*,\s*(-?\d+)', solution_text):
            coord_match = re.search(r'(-?\d+)\s*,\s*(-?\d+)', solution_text)
            parsed_answer = [int(coord_match.group(1)), int(coord_match.group(2))]
    if isinstance(ground_truth, list) and parsed_answer is not None and len(ground_truth) == 2:
        return int(parsed_answer[0] == ground_truth[0] and parsed_answer[1] == ground_truth[1])
    if debug:
        print('INCORRECT', ground_truth, parsed_answer)
    return 0


def _sudoku(ground_truth: str, llm_answer: str, debug: bool = False) -> float:
    match = re.search(r'<solution>\s*(.*?)\s*</solution>', llm_answer, re.IGNORECASE | re.DOTALL)
    extracted_solution = re.sub(r'\s+', '', match.group(1).strip()) if match else ''
    normalized_ground_truth = re.sub(r'\s+', '', str(ground_truth))
    normalized_solution = re.sub(r'\s+', '', extracted_solution)
    if len(normalized_solution) != len(normalized_ground_truth):
        return 0.0
    return float(normalized_solution == normalized_ground_truth)


def _theory_of_mind(ground_truth: str, llm_answer: str, debug: bool = False) -> int:
    solution_matches = _solution_matches(llm_answer)
    if not solution_matches:
        last_line = llm_answer.strip().split('\n')[-1].strip()
        for prefix in ['Answer:', 'answer:', 'The answer is', 'the answer is']:
            if last_line.startswith(prefix):
                last_line = last_line[len(prefix):].strip()
        solution_matches.append(last_line)
    if not solution_matches:
        return 0
    llm_solution = re.sub(r'[.,!?;:]', '', solution_matches[-1].strip().lower()).strip()
    gt = str(ground_truth).strip().lower()
    return int(llm_solution == gt or gt in llm_solution)


def _zebra_puzzle_old(ground_truth: str, llm_answer: str, debug: bool = False) -> int:
    number_to_word = {
        '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    bold_words = re.findall(r'\*\*\*(\w+)\*\*\*', llm_answer)
    if bold_words:
        last_word = bold_words[-1]
    else:
        words = re.findall(r'\b\w+\b', llm_answer)
        last_word = words[-1] if words else ''
    gt = str(ground_truth).lower()
    return int(
        last_word.lower() == gt or
        (last_word in number_to_word and number_to_word[last_word].lower() == gt) or
        last_word.lower() + ' movies' == gt)


def _zebra_puzzle(ground_truth: str, llm_answer: str, debug: bool = False) -> float:
    ground_truth_parts = [part.strip() for part in str(ground_truth).split(',')]
    solution_matches = _solution_matches(llm_answer)
    if not solution_matches:
        normalized = llm_answer.replace('\\\\fbox{', '\\\\boxed{')
        last_boxed = _last_boxed_only_string(normalized)
        if last_boxed:
            boxed_removed = _remove_boxed(last_boxed)
            boxed_removed = boxed_removed.replace('\\text{', '').replace('}', '').replace('\\', '')
            solution_matches.append(boxed_removed)
    if not solution_matches:
        last_line = llm_answer.strip().split('\n')[-1]
        if last_line.count(',') == len(ground_truth_parts) - 1:
            solution_matches.append(last_line)
    if not solution_matches:
        return 0.0
    if len(solution_matches) > 1:
        all_solution_text = []
        for match in solution_matches:
            all_solution_text += match.split(',')
        solution_text = all_solution_text[-len(ground_truth_parts):]
    else:
        solution_text = solution_matches[-1].split(',')
    num_correct = 0
    total = len(ground_truth_parts)
    for i in range(total):
        gt_word = ground_truth_parts[i].strip().lower().replace('-', ' ')
        if i >= len(solution_text):
            continue
        llm_word = solution_text[i].strip().lower().replace('-', ' ').replace('position', '')
        if gt_word == llm_word or gt_word in llm_word:
            num_correct += 1
    return ((num_correct == total) + num_correct / total) / 2


def _get_zebra_evaluator(release_date: str):
    if release_date and release_date < '2024-11-25':
        return _zebra_puzzle_old
    return _zebra_puzzle


def _extract_mathcontest_answer(statement: str, letter: str):
    pattern = r'\\textbf{\(([A-E])\)\s?}(.*?)(?:\\qquad|\$)'
    matches = re.findall(pattern, statement)
    answers = {match[0]: match[1].strip() for match in matches}
    answer = answers.get(letter, None) or 'FAILURE'
    return answer.strip().strip('$').strip('~')


def _mathcontest(ground_truth: str, llm_answer: str, question_text: str, debug: bool = False) -> int:
    score = 0
    if not (isinstance(ground_truth, str) and len(ground_truth) == 1 and 'A' <= ground_truth <= 'E'):
        raise ValueError('amc_answer must be a single capital letter between A and E.')
    solution_matches = _solution_matches(llm_answer)
    if solution_matches:
        solution_match = solution_matches[-1]
        if len(set(solution_match)) == 1 and next(iter(set(solution_match))).lower() == ground_truth.lower():
            score = 1
    if ground_truth * 4 in llm_answer:
        score = 1
    parsed_answer = None
    if score == 0:
        normalized = llm_answer.replace('\\\\fbox{', '\\\\boxed{')
        last_boxed = _last_boxed_only_string(normalized)
        if last_boxed:
            boxed = _remove_boxed(last_boxed).replace('\\text{', '').replace('}', '').replace('\\', '').lower()
            if boxed in {'a', 'b', 'c', 'd', 'e'}:
                parsed_answer = boxed
            if parsed_answer == ground_truth.lower():
                score = 1
    if score == 0:
        value = _extract_mathcontest_answer(question_text, ground_truth)
        length_to_check = 20 + len(value)
        if value in llm_answer[-length_to_check:]:
            score = 1
    if score == 0:
        last_line = llm_answer.strip().split('\n')[-1]
        if last_line.strip().replace('*', '').lower() == ground_truth.lower():
            score = 1
        elif '(' in last_line and ')' in last_line:
            val = last_line.split('(')[1].split(')')[0]
            if val.lower() == ground_truth.lower():
                score = 1
    return score


def _aime(ground_truth: str, llm_answer: str, debug: bool = False) -> int:
    score = 0
    solution_matches = _solution_matches(llm_answer)
    if solution_matches:
        solution_match = solution_matches[-1]
        if len(set(solution_match)) == 1 and next(iter(set(solution_match))).lower() == str(ground_truth).lower():
            score = 1
    if score == 0 and str(ground_truth) in llm_answer[-50:]:
        score = 1
    return score


def _remove_nonnumeric_chars_at_ends(s: str):
    start_index = 0
    while start_index < len(s) and not s[start_index].isdigit():
        start_index += 1
    end_index = start_index
    while end_index < len(s) and s[end_index].isdigit():
        end_index += 1
    return s[start_index:end_index], len(s) - (end_index - start_index)


def _normalize_numeric_token(token: str) -> str:
    normalized = []
    for char in token.strip():
        if char in '+-':
            normalized.append(char)
            continue
        try:
            normalized.append(str(unicodedata.digit(char)))
        except (TypeError, ValueError):
            normalized.append(char)
    return ''.join(normalized)


def _parse_int_token(token: str) -> int:
    normalized = _normalize_numeric_token(token).strip()
    if normalized in {'', '+', '-'}:
        raise ValueError('empty numeric token')
    return int(normalized)


def _extract_expression_completions_from_generation(generation: str, debug: bool = False):
    numbers = None
    if 'answer:' in generation.lower():
        lines = generation.lower().strip().split('\n')
        answer_line = None
        answer_index = None
        for i, line in enumerate(lines):
            if 'answer:' in line:
                answer_line = line
                answer_index = i
        if answer_line is not None:
            answer_str = answer_line.split('answer:')[1].replace('answer:', '').replace('**', '').replace('.', '').strip()
            if answer_str == '' and answer_index is not None and answer_index < len(lines) - 1:
                answer_str = lines[answer_index + 1].replace('answer:', '').replace('**', '').replace('.', '').strip()
            numbers = []
            for n in answer_str.split(','):
                n = n.strip().split(' ')[-1].replace('$', '').replace('{', '').replace('}', '').replace('\\', '').replace('boxed', '').replace('<', '').replace('>', '')
                try:
                    numbers.append(_parse_int_token(n))
                except Exception:
                    numbers.append('NO ANSWER')
            if len(numbers) == 0 or set(numbers) == {'NO ANSWER'}:
                numbers = None
    if numbers is None and '\\boxed' in generation:
        boxed = _last_boxed_only_string(generation)
        string = _remove_boxed(boxed) if boxed is not None else generation
        string = string.replace('\\text{', '').replace('}', '').replace('\\', '')
        numbers = []
        for n in string.strip().split(','):
            try:
                numbers.append(_parse_int_token(n))
            except Exception:
                numbers.append('NO ANSWER')
        if len(numbers) == 0 or set(numbers) == {'NO ANSWER'}:
            numbers = None
    if numbers is None:
        last_line = generation.strip().lower().split('\n')[-1]
        numbers = []
        for n in last_line.strip().split(','):
            n, _ = _remove_nonnumeric_chars_at_ends(n)
            if len(n.strip()) == 0:
                continue
            try:
                numbers.append(_parse_int_token(n))
            except Exception:
                numbers.append('NO ANSWER')
        if len(numbers) == 0 or set(numbers) == {'NO ANSWER'}:
            numbers = None
    if numbers is None:
        numbers = [k.strip() for k in generation.lower().split('answer:')[-1].split(',')]
        new_numbers = []
        for i, n in enumerate(numbers):
            n, num_removed = _remove_nonnumeric_chars_at_ends(n)
            if n != '':
                new_numbers.append(_parse_int_token(n))
            if i > 0 and num_removed > 0:
                break
        numbers = new_numbers
    return numbers


def _sequence_edit_distance(seq1, seq2):
    seq1 = list(seq1 or [])
    seq2 = list(seq2 or [])
    dp = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    for i in range(len(seq1) + 1):
        dp[i][0] = i
    for j in range(len(seq2) + 1):
        dp[0][j] = j
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]


def _proof_rearrangement(ground_truth: str, llm_answer: str, edit_distance: bool = False, debug: bool = False) -> float:
    ground_truth = [int(n) for n in str(ground_truth).split(',')]
    completions = _extract_expression_completions_from_generation(llm_answer, debug)
    if edit_distance:
        match = _sequence_edit_distance(completions, ground_truth)
        frac_matches = 1 - (match / max(len(completions), len(ground_truth), 1))
    else:
        match = [
            (completions[i] == ground_truth[i]) if i < len(ground_truth) else 0
            for i in range(len(completions))
        ]
        frac_matches = sum(match) / len(match) if match else 0
    return frac_matches


def _run_with_timeout(func, args=(), timeout=8):
    def wrapper(queue):
        try:
            queue.put(func(*args))
        except Exception as e:
            queue.put(e)

    queue = Queue()
    process = Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError('Operation timed out')
    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result


def _amps_parse(x: str):
    import sympy
    from sympy.parsing.latex import parse_latex

    try:
        parsed = parse_latex(x, backend='lark')
    except Exception:
        try:
            parsed = parse_latex(x.replace('\\\\', '\\'), backend='lark')
        except Exception:
            try:
                parsed = parse_latex(x)
            except Exception:
                warnings.warn(f"couldn't parse {x}")
                return []
    if hasattr(parsed, 'children'):
        return parsed.children
    return [parsed]


def _amps_is_equiv(x1: str, x2: str) -> bool:
    import sympy

    parsed_x1s = _amps_parse(x1)
    parsed_x2s = _amps_parse(x2)
    if not parsed_x1s or not parsed_x2s:
        return False
    for parsed_x1 in parsed_x1s:
        for parsed_x2 in parsed_x2s:
            try:
                diff = parsed_x1 - parsed_x2
                simplified = _run_with_timeout(sympy.simplify, args=(diff,), timeout=10)
                if simplified == 0:
                    return True
                if sympy.Abs(simplified) < 0.001:
                    return True
            except Exception:
                continue
    return False


def _amps_normalize(final_answer: str) -> str:
    final_answer = final_answer.split('=')[-1]
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{\[])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')
    return final_answer


def _amps_hard(ground_truth: str, llm_answer: str, debug: bool = False) -> int:
    parsed_answer = None
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[-1]
    llm_answer = llm_answer.replace('+C', '').replace('+ C', '').replace('+ c', '').replace('+c', '')
    llm_answer = llm_answer.replace('\\\\fbox{', '\\\\boxed{').replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    llm_answer = llm_answer.replace('\\left', '').replace('\\right', '').replace('\\bigl', '').replace('\\bigr', '')
    llm_answer = llm_answer.replace('\\Bigl', '').replace('\\Bigr', '').replace('\\,', '').replace('\\;', '')
    llm_answer = llm_answer.replace('\\cdot', '*')
    ground_truth = str(ground_truth).replace('\\left', '').replace('\\right', '').replace(' ^', '^').replace('\\ ', '*')
    last_boxed = _last_boxed_only_string(llm_answer)
    if last_boxed:
        parsed_answer = _amps_normalize(_remove_boxed(last_boxed))
    if parsed_answer is None:
        last_line = llm_answer.split('\n')[-1]
        if last_line.count('$') >= 2:
            close_pos = last_line.rfind('$')
            if close_pos > 0 and last_line[close_pos - 1] == '$':
                close_pos -= 1
            open_pos = last_line.rfind('$', 0, close_pos)
            math = last_line[open_pos + 1:close_pos]
            if '=' in math:
                math = math.split('=')[-1].strip()
            elif '\\quad \\text{or} \\quad' in math:
                math = math.split('\\quad \\text{or} \\quad')[-1].strip()
            parsed_answer = _amps_normalize(math)
    if parsed_answer is not None:
        try:
            if _amps_is_equiv(ground_truth, parsed_answer):
                return 1
        except Exception:
            pass
    if len(llm_answer) > 0 and llm_answer[-1] == '.':
        llm_answer = llm_answer[:-1]
    return int(ground_truth == llm_answer[-len(ground_truth):])


def _integrals_normalize(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace('\\\\', '\\').replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    answer = answer.replace('\\left', '').replace('\\right', '').replace('\\bigl', '').replace('\\bigr', '')
    answer = answer.replace('\\Bigl', '').replace('\\Bigr', '').replace('\\cdot', '*')
    answer = answer.replace('\\,', '').replace('\\;', '').replace('\n', '').replace(' ', '')
    return answer


def _integrals_parse(answer: str):
    answer = _integrals_normalize(answer)
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
    except Exception:
        sympy = None
        parse_latex = None

    try:
        if '/' in answer and '\\' not in answer:
            num, denom = answer.split('/')
            return ('frac', int(num), int(denom))
    except Exception:
        pass
    try:
        return ('int', int(answer))
    except Exception:
        pass
    try:
        if sympy is not None:
            return ('rational', sympy.Rational(answer).limit_denominator(10000000))
    except Exception:
        pass
    frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', answer)
    if frac_match:
        try:
            num = int(frac_match.group(1))
            denom = int(frac_match.group(2))
            return ('frac', num, denom)
        except Exception:
            pass
    if parse_latex is not None:
        try:
            parsed = parse_latex(answer)
            return ('sympy', parsed)
        except Exception:
            try:
                parsed = parse_latex(answer, backend='lark')
                if hasattr(parsed, 'children'):
                    parsed = parsed.children[0]
                return ('sympy', parsed)
            except Exception:
                pass
    return ('str', answer)


def _integrals_equiv(x1, x2) -> bool:
    if x1[0] in {'int', 'frac', 'str'} and x1 == x2:
        return True
    if x1[0] == 'rational' and x2[0] == 'rational':
        return x1[1] == x2[1]
    if x1[0] == 'sympy' and x2[0] == 'sympy':
        try:
            import sympy
            simplified = sympy.simplify(x1[1] - x2[1])
            return simplified == 0 or sympy.Abs(simplified).evalf() < 1e-5
        except Exception:
            return False
    if x1[0] == 'frac' and x2[0] == 'frac':
        return x1[1] * x2[2] == x2[1] * x1[2]
    return False


def _integrals_with_game(ground_truth: str, llm_answer: str, debug: bool = False) -> int:
    gt_parsed = _integrals_parse(str(ground_truth))
    solution_matches = _solution_matches(llm_answer)
    parsed_model_answer = None
    if solution_matches:
        parsed_model_answer = _integrals_parse(solution_matches[-1].strip())
        if _integrals_equiv(gt_parsed, parsed_model_answer):
            return 1
    normalized = llm_answer.replace('\\\\fbox{', '\\\\boxed{')
    last_boxed = _last_boxed_only_string(normalized)
    if last_boxed:
        parsed_model_answer = _integrals_parse(_remove_boxed(last_boxed))
        if _integrals_equiv(gt_parsed, parsed_model_answer):
            return 1
    last_part = llm_answer[-200:] if len(llm_answer) > 200 else llm_answer
    return int(_integrals_normalize(str(ground_truth)) in _integrals_normalize(last_part))


def _score_livebench(sample, prediction: str, debug: bool = False) -> float:
    task = str(sample.get('task', ''))
    task_or_subtask = str(sample.get('subtask') or task)
    question_text = str(sample.get('question', ''))
    ground_truth = sample.get('ground_truth', sample.get('answer', ''))
    splits = task_or_subtask.split('_')

    if len(splits) > 0 and (splits[0] in ['amc', 'smc', 'aime', 'imo', 'usamo'] or (len(splits) > 1 and splits[1] == 'amc')):
        if splits[0] in ['amc', 'smc'] or (len(splits) > 1 and splits[1] == 'amc'):
            return _mathcontest(str(ground_truth), prediction, question_text, debug)
        if splits[0] == 'aime':
            return _aime(str(ground_truth), prediction, debug)
        if splits[0] in ['imo', 'usamo']:
            return _proof_rearrangement(str(ground_truth), prediction, edit_distance=True, debug=debug)
    if task_or_subtask == 'integrals_with_game':
        return _integrals_with_game(str(ground_truth), prediction, debug)
    if 'amps_hard' in task_or_subtask or 'amps_hard' in task:
        return _amps_hard(str(ground_truth), prediction, debug)
    if task_or_subtask == 'web_of_lies_v2':
        return _web_of_lies_v2(str(ground_truth), prediction, debug)
    if task_or_subtask == 'web_of_lies_v3':
        return _web_of_lies_v3(str(ground_truth), prediction, debug)
    if task_or_subtask == 'house_traversal':
        return _house_traversal(str(ground_truth), prediction, debug)
    if 'zebra_puzzle' in task_or_subtask:
        evaluator = _get_zebra_evaluator(str(sample.get('livebench_release_date', '')))
        return evaluator(str(ground_truth), prediction, debug)
    if task_or_subtask == 'spatial':
        return _spatial(str(ground_truth), prediction, debug)
    if task_or_subtask == 'theory_of_mind':
        return _theory_of_mind(str(ground_truth), prediction, debug)
    if task_or_subtask == 'logic_with_navigation':
        return _logic_with_navigation(ground_truth, prediction, debug)
    if task_or_subtask == 'sudoku':
        return _sudoku(str(ground_truth), prediction, debug)
    raise NotImplementedError(f'Unsupported LiveBench task: {task_or_subtask}')


@ICL_EVALUATORS.register_module()
class LiveBenchEvaluator(BaseEvaluator):

    def score(self, predictions, references, test_set):
        scores = []
        for sample, prediction in zip(test_set, predictions):
            cleaned_prediction = _strip_think_tags(prediction)
            scores.append(_score_livebench(sample, cleaned_prediction))
        avg = sum(scores) / len(scores) if scores else 0
        return {
            'score': avg * 100,
            'accuracy': avg * 100,
        }

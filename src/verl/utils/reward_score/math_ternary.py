# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def wrap_boxed(solution_str):
    if "\\boxed" not in solution_str:
        return "\\boxed{" + solution_str + "}"
    return solution_str


def remove_thinking_draft(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
        if len(text) > 0:
            return text
    if "</seed:think>" in text:
        text = text.split("</seed:think>")[-1].strip()
        if len(text) > 0:
            return text
    return text


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def check_inability(original_response):
    answer = remove_thinking_draft(original_response)
    if isinstance(answer, str):
        answer = last_boxed_only_string(answer)
        if isinstance(answer, str):
            answer = remove_boxed(answer)
            if isinstance(answer, str):
                if "CANNOT SOLVE" in answer.upper():
                    return True
                else:
                    return False
    return None


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:

    if check_inability(model_output):
        return 0.0

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    model_output = re.sub(r"√(\d+)", r"\\sqrt{\1}", model_output.strip())
    ground_truth = re.sub(r"√(\d+)", r"\\sqrt{\1}", ground_truth.strip())
    # remove \bigl and \bigr
    model_output = model_output.replace("\\bigl", "")
    model_output = model_output.replace("\\bigr", "")
    ground_truth = ground_truth.replace("\\bigl", "")
    ground_truth = ground_truth.replace("\\bigr", "")
    # remove percentage
    model_output = model_output.replace("\\%", "")
    model_output = model_output.replace("\%", "")  # noqa: W605
    ground_truth = ground_truth.replace("\\%", "")
    ground_truth = ground_truth.replace("\%", "")  # noqa: W605
    # wrap the ground truth in \boxed{} format
    ground_truth_boxed = wrap_boxed(ground_truth)
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    if ret_score == 1.0:
        return 1.0
    else:
        return -1.0
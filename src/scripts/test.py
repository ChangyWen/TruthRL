from curses import raw
import os
import json
import argparse
import jsonlines
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from vllm import LLM, SamplingParams
from typing import Optional, List
from uuid import uuid4
import re
import pandas as pd
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


def load_dataset(dataset_file, tokenizer):
    data = pd.read_parquet(dataset_file)

    idxs = []
    prompts = []
    raw_prompts = []
    ground_truths = []
    for index, row in data.iterrows():
        idx = row["data_source"] + "_" + str(index)
        raw_prompt = row["prompt"][0]["content"]
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True)
        ground_truth = row["reward_model"]["ground_truth"]
        idxs.append(idx)
        prompts.append(prompt)
        raw_prompts.append(raw_prompt)
        ground_truths.append(ground_truth)
    return idxs, prompts, raw_prompts, ground_truths


def main(
    llm,
    sampling_params,
    tokenizer,
    dataset_file,
    save_file,
):
    idxs, prompts, raw_prompts, ground_truths = load_dataset(dataset_file, tokenizer)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    res = []
    for i in range(len(outputs)):
        all_results = []
        all_scores = []
        all_lengths = []
        for j in range(len(outputs[i].outputs)):
            output = outputs[i].outputs[j].text.strip()
            score = compute_score(output, ground_truths[i])
            length = len(tokenizer.encode(output))
            all_results.append(output)
            all_scores.append(score)
            all_lengths.append(length)
        res.append({
            "idx": idxs[i],
            "prompt": raw_prompts[i],
            "ground_truth": ground_truths[i],
            "scores": all_scores,
            "lengths": all_lengths,
            "results": all_results,
        })

    # save to jsonl file
    with open(save_file, "a") as f:
        for item in res:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, required=False, default=8)
    parser.add_argument("--max_prompt_length", type=int, required=False, default=512)
    parser.add_argument("--max_completion_length", type=int, required=False, default=32256)
    parser.add_argument("--n_samples", type=int, required=False, default=16)
    parser.add_argument("--temperature", type=float, required=False, default=0.6)
    args = parser.parse_args()

    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_prompt_length + args.max_completion_length,
        max_num_batched_tokens=args.max_prompt_length + args.max_completion_length,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        n=1 if args.temperature == 0.0 else args.n_samples,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_completion_length,
    )

    tokenizer = llm.get_tokenizer()

    datasets = [
        # ("/mnt/blob_output/v-dachengwen/data/gsm8k/test.parquet", "gsm8k"),
        # ("/mnt/blob_output/v-dachengwen/data/math/amc.parquet", "amc"),
        # ("/mnt/blob_output/v-dachengwen/data/math/math-500.parquet", "math-500"),
        # ("/mnt/blob_output/v-dachengwen/data/math/minerva.parquet", "minerva"),
        # ("/mnt/blob_output/v-dachengwen/data/math/olympiad_bench.parquet", "olympiad_bench"),
        ("/mnt/blob_output/v-dachengwen/data/math/aime.parquet", "aime"),
        ("/mnt/blob_output/v-dachengwen/data/math/aime_2025.parquet", "aime_2025"),
        ("/mnt/blob_output/v-dachengwen/data/math/beyond_aime.parquet", "beyond_aime"),
    ]

    if "actor/huggingface" in args.model_name:
        save_dir = args.model_name.replace("actor/huggingface", "results")
    else:
        save_dir = "/mnt/blob_output/v-dachengwen/TruthRL/results-" + args.model_name.split("/")[-1]

    os.makedirs(save_dir, exist_ok=True)

    for dataset in datasets:
        dataset_file = dataset[0]
        dataset_name = dataset[1]
        save_file = os.path.join(save_dir, f"{dataset_name}_temperature_{args.temperature}_repeat_{args.n_samples}.jsonl")
        if os.path.exists(save_file):
            print(f"[{save_file}] already exists, skipping ...")
            continue
        print(f"Processing [{dataset_name}] on [{dataset_file}] ...")
        print(f"Results will be saved to [{save_file}] ...")
        main(llm, sampling_params, tokenizer, dataset_file, save_file)

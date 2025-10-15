import pandas as pd
import json


def get_prompt(problem: str):
    return f"""
{problem}

Please reason step by step and then provide the final answer. The reasoning process must be enclosed within <think> </think> tags. The final answer MUST be put in \\boxed{{}}.
At any point during your reasoning, if you become highly unsure, or find the problem unsolvable or beyond your capability, stop attempting a solution.
In this case, instead of generating a possibly incorrect solution or guessing an answer, just honestly output \\boxed{{Cannot Solve}} with some brief explanation.
    """.strip()


if __name__ == "__main__":
    df = []
    with open("data/omni_math_rule/omni_math_rule.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            problem = item["problem"]
            ground_truth = item["answer"]
            answer = item["solution"]
            prompt = [{"role": "user", "content": get_prompt(problem)}]
            df.append({
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "ground_truth": ground_truth,
                    "style": "rule",
                },
                "extra_info": {
                    "answer": answer,
                    "split": "train"
                },
            })
    df = pd.DataFrame(df)
    df.to_parquet("data/omni_math_rule/omni_math_rule.parquet")

import pandas as pd


def get_prompt(problem: str):
    return f"""
{problem}

Please reason step by step and then provide the final answer. The reasoning process must be enclosed within <think> </think> tags. The final answer MUST be put in \\boxed{{}}.
At any point during your reasoning, if you become highly unsure, or find the problem unsolvable or beyond your capability, stop attempting a solution.
In this case, instead of generating a possibly incorrect solution or guessing an answer, just honestly output \\boxed{{Cannot Solve}} with some brief explanation.
    """.strip()


if __name__ == "__main__":
    files = [
        "data/gsm8k/train.parquet",
        "data/gsm8k/test.parquet",
        "data/math/aime.parquet",
        "data/math/amc.parquet",
        "data/math/math-500.parquet",
        "data/math/minerva.parquet",
        "data/math/olympiad_bench.parquet",
        "data/math/test.parquet",
        "data/math/train.parquet",
    ]


    for file in files:
        data = pd.read_parquet(file)
        for index, row in data.iterrows():
            original_prompt = row["prompt"]
            if len(original_prompt) != 2:
                print(f"original_prompt={original_prompt}")
                raise ValueError(f"original_prompt={original_prompt}")
            assert original_prompt[0]["role"] == "system"
            assert original_prompt[1]["role"] == "user"
            assert "reason step by step" in original_prompt[0]["content"]
            assert "reason step by step" not in original_prompt[1]["content"]
            new_prompt = [{"role": "user", "content": get_prompt(original_prompt[1]["content"])}]
            row["prompt"] = new_prompt
        data.to_parquet(file)

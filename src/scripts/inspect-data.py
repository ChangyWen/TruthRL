import pandas as pd


if __name__ == "__main__":
    data = pd.read_parquet("data/gsm8k/train.parquet")
    for index, row in data.iterrows():
        print(row["prompt"])
        print(row["reward_model"])
        print(row["extra_info"])
        break
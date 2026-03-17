import json
import numpy as np
import sys
from ..constant import *



def recalculate(filepath, alpha):
    with open(filepath, "r") as f:
        data = json.load(f)

    for qid, val in data.items():
        avg_f1 = val["Average"]["f1"]
        avg_em = val["Average"]["exact_match"]
        val["Diff Score"] = 1.0 - (alpha * avg_f1 + (1 - alpha) * avg_em)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    scores = [val["Diff Score"] for val in data.values()]
    return scores


if __name__ == "__main__":
    ALPHA = 0.3

    FILES = [
        f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_0/scores.json",
        f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_1/scores.json",
        f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_2/scores.json",
        f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_3/scores.json",
    ]


    print(f"Alpha = {ALPHA}\n")

    for filepath in FILES:
        scores = recalculate(filepath, ALPHA)
        arr = np.array(scores)
        print(f"{filepath}:")
        print(f"  Mean={arr.mean():.4f}  Median={np.median(arr):.4f}  Min={arr.min():.4f}  Max={arr.max():.4f}")
        print(f"  Saved.\n")
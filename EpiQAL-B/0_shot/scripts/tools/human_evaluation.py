import matplotlib
matplotlib.use('Agg')
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from ..constant import *
import random



if __name__ == "__main__":
    B_FINAL = f"{RESULT_FILE_PATH}/final_results.json"

    os.makedirs("./scripts/tools/human_evaluation", exist_ok=True)

    with open(B_FINAL) as f:
        data = json.load(f)

    scores = []
    for item in data:
        ds = item.get("question", {}).get("Difficulty Score", None)
        if ds is not None:
            scores.append(ds)

    arr = np.array(scores)
    print(f"\n{'='*50}")
    print(f"  B Final Difficulty Score (n={len(arr)})")
    print(f"{'='*50}")
    print(f"  Mean:   {arr.mean():.4f}")
    print(f"  Median: {np.median(arr):.4f}")
    print(f"  Std:    {arr.std():.4f}")
    print(f"  Min:    {arr.min():.4f}")
    print(f"  Max:    {arr.max():.4f}")
    for p in [10, 25, 33, 50, 67, 75, 90]:
        print(f"  P{p:02d}:    {np.percentile(arr, p):.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, 1.05, 0.05)
    ax.hist(arr, bins=bins, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.axvline(arr.mean(), color="red", linestyle="--", label=f"Mean={arr.mean():.3f}")
    ax.axvline(np.median(arr), color="orange", linestyle="--", label=f"Median={np.median(arr):.3f}")
    p33 = np.percentile(arr, 33)
    p67 = np.percentile(arr, 67)
    ax.axvline(p33, color="green", linestyle=":", label=f"P33={p33:.3f}")
    ax.axvline(p67, color="purple", linestyle=":", label=f"P67={p67:.3f}")
    ax.set_title("EpiQAL-B: Final DiffScore Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Difficulty Score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig("./scripts/tools/human_evaluation/b_final_diffscore.png", dpi=150, bbox_inches="tight")
    print("\n[Saved] ./scripts/tools/human_evaluation/b_final_diffscore.png")
    

    with open(B_FINAL) as f:
        data = json.load(f)

    easy, medium, hard = [], [], []
    for item in data:
        ds = item.get("question", {}).get("Difficulty Score", None)
        if ds is None or ds == 0:
            easy.append(item)
        elif ds <= 0.4:
            medium.append(item)
        else:
            hard.append(item)

    print(f"B Distribution: Easy={len(easy)}, Medium={len(medium)}, Hard={len(hard)}")

    sampled = []
    for tier, name, n in [(easy, "Easy", 6), (medium, "Medium", 7), (hard, "Hard", 7)]:
        picked = random.sample(tier, min(n, len(tier)))
        for item in picked:
            item["_difficulty_tier"] = name
        sampled.extend(picked)
        print(f"  Sampled {len(picked)} from {name}")

    random.shuffle(sampled)

    eval_items = []
    for i, item in enumerate(sampled):
        eval_items.append({
            "eval_id": i,
            "idx": item["idx"],
            "difficulty_tier": item["_difficulty_tier"],
            "difficulty_score": item.get("question", {}).get("Difficulty Score", None),
            "paragraph": item["paragraph"],
            "external_info": item["external_info"],
            "question": item["question"]["Revised Question"],
            "ori_question": item.get("ori_question", {}).get("Question", ""),
            "evidence": item.get("ori_question", {}).get("Evidence", []),
            "rationale": item.get("ori_question", {}).get("Rationale", ""),
            "choices": item["choices"],
            "ref_answers": item["ref_answers"],
            "evaluation": {
                "Answer Correctness": "",
                "Distractor Quality": "",
                "Question Clarity": "",
                "Evidence Sufficiency": "",
                "Reasoning Depth": "",
                "Other Comments": ""
            }
        })

    output_path = "./scripts/tools/human_evaluation/b_eval_sample.json"
    with open(output_path, "w") as f:
        json.dump(eval_items, f, indent=4)

    print(f"\n[Saved] {output_path} ({len(eval_items)} items)")
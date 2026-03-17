import matplotlib
matplotlib.use('Agg')
import json
import os
from ..constant import *
import random



if __name__ == "__main__":
    A_FINAL = f"{RESULT_FILE_PATH}/final_results.json"

    os.makedirs("./scripts/tools/human_evaluation", exist_ok=True)

    with open(A_FINAL) as f:
        data = json.load(f)

    sampled = random.sample(data, min(20, len(data)))
    random.shuffle(sampled)

    print(f"A: Sampled {len(sampled)} from {len(data)} total")

    eval_items = []
    for i, item in enumerate(sampled):
        eval_items.append({
            "eval_id": i,
            "idx": item["idx"],
            "paragraph": item["paragraph"],
            "question": item["question"]["Question"],
            "evidence": item["question"].get("Evidence", []),
            "rationale": item["question"].get("Rationale", ""),
            "choices": item["choices"],
            "ref_answers": item["ref_answers"],
            "evaluation": {
                "Answer Correctness": "",
                "Distractor Quality": "",
                "Question Clarity": "",
                "Evidence Sufficiency": "",
                "Other Comments": ""
            }
        })

    output_path = "./scripts/tools/human_evaluation/a_eval_sample.json"
    with open(output_path, "w") as f:
        json.dump(eval_items, f, indent=4)

    print(f"\n[Saved] {output_path} ({len(eval_items)} items)")
# distractor_analysis.py
import json
import os
from collections import Counter, defaultdict

MODEL_DIR = "../../output/evaluation_GroupA/noCOT/model"
DISTRACTOR_CAT = "../../output/distractor_category.json"
FINAL_QA = "../../output/final_qa.json"

with open(DISTRACTOR_CAT) as f:
    distractor_cats = json.load(f)

with open(FINAL_QA) as f:
    final_qa_list = json.load(f)
    final_qa = {item["idx"]: item for item in final_qa_list}

model_answers = {}
for fname in sorted(os.listdir(MODEL_DIR)):
    if not fname.endswith(".json"):
        continue
    model_name = fname.replace(".json", "")
    with open(os.path.join(MODEL_DIR, fname)) as f:
        model_answers[model_name] = json.load(f)

print(f"Models: {len(model_answers)}")
print(f"Questions: {len(final_qa)}")


CATEGORY_MAP = {
    "wrong entity/role": "Wrong entity/role",
    "wrong context": "Wrong context",
    "wrong metric": "Wrong metric",
    "semantic near-miss": "Semantic near-miss",
}

def clean_category(cat):
    for key, val in CATEGORY_MAP.items():
        if key in cat.lower():
            return val
    return "unknown"

for idx in distractor_cats:
    for opt_idx in distractor_cats[idx]:
        distractor_cats[idx][opt_idx][0] = clean_category(distractor_cats[idx][opt_idx][0])

all_categories = []
for idx, distractors in distractor_cats.items():
    for opt_idx, (category, text, reason) in distractors.items():
        all_categories.append(category)

cat_counts = Counter(all_categories)
valid_cats = sorted([c for c in cat_counts.keys() if c != "unknown"])

print(f"\n{'='*60}")
print(f"  Distractor Category Distribution (n={len(all_categories)})")
print(f"{'='*60}")
for cat, cnt in cat_counts.most_common():
    print(f"  {cat}: {cnt} ({cnt/len(all_categories)*100:.1f}%)")

model_error_cats = defaultdict(lambda: Counter())
model_total_errors = Counter()
overall_error_cats = Counter()

for model_name, answers in model_answers.items():
    for idx, ans_list in answers.items():
        if idx not in final_qa:
            continue
        ref_set = set(final_qa[idx]["ref_answers"])
        ans_set = set(str(a) for a in ans_list)

        false_positives = ans_set - ref_set
        for fp in false_positives:
            if idx in distractor_cats and str(fp) in distractor_cats[idx]:
                cat = distractor_cats[idx][str(fp)][0]
                model_error_cats[model_name][cat] += 1
                overall_error_cats[cat] += 1
            else:
                model_error_cats[model_name]["unknown"] += 1
                overall_error_cats["unknown"] += 1
            model_total_errors[model_name] += 1

print(f"\n{'='*60}")
print(f"  Overall Deception Rate by Category")
print(f"  (misselected across all models / total exposure)")
print(f"{'='*60}")
for cat in valid_cats:
    misselected = overall_error_cats.get(cat, 0)
    exposed = cat_counts.get(cat, 0) * len(model_answers)
    rate = misselected / exposed * 100 if exposed > 0 else 0
    print(f"  {cat}: {misselected}/{exposed} ({rate:.1f}%)")

print(f"\n{'='*60}")
print(f"  Per-Model Deception Rate by Category")
print(f"  (misselected / total distractors in that category)")
print(f"{'='*60}")

header = f"{'Model':<35}" + "".join(f"{cat:<20}" for cat in valid_cats) + f"{'Overall':<10}"
print(header)
print("-" * len(header))

for model_name in sorted(model_answers.keys()):
    row = f"{model_name:<35}"
    total_mis = model_total_errors[model_name]
    total_distractors = sum(cat_counts.get(c, 0) for c in valid_cats)
    for cat in valid_cats:
        cnt = model_error_cats[model_name][cat]
        total_cat = cat_counts.get(cat, 1)
        rate = cnt / total_cat * 100
        row += f"{rate:.1f}%".ljust(20)
    overall_rate = total_mis / total_distractors * 100 if total_distractors > 0 else 0
    row += f"{overall_rate:.1f}%"
    print(row)

output = {
    "distractor_distribution": dict(cat_counts.most_common()),
    "overall_deception_rate": {
        cat: {
            "misselected": overall_error_cats.get(cat, 0),
            "exposed": cat_counts.get(cat, 0) * len(model_answers),
            "rate": overall_error_cats.get(cat, 0) / (cat_counts.get(cat, 0) * len(model_answers))
                  if cat_counts.get(cat, 0) > 0 else 0
        }
        for cat in valid_cats
    },
    "per_model_deception_rate": {
        m: {
            **{
                cat: {
                    "misselected": model_error_cats[m][cat],
                    "total_in_category": cat_counts.get(cat, 0),
                    "rate": model_error_cats[m][cat] / cat_counts.get(cat, 1)
                }
                for cat in valid_cats
            },
            "overall": {
                "misselected": model_total_errors[m],
                "total_distractors": sum(cat_counts.get(c, 0) for c in valid_cats),
                "rate": model_total_errors[m] / sum(cat_counts.get(c, 0) for c in valid_cats)
                        if sum(cat_counts.get(c, 0) for c in valid_cats) > 0 else 0
            }
        }
        for m in sorted(model_answers.keys())
    },
    "per_model_total_errors": dict(model_total_errors.most_common()),
}

output_path = os.path.join(os.path.dirname(MODEL_DIR), "distractor_analysis.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"\n[Saved] {output_path}")
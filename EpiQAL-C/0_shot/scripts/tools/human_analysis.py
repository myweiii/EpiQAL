import json, glob, os
from collections import defaultdict
import numpy as np


def krippendorff_alpha(data_matrix, level="ordinal"):
    R = len(data_matrix)
    N = len(data_matrix[0])
    mat = np.full((R, N), np.nan)
    for i in range(R):
        for j in range(N):
            if data_matrix[i][j] is not None:
                mat[i, j] = data_matrix[i][j]
    values = sorted(set(v for row in mat for v in row if not np.isnan(v)))
    if len(values) <= 1:
        return 1.0
    if level == "nominal":
        diff_func = lambda a, b: 0.0 if a == b else 1.0
    else:
        val2rank = {v: i for i, v in enumerate(values)}
        def diff_func(a, b):
            lo, hi = min(val2rank[a], val2rank[b]), max(val2rank[a], val2rank[b])
            return (sum(1 for k in range(lo, hi + 1)) - 1) ** 2
    D_o = D_o_pairs = 0.0
    for j in range(N):
        col = [mat[i, j] for i in range(R) if not np.isnan(mat[i, j])]
        if len(col) < 2: continue
        for a in range(len(col)):
            for b in range(a + 1, len(col)):
                D_o += diff_func(col[a], col[b]); D_o_pairs += 1
    if D_o_pairs == 0: return 1.0
    all_vals, weights = [], []
    for j in range(N):
        col = [mat[i, j] for i in range(R) if not np.isnan(mat[i, j])]
        if len(col) < 2: continue
        for v in col: all_vals.append(v); weights.append(1.0 / (len(col) - 1))
    D_e = D_e_pairs = 0.0
    for i in range(len(all_vals)):
        for k in range(i + 1, len(all_vals)):
            d = diff_func(all_vals[i], all_vals[k])
            D_e += d * weights[i] * weights[k]; D_e_pairs += weights[i] * weights[k]
    if D_e_pairs == 0 or D_e == 0: return 1.0
    return 1.0 - (D_o / D_o_pairs) / (D_e / D_e_pairs)

EVAL_DIR = "./human_evaluation_1"
SUBSET = "c"
DIMS = ["Answer Correctness", "Distractor Quality", "Question Clarity", "Evidence Sufficiency", "Reasoning Depth", "Answerability"]

files = sorted(glob.glob(os.path.join(EVAL_DIR, f"{SUBSET}_eval_sample_*.json")))
all_data = [(os.path.basename(fp), json.load(open(fp, encoding="utf-8"))) for fp in files]

print(f"Subset {SUBSET.upper()}: {len(files)} annotator file(s)\n")

for fname, data in all_data:
    dim_scores = defaultdict(list)
    for item in data:
        ev = item.get("evaluation", {})
        for d in DIMS:
            v = ev.get(d, "")
            if v != "" and v is not None:
                dim_scores[d].append(int(v))
    print(f"  {fname} ({len(data)} samples)")
    for d in DIMS:
        s = dim_scores[d]
        dist = {str(i): s.count(i) for i in sorted(set(s))}
        print(f"    {d:25s}  mean={sum(s)/len(s):.2f}  n={len(s)}  dist={dist}")
    print()

id_dim = defaultdict(lambda: defaultdict(list))
for fname, data in all_data:
    for item in data:
        eid = item.get("eval_id")
        ev = item.get("evaluation", {})
        for d in DIMS:
            v = ev.get(d, "")
            if v != "" and v is not None:
                id_dim[eid][d].append(int(v))

print("  Aggregated:")
for d in DIMS:
    means = [sum(id_dim[e][d])/len(id_dim[e][d]) for e in id_dim if id_dim[e].get(d)]
    print(f"    {d:25s}  mean={sum(means)/len(means):.2f}  items={len(means)}")

if len(all_data) >= 2:
    print("\n  Inter-Annotator Agreement:")
    for d in DIMS:
        total = agree = 0
        for e in id_dim:
            sc = id_dim[e].get(d, [])
            if len(sc) >= 2:
                total += 1
                if len(set(sc)) == 1: agree += 1
        print(f"    {d:25s}  {agree/total:.1%}  ({total} items)" if total else f"    {d:25s}  N/A")

    print("\n  Krippendorff's alpha (ordinal):")
    all_eids = sorted(id_dim.keys())
    all_alphas = []
    for d in DIMS:
        matrix = []
        for idx, (fname, data) in enumerate(all_data):
            eid_to_score = {}
            for item in data:
                ev = item.get("evaluation", {})
                v = ev.get(d, "")
                if v != "" and v is not None:
                    eid_to_score[item["eval_id"]] = int(v)
            matrix.append([eid_to_score.get(eid, None) for eid in all_eids])
        alpha = krippendorff_alpha(matrix, level="ordinal")
        all_alphas.append(alpha)
        print(f"    {d:25s}  alpha={alpha:.3f}")
    print(f"    {'Overall (mean)':25s}  alpha={np.mean(all_alphas):.3f}")
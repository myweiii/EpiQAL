import matplotlib
matplotlib.use('Agg')
from ..constant import *
 
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
 

FONT_PATH = "./scripts/tools/distribution/Times New Roman.ttf"

_font_prop = fm.FontProperties(fname=FONT_PATH)
_font_name = _font_prop.get_name()
fm.fontManager.addfont(FONT_PATH)
matplotlib.rcParams['font.family'] = _font_name
 
 
def load_diff_scores(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    scores = {}
    for qid, val in data.items():
        if "Diff Score" in val:
            scores[qid] = val["Diff Score"]
    return scores
 
def print_stats(name, scores):
    arr = np.array(list(scores.values()))
    print(f"\n{'='*50}")
    print(f"  {name}  (n={len(arr)})")
    print(f"{'='*50}")
    print(f"  Mean:   {arr.mean():.4f}")
    print(f"  Median: {np.median(arr):.4f}")
    print(f"  Std:    {arr.std():.4f}")
    print(f"  Min:    {arr.min():.4f}")
    print(f"  Max:    {arr.max():.4f}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p:02d}:    {np.percentile(arr, p):.4f}")

    for theta in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        n_easy = np.sum(arr < theta)
        print(f"  θ_d={theta:.2f}: {n_easy}/{len(arr)} easy ({n_easy/len(arr)*100:.1f}%)")
 
if __name__ == "__main__":
    FILES = {
        "Original":  f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_0/scores.json",
        "Iter 1":    f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_1/scores.json",
        "Iter 2":    f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_2/scores.json",
        "Iter 3":    f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_3/scores.json",
    }
 
    os.makedirs("./scripts/tools/diff_selection", exist_ok=True)
 
 
    all_scores = {}
    for name, path in FILES.items():
        all_scores[name] = load_diff_scores(path)
        print_stats(f"B - {name}", all_scores[name])
 
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
 
    iter_names = list(FILES.keys())
    colors = ["#7EAAC8", "#A8D5BA", "#E6B89C", "#B8A9C9"]
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5),
                                    height_ratios=[3, 2], sharex=True)
 
    data_arrays = [np.array(list(all_scores[name].values())) for name in iter_names]
 
    bp = ax1.boxplot(data_arrays, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', linewidth=1.5),
                     whiskerprops=dict(linewidth=1.0),
                     flierprops=dict(marker='.', markersize=2, alpha=0.3))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
 
    means = [d.mean() for d in data_arrays]
    ax1.scatter(range(1, 5), means, color='#2F4858', marker='D', s=30, zorder=5, label='Mean')
    ax1.axhline(y=0.2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label=r'$\theta_d = 0.2$')
 
    ax1.set_ylabel("DiffScore")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
 
    stderrs = [d.std() / np.sqrt(len(d)) for d in data_arrays]
    easy_pcts = [np.sum(d < 0.2) / len(d) * 100 for d in data_arrays]
 
    ax2.errorbar(range(1, 5), means, yerr=stderrs, fmt='D-', color='#2F4858',
                 markersize=6, linewidth=1.5, capsize=4, capthick=1.2, label='Mean ± SE')

    for i, (m, pct) in enumerate(zip(means, easy_pcts)):
        ax2.annotate(f'{m:.3f}\n({pct:.1f}% easy)',
                     xy=(i + 1, m), xytext=(0, 12), textcoords='offset points',
                     ha='center', fontsize=8, color='#333333')
 
    ax2.axhline(y=0.2, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax2.set_xticks(range(1, 5))
    ax2.set_xticklabels(iter_names)
    ax2.set_ylabel("Mean DiffScore")
    y_lo = min(means) - 0.04
    y_hi = max(means) + 0.06
    ax2.set_ylim(y_lo, y_hi)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(axis='y', alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("./scripts/tools/diff_selection/b_diffscore_boxplot.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("\n[Saved] ./scripts/tools/diff_selection/b_diffscore_boxplot.pdf")

    common_ids = sorted(set.intersection(*[set(s.keys()) for s in all_scores.values()]))
 
    fig, ax = plt.subplots(figsize=(12, 6))
    matrix = np.array([[all_scores[name][qid] for name in iter_names] for qid in common_ids])
 
    for i in range(len(common_ids)):
        ax.plot(range(4), matrix[i], alpha=0.15, color="gray", linewidth=0.8)
 
    means = matrix.mean(axis=0)
    ax.plot(range(4), means, color="red", linewidth=2.5, marker="o", markersize=8, label="Mean", zorder=5)
    medians = np.median(matrix, axis=0)
    ax.plot(range(4), medians, color="orange", linewidth=2.5, marker="s", markersize=8, label="Median", zorder=5)
 
    ax.set_xticks(range(4))
    ax.set_xticklabels(iter_names)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Diff Score")
    ax.set_title(f"EpiQAL-B: DiffScore Across Iterations (n={len(common_ids)})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("./scripts/tools/diff_selection/b_diffscore_trends.png", dpi=150, bbox_inches="tight")
    print("[Saved] ./scripts/tools/diff_selection/b_diffscore_trends.png")
 
 
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    selection_stats = {theta: {"Original": 0, "Iter 1": 0, "Iter 2": 0, "Iter 3": 0, "None": 0} for theta in thresholds}
 
    for theta in thresholds:
        for qid in common_ids:
            selected = "None"
            for name in iter_names:
                if all_scores[name][qid] >= theta:
                    selected = name
                    break
            selection_stats[theta][selected] += 1
 
        print(f"\n  θ_d = {theta:.2f}:")
        for name in iter_names + ["None"]:
            cnt = selection_stats[theta][name]
            print(f"    {name:10s}: {cnt:4d} ({cnt/len(common_ids)*100:5.1f}%)")
 
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
 
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(thresholds))
    width = 0.55
    bottom = np.zeros(len(thresholds))
    colors_bar = ["#7EAAC8", "#A8D5BA", "#E6B89C", "#B8A9C9", "#DEDEDE"]
    labels = ["Original", "Iter 1", "Iter 2", "Iter 3", "Fallback (Iter 3)"]
 
    for i, name in enumerate(iter_names + ["None"]):
        vals = np.array([selection_stats[t][name] for t in thresholds])
        ax.bar(x, vals, width, bottom=bottom, label=labels[i], color=colors_bar[i], edgecolor='white', linewidth=0.5)
        bottom += vals
 
    # Highlight θ_d = 0.2
    idx_02 = thresholds.index(0.2)
    ax.axvline(x=idx_02, color='#2F4858', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.annotate(r'$\theta_d\!=\!0.2$', xy=(idx_02 + 0.15, len(common_ids) * 0.92), fontsize=8, color='#2F4858')
 
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_xlabel(r"Threshold $\theta_d$")
    ax.set_ylabel("Number of instances")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("./scripts/tools/diff_selection/b_version_selection.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print("\n[Saved] ./scripts/tools/diff_selection/b_version_selection.pdf")
    
    
    
    QA_FILES = [
        f"{RESULT_FILE_PATH}/tmp/questions.json",
        f"{RESULT_FILE_PATH}/tmp/revision/revision_1/questions.json",
        f"{RESULT_FILE_PATH}/tmp/revision/revision_2/questions.json",
        f"{RESULT_FILE_PATH}/tmp/revision/revision_3/questions.json",
    ]

    FINAL_RESULT = f"{RESULT_FILE_PATH}/final_results.json"

    scores = []
    for path in FILES.values():
        with open(path) as f:
            scores.append(json.load(f))

    qas = []
    for path in QA_FILES:
        with open(path) as f:
            qas.append(json.load(f))

    with open(FINAL_RESULT) as f:
        final_res = json.load(f)

    counts = [0, 0, 0, 0]
    for item in final_res:
        qid = item["idx"]
        selected = 3  # fallback
        item["ori_question"]["Question"] = qas[0][qid]["Question"]
        
        for i in range(4):
            if scores[i][qid]["Diff Score"] >= DIFFICULTY_THRESHOLD:
                selected = i
                break
        counts[selected] += 1
        item["question"] = {"Revised Question": qas[selected][qid]["Question"],
                            "Revised Time": selected,
                            "Difficulty Score": scores[selected][qid]["Diff Score"]}

    with open(FINAL_RESULT, "w") as f:
        json.dump(final_res, f, indent=4)

    print(f"Done. Selection: Original={counts[0]}, Iter1={counts[1]}, Iter2={counts[2]}, Iter3={counts[3]}")

    final_qa = []
    for item in final_res:
        final_qa.append({
            "idx": item["idx"],
            "paragraph": item["paragraph"],
            "question": item["question"]["Revised Question"],
            "choices": item["choices"],
            "ref_answers": item["ref_answers"],
        })

    output_path = f"{RESULT_FILE_PATH}/final_qa.json"
    with open(output_path, "w") as f:
        json.dump(final_qa, f, indent=4)

    print(f"[Saved] {output_path} ({len(final_qa)} items)")
    

    SKIP_KEYS = {"Average", "Variance", "Diff Score"}
    model_f1s = {}
    model_ems = {}

    for item in final_res:
        qid = item["idx"]
        selected = item["question"]["Revised Time"]
        question_scores = scores[selected][qid]
        for model, metrics in question_scores.items():
            if model in SKIP_KEYS:
                continue
            if metrics["f1"] < 0 or metrics["exact_match"] < 0:
                continue
            if model not in model_f1s:
                model_f1s[model] = []
                model_ems[model] = []
            model_f1s[model].append(metrics["f1"])
            model_ems[model].append(metrics["exact_match"])

    print(f"\n{'='*60}")
    print(f"  Final Model Scores (after stem selection)")
    print(f"{'='*60}")
    for model in model_f1s:
        avg_f1 = np.mean(model_f1s[model])
        avg_em = np.mean(model_ems[model])
        print(f"  {model}:")
        print(f"    F1={avg_f1:.4f}  EM={avg_em:.4f}  (n={len(model_f1s[model])})")

    overall_f1 = np.mean([np.mean(vals) for vals in model_f1s.values()])
    overall_em = np.mean([np.mean(vals) for vals in model_ems.values()])
    print(f"\n  Overall: F1={overall_f1:.4f}  EM={overall_em:.4f}")
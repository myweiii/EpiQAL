from ..constant import *
import json
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
from matplotlib.colors import ListedColormap
import numpy as np
from collections import Counter
import math

MORANDI_PALETTE = [
    "#7EAAC8", 
    "#A8D5BA", 
    "#E6B89C", 
    "#B8A9C9", 
    "#F2C4C4", 
    "#8FBCBB", 
    "#D4A5A5", 
    "#9DC3C1", 
    "#C9B8D9", 
    "#E8D5B7", 
    "#A3C4BC", 
    "#D5C4A1", 
    "#B5CFD8", 
    "#E2BFC4", 
    "#C8D5B9", 
    "#D1B9A0",
    "#A7B5C8", 
    "#C4D8C0", 
    "#DFBFA8", 
    "#B0A8C0", 
    "#DEDEDE",
]

COLOR_A = "#7EAAC8" 
COLOR_B = "#E6B89C"


if __name__ == "__main__":
    with open(f"{RESULT_FILE_PATH}/final_qa.json", "r") as f:
        final_qa = json.load(f)
    with open(f"{RESULT_FILE_PATH}/tmp/classes.json", "r") as f:
        classes = json.load(f)
    with open(f"{RESULT_FILE_PATH}/tmp/topics.json", "r") as f:
        topics = json.load(f)

    class_distribution = {}
    topic_distribution = {}

    for item in final_qa:
        doc_id = item["idx"]
        current_class = classes[doc_id]["Class"]
        current_topic = topics[doc_id]["Topic"]
        class_distribution[current_class] = class_distribution.get(current_class, 0) + 1
        topic_distribution[current_topic] = topic_distribution.get(current_topic, 0) + 1
        
    total_options = 0
    total_correct = 0
    for item in final_qa:
        total_options += len(item["choices"])
        total_correct += len(item["ref_answers"])

    print(f"Samples: {len(final_qa)}")
    print(f"Avg. #Options: {total_options / len(final_qa):.3f}")
    print(f"Avg. #Correct: {total_correct / len(final_qa):.3f}")

    with open(f"./scripts/tools/distribution/class_distribution_A.json", "w") as f:
        json.dump(class_distribution, f, indent=4)
    with open(f"./scripts/tools/distribution/topic_distribution_A.json", "w") as f:
        json.dump(topic_distribution, f, indent=4)

    print(f"Classes: {len(class_distribution)}, Topics: {len(topic_distribution)}")

    font_manager.fontManager.addfont("./scripts/tools/distribution/Times New Roman.ttf")
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    with open(f"./scripts/tools/distribution/class_distribution_A.json", "r") as f:
        class_A = json.load(f)
    with open(f"./scripts/tools/distribution/class_distribution_A.json", "r") as f:
        class_B = json.load(f)

    all_labels = sorted(set(class_A) | set(class_B))
    A = np.array([class_A.get(k, 0) for k in all_labels], dtype=float)
    B = np.array([class_B.get(k, 0) for k in all_labels], dtype=float)

    order = np.argsort(-(A + B))
    labels = [all_labels[i] for i in order]
    A, B = A[order], B[order]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(labels))
    h = 0.3

    barsA = ax.barh(y - h/2, A, height=h, label="Dataset A", color=COLOR_A)
    barsB = ax.barh(y + h/2, B, height=h, label="Dataset B", color=COLOR_B)

    ax.bar_label(barsA, fmt="%.0f", padding=2, fontsize=7)
    ax.bar_label(barsB, fmt="%.0f", padding=2, fontsize=7)

    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number")
    ax.set_ylabel("Class Name")
    ax.set_title("Class Distribution")
    ax.xaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=False)

    plt.tight_layout()
    plt.savefig("./scripts/tools/distribution/class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(f"./scripts/tools/distribution/topic_distribution_A.json", "r") as f:
        topic_A = json.load(f)
    with open(f"./scripts/tools/distribution/topic_distribution_A.json", "r") as f:
        topic_B = json.load(f)

    all_labels = sorted(set(topic_A) | set(topic_B))
    A = np.array([topic_A.get(k, 0) for k in all_labels], dtype=float)
    B = np.array([topic_B.get(k, 0) for k in all_labels], dtype=float)

    order = np.argsort(-(A + B))
    labels = [all_labels[i] for i in order]
    A, B = A[order], B[order]

    total = float((A + B).sum()) or 1.0
    keep = (A + B) / total >= 0.03

    labels_kept = [l for l, k in zip(labels, keep) if k]
    A_kept = [v for v, k in zip(A, keep) if k]
    B_kept = [v for v, k in zip(B, keep) if k]

    other_A = float(A[~keep].sum())
    other_B = float(B[~keep].sum())
    if other_A > 0 or other_B > 0:
        labels_kept.append("Other")
        A_kept.append(other_A)
        B_kept.append(other_B)

    colors = [MORANDI_PALETTE[i % len(MORANDI_PALETTE)] for i in range(len(labels_kept))]
    color_map = {lab: c for lab, c in zip(labels_kept, colors)}

    def autopct_hide_small(p):
        return f"{p:.1f}%" if p >= 3 else ""

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].pie(A_kept, labels=None, colors=colors, autopct=autopct_hide_small,
                startangle=90, counterclock=False, radius=0.7, pctdistance=0.75)
    axes[0].set_title("Topic Distribution for Dataset A", pad=2, y=0.98)
    axes[0].axis("equal")

    axes[1].pie(B_kept, labels=None, colors=colors, autopct=autopct_hide_small,
                startangle=90, counterclock=False, radius=0.7, pctdistance=0.75)
    axes[1].set_title("Topic Distribution for Dataset B", pad=2, y=0.98)
    axes[1].axis("equal")

    handles = [patches.Patch(facecolor=color_map[lab], edgecolor="none", label=lab) for lab in labels_kept]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.85, 0.5), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.86, 1])
    plt.savefig("./scripts/tools/distribution/topic_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
from ..constant import *
import json
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
import numpy as np
from collections import Counter
import math

if __name__ == "__main__":
    with open(f"{RESULT_FILE_PATH}/final_qa.json", "r") as f:
        final_qa = json.load(f)

    total_options = 0
    total_correct = 0
    for item in final_qa:
        total_options += len(item["choices"])
        total_correct += len(item["ref_answers"])

    print(f"Samples: {len(final_qa)}")
    print(f"Avg. #Options: {total_options / len(final_qa):.3f}")
    print(f"Avg. #Correct: {total_correct / len(final_qa):.3f}")

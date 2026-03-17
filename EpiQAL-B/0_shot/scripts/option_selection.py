from .func import *
from .constant import *
import os
import json
import math

def option_selection_pipeline(input_para, correct_option_checking, distractor_checking, review_options=None):
    ref_answers = {}
    selected_options = {}
    
    accept_threshold = math.ceil(CHECK_VOTE_THRES * CHECK_MODEL_NUM)
    review_threshold = math.ceil(HUMAN_REVIEW_THRES * CHECK_MODEL_NUM)
    
    total = 0
    discarded = 0
    
    ################### Option Slection ###################
    for i in tqdm(range(len(input_para)), desc=f"Option Selection"):
        current_idx = input_para[i]["idx"]
        total += 1
        
        correct_choices = []
        if current_idx not in correct_option_checking.keys():
            correct_option_checking[current_idx] = {}
        for correct_option in correct_option_checking[current_idx].keys():
            votes = correct_option_checking[current_idx][correct_option]
            #print(CHECK_MODEL_NUM * CHECK_VOTE_THRES)
            if votes >= accept_threshold:
                correct_choices.append(correct_option)
            elif votes >= review_threshold and review_options:
                if current_idx in review_options and correct_option in review_options[current_idx]:
                    decision = review_options[current_idx][correct_option].get("decision", "")
                    if decision == "accept":
                        correct_choices.append(correct_option)
        correct_idx = list(range(len(correct_choices)))

        distractor_choices = []
        if current_idx not in distractor_checking.keys():
            distractor_checking[current_idx] = {}
        for distractor in distractor_checking[current_idx].keys():
            votes = distractor_checking[current_idx][distractor]
            if votes >= accept_threshold:
                distractor_choices.append(distractor)
            elif votes >= review_threshold and review_options:
                if current_idx in review_options and distractor in review_options[current_idx]:
                    decision = review_options[current_idx][distractor].get("decision", "")
                    if decision == "accept":
                        distractor_choices.append(distractor)
        
        if len(correct_choices) == 0:
            discarded += 1
            continue
        choices = correct_choices + distractor_choices

        random_idx = list(range(len(choices)))
        random.shuffle(random_idx)
        #correct_idx = random_idx.index(correct_idx)
        correct_option_idx = [str(random_idx.index(i)) for i in correct_idx]
        shuffled_choices = [{"Index": i, "Option": choices[random_idx[i]]} for i in range(len(random_idx))]
        #print(random_idx)
        #print(correct_option_idx)
        
        ref_answers[current_idx] = sorted(correct_option_idx) #.sort()
        selected_options[current_idx] = shuffled_choices
    
    ref_answers = sort_dict(ref_answers)
    with open(f"{RESULT_FILE_PATH}/tmp/ref_answers.json", "w") as f:
        json.dump(ref_answers, f, indent=4)
    
    selected_options = sort_dict(selected_options)
    with open(f"{RESULT_FILE_PATH}/tmp/selected_options.json", "w") as f:
        json.dump(selected_options, f, indent=4)
    
    print(f"\n=== Option Selection Summary ===")
    print(f"Total samples:     {total}")
    print(f"Discarded:         {discarded} ({discarded/total*100:.1f}%)")
    print(f"Retained:          {total - discarded} ({(total - discarded)/total*100:.1f}%)")


    return ref_answers, selected_options

if __name__ == "__main__":
    
    train, val, test = get_data()
    print(len(train), len(val), len(test))
    
    input_para = train + val + test
    input_para = input_para[:500]

    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/correct_option_checking.json", "r") as f:
        correct_option_checking = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/distractor_checking.json", "r") as f:
        distractor_checking = json.load(f)
        
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/review_options_finished.json", "r") as f:
        review_options = json.load(f)

    option_selection_pipeline(input_para, correct_option_checking, distractor_checking, review_options)  
    
    
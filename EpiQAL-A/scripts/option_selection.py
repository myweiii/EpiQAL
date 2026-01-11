from .func import *
from .constant import *
import os
import json


def option_selection_pipeline(input_para, correct_option_checking, distractor_checking):
    ref_answers = {}
    selected_options = {}
    
    ################### Option Slection ###################
    for i in tqdm(range(len(input_para)), desc=f"Option Selection"):
        current_idx = input_para[i]["idx"]
        
        correct_choices = []
        if current_idx not in correct_option_checking.keys():
            correct_option_checking[current_idx] = {}
        for correct_option in correct_option_checking[current_idx].keys():
            votes = correct_option_checking[current_idx][correct_option]
            #print(CHECK_MODEL_NUM * CHECK_VOTE_THRES)
            if votes >= CHECK_MODEL_NUM * CHECK_VOTE_THRES:
                correct_choices.append(correct_option)
        correct_idx = list(range(len(correct_choices)))

        distractor_choices = []
        if current_idx not in distractor_checking.keys():
            distractor_checking[current_idx] = {}
        for distractor in distractor_checking[current_idx].keys():
            votes = distractor_checking[current_idx][distractor]
            if votes >= CHECK_MODEL_NUM * CHECK_VOTE_THRES:
                distractor_choices.append(distractor)
        
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

    return ref_answers, selected_options

if __name__ == "__main__":
    
    train, val, test = get_data()
    print(len(train), len(val), len(test))
    
    input_para = train + val + test
    input_para = input_para[100:106]

    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/correct_option_checking.json", "r") as f:
        correct_option_checking = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/distractor_checking.json", "r") as f:
        distractor_checking = json.load(f)
        
    option_selection_pipeline(input_para, correct_option_checking, distractor_checking)
    
    
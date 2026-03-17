from ..constant import *
import json
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
import numpy as np
from collections import Counter, defaultdict
import math

if __name__ == "__main__":
    with open(f"{RESULT_FILE_PATH}/input_para.json", "r") as f:
        initial_input_para = json.load(f)
        
    initial_input_para_json = {}
    replaced_entity = defaultdict(list)
    for item in initial_input_para:
        current_idx = item["idx"]
        initial_input_para_json[current_idx] = item
        
    with open(f"{RESULT_FILE_PATH}/input_para_json.json", "w") as f:
        json.dump(initial_input_para_json, f, indent=4)
        
    with open(f"{RESULT_FILE_PATH}/input_para_json.json", "r") as f:
        input_para_json = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/questions.json", "r") as f:
        questions = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/coherence_correct_option.json", "r") as f:
        coherence_correct_option = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/coherence_distractor.json", "r") as f:
        coherence_distractor = json.load(f)
        
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/correct_option_checking.json", "r") as f:
        correct_option_checking = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/distractor_checking.json", "r") as f:
        distractor_checking = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/option/correct_options.json", "r") as f:
        correct_options = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/option/distractors.json", "r") as f:
        distractors = json.load(f)

    
    total_num = 0

    # Option-level
    total_correct_accept = 0
    total_correct_reject = 0
    total_correct_review = 0
    total_distractor_accept = 0
    total_distractor_reject = 0
    total_distractor_review = 0

    # Instance-level
    instance_accept = 0
    instance_reject = 0
    instance_review = 0
    instance_has_rejected_option = 0
    instance_all_correct_rejected = 0
    instance_all_distractor_rejected = 0
    instance_discarded = 0

    review_vote_distribution = Counter()
    all_vote_distribution = Counter()

    review_options = defaultdict(dict)
    
    for doc_id in input_para_json.keys():
        total_num += 1

        inst_statuses = []
        correct_statuses = []
        distractor_statuses = []
        
        # Correct options
        if doc_id in correct_option_checking:
            for option, vote_count in correct_option_checking[doc_id].items():
                all_vote_distribution[vote_count] += 1
                accept_threshold = math.ceil(CHECK_VOTE_THRES * CHECK_MODEL_NUM)
                review_threshold = math.ceil(HUMAN_REVIEW_THRES * CHECK_MODEL_NUM)
                if vote_count < review_threshold:
                    total_correct_reject += 1
                    inst_statuses.append("reject")
                    correct_statuses.append("reject")
                elif vote_count >= accept_threshold:
                    total_correct_accept += 1
                    inst_statuses.append("accept")
                    correct_statuses.append("accept")
                else:
                    total_correct_review += 1
                    inst_statuses.append("review")
                    correct_statuses.append("review")
                    review_vote_distribution[vote_count] += 1
                    
                    review_options[doc_id]["Passage"] = input_para_json[doc_id]["inputs"]
                    review_options[doc_id]["Question"] = questions[doc_id]
                    
                    correct_options_list = []
                    for item in correct_options.get(doc_id, []):
                        correct_options_list.append(item.get("Option", []))
                    review_options[doc_id]["Correct Option"] = correct_options_list
                                   
                    failed_rationales = []
                    for model_name, results in coherence_correct_option[doc_id].items():
                        for r in results:
                            if r["Option"] == option and r["Coherence"] == "No":
                                failed_rationales.append({"model": model_name, "rationale": r["Rationale"]})
                    
                    gen_evidence = []
                    gen_rationale = ""
                    for item in correct_options.get(doc_id, []):
                        if item["Option"] == option:
                            gen_evidence = item.get("Evidence", [])
                            gen_rationale = item.get("Rationale", "")
                            break

                    review_options[doc_id][option] = {
                                                    "option": option,
                                                    "category": "Correct Option",
                                                    "votes": f"{vote_count}/{CHECK_MODEL_NUM}",
                                                    "evidence": gen_evidence,
                                                    "gen_rationale": gen_rationale,
                                                    "failed_rationales": failed_rationales,
                                                    "decision": ""
                                                    }
        
        # Distractors
        if doc_id in distractor_checking:
            for option, vote_count in distractor_checking[doc_id].items():
                all_vote_distribution[vote_count] += 1
                accept_threshold = math.ceil(CHECK_VOTE_THRES * CHECK_MODEL_NUM)
                review_threshold = math.ceil(HUMAN_REVIEW_THRES * CHECK_MODEL_NUM)
                if vote_count < review_threshold:
                    total_distractor_reject += 1
                    inst_statuses.append("reject")
                    distractor_statuses.append("reject")
                elif vote_count >= accept_threshold:
                    total_distractor_accept += 1
                    inst_statuses.append("accept")
                    distractor_statuses.append("accept")
                else:
                    total_distractor_review += 1
                    inst_statuses.append("review")
                    distractor_statuses.append("review")
                    review_vote_distribution[vote_count] += 1
                    
                    review_options[doc_id]["Passage"] = input_para_json[doc_id]["inputs"]
                    review_options[doc_id]["Question"] = questions[doc_id]
                    
                    correct_options_list = []
                    for item in correct_options.get(doc_id, []):
                        correct_options_list.append(item.get("Option", []))
                    review_options[doc_id]["Correct Option"] = correct_options_list
                    
                    failed_rationales = []
                    for model_name, results in coherence_distractor[doc_id].items():
                        for r in results:
                            if r["Option"] == option and r["Coherence"] == "No":
                                failed_rationales.append({"model": model_name, "rationale": r["Rationale"]})

                    gen_evidence = []
                    gen_rationale = ""
                    for item in distractors.get(doc_id, []):
                        if item["Option"] == option:
                            gen_evidence = item.get("Evidence", [])
                            gen_rationale = item.get("Rationale", "")
                            break
                        
                    review_options[doc_id][option] = {
                                                    "option": option,
                                                    "category": "Distractor",
                                                    "votes": f"{vote_count}/{CHECK_MODEL_NUM}",
                                                    "evidence": gen_evidence,
                                                    "gen_rationale": gen_rationale,
                                                    "failed_rationales": failed_rationales,
                                                    "decision": ""
                                                    }

        has_review = "review" in inst_statuses
        has_reject = "reject" in inst_statuses
        all_correct_rej = correct_statuses and all(s == "reject" for s in correct_statuses)
        all_distract_rej = distractor_statuses and all(s == "reject" for s in distractor_statuses)
        
        if all_correct_rej:
            instance_all_correct_rejected += 1
        if all_distract_rej:
            instance_all_distractor_rejected += 1
        if has_reject:
            instance_has_rejected_option += 1

        if all_correct_rej:
            instance_discarded += 1
        elif has_review:
            instance_review += 1
        elif has_reject:
            instance_reject += 1
        else:
            instance_accept += 1

    total_human_review = total_correct_review + total_distractor_review
    total_all_options = (total_correct_accept + total_correct_reject + total_correct_review +
                        total_distractor_accept + total_distractor_reject + total_distractor_review)
    instance_usable = instance_accept + instance_reject + instance_review

    print(f"=== Option-Level Statistics ===")
    print(f"HUMAN_REVIEW_THRES={HUMAN_REVIEW_THRES}, CHECK_VOTE_THRES={CHECK_VOTE_THRES}, CHECK_MODEL_NUM={CHECK_MODEL_NUM}")
    print()
    print(f"{'':20s} {'Accept':>8s} {'Reject':>8s} {'Review':>8s} {'Total':>8s}")
    print(f"{'-'*52}")
    print(f"{'Correct Option':20s} {total_correct_accept:>8d} {total_correct_reject:>8d} {total_correct_review:>8d} {total_correct_accept+total_correct_reject+total_correct_review:>8d}")
    print(f"{'Distractor':20s} {total_distractor_accept:>8d} {total_distractor_reject:>8d} {total_distractor_review:>8d} {total_distractor_accept+total_distractor_reject+total_distractor_review:>8d}")
    print(f"{'All':20s} {total_correct_accept+total_distractor_accept:>8d} {total_correct_reject+total_distractor_reject:>8d} {total_human_review:>8d} {total_all_options:>8d}")
    print()
    if total_all_options > 0:
        print(f"Option accept rate:  {(total_correct_accept+total_distractor_accept)/total_all_options*100:.1f}%")
        print(f"Option reject rate:  {(total_correct_reject+total_distractor_reject)/total_all_options*100:.1f}%")
        print(f"Option review rate:  {total_human_review/total_all_options*100:.1f}%")

    print(f"\n=== Instance-Level Statistics ===")
    print(f"{'':20s} {'Count':>8s} {'Pct':>8s}")
    print(f"{'-'*36}")
    print(f"{'Discarded':20s} {instance_discarded:>8d} {instance_discarded/total_num*100:>7.1f}%  (all correct options rejected)")
    print(f"{'All accept':20s} {instance_accept:>8d} {instance_accept/total_num*100:>7.1f}%")
    print(f"{'Partial reject':20s} {instance_reject:>8d} {instance_reject/total_num*100:>7.1f}%  (reject bad options, keep rest)")
    print(f"{'Needs review':20s} {instance_review:>8d} {instance_review/total_num*100:>7.1f}%")
    print(f"{'Total instances':20s} {total_num:>8d}")
    print(f"{'Usable instances':20s} {instance_usable:>8d} {instance_usable/total_num*100:>7.1f}%  (excluding discarded)")

    print(f"\n=== Vote Distribution (All Options) ===")
    for votes in sorted(all_vote_distribution.keys()):
        count = all_vote_distribution[votes]
        print(f"  {votes}/9 votes: {count:>6d} ({count/total_all_options*100:>5.1f}%)")

    if review_vote_distribution:
        print(f"\n=== Vote Distribution (Review Zone Only) ===")
        for votes in sorted(review_vote_distribution.keys()):
            count = review_vote_distribution[votes]
            print(f"  {votes}/9 votes: {count:>6d} ({count/sum(review_vote_distribution.values())*100:>5.1f}%)")
    
    
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/review_options.json", "w") as f:
            json.dump(review_options, f, indent=4)
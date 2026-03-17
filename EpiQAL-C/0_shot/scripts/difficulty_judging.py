from .func import *
from .constant import *
import os
import json
import math
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm import LLM, SamplingParams
from collections import defaultdict
import numpy as np
import torch 
import gc
from transformers import AutoConfig
from typing import List, Literal
from pydantic import BaseModel, Field
from vllm.sampling_params import StructuredOutputsParams
import logging

class DifficultyJudgingAnalysis(BaseModel):
    Index: str = Field(description="The option's original index as a string.")
    Option: str = Field(description="The exact option text.")
    Category: Literal["Correct", "Incorrect"] = Field(description="Either 'Correct' or 'Incorrect' indicating whether the option can be logically derived from the Passage Body by applying epidemiological principles.")
    Evidence: List[str] = Field(description="A list of verbatim quotes from the Passage Body that support the judgment. Include all quotes necessary to support the reasoning chain; for complex judgments requiring integration of multiple facts, this will naturally involve multiple pieces of evidence.")
    Rationale: str = Field(description="Explains the reasoning for why the option is correct or incorrect, demonstrating how the evidence logically supports or refutes the option through epidemiological reasoning.")
    
class DifficultyJudgingResponse(BaseModel):
    results: List[DifficultyJudgingAnalysis]
        
def difficulty_judging_pipeline(input_para, questions, selected_options, ref_answers, times=0, score_per_input=None):
    
    answers = defaultdict(dict)

    os.makedirs(f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_{times}/model", exist_ok=True)
                    
    for key in JUDGE_MODEL_GROUP.keys():
        for judge_model_args in JUDGE_MODEL_GROUP[key]:
            if key == "API":
                judge_model_name = judge_model_args[2]
                client = OpenAI(api_key=judge_model_args[1], base_url=judge_model_args[0])
                batch_size = judge_model_args[3]
            else:
                judge_model_name = judge_model_args[0]
                config = AutoConfig.from_pretrained(judge_model_name)
                #print(config.max_position_embeddings)
                batch_size = judge_model_args[2]
                
                llm_model = LLM(model=judge_model_name,
                    tensor_parallel_size=LOCAL_TENSOR_PARALLEL_SIZE,
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                    quantization=judge_model_args[1],
                    max_model_len=min(LOCAL_MAX_MODEL_LEN, config.max_position_embeddings),
                    max_num_seqs=batch_size)

                json_schema = DifficultyJudgingResponse.model_json_schema()
                structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
                
                sampling_params = SamplingParams(temperature=JUDGE_TEMPRATURE, max_tokens=JUDGE_MAX_TOKENS, top_p=JUDGE_TOP_P, structured_outputs=structured_outputs_params_json)
            
            answers_per_model = {}
            
            batch_num = math.ceil(len(input_para) / batch_size)
            ################### Judging ###################
            print("#"*30)
            for i in range(0, len(input_para), batch_size):
                print(f"{judge_model_name} - Judging {times}: Batch {i//batch_size+1}/{batch_num}")
                input_batch_para = input_para[i:i+batch_size]
                attempts = 0
                            
                                        
                while(1):
                    if attempts == 0:
                        input_list = range(len(input_batch_para))
                        difficulty_judging_prompt_list = difficulty_judging_prompt(input_batch_para, questions, selected_options)
                    else:
                        print(f"Some are in regeneration... Attempt {attempts}...")
                        #for idx in repeat_list:
                        input_list = repeat_list
                        difficulty_judging_prompt_list = difficulty_judging_prompt([input_batch_para[a] for a in repeat_list] , questions, selected_options, err)
            
            
                    if key == "API":
                        output_list = [None] * len(difficulty_judging_prompt_list)
                        with ThreadPoolExecutor(max_workers=batch_size) as pool:
                            futures = {pool.submit(call_llm, client, p, judge_model_name, JUDGE_TEMPRATURE, JUDGE_MAX_TOKENS, JUDGE_TOP_P, DifficultyJudgingResponse): idx for idx, p in enumerate(difficulty_judging_prompt_list)}

                            for future in tqdm(as_completed(futures), total=len(difficulty_judging_prompt_list)):
                                idx = futures[future]
                                output_list[idx] = future.result()
                    else:
                        output_list = llm_model.chat(difficulty_judging_prompt_list, sampling_params)
                    
                    repeat_list = []
                    err = []
                    #print(f"\n{attempts}: {len(input_list)}")
                    for idx in range(len(input_list)):
                        try:
                            llm_response = output_list[idx]
                            current_idx = input_batch_para[input_list[idx]]['idx']
                            
                            if len(selected_options) == 0:
                                answers[current_idx][judge_model_name] = [{"Index": "-1",
                                                        "Category": "Failed to answer",
                                                        "Option": "Failed to answer",
                                                        "Reason": "Failed to answer"
                                                        }]
                                continue
                    
                            if key == "API":
                                llm_response = llm_response.choices[0].message.content.strip()
                            else:
                                llm_response = llm_response.outputs[0].text.strip().split('</think>')[-1]
                            
                            #print(llm_response)
                            answers[current_idx][judge_model_name] = json.loads(llm_response)["results"]
                            answers_per_model[current_idx] = answers[current_idx][judge_model_name]
                            #print("111", current_idx, llm_response)
                            #print(answers_per_model[current_idx])
                        except Exception as err_info:
                            print(f"\n{attempts}: idx {current_idx} - Failed to answer... \n\n {err_info}")
                            repeat_list.append(input_list[idx])
                            err_text = f"Previous response: \n{{{llm_response}}} \n Previous error: {{{err_info}}}"
                            if "<think>" in str(llm_response):
                                err_text = err_text + "\n Suggestion: REDUCE YOUR THINKING TIME."
                            err.append(err_text)
                            
                        #print(output.outputs[0].text)
                        #print("-"*30)

                    attempts += 1
                    if len(repeat_list) == 0 or attempts == MAX_GENERATE_ATTEMPT:
                        break
                    
                for idx, fail_idx in enumerate(repeat_list):
                    current_idx = input_batch_para[fail_idx]['idx']
                    print(f"\nidx {current_idx} - Failed to answer... \n\n {err[idx]}")
                    print("-"*30)
                    answers[current_idx][judge_model_name] = [{"Index": "-1",
                                                        "Category": "Failed to answer",
                                                        "Option": "Failed to answer",
                                                        "Reason": "Failed to answer"
                                                        }]

                    answers_per_model[current_idx] = answers[current_idx][judge_model_name]
            if key == "LOCAL":
                del llm_model
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
            
            answers_per_model = sort_dict(answers_per_model)
            with open(f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_{times}/model/{judge_model_name.split('/')[-1]}.json", "w") as f:
                json.dump(answers_per_model, f, indent=4)
    
    answers = sort_dict(answers)
    with open(f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_{times}/answers.json", "w") as f:
        json.dump(answers, f, indent=4)

    ################### Calculating Scores ###################
    
    if score_per_input == None:
        score_per_input = defaultdict(dict)
    ans_comp = defaultdict(dict)
    
    for i in tqdm(range(0, len(input_para)), desc=f"Calculating Scores"):
        current_idx = input_para[i]["idx"]
        
        for key in JUDGE_MODEL_GROUP.keys():
            for judge_model_args in JUDGE_MODEL_GROUP[key]:
                if key == "API":
                    judge_model_name = judge_model_args[2]
                else:
                    judge_model_name = judge_model_args[0]

                
                try:
                    if answers[current_idx][judge_model_name][0]["Index"] == "-1":
                        score_per_input[current_idx][judge_model_name] = {"f1": -1, "exact_match": -1}
                        continue
                    item_list = []
                    for item in answers[current_idx][judge_model_name]:
                        new_item = {}
                        for gen_key in list(item.keys()):
                            if "Category" in gen_key:
                                new_item["Category"] = item[gen_key]
                            else:
                                new_item[gen_key.strip()] = item[gen_key]
                        item_list.append(new_item)
                    answers[current_idx][judge_model_name] = item_list

                    ans_list = []
                    for ans in answers[current_idx][judge_model_name]:
                        if "correct" == ans["Category"].lower().strip():
                            ans_list.append(str(ans["Index"]))
                    
                    ref_set, ans_set = set(ref_answers[current_idx]), set(ans_list)
                    
                    #print(ref_set, ans_set)
                    inter = len(ref_set & ans_set)
                    precision = inter / len(ans_set) if ans_set else 0
                    recall = inter / len(ref_set) if ref_set else 0
                    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
                    if ref_set == ans_set:
                        exact_match = 1
                    else:
                        exact_match = 0
                    
                    #f1_per_input.append(f1)
                    #em_per_input.append(exact_match)
                    
                    #f1_per_model[judge_model_name].append(f1)
                    #em_per_model[judge_model_name].append(exact_match)
                    
                    ans_comp[current_idx]["ref_ans"] = list(ref_set)
                    ans_comp[current_idx][judge_model_name] = list(ans_set)
                    
                except Exception as err:
                    print("Checking warning!", err)
                    f1 = -1
                    exact_match = -1
                    ans_set = set()
                
                
                score_per_input[current_idx][judge_model_name] = {"f1": f1, "exact_match": exact_match}
        
    
    
    f1_per_model = defaultdict(list)
    em_per_model = defaultdict(list)
    
    for current_idx in tqdm(score_per_input.keys(), desc=f"Formatting Scores"):
        f1_per_input = []
        em_per_input = []
        
        for model_key in list(score_per_input[current_idx].keys()):
            if model_key in ["Average", "Variance", "Diff Score"]:
                continue
            f1 = score_per_input[current_idx][model_key]["f1"]
            exact_match = score_per_input[current_idx][model_key]["exact_match"]
            if f1 != -1 and exact_match != -1:
                f1_per_input.append(f1)
                em_per_input.append(exact_match)
                
                f1_per_model[model_key].append(f1)
                em_per_model[model_key].append(exact_match)
        
        avg_f1 = np.mean(np.array(f1_per_input))
        avg_em = np.mean(np.array(em_per_input))
        diff_score = 1 - (ALPHA * avg_f1 + (1 - ALPHA) * avg_em)
        
        score_per_input[current_idx]["Average"] = {"f1": avg_f1, "exact_match": avg_em}
        score_per_input[current_idx]["Variance"] = {"f1": np.var(np.array(f1_per_input)), "exact_match": np.var(np.array(em_per_input))}
        score_per_input[current_idx]["Diff Score"] = diff_score
        
    score_per_model = {"Score": {}, "Model": {}}
    avg_f1_per_model = []
    avg_em_per_model = []
    for key in f1_per_model.keys():
        score_per_model["Model"][key] = {"f1": np.mean(np.array(f1_per_model[key])), "exact_match": np.mean(np.array(em_per_model[key]))}
        avg_f1_per_model.append(np.mean(np.array(f1_per_model[key])))
        avg_em_per_model.append(np.mean(np.array(em_per_model[key])))
    score_per_model["Model"] = sort_dict(score_per_model["Model"])
    
    score_per_model["Score"] = {"f1": np.mean(np.array(avg_f1_per_model)), "exact_match": np.mean(np.array(avg_em_per_model))}
                                
    
    
    score_per_input = sort_dict(score_per_input)
    with open(f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_{times}/scores.json", "w") as f:
        json.dump(score_per_input, f, indent=4)
    
    #score_per_model = sort_dict(score_per_model)
    with open(f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_{times}/score_per_model.json", "w") as f:
        json.dump(score_per_model, f, indent=4)
    
    ans_comp = sort_dict(ans_comp)
    with open(f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_{times}/answer_comparison.json", "w") as f:
        json.dump(ans_comp, f, indent=4)
    
    
    return score_per_input


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.ERROR)
    train, val, test = get_data()
    print(len(train), len(val), len(test))
    
    input_para = train + val + test
    input_para = input_para[94:104]

    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    
    with open(f"{RESULT_FILE_PATH}/tmp/questions.json", "r") as f:
        questions = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/selected_options.json", "r") as f:
        selected_options = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/ref_answers.json", "r") as f:
        ref_answers = json.load(f)
        
    difficulty_judging_pipeline(input_para, questions, selected_options, ref_answers)
    
    
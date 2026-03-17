from .func import *
from .constant import *
import os
import json
import math
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm import LLM, SamplingParams
from collections import defaultdict
import torch
import gc
from transformers import AutoConfig
from typing import List, Literal
from pydantic import BaseModel, Field
from vllm.sampling_params import StructuredOutputsParams
import logging

class CorrectOptionCheckingAnalysis(BaseModel):
    Category: Literal["Correct Option"] = Field(description="The type of option being checked, which must strictly be 'Correct Option'.")
    Option: str = Field(description="The exact text of the candidate option being evaluated.")
    Coherence: Literal["Yes", "No"] = Field(description="Return 'Yes' if all checks pass: evidence is verbatim from source materials, option is not direct retrieval, option addresses the question, evidence logically supports the option, rationale is sound, and option is epidemiologically correct as an answer. Return 'No' if any check fails.")
    Rationale: str = Field(description="A detailed justification explaining the evaluation of each step. If Coherence is No, specify which step failed and why. If Coherence is Yes, briefly confirm how each check was satisfied.")
    
class CorrectOptionCheckingResponse(BaseModel):
    results: List[CorrectOptionCheckingAnalysis]
    
    
def correct_option_checking_pipeline(input_para, questions, external_info, correct_options):
    
    coherence_correct_option = defaultdict(dict)
    correct_option_checking = defaultdict(dict)
    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp/coherence/correct_option", exist_ok=True)
    
    for key in CHECK_MODEL_GROUP.keys():
        for check_model_args in CHECK_MODEL_GROUP[key]:
            if key == "API":
                check_model_name = check_model_args[2]
                client = OpenAI(api_key=check_model_args[1], base_url=check_model_args[0])
                batch_size = check_model_args[3]
            else:
                check_model_name = check_model_args[0]
                config = AutoConfig.from_pretrained(check_model_name)
                batch_size = check_model_args[2]
                
                llm_model = LLM(model=check_model_name,
                    tensor_parallel_size=LOCAL_TENSOR_PARALLEL_SIZE,
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                    quantization=check_model_args[1],
                    max_model_len=min(LOCAL_MAX_MODEL_LEN, config.max_position_embeddings),
                    max_num_seqs=batch_size)
                
                json_schema = CorrectOptionCheckingResponse.model_json_schema()
                structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
                
                sampling_params = SamplingParams(temperature=CHECK_TEMPRATURE, max_tokens=CHECK_MAX_TOKENS, top_p=CHECK_TOP_P, structured_outputs=structured_outputs_params_json)
            
            model_coherence = defaultdict(dict)
            
            batch_num = math.ceil(len(input_para) / batch_size)
            ################### Correct Option Checking ###################
            print("#"*30)
            for i in range(0, len(input_para), batch_size):
                print(f"{check_model_name} - Correct Option Checking: Batch {i//batch_size+1}/{batch_num}")
                input_batch_para = input_para[i:i+batch_size]
                 
                
                for generate_time in range(1, CHECK_TIME+1):
                    attempts = 0
                    print(f"{check_model_name} - Correct Option Checking: {generate_time}/{CHECK_TIME}")
                    while(1):
                        if attempts == 0:
                            input_list = range(len(input_batch_para))
                            answer_checking_prompt_list = answer_checking_prompt(input_batch_para, questions, external_info, correct_options, "Correct Option")
                        else:
                            print(f"Some are in regeneration... Attempt {attempts}...")
                            input_list = repeat_list
                            answer_checking_prompt_list = answer_checking_prompt([input_batch_para[a] for a in repeat_list], questions, external_info, correct_options, "Correct Option", err)
                

                        if key == "API":
                            output_list = [None] * len(answer_checking_prompt_list)
                            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                                futures = {pool.submit(call_llm, client, p, check_model_name, CHECK_TEMPRATURE, CHECK_MAX_TOKENS, CHECK_TOP_P, CorrectOptionCheckingResponse): idx for idx, p in enumerate(answer_checking_prompt_list)}

                                for future in tqdm(as_completed(futures), total=len(answer_checking_prompt_list)):
                                    idx = futures[future]
                                    output_list[idx] = future.result()
                        else:
                            output_list = llm_model.chat(answer_checking_prompt_list, sampling_params)
                        
                        repeat_list = []
                        err = []
                        for idx in range(len(input_list)):
                            try:
                                llm_response = output_list[idx]
                                #print(llm_response)
                                current_idx = input_batch_para[input_list[idx]]['idx']
                                if key == "API":
                                    llm_response = llm_response.choices[0].message.content.strip()
                                    
                                else:
                                    llm_response = llm_response.outputs[0].text.strip().split('</think>')[-1]
                                #
                                llm_response = llm_response.split("```json")[-1].split("```")[0]
                                #print(llm_response)
                                coherence_correct_option[current_idx][f"{check_model_name}_{generate_time}"] = json.loads(llm_response)["results"]
                                model_coherence[current_idx][f"{check_model_name}_{generate_time}"] = coherence_correct_option[current_idx][f"{check_model_name}_{generate_time}"]
                                
                                item_list = []
                                for item in coherence_correct_option[current_idx][f"{check_model_name}_{generate_time}"]:
                                    new_item = {}
                                    for gen_key in item.keys():
                                        if "Option" in gen_key:
                                            new_item["Option"] = item[gen_key]
                                        elif "Coherence" in gen_key:
                                            new_item["Coherence"] = item[gen_key]
                                        else:
                                            new_item[gen_key] = item[gen_key]
                                            
                                    item_list.append(new_item)
                                coherence_correct_option[current_idx][f"{check_model_name}_{generate_time}"] = item_list
                                
                                '''
                                for gen_key in coherence_correct_option[current_idx][f"{check_model_name}_{generate_time}"].keys():
                                    if "Option" in gen_key:
                                        coherence_correct_option[current_idx][f"{check_model_name}_{generate_time}"]["Option"] = coherence_correct_option[current_idx][f"{check_model_name}_{generate_time}"][gen_key]
                                '''
                                
                                for item in coherence_correct_option[current_idx][f"{check_model_name}_{generate_time}"]:
                                    option = item["Option"].strip()
                                    if "Yes" in item["Coherence"]:
                                        if option not in correct_option_checking[current_idx].keys():
                                            correct_option_checking[current_idx][option] = 0
                                        correct_option_checking[current_idx][option] += 1
                                        
                                
                            except Exception as err_info:
                                repeat_list.append(input_list[idx])
                                err.append(f"Previous response: \n{{{llm_response}}} \n Previous error: {{{err_info}}}")
                                print(f"\n{attempts}: idx {current_idx} - Failed to check correct options... \n\n {err_info}")
                            #print(output.outputs[0].text)
                            #print("-"*30)
                        
                        attempts += 1
                        if len(repeat_list) == 0 or attempts == MAX_GENERATE_ATTEMPT:
                            break
                        
                    for idx, fail_idx in enumerate(repeat_list):
                        current_idx = input_batch_para[fail_idx]['idx']
                        print(f"\nidx {current_idx} - Failed to check correct options... \n\n {err[idx]}")
                        print("-"*30)
                        coherence_correct_option[current_idx][f"{check_model_name}_{generate_time}"] = [{"Category": "Failed to check correct options",
                                                                "Option": "Failed to check correct options",
                                                                "Coherence": "Failed to check correct options",
                                                                "Reason": "Failed to check correct options"
                                                                }]
                        model_coherence[current_idx][f"{check_model_name}_{generate_time}"] = coherence_correct_option[current_idx][f"{check_model_name}_{generate_time}"]
            
            
                
            if key == "LOCAL":
                del llm_model
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
            
            model_coherence = sort_dict(model_coherence)
            with open(f"{RESULT_FILE_PATH}/tmp/coherence/correct_option/{check_model_name.split('/')[-1]}.json", "w") as f:
                json.dump(model_coherence, f, indent=4)
    
    coherence_correct_option = sort_dict(coherence_correct_option)
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/coherence_correct_option.json", "w") as f:
        json.dump(coherence_correct_option, f, indent=4)
    
    correct_option_checking = sort_dict(correct_option_checking)
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/correct_option_checking.json", "w") as f:
        json.dump(correct_option_checking, f, indent=4)
        
    return correct_option_checking


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.ERROR)
    train, val, test = get_data()
    print(len(train), len(val), len(test))
    
    input_para = train + val + test
    input_para = input_para[:100]

    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    
    with open(f"{RESULT_FILE_PATH}/tmp/questions.json", "r") as f:
        questions = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/external_info.json", "r") as f:
        external_info = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/option/correct_options.json", "r") as f:
        correct_options = json.load(f)
        
    correct_option_checking_pipeline(input_para, questions, external_info, correct_options)
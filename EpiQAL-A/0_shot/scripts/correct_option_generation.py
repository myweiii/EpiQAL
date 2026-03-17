from .func import *
from .constant import *
import os
import json
import math
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import List, Literal
from pydantic import BaseModel, Field
import logging
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoConfig
import torch

class CorrectOptionGenerationAnalysis(BaseModel):
    Category: Literal["Correct Option"] = Field(description="The classification of this option, which must always be 'Correct Option'.")
    Option: str = Field(description="A concise answer that directly answers the question. Must be derived from explicit information in the passage. Should be semantically complete but does not need to be a full sentence. Must not copy the entire evidence sentence.")
    Evidence: List[str] = Field(description="A list of exact verbatim quotes from the passage that directly support this option. Each quote must be copied exactly without editing or paraphrasing.")
    Rationale: str = Field(description="Explanation of why this option correctly answers the question, including how the evidence explicitly supports the answer.")
        
class CorrectOptionGenerationResponse(BaseModel):
    results: List[CorrectOptionGenerationAnalysis]

def correct_option_generation_pipeline(input_para, questions, client):
    batch_num = math.ceil(len(input_para) / BATCH_SIZE)
    correct_options = {}
    option_reasoning = defaultdict(dict)
    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp/option", exist_ok=True)
    
    if GENERATION_MODEL_TYPE == "LOCAL":
        json_schema = CorrectOptionGenerationResponse.model_json_schema()
        structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
        sampling_params = SamplingParams(temperature=GENERATION_TEMPRATURE, max_tokens=GENERATION_MAX_TOKENS, top_p=GENERATION_TOP_P, structured_outputs=structured_outputs_params_json)
    
    ################### Correct Option Generation ###################
    print("#"*30)
    for i in range(0, len(input_para), BATCH_SIZE):
        print(f"Correct Option Generation: Batch {i//BATCH_SIZE+1}/{batch_num}")
        input_batch_para = input_para[i:i+BATCH_SIZE]
        attempts = 0   
        
        while(1):
            if attempts == 0:
                input_list = range(len(input_batch_para))
                correct_option_generation_prompt_list = correct_option_generation_prompt(input_batch_para, questions)
            else:
                print(f"Some are in regeneration... Attempt {attempts}...")
                input_list = repeat_list
                correct_option_generation_prompt_list = correct_option_generation_prompt([input_batch_para[a] for a in repeat_list], questions, err)
    
    
            
            
            if GENERATION_MODEL_TYPE == "API":
                output_list = [None] * len(correct_option_generation_prompt_list)
                with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
                    futures = {pool.submit(call_llm, client, p, ResponseWrapper=CorrectOptionGenerationResponse): idx for idx, p in enumerate(correct_option_generation_prompt_list)}

                    for future in tqdm(as_completed(futures), total=len(correct_option_generation_prompt_list)):
                        idx = futures[future]
                        output_list[idx] = future.result()
            else:
                output_list = client.chat(correct_option_generation_prompt_list, sampling_params)
            
            repeat_list = []
            err = []
            for idx in range(len(input_list)):
                try:
                    llm_response = output_list[idx]
                    current_idx = input_batch_para[input_list[idx]]['idx']
                    if GENERATION_MODEL_TYPE == "API":
                        llm_response = llm_response.choices[0].message.content.strip()
                    else:
                        llm_response = llm_response.outputs[0].text.strip().split('</think>')[-1]
                    #llm_response = output.outputs[0].text.split('</think>')[-1].strip()
                    #print(llm_response)
                    correct_options[current_idx] = json.loads(llm_response)["results"]
                    #option_reasoning[current_idx]["Correct Options"] = output_list[idx].choices[0].message.reasoning
                except Exception as err_info:
                    repeat_list.append(input_list[idx])
                    err.append(f"Previous response: \n{{{llm_response}}} \n Previous error: {{{err_info}}}")
                    print(f"\n{attempts}: idx {current_idx} - Failed to generate correct options... \n\n {err_info}")
                #print(output.outputs[0].text)
                #print("-"*30)
            
            attempts += 1
            if len(repeat_list) == 0 or attempts == MAX_GENERATE_ATTEMPT:
                break
            
        for idx, fail_idx in enumerate(repeat_list):
            current_idx = input_batch_para[fail_idx]['idx']
            print(f"\nidx {current_idx} - Failed to generate correct options... \n\n {err[idx]}")
            print("-"*30)
            correct_options[current_idx] = [{"Category": "Failed to generate correct options.",
                                            "Option": "Failed to generate correct options.",
                                            "Evidence": "Failed to generate correct options.",
                                            "Reason": "Failed to generate correct options."
                                            }]
    
    correct_options = sort_dict(correct_options)  
    with open(f"{RESULT_FILE_PATH}/tmp/option/correct_options.json", "w") as f:
        json.dump(correct_options, f, indent=4)
    
    '''
    option_reasoning = sort_dict(option_reasoning)
    with open(f"{RESULT_FILE_PATH}/tmp/option/option_reasoning.json", "w") as f:
        json.dump(option_reasoning, f, indent=4)
    '''
    
    return correct_options  #, option_reasoning


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.ERROR)
    train, val, test = get_data()
    print(len(train), len(val), len(test))
    
    input_para = train + val + test
    input_para = input_para[90:106]
    
    if GENERATION_MODEL_TYPE == "API":
        client = OpenAI(api_key=API_KEY, base_url=OPENAI_BASE_URL)
    else:
        config = AutoConfig.from_pretrained(GENERATION_MODEL_NAME)
        
        client = LLM(model=GENERATION_MODEL_NAME,
            tensor_parallel_size=LOCAL_TENSOR_PARALLEL_SIZE,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization=LOCAL_QUANTIZATION,
            max_model_len=min(LOCAL_MAX_MODEL_LEN, config.max_position_embeddings),
            max_num_seqs=BATCH_SIZE)
    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    
    with open(f"{RESULT_FILE_PATH}/tmp/questions.json", "r") as f:
        questions = json.load(f)
    
        
    correct_option_generation_pipeline(input_para, questions, client)
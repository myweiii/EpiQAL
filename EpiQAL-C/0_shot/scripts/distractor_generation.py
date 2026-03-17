from .func import *
from .constant import *
import os
import json
import math
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from transformers import AutoConfig
import torch

class DistractorGenerationAnalysis(BaseModel):
    Category: Literal["Distractor"] = Field(description="The classification of this option, which must always be Distractor.")
    Option: str = Field(description="The distractor sentence, self-contained and grammatically parallel to the correct option.")
    Discussion_Source: str = Field(description="The original sentence from the Discussion quoted verbatim, or the source used for causal reversal.")
    Evidence: List[str] = Field(description="Verbatim quotes showing why this conclusion cannot be derived from the Passage Body alone.")
    Rationale: str = Field(description="Explanation of why this is a valid distractor that cannot answer the question.")
    
class DistractorGenerationResponse(BaseModel):
    results: List[DistractorGenerationAnalysis]

def distractor_generation_pipeline(input_para, questions, correct_options, client):
    batch_num = math.ceil(len(input_para) / BATCH_SIZE)
    distractors = {}

    os.makedirs(f"{RESULT_FILE_PATH}/tmp/option", exist_ok=True)
    
    if GENERATION_MODEL_TYPE == "LOCAL":
        json_schema = DistractorGenerationResponse.model_json_schema()
        structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
        sampling_params = SamplingParams(temperature=GENERATION_TEMPRATURE, max_tokens=GENERATION_MAX_TOKENS, top_p=GENERATION_TOP_P, structured_outputs=structured_outputs_params_json)
        
    ################### Distractor Generation ###################
    print("#"*30)
    for i in range(0, len(input_para), BATCH_SIZE):
        print(f"Distractor Generation: Batch {i//BATCH_SIZE+1}/{batch_num}")
        input_batch_para = input_para[i:i+BATCH_SIZE]
        attempts = 0   
        
        while(1):
            if attempts == 0:
                input_list = range(len(input_batch_para))
                distractor_generation_prompt_list = distractor_generation_prompt(input_batch_para, questions, correct_options)
            else:
                print(f"Some are in regeneration... Attempt {attempts}...")
                input_list = repeat_list
                distractor_generation_prompt_list = distractor_generation_prompt([input_batch_para[a] for a in repeat_list], questions, correct_options, err)
    
            
            if GENERATION_MODEL_TYPE == "API":
                output_list = [None] * len(distractor_generation_prompt_list)
                with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
                    futures = {pool.submit(call_llm, client, p, ResponseWrapper=DistractorGenerationResponse): idx for idx, p in enumerate(distractor_generation_prompt_list)}

                    for future in tqdm(as_completed(futures), total=len(distractor_generation_prompt_list)):
                        idx = futures[future]
                        output_list[idx] = future.result()
            else:
                output_list = client.chat(distractor_generation_prompt_list, sampling_params)
                    
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
                    distractors[current_idx] = json.loads(llm_response)["results"]
                    #option_reasoning[current_idx]["Distractors"] = output_list[idx].choices[0].message.reasoning
                except Exception as err_info:
                    repeat_list.append(input_list[idx])
                    err.append(f"Previous response: \n{{{llm_response}}} \n Previous error: {{{err_info}}}")
                    print(f"\n{attempts}: idx {current_idx} - Failed to generate distractors... \n\n {err_info}")
                #print(output.outputs[0].text)
                #print("-"*30)
            
            attempts += 1
            if len(repeat_list) == 0 or attempts == MAX_GENERATE_ATTEMPT:
                break
            
        for idx, fail_idx in enumerate(repeat_list):
            current_idx = input_batch_para[fail_idx]['idx']
            print(f"\nidx {current_idx} - Failed to generate distractors... \n\n {err[idx]}")
            print("-"*30)
            distractors[current_idx] = [{"Category": "Failed to generate distractors.",
                                            "Option": "Failed to generate distractors.",
                                            "Evidence": "Failed to generate distractors.",
                                            "Reasoning Process": "Failed to generate correct options."
                                            }]
    
    distractors = sort_dict(distractors)
    with open(f"{RESULT_FILE_PATH}/tmp/option/distractors.json", "w") as f:
        json.dump(distractors, f, indent=4)
    
    '''
    option_reasoning = sort_dict(option_reasoning)
    with open(f"{RESULT_FILE_PATH}/tmp/option/option_reasoning.json", "w") as f:
        json.dump(option_reasoning, f, indent=4)
    '''
    
    return distractors      #, option_reasoning


if __name__ == "__main__":
    
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
        
    with open(f"{RESULT_FILE_PATH}/tmp/option/correct_options.json", "r") as f:
        correct_options = json.load(f)
        
    distractor_generation_pipeline(input_para, questions, correct_options, client)
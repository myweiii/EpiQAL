from .func import *
from .constant import *
import os
import json
import math
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm import LLM, SamplingParams
import torch
from transformers import AutoConfig
import gc

def external_information_generation_pipeline(input_para, relevant_triples):
    batch_num = math.ceil(len(input_para) / KG_BATCH_SIZE)
    external_info = {}
    
    if KG_MODEL_TYPE == "API":
        client = OpenAI(api_key=API_KEY, base_url=OPENAI_BASE_URL)
    else:
        config = AutoConfig.from_pretrained(KG_MODEL_NAME)
        
        client = LLM(model=KG_MODEL_NAME,
            tensor_parallel_size=LOCAL_TENSOR_PARALLEL_SIZE,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization=KG_LOCAL_QUANTIZATION,
            max_model_len=min(LOCAL_MAX_MODEL_LEN, config.max_position_embeddings),
            max_num_seqs=KG_BATCH_SIZE)
        
        sampling_params = SamplingParams(temperature=GENERATION_TEMPRATURE, max_tokens=GENERATION_MAX_TOKENS, top_p=GENERATION_TOP_P)
        
    print("#"*30)
    for i in range(0, len(input_para), KG_BATCH_SIZE):
        print(f"External Information Generation: Batch {i//KG_BATCH_SIZE+1}/{batch_num}")
        input_batch_para = input_para[i:i+KG_BATCH_SIZE]
        attempts = 0
                    
        while(1):
            if attempts == 0:
                input_list = range(len(input_batch_para))
                external_info_generation_prompt_list = external_info_generation_prompt(input_batch_para, relevant_triples)
            else:
                print(f"Some are in regeneration... Attempt {attempts}...")
                input_list = repeat_list
                external_info_generation_prompt_list = external_info_generation_prompt([input_batch_para[a] for a in repeat_list], relevant_triples, err)
    
            if KG_MODEL_TYPE == "API":
                output_list = [None] * len(external_info_generation_prompt_list)
                with ThreadPoolExecutor(max_workers=KG_BATCH_SIZE) as pool:
                    futures = {pool.submit(call_llm, client, p, KG_MODEL_NAME): idx for idx, p in enumerate(external_info_generation_prompt_list)}

                    for future in tqdm(as_completed(futures), total=len(external_info_generation_prompt_list)):
                        idx = futures[future]
                        output_list[idx] = future.result()
            else:
                output_list = client.chat(external_info_generation_prompt_list, sampling_params)


            repeat_list = []
            err = []
            for idx in range(len(input_list)):
                try:
                    llm_response = output_list[idx]
                    current_idx = input_batch_para[input_list[idx]]['idx']
                    #llm_response = output.choices[0].message.content
                    if KG_MODEL_TYPE == "API":
                        llm_response = llm_response.choices[0].message.content.strip()
                    else:
                        llm_response = llm_response.outputs[0].text.strip().split('</think>')[-1]
                    #print(llm_response)
                    external_info[current_idx] = llm_response
                except Exception as err_info:
                    repeat_list.append(input_list[idx])
                    err.append(f"Previous response: \n{{{llm_response}}} \n Previous error: {{{err_info}}}")
                    print(f"\n{attempts}: idx {current_idx} - Failed to generate external information... \n\n {err_info}")
                #print(output.outputs[0].text)
                #print("-"*30)
            
            attempts += 1
            if len(repeat_list) == 0 or attempts == MAX_GENERATE_ATTEMPT:
                break
            
        for idx, fail_idx in enumerate(repeat_list):
            current_idx = input_batch_para[fail_idx]['idx']
            print(f"\nidx {current_idx} - Failed to generate external information... \n\n {err[idx]}")
            print("-"*30)
            external_info[current_idx] = ["Failed to generate external information."] 
    
    external_info = sort_dict(external_info)
    with open(f"{RESULT_FILE_PATH}/tmp/external_info.json", "w") as f:
        json.dump(external_info, f, indent=4)
    
    del client
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    return external_info


if __name__ == "__main__":
    
    train, val, test = get_data()
    print(len(train), len(val), len(test))
    
    input_para = train + val + test
    input_para = input_para[94:102]
    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    
    with open(f"{RESULT_FILE_PATH}/tmp/kg/relevant_triples.json", "r") as f:
        relevant_triples = json.load(f)
        
    external_information_generation_pipeline(input_para, relevant_triples)
    
    
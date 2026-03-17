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

class TopicChosenResponse(BaseModel):
    Index: str = Field(description="The index of the chosen topic as a string, copied exactly from the provided list.")
    Topic: str = Field(description="The topic name, copied exactly as it appears in the provided list.")
    Description: str = Field(description="The description of the selected topic, copied exactly as it appears in the provided list.")
    Rationale: str = Field(description="Explanation of why this topic best supports the generation of multi-step inference questions based on the passage content, compared with other topics.")    

def topic_chosen_pipeline(input_para, classes, client):
    batch_num = math.ceil(len(input_para) / BATCH_SIZE)
    topics = {}
    
    if GENERATION_MODEL_TYPE == "LOCAL":
        json_schema = TopicChosenResponse.model_json_schema()
        structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
        sampling_params = SamplingParams(temperature=GENERATION_TEMPRATURE, max_tokens=GENERATION_MAX_TOKENS, top_p=GENERATION_TOP_P, structured_outputs=structured_outputs_params_json)
        
    print("#"*30)
    for i in range(0, len(input_para), BATCH_SIZE):
        print(f"Topic chosen: Batch {i//BATCH_SIZE+1}/{batch_num}")
        input_batch_para = input_para[i:i+BATCH_SIZE]
        attempts = 0
        
        while(1):
            if attempts == 0:
                input_list = range(len(input_batch_para))
                topic_chosen_prompt_list = topic_chosen_prompt(input_batch_para, classes, QUESTION_TOPIC)
            else:
                print(f"Some are in regeneration... Attempt {attempts}...")
                input_list = repeat_list
                topic_chosen_prompt_list = topic_chosen_prompt([input_batch_para[a] for a in repeat_list], classes, QUESTION_TOPIC, err)
    
            
            if GENERATION_MODEL_TYPE == "API":
                output_list = [None] * len(topic_chosen_prompt_list)
                with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
                    futures = {pool.submit(call_llm, client, p, ResponseWrapper=TopicChosenResponse): idx for idx, p in enumerate(topic_chosen_prompt_list)}

                    for future in tqdm(as_completed(futures), total=len(topic_chosen_prompt_list)):
                        idx = futures[future]
                        output_list[idx] = future.result()
            else: 
                output_list = client.chat(topic_chosen_prompt_list, sampling_params)
            
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
                    topics[current_idx] = json.loads(llm_response)
                except Exception as err_info:
                    repeat_list.append(input_list[idx])
                    err.append(f"Previous response: \n{{{llm_response}}} \n Previous error: {{{err_info}}}")
                    print(f"\n{attempts}: idx {current_idx} - Failed to choose topics... \n\n {err_info}")
                #print(output.outputs[0].text)
                #print("-"*30)
            
            attempts += 1
            if len(repeat_list) == 0 or attempts == MAX_GENERATE_ATTEMPT:
                break
            
        for idx, fail_idx in enumerate(repeat_list):
            current_idx = input_batch_para[fail_idx]['idx']
            print(f"\nidx {current_idx} - Failed to choose topics... \n\n {err[idx]}")
            print("-"*30)
            topics[current_idx] = {"Index": "-1",
                                            "Topic": "Failed to choose topics.",
                                            "Description": "Failed to choose topics.",
                                            "Reason": "Failed to choose topics."
                                            }   
    
    topics = sort_dict(topics)
    
    with open(f"{RESULT_FILE_PATH}/tmp/topics.json", "w") as f:
        json.dump(topics, f, indent=4)
    
    return topics


if __name__ == "__main__":
    
    train, val, test = get_data()
    print(len(train), len(val), len(test))
    
    input_para = train + val + test
    input_para = input_para[94:102]
    
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
    
    with open(f"{RESULT_FILE_PATH}/tmp/classes.json", "r") as f:
        classes = json.load(f)
    topic_chosen_pipeline(input_para, classes, client)
    
    
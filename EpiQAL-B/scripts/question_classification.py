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

class QuestionClassificationResponse(BaseModel):
    Index: str = Field(description="The index of the chosen class as a string, copied exactly from the provided list.")
    Class: str = Field(description="The class name, copied exactly as it appears in the provided list.")
    Description: str = Field(description="The description of the selected class, copied exactly as it appears in the provided list.")
    Rationale: str = Field(description="Explanation of why this class best supports the generation of multi-step inference questions based on the passage content, compared with other classes.")
    
def question_classification_pipeline(input_para, client):
    input_classes_str = json.dumps(QUESTION_CLASS)
    batch_num = math.ceil(len(input_para) / BATCH_SIZE)
    classes = {}
    
    if GENERATION_MODEL_TYPE == "LOCAL":
        json_schema = QuestionClassificationResponse.model_json_schema()
        structured_outputs_params_json = StructuredOutputsParams(json=json_schema)
        sampling_params = SamplingParams(temperature=GENERATION_TEMPRATURE, max_tokens=GENERATION_MAX_TOKENS, top_p=GENERATION_TOP_P, structured_outputs=structured_outputs_params_json)
    
    print("#"*30)
    for i in range(0, len(input_para), BATCH_SIZE):
        print(f"Question classification: Batch {i//BATCH_SIZE+1}/{batch_num}")
        input_batch_para = input_para[i:i+BATCH_SIZE]
        attempts = 0
        
        while(1):
            if attempts == 0:
                input_list = range(len(input_batch_para))
                question_classification_prompt_list = question_classification_prompt(input_batch_para, input_classes_str)
            else:
                print(f"Some are in regeneration... Attempt {attempts}...")
                #for idx in repeat_list:
                input_list = repeat_list
                question_classification_prompt_list = question_classification_prompt([input_batch_para[a] for a in repeat_list] , input_classes_str, err)
    
    
            #output_list = llm_model.chat(question_classification_prompt_list, sampling_params)
            
            if GENERATION_MODEL_TYPE == "API":
                output_list = [None] * len(question_classification_prompt_list)
                with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
                    futures = {pool.submit(call_llm, client, p, ResponseWrapper=QuestionClassificationResponse): idx for idx, p in enumerate(question_classification_prompt_list)}

                    for future in tqdm(as_completed(futures), total=len(question_classification_prompt_list)):
                        idx = futures[future]
                        output_list[idx] = future.result()
            else:
                output_list = client.chat(question_classification_prompt_list, sampling_params)
        
            
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
                    classes[current_idx] = json.loads(llm_response)
                except Exception as err_info:
                    repeat_list.append(input_list[idx])
                    err.append(f"Previous response: \n{{{llm_response}}} \n Previous error: {{{err_info}}}")
                    print(f"\n{attempts}: idx {current_idx} - Failed to choose categories... \n\n {err_info}")
                #print(output.outputs[0].text)
                #print("-"*30)
            
            attempts += 1
            if len(repeat_list) == 0 or attempts == MAX_GENERATE_ATTEMPT:
                break
            
        for idx, fail_idx in enumerate(repeat_list):
            current_idx = input_batch_para[fail_idx]['idx']
            print(f"\nidx {current_idx} - Failed to choose categories... \n\n {err[idx]}")
            print("-"*30)
            classes[current_idx] = {"Index": "-1",
                                    "Class": "Failed to choose categories.",
                                    "Description": "Failed to choose categories.",
                                    "Reason": "Failed to choose categories."
                                    }

    classes = sort_dict(classes)
    with open(f"{RESULT_FILE_PATH}/tmp/classes.json", "w") as f:
        json.dump(classes, f, indent=4)
    
    return classes


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
    
    question_classification_pipeline(input_para, client)
    
    
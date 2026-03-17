from .func import *
from .constant import *
import os
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm import LLM, SamplingParams
import torch
import requests
from .difficulty_judging import *
from collections import defaultdict
from typing import List, Literal
from pydantic import BaseModel, Field
from vllm.sampling_params import StructuredOutputsParams
import logging
from gliner import GLiNER

class EntityQueryResponse(BaseModel):
    Entity: str = Field(description="The specific epidemiological entity extracted verbatim from the question. This may be a disease, pathogen, intervention, vector, population group, or study design, selected based on priority ranking and centrality to the question's reasoning.")
    Rationale: str = Field(description="The rationale explaining why this entity was selected, including its priority classification, its centrality to the question's reasoning, and whether the priority ranking was overridden.")
        
class QuestionRefinementResponse(BaseModel):
    Replaced_Entity: str = Field(description="The original entity that was replaced, copied exactly from the input.")
    Snippet_Summary: str = Field(description="The concise descriptive phrase synthesized from the snippets that replaces the entity. This should be specific enough to identify the entity for someone with domain knowledge.")
    New_Question: str = Field(description="The complete rewritten question with the original entity replaced by the snippet summary, grammatically correct and naturally flowing.")
    Rationale: str = Field(description="Explanation of what specific details were selected from the snippets, why these details can uniquely identify the entity, and how the descriptive phrase was constructed.")
    
def question_revision_pipeline(initial_input_para, score_per_input, questions, selected_options, ref_answers):
    replaced_entity = defaultdict(list)
    
    '''
    initial_input_para_json = {}
    for item in initial_input_para:
        current_idx = item["idx"]
        initial_input_para_json[current_idx] = item
        
    with open(f"{RESULT_FILE_PATH}/input_para_json.json", "w") as f:
        json.dump(initial_input_para_json, f, indent=4)
    '''
    
    with open(f"{RESULT_FILE_PATH}/input_para_json.json", "r") as f:
        initial_input_para_json = json.load(f)
    
    ################### Question Entity Query Generation ###################
    for times in range(1, MAX_REVISION_TIMES+1):
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

            query_json_schema = EntityQueryResponse.model_json_schema()
            query_structured_outputs_params_json = StructuredOutputsParams(json=query_json_schema)
            query_sampling_params = SamplingParams(temperature=GENERATION_TEMPRATURE, max_tokens=GENERATION_MAX_TOKENS, top_p=GENERATION_TOP_P, structured_outputs=query_structured_outputs_params_json)
            
            refine_json_schema = QuestionRefinementResponse.model_json_schema()
            refine_structured_outputs_params_json = StructuredOutputsParams(json=refine_json_schema)
            refine_sampling_params = SamplingParams(temperature=GENERATION_TEMPRATURE, max_tokens=GENERATION_MAX_TOKENS, top_p=GENERATION_TOP_P, structured_outputs=refine_structured_outputs_params_json)


        input_para = initial_input_para #[]
        '''
        for current_idx in score_per_input.keys():
            f1_score = score_per_input[current_idx]["Average"]["f1"]
            em_score = score_per_input[current_idx]["Average"]["exact_match"]
            
            diff_score = 1 - (ALPHA * f1_score + (1 - ALPHA) * em_score)
            
            if diff_score < DIFFICULTY_THRESHOLD:
                input_para.append(initial_input_para_json[current_idx])
        '''
            
        batch_num = math.ceil(len(input_para) / BATCH_SIZE)
        os.makedirs(f"{RESULT_FILE_PATH}/tmp/revision/revision_{times}", exist_ok=True)
        
        with open(f"{RESULT_FILE_PATH}/tmp/revision/revision_{times}/input_para.json", "w") as f:
            json.dump(input_para, f, indent=4)
        
        print("#"*30)
        question_entity_queries = {}
        
        for i in range(0, len(input_para), BATCH_SIZE):
            print(f"Question Entity Query Generation - {times}: Batch {i//BATCH_SIZE+1}/{batch_num}")
            input_batch_para = input_para[i:i+BATCH_SIZE]
            attempts = 0
            
            while(1):
                if attempts == 0:
                    input_list = range(len(input_batch_para))
                    question_entityy_query_generation_prompt_list = question_entity_query_generation_prompt(input_batch_para, questions)
                else:
                    print(f"Some are in regeneration... Attempt {attempts}...")
                    input_list = repeat_list
                    question_entityy_query_generation_prompt_list = question_entity_query_generation_prompt([input_batch_para[a] for a in repeat_list], questions, err)

                
                if GENERATION_MODEL_TYPE == "API":
                    output_list = [None] * len(question_entityy_query_generation_prompt_list)
                    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
                        futures = {pool.submit(call_llm, client, p, ResponseWrapper=EntityQueryResponse): idx for idx, p in enumerate(question_entityy_query_generation_prompt_list)}

                        for future in tqdm(as_completed(futures), total=len(question_entityy_query_generation_prompt_list)):
                            idx = futures[future]
                            output_list[idx] = future.result()
                else:
                    output_list = client.chat(question_entityy_query_generation_prompt_list, query_sampling_params)
                
                
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
                        question_entity_queries[current_idx] = json.loads(llm_response)
                        question_entity_queries[current_idx]["Query"] = f"definition characteristic {question_entity_queries[current_idx]['Entity']} epidemiology"
                    except Exception as err_info:
                        repeat_list.append(input_list[idx])
                        err.append(f"Previous response: \n{{{llm_response}}} \n Previous error: {{{err_info}}}")
                        print(f"\n{attempts}: idx {current_idx} - Failed to generate question entity query... \n\n {err_info}")
                    #print(output.outputs[0].text)
                    #print("-"*30)
                
                attempts += 1
                if len(repeat_list) == 0 or attempts == MAX_GENERATE_ATTEMPT:
                    break
                
            for idx, fail_idx in enumerate(repeat_list):
                current_idx = input_batch_para[fail_idx]['idx']
                print(f"\nidx {current_idx} - Failed to extract entities... \n\n {err[idx]}")
                print("-"*30)
        
        #exit()
        question_entity_queries = sort_dict(question_entity_queries)
        with open(f"{RESULT_FILE_PATH}/tmp/revision/revision_{times}/question_entity_queries.json", "w") as f:
            json.dump(question_entity_queries, f, indent=4)
        
        
        ################### Core Entity Definition Search ###################
        searched_snippet = defaultdict(dict)
        for i in tqdm(range(0, len(list(question_entity_queries.items())), BATCH_SIZE), desc="Search on Google..."):
            question_entity_queries_batch = list(question_entity_queries.items())[i:i+BATCH_SIZE]
            
            query_list = []
            for current_idx, value in question_entity_queries_batch:
                query = value["Query"]
                query_list.append({"q": query})
                searched_snippet[current_idx]["Entity"] = value["Entity"]
            payload = json.dumps(query_list)
            
            headers = {
            'X-API-KEY': DEFINITION_SEARCH_API_KEY,
            'Content-Type': 'application/json'
            }

            for fail in range(5):
                try:
                    response = requests.request("POST", DEFINITION_SEARCH_URL, headers=headers, data=payload)
                    response_json = response.json()
                    break
                except Exception as e:
                    print(f"{fail}, {e}")
                    pass
                
            if fail == 4:
                print(f"Search timeout.")
                continue
            
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                continue
            
            
            for idx, res in enumerate(response_json):
                current_idx = question_entity_queries_batch[idx][0]
                snippet_list = []

                answer_box_dict = res.get("answerBox", {})
                try:
                    snippet_list.append(answer_box_dict["snippet"])
                except:
                    pass
                    
                for snippet_json in res.get("organic", []):
                    snippet = snippet_json.get("snippet", "")
                    if snippet:
                        snippet_list.append(snippet)
                    if len(snippet_list) >= DEFINITION_SEARCH_MAX_SNIPPET:
                        break
                searched_snippet[current_idx]["Snippet"] = snippet_list
                
        for idx in list(searched_snippet.keys()):
            if "Snippet" not in searched_snippet[idx].keys():
                del searched_snippet[idx]
                
        searched_snippet = sort_dict(searched_snippet)
        with open(f"{RESULT_FILE_PATH}/tmp/revision/revision_{times}/searched_snippet.json", "w") as f:
            json.dump(searched_snippet, f, indent=4)
            
        
        ################### Question Reconstruction ###################
        reconstruction_result = {}
        for i in range(0, len(list(searched_snippet.items())), BATCH_SIZE):
            batch_num = math.ceil(len(list(searched_snippet.items())) / BATCH_SIZE)
            print(f"Question Reconstruction - {times}: Batch {i//BATCH_SIZE+1}/{batch_num}")
            searched_snippet_batch = list(searched_snippet.items())[i:i+BATCH_SIZE]
            attempts = 0
            
            while(1):
                if attempts == 0:
                    input_list = range(len(searched_snippet_batch))
                    question_reconstruction_prompt_list = question_reconstruction_prompt(searched_snippet_batch, questions, replaced_entity)
                else:
                    print(f"Some are in regeneration... Attempt {attempts}...")
                    input_list = repeat_list
                    question_reconstruction_prompt_list = question_reconstruction_prompt([searched_snippet_batch[a] for a in repeat_list], questions, replaced_entity, err)
        
                if GENERATION_MODEL_TYPE == "API":
                    output_list = [None] * len(question_reconstruction_prompt_list)
                    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
                        futures = {pool.submit(call_llm, client, p, ResponseWrapper=QuestionRefinementResponse): idx for idx, p in enumerate(question_reconstruction_prompt_list)}

                        for future in tqdm(as_completed(futures), total=len(question_reconstruction_prompt_list)):
                            idx = futures[future]
                            output_list[idx] = future.result()
                else:
                    output_list = client.chat(question_reconstruction_prompt_list, refine_sampling_params)
                
                
                repeat_list = []
                err = []
                for idx in range(len(input_list)):
                    llm_response = output_list[idx]
                    current_idx = searched_snippet_batch[input_list[idx]][0]
                    if GENERATION_MODEL_TYPE == "API":
                        llm_response = llm_response.choices[0].message.content.strip()
                    else:
                        llm_response = llm_response.outputs[0].text.strip().split('</think>')[-1]
                    try:
                        reconstruction_result[current_idx] = json.loads(llm_response)
                        questions[current_idx]["Question"] = reconstruction_result[current_idx]["New_Question"]
                        replaced_entity[current_idx].append(searched_snippet_batch[input_list[idx]][1]["Entity"])
                    except Exception as err_info:
                        repeat_list.append(input_list[idx])
                        err.append(f"Previous response: \n{{{llm_response}}} \n Previous error: {{{err_info}}}")
                        print(f"\n{attempts}: idx {current_idx} - Failed to generate question entity query... \n\n {err_info}")
                    #print(output.outputs[0].text)
                    #print("-"*30)
                
                attempts += 1
                if len(repeat_list) == 0 or attempts == MAX_GENERATE_ATTEMPT:
                    break
                
            for idx, fail_idx in enumerate(repeat_list):
                current_idx = searched_snippet_batch[fail_idx][0]
                print(f"\nidx {current_idx} - Failed to extract entities... \n\n {err[idx]}")
                print("-"*30)
                
        questions = sort_dict(questions)
        with open(f"{RESULT_FILE_PATH}/tmp/revision/revision_{times}/questions.json", "w") as f:
            json.dump(questions, f, indent=4)
        
        reconstruction_result = sort_dict(reconstruction_result)
        with open(f"{RESULT_FILE_PATH}/tmp/revision/revision_{times}/reconstruction_result.json", "w") as f:
            json.dump(reconstruction_result, f, indent=4)
        
        replaced_entity = sort_dict(replaced_entity)
        with open(f"{RESULT_FILE_PATH}/tmp/revision/revision_{times}/replaced_entity.json", "w") as f:
            json.dump(replaced_entity, f, indent=4)
        
        
        del client
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        
        score_per_input = difficulty_judging_pipeline(input_para, questions, selected_options, ref_answers, times, score_per_input)
        
        score_per_input = sort_dict(score_per_input)
        with open(f"{RESULT_FILE_PATH}/tmp/revision/revision_{times}/scores.json", "w") as f:
            json.dump(score_per_input, f, indent=4)
        
    with open(f"{RESULT_FILE_PATH}/tmp/revision/replaced_entity.json", "w") as f:
        json.dump(replaced_entity, f, indent=4)
            
    with open(f"{RESULT_FILE_PATH}/tmp/questions_news.json", "w") as f:
        json.dump(questions, f, indent=4)
        
    return questions


if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.ERROR)
    train, val, test = get_data()
    print(len(train), len(val), len(test))
    
    input_para = train + val + test
    input_para = input_para[90:106]
    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    
    with open(f"{RESULT_FILE_PATH}/tmp/questions.json", "r") as f:
        questions = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/selected_options.json", "r") as f:
        selected_options = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/ref_answers.json", "r") as f:
        ref_answers = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/diff_judge/diff_judge_0/scores.json", "r") as f:
        score_per_input = json.load(f)
        
    question_revision_pipeline(input_para, score_per_input, questions, selected_options, ref_answers)
    
    
from scripts import *
import logging
import time



def main():
    #logging.getLogger().setLevel(logging.ERROR)
    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    input_para_path = f"{RESULT_FILE_PATH}/input_para.json"
    if os.path.exists(input_para_path):
        with open(input_para_path, "r") as f:
            input_para = json.load(f)
    else:
        input_para, _, _ = get_data()
        with open(input_para_path, "w") as f:
            json.dump(input_para, f, indent=4)
    print(len(input_para))
    
    input_para = input_para[:500]
    
    input_para_json = {}
    for item in input_para:
        current_idx = item["idx"]
        input_para_json[current_idx] = item
        
    with open(f"{RESULT_FILE_PATH}/input_para_json.json", "w") as f:
        json.dump(input_para_json, f, indent=4)
        
    
    start_time = time.perf_counter()
    '''
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
    
    cg_start_time = time.perf_counter()
    correct_options = correct_option_generation_pipeline(input_para, client)
    cg_end_time = time.perf_counter()
    cg_time = cg_end_time - cg_start_time
    
    qg_start_time = time.perf_counter()
    questions = question_generation_pipeline(input_para, correct_options, client)
    qg_end_time = time.perf_counter()
    qg_time = qg_end_time - qg_start_time
    
    dg_start_time = time.perf_counter()
    distractors = distractor_generation_pipeline(input_para, questions, correct_options, client)
    dg_end_time = time.perf_counter()
    dg_time = dg_end_time - dg_start_time
    
    del client
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    
    with open(f"{RESULT_FILE_PATH}/tmp/questions.json", "r") as f:
        questions = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/option/correct_options.json", "r") as f:
        correct_options = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/option/distractors.json", "r") as f:
        distractors = json.load(f)
        
    cc_start_time = time.perf_counter()
    correct_option_checking = correct_option_checking_pipeline(input_para, questions, correct_options)
    cc_end_time = time.perf_counter()
    cc_time = cc_end_time - cc_start_time
    
    dc_start_time = time.perf_counter()
    distractor_checking = distractor_checking_pipeline(input_para, questions, distractors)
    dc_end_time = time.perf_counter()
    dc_time = dc_end_time - dc_start_time
    
    with open(f"{RESULT_FILE_PATH}/checking_time.json", "w") as f:
        json.dump({
                   "Correct Options Check": cc_time,
                   "Distrators Check": dc_time,}, f, indent=4)
    
    '''
    with open(f"{RESULT_FILE_PATH}/tmp/questions.json", "r") as f:
        questions = json.load(f)
        
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/correct_option_checking.json", "r") as f:
        correct_option_checking = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/distractor_checking.json", "r") as f:
        distractor_checking = json.load(f)
    
    with open(f"{RESULT_FILE_PATH}/tmp/coherence/review_options_finished.json", "r") as f:
        review_options = json.load(f)
        
    os_start_time = time.perf_counter()
    ref_answers, selected_options = option_selection_pipeline(input_para, correct_option_checking, distractor_checking, review_options)
    os_end_time = time.perf_counter()
    os_time = os_end_time - os_start_time
    
    input_para = []
    for key in selected_options.keys():
        input_para.append(input_para_json[key])
    
    dj_start_time = time.perf_counter()
    score_per_input = difficulty_judging_pipeline(input_para, questions, selected_options, ref_answers)
    dj_end_time = time.perf_counter()
    dj_time = dj_end_time - dj_start_time
    
    qr_start_time = time.perf_counter()
    questions_news = question_revision_pipeline(input_para, score_per_input, questions, selected_options, ref_answers)
    qr_end_time = time.perf_counter()
    qr_time = qr_end_time - qr_start_time
    
    end_time = time.perf_counter()
    
    
    with open(f"{RESULT_FILE_PATH}/tmp/questions.json", "r") as f:
        questions = json.load(f)
        
    results = []
    qa_results = []
    for i in tqdm(range(len(input_para)), desc="Save the results"):
        results.append({"idx": input_para[i]["idx"],
                        "paragraph": input_para[i]["inputs"],
                        "ori_question": questions[input_para[i]["idx"]],
                        "question": questions_news[input_para[i]["idx"]],
                        "choices": selected_options[input_para[i]["idx"]],
                        "ref_answers": ref_answers[input_para[i]["idx"]],})
                        #"score_per_input": score_per_input[input_para[i]["idx"]]})
                        
        qa_results.append({"idx": input_para[i]["idx"],
                        "paragraph": input_para[i]["inputs"],
                        "question": questions_news[input_para[i]["idx"]]["Question"],
                        "choices": selected_options[input_para[i]["idx"]],
                        "ref_answers": ref_answers[input_para[i]["idx"]],})
        
    with open(f"{RESULT_FILE_PATH}/final_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    with open(f"{RESULT_FILE_PATH}/final_qa.json", "w") as f:
        json.dump(qa_results, f, indent=4)
    
    all_time = end_time - start_time
    '''
    with open(f"{RESULT_FILE_PATH}/efficiency_time.json", "w") as f:
        json.dump({"Time": all_time,
                   "Question Generation": qg_time,
                   "Correct Options Gen": cg_time,
                   "Distractors Gen": dg_time,
                   "Correct Options Check": cc_time,
                   "Distrators Check": dc_time,
                   "Option Selection": os_time,
                   "Difficulty Judging": dj_time,
                   "Question Revision": qr_time}, f, indent=4)
    '''
    return 0


if __name__ == "__main__":
    main()
    
from scripts import *
import logging
import time



def main():
    #logging.getLogger().setLevel(logging.ERROR)
    input_para, _, _ = get_data()
    print(len(input_para))

    input_para = input_para[:500]
    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    with open(f"{RESULT_FILE_PATH}/input_para.json", "w") as f:
        json.dump(input_para, f, indent=4)

    start_time = time.perf_counter()

    kg_start_time = time.perf_counter()
    entities_extracted = disease_extraction_ner_pipeline(input_para)
    relevant_triples = entity_alignment_pipeline(input_para, entities_extracted)
    external_info = external_information_generation_pipeline(input_para, relevant_triples)
    kg_end_time = time.perf_counter()
    kg_time = kg_end_time - kg_start_time

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
    
    qc_start_time = time.perf_counter()
    classes = question_classification_pipeline(input_para, client)
    qc_end_time = time.perf_counter()
    qc_time = qc_end_time - qc_start_time
    
    tc_start_time = time.perf_counter()
    topics = topic_chosen_pipeline(input_para, classes, client)
    tc_end_time = time.perf_counter()
    tc_time = tc_end_time - tc_start_time
    
    qg_start_time = time.perf_counter()
    questions = question_generation_pipeline(input_para, topics, external_info, client)
    qg_end_time = time.perf_counter()
    qg_time = qg_end_time - qg_start_time
    
    cg_start_time = time.perf_counter()
    correct_options = correct_option_generation_pipeline(input_para, questions, external_info, client)
    cg_end_time = time.perf_counter()
    cg_time = cg_end_time - cg_start_time
    
    dg_start_time = time.perf_counter()
    distractors = distractor_generation_pipeline(input_para, questions, external_info, correct_options, client)
    dg_end_time = time.perf_counter()
    dg_time = dg_end_time - dg_start_time
    
    del client
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    cc_start_time = time.perf_counter()
    correct_option_checking = correct_option_checking_pipeline(input_para, questions, external_info, correct_options)
    cc_end_time = time.perf_counter()
    cc_time = cc_end_time - cc_start_time
    
    dc_start_time = time.perf_counter()
    distractor_checking = distractor_checking_pipeline(input_para, questions, external_info, distractors)
    dc_end_time = time.perf_counter()
    dc_time = dc_end_time - dc_start_time
    
    os_start_time = time.perf_counter()
    ref_answers, selected_options = option_selection_pipeline(input_para, correct_option_checking, distractor_checking)
    os_end_time = time.perf_counter()
    os_time = os_end_time - os_start_time
        
    dj_start_time = time.perf_counter()
    score_per_input = difficulty_judging_pipeline(input_para, questions, selected_options, ref_answers)
    dj_end_time = time.perf_counter()
    dj_time = dj_end_time - dj_start_time
    
    qr_start_time = time.perf_counter()
    questions_news = question_revision_pipeline(input_para, score_per_input, questions, selected_options, ref_answers)
    qr_end_time = time.perf_counter()
    qr_time = qr_end_time - qr_start_time
    
    end_time = time.perf_counter()
    
    
    results = []
    qa_results = []
    for i in tqdm(range(len(input_para)), desc="Save the results"):
        results.append({"idx": input_para[i]["idx"],
                        "paragraph": input_para[i]["inputs"],
                        "category": classes[input_para[i]["idx"]],
                        "topic": topics[input_para[i]["idx"]],
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

    with open(f"{RESULT_FILE_PATH}/efficiency_time.json", "w") as f:
        json.dump({"Time": all_time,
                   "KG": kg_time,
                   "Classification": qc_time,
                   "Topic Chosen": tc_time,
                   "Question Generation": qg_time,
                   "Correct Options Gen": cg_time,
                   "Distractors Gen": dg_time,
                   "Correct Options Check": cc_time,
                   "Distrators Check": dc_time,
                   "Option Selection": os_time,
                   "Difficulty Judging": dj_time,
                   "Question Revision": qr_time}, f, indent=4)
    
    with open(f"{RESULT_FILE_PATH}/efficiency_time.json", "w") as f:
        json.dump({"Time": all_time,
                   "Difficulty Judging": dj_time,
                   "Question Revision": qr_time}, f, indent=4)
    
    return 0


if __name__ == "__main__":
    main()
    
from .constant import *
import random
import os
from tqdm import tqdm
import json
from natsort import natsorted
import time
import logging

def sort_dict(d):
    sorted_keys = natsorted(d.keys())
    sorted_d = {k: d[k] for k in sorted_keys}
    return sorted_d


def get_data(train_ratio=1, val_ratio=0, sub_ratio=0.1):
    journal_dirs = []
    for dirpath, dirnames, filenames in os.walk(DATA_PATH):
        for dirname in dirnames:
            if dirname.startswith("journal"):
                journal_dirs.append(os.path.join(dirpath, dirname))

    random.seed(42)
    journal_dirs = random.sample(journal_dirs, len(journal_dirs))

    id = 0
    inputs = []
    for path in tqdm(journal_dirs):
        try:
            target_path = os.path.join(path, 'author_summary.txt')
            with open (target_path, 'r') as target_file:
                target = target_file.readlines()
            
            #target = " ".join([line.strip() for line in target])
            
            
            
            abstract_path = os.path.join(path, 'abstract.json')
            with open (abstract_path, 'r') as abstract_file:
                abstract = json.load(abstract_file)
            abstract_string = ""
            for key in abstract.keys():
                section = abstract[key].strip().replace("\n", "")
                abstract_string = abstract_string + section #key + ": " + section
            
            target = abstract_string
            #print(target)
            
            
            content_path = os.path.join(path, 'content.json')
            with open (content_path, 'r') as content_file:
                content = json.load(content_file)
            
            content_string = ""
            for key in content.keys():
                section = content[key].strip()#.replace("\n", "")
                content_string = content_string + section + "\n" #"---" + key + "---\n" + section
            
            #print(content_string)
            
            
            #print(content_string)
            
            inputs.append({"idx": str(id), "inputs": content_string, "target": target})
            id += 1
            if id > len(journal_dirs)*sub_ratio:
                break
            #break
        except Exception as e:
            print("Error files: ", path)
            print("Error message: ", e)
            #break

    total = len(inputs)

    train_input = inputs[0:int(total*train_ratio)]
    val_input = inputs[int(total*train_ratio):int(total*train_ratio)+int(total*val_ratio)]
    test_input = inputs[int(total*train_ratio)+int(total*val_ratio):]
    
    return train_input, val_input, test_input





def call_llm(client, p, model_name=GENERATION_MODEL_NAME, temp=GENERATION_TEMPRATURE, max_tokens=GENERATION_MAX_TOKENS, top_p=GENERATION_TOP_P, ResponseWrapper=None):
    #time.sleep(random.uniform(0.05, 0.2))
    for i in range(5):
        time.sleep(random.uniform(0.05, 0.2))
        httpx_logger = logging.getLogger("httpx")
        httpx_logger.setLevel(logging.WARNING)
        try:        
            if ResponseWrapper:
                output = client.chat.completions.parse(
                    model=model_name,
                    messages=p,
                    temperature=temp,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    response_format=ResponseWrapper
                    #timeout=60
                )
            else:
                output = client.chat.completions.create(
                    model=model_name,
                    messages=p,
                    temperature=temp,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    #response_format=ResponseWrapper
                    #timeout=60
                )
            #print(output)
            return output
        except Exception as err:
            err_info = err
            pass
    
    return err_info



def question_classification_prompt(input_para_list, question_class, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""
        
        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to classify an epidemiological passage into the most appropriate class for guiding question generation. The questions to be generated are text-grounded factual recall questions where answers can be directly found or located within the passage, rather than questions requiring external knowledge or multi-step inference.
                            You will be given a list of classes in this format:
                                {{
                                    {question_class}
                                }}

                            Follow these steps to complete the task.
                            Step 1. Read the passage carefully. Identify its main epidemiological focus, including what phenomena it describes, what methods it uses, and what findings it reports.
                            Step 2. Read through all provided classes and their descriptions. Understand what each class is intended to capture.
                            Step 3. Compare the passage content against each class. Consider which class would best support the generation of factual questions whose answers are explicitly stated in the passage.
                            Step 4. Select the single most appropriate class. If the passage touches on multiple classes, choose the one that reflects its primary focus and contains the richest factual content for question generation.
                            Step 5. Write your output in the following format. Copy the class index, class name, description exactly as they appear in the provided list, and rationale why you choose it.
                            
                            {self_check}
                        """
        exp_prompt = f"""
                        Passage: 
                            {{
                                During the winter, Atlanta reported increased COVID 19 transmission, with seventy two new cases identified over two weeks. 
                                The source of these cases remains undetermined. Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission. 
                                The two locations have frequent travel between them, and preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days.
                            }}
                    """
        exp_output = f""" 
                        {{
                            "Index": "1",
                            "Class": "Surveillance & Descriptive Epidemiology",
                            "Description": "Describes population occurrence from routine data (rates, time-place-person, aberration signals) and basic system performance, without causal analysis or forecasting.",
                            "Rationale": "I selected this class because the passage focuses entirely on reporting observed data trends ('increased COVID 19 transmission,' 'seventy two new cases,' 'outbreak of fifty cases') within specific timeframes ('winter,' 'two weeks') and locations ('Atlanta,' 'Florida'). It describes the Time-Place-Person patterns—specifically the spatial link between the two cities via travel history—without diving into complex causal modeling, specific control measures, or future forecasting. This aligns perfectly with the scope of descriptive epidemiology."
                        }}                    
                    """
        usr_prompt = f"""
                    Passage:  
                        {{ 
                            {input_para["inputs"]} 
                        }}
                    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exp_prompt},
            {"role": "assistant", "content": exp_output},
            {"role": "user", "content": usr_prompt},
        ]
        prompt_list.append(messages)
    return prompt_list




def topic_chosen_prompt(input_para_list, classes, topic_dict, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_class = classes[current_idx]["Index"]
        try:
            question_topic = json.dumps(topic_dict[current_class])
        except:
            prompt_list.append(["ERROR"])
            continue
            
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to classify an epidemiological passage into the most appropriate topic for guiding question generation. The questions to be generated are text-grounded factual recall questions where answers can be directly found or located within the passage, rather than questions requiring external knowledge or multi-step inference.
                            You will be given a list of topics in this format:
                                {{
                                    {question_topic}
                                }}

                            Follow these steps to complete the task.
                            Step 1. Read the passage carefully. Identify its main epidemiological focus, including what phenomena it describes, what methods it uses, and what findings it reports.
                            Step 2. Read through all provided topics and their descriptions. Understand what each topic is intended to capture.
                            Step 3. Compare the passage content against each topic. Consider which topic would best support the generation of factual questions whose answers are explicitly stated in the passage.
                            Step 4. Select the single most appropriate topic. If the passage touches on multiple topics, choose the one that reflects its primary focus and contains the richest factual content for question generation.
                            Step 5. Write your output in the following format. Copy the topic index, topic name, description exactly as they appear in the provided list, and rationale why you choose it.

                            {self_check}
                        """
        exp_prompt = f"""
                        Passage: 
                            {{
                                During the winter, Atlanta reported increased COVID 19 transmission, with seventy two new cases identified over two weeks. 
                                The source of these cases remains undetermined. Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission. 
                                The two locations have frequent travel between them, and preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days.
                            }}
                    """
        exp_output = f"""
                        {{
                            "Index": "2",
                            "Topic": "Time-Place-Person patterns, seasonality & clustering",
                            "Description": "Describes temporal trends, spatial distribution, and demographic profiles using routine population surveillance.",
                            "Rationale": "I selected this topic because the passage focuses primarily on describing the descriptive characteristics of the outbreak in terms of Time ('During the winter,' 'over two weeks'), Place ('Atlanta,' 'Florida'), and Person/Activity (travel history between the two locations). It highlights the spatial link and seasonal context rather than focusing on statistical detection methods or calculating standardized rates."
                        }}              
                    """
        usr_prompt = f"""
                    Passage:  
                        {{ 
                            {input_para["inputs"]} 
                        }}
                    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exp_prompt},
            {"role": "assistant", "content": exp_output},
            {"role": "user", "content": usr_prompt},
        ]
        prompt_list.append(messages)
    return prompt_list



def call_ner(model, text):
    chunks = [text[i:i+384] for i in range(0, len(text), 384)]
    all_entities = []
    for chunk in chunks:
        entities = model.predict_entities(chunk, ["disease"], ner_threshold=0.98)
        for entity in entities:
            all_entities.append(entity["text"])
    return list(set(all_entities))


def question_generation_prompt(input_para_list, topics, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_topic = json.dumps({k: topics[current_idx].get(k) for k in ("Topic", "Description")})
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to generate a retrieval-based question using the provided passage. The question should be answerable by directly locating information in the passage, without requiring inference or external knowledge.

                            Follow these steps exactly:
                            Step 1: Read the passage carefully and identify factual content that can support a retrieval-based question. Good candidates include specific numbers, dates, definitions, names, or clearly stated facts.
                            Step 2: Use the topic to constrain the scope of the question, but do not explicitly ask about the topic name itself. The question should focus on specific content within that topic area.
                            Step 3: Write one question that requires readers to locate and retrieve specific information from the passage. The question should have a clear, unambiguous answer that appears explicitly in the passage.
                            Step 4: Apply quality requirements. A good retrieval question should target specific factual content rather than vague or general information, have an answer that is explicitly stated in the passage in a locatable form, and not be answerable by general knowledge alone without reading the passage.
                            Step 5: Apply question stem constraints. The question stem should not copy phrases directly from the passage that would make the answer obvious, should not be so broad that multiple unrelated answers could apply, and should be grammatically complete and clear.
                            Step 6: Identify the evidence from the passage that contains the answer. There may be one or multiple pieces of evidence if the question can be answered from different parts of the passage.
                            Step 7: Do not include answers or options in your output. Do not imply how many correct answers there are.
                            Step 8: Output your result with three fields. Question is the question stem. Evidence is a list of verbatim quotes from the passage that contain the answer. Rationale explains why this is a good retrieval-based question and what factual content it targets.
                                                        
                            {self_check}
                        """
        exp_prompt = f"""
                        Passage:
                            {{
                                During the winter, Atlanta reported increased COVID 19 transmission, with seventy two new cases identified over two weeks. 
                                The source of these cases remains undetermined. Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission. 
                                The two locations have frequent travel between them, and preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days.
                            }}
                        Topic:
                            {{
                                "Topic": "Time-Place-Person patterns, seasonality & clustering",
                                "Description": "Describes temporal trends, spatial distribution, and demographic profiles using routine population surveillance."
                            }}
                    """
        exp_output = """
                        {
                            "Question": "What was the reporting period during which Atlanta identified its new COVID-19 cases?",
                            "Evidence": [
                                    "seventy two new cases identified over two weeks"
                                ],
                            "Rationale": "This is a good retrieval-based question because it targets a specific temporal fact (the two-week surveillance period) that is explicitly stated in the passage. The question aligns with the topic of time-place patterns by focusing on the temporal dimension of case surveillance without copying key phrases from the passage verbatim. The answer ('two weeks') is clearly locatable in the text and cannot be inferred from general epidemiological knowledge alone—readers must consult the passage to find this specific reporting timeframe for Atlanta's outbreak."
                        }
                    """
        usr_prompt = f"""
                    Passage: 
                        {{
                            {input_para["inputs"]}
                        }}
                    Topic:
                        {{
                            {current_topic}
                        }}
                    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exp_prompt},
            {"role": "assistant", "content": exp_output},
            {"role": "user", "content": usr_prompt},
        ]
        prompt_list.append(messages)
    return prompt_list


def correct_option_generation_prompt(input_para_list, questions, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_question = json.dumps(questions[current_idx]) #["Question"]
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to generate correct options for a retrieval-based question. The correct options should be answers that can be directly found in the passage.
                            You will be given the passage, the question, and the evidence from question generation.

                            Follow these steps exactly:
                            Step 1: Read the passage, the question, and the provided evidence carefully.
                            Step 2: Identify the specific information in the passage that directly answers the question. Use the provided evidence as guidance.
                            Step 3: Generate one or more correct options. Each option must be directly supported by explicit text in the passage. Do not infer or add information not present in the passage.
                            Step 4: Apply option constraints. Each option should use concise wording that captures the answer without copying the entire evidence sentence. Each option should be semantically complete, though it does not need to be a full sentence. Each option must not contradict any information in the passage.
                            Step 5: If generating multiple options, ensure each represents a distinct correct answer from different parts of the passage. Options should not overlap or be redundant.
                            Step 6: For each option, quote the exact evidence from the passage that supports it.
                            Step 7: Output your result as a JSON object with a single key "result" whose value is a list of option dictionaries. Each dictionary should have four fields: Category is always Correct Option, Option is the answer text, Evidence is a list of verbatim quotes from the passage that support this option, and Rationale explains why this option correctly answers the question. Output only the JSON object with no additional text.
                                                        
                            {self_check}
                    """
        exp_prompt = f"""
                        Passage:
                            {{
                                During the winter, Atlanta reported increased COVID 19 transmission, with seventy two new cases identified over two weeks. 
                                The source of these cases remains undetermined. Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission. 
                                The two locations have frequent travel between them, and preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days.
                            }}
                        Question:
                            {{
                                "Question": "What was the reporting period during which Atlanta identified its new COVID-19 cases?",
                                "Evidence": [
                                        "seventy two new cases identified over two weeks"
                                    ],
                                "Rationale": "This is a good retrieval-based question because it targets a specific temporal fact (the two-week surveillance period) that is explicitly stated in the passage. The question aligns with the topic of time-place patterns by focusing on the temporal dimension of case surveillance without copying key phrases from the passage verbatim. The answer ('two weeks') is clearly locatable in the text and cannot be inferred from general epidemiological knowledge alone—readers must consult the passage to find this specific reporting timeframe for Atlanta's outbreak."
                            }}
                    """
        exp_output = """
                        {
                            "result": [
                                {
                                    "Category": "Correct Option",
                                    "Option": "Two weeks",
                                    "Evidence": [
                                            "seventy two new cases identified over two weeks"
                                        ],
                                    "Rationale": "The passage explicitly states that Atlanta's seventy-two new COVID-19 cases were 'identified over two weeks,' directly answering the question about the reporting period. This is the only temporal duration mentioned for Atlanta's case identification."
                                }
                            ]
                        }
                    """
        usr_prompt = f"""
                    Passage: 
                        {{
                            {input_para["inputs"]}
                        }}
                    Question: 
                        {current_question}
                    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exp_prompt},
            {"role": "assistant", "content": exp_output},
            {"role": "user", "content": usr_prompt},
        ]
        prompt_list.append(messages)
    return prompt_list


def distractor_generation_prompt(input_para_list, questions, correct_options, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_question = json.dumps(questions[current_idx]) #["Question"]
        #current_correct_options = "\n".join([item["Option"] for item in correct_options[current_idx]])
        current_correct_options = json.dumps([{k: item.get(k) for k in ("Category", "Option")} for item in correct_options[current_idx]])
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to generate distractors for a retrieval-based question. Distractors should be plausible-sounding answers that appear in the passage but do not correctly answer the specific question asked. They test whether readers can precisely locate the correct information rather than guessing based on keyword matching.
                            You will be given the passage, the question, and the correct options.

                            Follow these steps exactly:
                            Step 1: Read the passage, the question, and the correct options carefully. Understand what specific information the question asks for.
                            Step 2: Identify content in the passage that could be confused with the correct answer. Good distractors share these characteristics:
                                - They belong to the same semantic category as the correct option such as both being locations, numbers, time periods, or names
                                - They appear in the passage and are factually accurate within the passage context
                                - They relate to a different entity, time, place, or context than what the question specifically asks about
                            Step 3: Generate distractors using only information from the passage. Each distractor must be a valid fact stated in the passage but incorrect as an answer to this specific question.
                            Step 4: Ensure each distractor is plausible. A reader who skims the passage or relies on keyword matching might mistakenly select it, but careful reading reveals it answers a different aspect than what the question asks.
                            Step 5: Match the style of the correct options. Distractors should have similar length, grammatical structure, and level of specificity as the correct options.
                            Step 6: Each distractor should be semantically complete, though it does not need to be a full sentence.
                            Step 7: If generating multiple distractors, ensure each comes from a distinct part of the passage or addresses a different potential confusion. Distractors should not overlap or be too similar to each other.
                            Step 8: For each distractor, quote the exact evidence from the passage that contains this information.
                            Step 9: Output your result as a JSON object with a single key "result" whose value is a list of distractor dictionaries. Each dictionary should have four fields: Category is always Distractor, Option is the distractor text, Evidence is a list of verbatim quotes from the passage, and Rationale explains why this is a plausible but incorrect answer. Output only the JSON object with no additional text.
                            
                            {self_check}
                    """
        exp_prompt = f"""
                        Passage:
                            {{
                                During the winter, Atlanta reported increased COVID 19 transmission, with seventy two new cases identified over two weeks. 
                                The source of these cases remains undetermined. Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission. 
                                The two locations have frequent travel between them, and preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days.
                            }}
                        Question:
                            {{
                                "Question": "What was the reporting period during which Atlanta identified its new COVID-19 cases?",
                                "Evidence": [
                                        "seventy two new cases identified over two weeks"
                                    ],
                                "Rationale": "This is a good retrieval-based question because it targets a specific temporal fact (the two-week surveillance period) that is explicitly stated in the passage. The question aligns with the topic of time-place patterns by focusing on the temporal dimension of case surveillance without copying key phrases from the passage verbatim. The answer ('two weeks') is clearly locatable in the text and cannot be inferred from general epidemiological knowledge alone—readers must consult the passage to find this specific reporting timeframe for Atlanta's outbreak."
                            }}
                        Correct Options:
                            {{
                                [ 
                                    {{ 
                                        "Category": "Correct Option", 
                                        "Option": "Two weeks" 
                                    }} 
                                ]
                            }}
                    """
        exp_output = """
                        {
                            "result": [
                                {
                                    "Category": "Distractor",
                                    "Option": "Fourteen days",
                                    "Evidence": [
                                        "preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days"
                                    ],
                                    "Rationale": "This is a plausible distractor because 'fourteen days' is a time period explicitly mentioned in the passage and belongs to the same semantic category as the correct answer. However, it refers to the travel exposure window (how recently Atlanta cases had visited Florida) rather than the reporting period during which Atlanta identified its cases. A reader relying on keyword matching might confuse this temporal reference with the reporting timeframe."
                                },
                                {
                                    "Category": "Distractor",
                                    "Option": "Earlier this month",
                                    "Evidence": [
                                        "Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission"
                                    ],
                                    "Rationale": "This is a plausible distractor because it is a temporal reference found in the passage. However, it describes when Florida reported its outbreak, not the duration of Atlanta's reporting period. A reader who skims the passage might mistakenly associate this time reference with Atlanta's case identification."
                                }
                            ]
                        }
                        """
        usr_prompt = f"""
                    Passage: 
                        {{
                            {input_para["inputs"]}
                        }}
                    Question: 
                        {{
                            {current_question}
                        }}
                    Correct Options:
                        {{
                            {current_correct_options}
                        }}
                    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exp_prompt},
            {"role": "assistant", "content": exp_output},
            {"role": "user", "content": usr_prompt},
        ]
        prompt_list.append(messages)
    return prompt_list
    


def answer_checking_prompt(input_para_list, questions, options, answer_type=None, err=None):
    example_options = {"Correct Option": """
                       [
                            {
                                "Category": "Correct Option",
                                "Option": "Two weeks",
                                "Evidence": [
                                        "seventy two new cases identified over two weeks"
                                    ],
                                "Rationale": "The passage explicitly states that Atlanta's seventy-two new COVID-19 cases were 'identified over two weeks,' directly answering the question about the reporting period. This is the only temporal duration mentioned for Atlanta's case identification."
                            }
                        ]""",
                        "Distractor": """
                        [
                            {
                                "Category": "Distractor",
                                "Option": "Fourteen days",
                                "Evidence": [
                                    "preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days"
                                ],
                                "Rationale": "This is a plausible distractor because 'fourteen days' is a time period explicitly mentioned in the passage and belongs to the same semantic category as the correct answer. However, it refers to the travel exposure window (how recently Atlanta cases had visited Florida) rather than the reporting period during which Atlanta identified its cases. A reader relying on keyword matching might confuse this temporal reference with the reporting timeframe."
                            },
                            {
                                "Category": "Distractor",
                                "Option": "Earlier this month",
                                "Evidence": [
                                    "Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission"
                                ],
                                "Rationale": "This is a plausible distractor because it is a temporal reference found in the passage. However, it describes when Florida reported its outbreak, not the duration of Atlanta's reporting period. A reader who skims the passage might mistakenly associate this time reference with Atlanta's case identification."
                            }
                        ]"""}

    example_outputs = {"Correct Option": """
                            {    
                                "result": [
                                    {
                                        "Category": "Correct Option",
                                        "Option": "Two weeks",
                                        "Coherence": "Yes",
                                        "Rationale": "All checks pass. Step 2: The evidence 'seventy two new cases identified over two weeks' is an exact verbatim quote from the passage. Step 3: The option 'Two weeks' is explicitly stated in the passage. Step 4: The option directly addresses the question about the reporting period for Atlanta's COVID-19 cases. Step 5: The option correctly answers the question, as the passage clearly states the cases were identified over two weeks. Step 6: The rationale is logically sound, correctly explaining that the passage explicitly provides this temporal duration for Atlanta's case identification."
                                    }
                                ]
                            }
                       """,
                       "Distractor": """
                       {
                            "result": [
                                {
                                    "Category": "Distractor",
                                    "Option": "Fourteen days",
                                    "Coherence": "Yes",
                                    "Rationale": "All checks pass. Step 2: The evidence 'preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days' is an exact verbatim quote from the passage. Step 3: 'Fourteen days' is explicitly mentioned in the passage. Step 4: The option addresses the question as it provides a time period relevant to a question about reporting duration. Step 5: The option is incorrect because 'fourteen days' refers to the travel exposure window for Atlanta cases visiting Florida, not the reporting period during which cases were identified (which was 'two weeks'). Step 6: The rationale correctly explains why this is a plausible but incorrect answer."
                                },
                                {
                                    "Category": "Distractor",
                                    "Option": "Earlier this month",
                                    "Coherence": "Yes",
                                    "Rationale": "All checks pass. Step 2: The evidence 'Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission' is an exact verbatim quote from the passage. Step 3: 'Earlier this month' is explicitly stated in the passage. Step 4: The option addresses the question as it provides a temporal reference relevant to a question about timing. Step 5: The option is incorrect because 'Earlier this month' refers to when Florida reported its outbreak, not the reporting period for Atlanta's COVID-19 cases. Step 6: The rationale correctly explains why this temporal reference could be mistakenly associated with Atlanta's case identification."
                                }
                            ]
                        }
                       """}


    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_question = questions[current_idx]["Question"]
        current_options = json.dumps(options[current_idx])
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to verify whether the provided options are valid as {answer_type} based on the given passage, question, evidence, and rationale. Evaluate based on the option's role relative to the question within this passage, not on external or real-world factual accuracy.

                            Follow these steps exactly:
                            Step 1: Read the passage, question, and all options with their evidence and rationales carefully.
                            Step 2: For each option, verify evidence source validity. Check whether the Evidence text is an exact verbatim quote from the passage. If the Evidence cannot be located in the passage, this check fails.
                            Step 3: For each option, verify the option content appears in or is supported by the passage. The option must be based on information explicitly stated in the passage, not fabricated or inferred.
                            Step 4: For each option, verify it addresses the question. The option must be relevant to what the question asks.
                            Step 5: {"For each option, verify it correctly answers the question. The information in the option must directly and accurately answer what the question asks based on the passage. If the option does not correctly answer the question, this check fails." if answer_type == "Correct Option" else "For each option, verify it is incorrect for this question. The option must relate to a different context, entity, time, place, or aspect than what the question specifically asks about. If the option could correctly answer the question, this check fails."}
                            Step 6: For each option, verify the rationale is logically sound. Check whether the provided Rationale correctly explains {"why this option correctly answers the question based on passage evidence." if answer_type == "Correct Option" else "why this option is plausible but incorrect for this specific question."}
                            Step 7: Make the final judgment. Return Yes in Coherence only if all checks pass. Otherwise return No and explain which check failed.
                            Step 8: Output your result as a JSON object with a single key "result" whose value is a list of dictionaries. Each dictionary should have four fields: Category is {answer_type}, Option is the option text being evaluated, Coherence is either Yes or No, and Rationale explains your judgment including which check failed if Coherence is No. Output only the JSON object with no additional text.
                            
                            {self_check}
                    """
        exp_prompt = f"""
                        Passage:
                            {{
                                During the winter, Atlanta reported increased COVID 19 transmission, with seventy two new cases identified over two weeks. 
                                The source of these cases remains undetermined. Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission. 
                                The two locations have frequent travel between them, and preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days.
                            }}
                        Question:
                            {{
                                What was the reporting period during which Atlanta identified its new COVID-19 cases?
                            }}
                        Options:
                                {example_options[answer_type]}    
                    """
        exp_output = f"""
                        {example_outputs[answer_type]}
                    """
        usr_prompt = f"""
                    Passage: 
                        {{
                            {input_para["inputs"]}
                        }}
                    Question: 
                        {{
                            {current_question}
                        }}
                    Options: 
                        {current_options}
                    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exp_prompt},
            {"role": "assistant", "content": exp_output},
            {"role": "user", "content": usr_prompt},
        ]
        prompt_list.append(messages)
    return prompt_list



def difficulty_judging_prompt(input_para_list, questions, selected_options, err=None):     
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_question = questions[current_idx]["Question"]
        current_choices = json.dumps(selected_options[current_idx])
        
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology. 
                            Your task is to read an epidemiology-related passage and answer a retrieval-based multiple-choice question. The correct answers can be directly found in the passage.

                            Follow these steps exactly:
                            Step 1: Read the passage and the question carefully.
                            Step 2: Read all provided options. The number of correct answers is not predetermined. There may be one correct answer, multiple correct answers, or no correct answer at all. You must evaluate each option independently.
                            Step 3: For each option, locate information in the passage that is relevant to the option. Determine whether the option matches information explicitly stated in the passage.
                            Step 4: Decide whether each option is Correct or Incorrect based on the passage. An option is Correct only if the information it contains is explicitly stated in the passage and correctly answers the specific question asked. An option is Incorrect if it contains information not in the passage, contradicts the passage, or answers a different aspect than what the question asks.
                            Step 5: You must output an analysis for every option provided, regardless of whether it is Correct or Incorrect.
                            Step 6: For each option, include evidence and a rationale. Evidence should be a list of exact verbatim quotes from the passage that are relevant to your judgment. Rationale explains why the option is correct or incorrect based on the evidence.
                            Step 7: Format your output as a JSON object with a single key "result" whose value is a list of option dictionaries. Each dictionary should contain five fields: Index, Option, Category, Evidence, and Rationale. Index is the option's original index as a string. Option is the exact option text. Category is either Correct or Incorrect. Evidence is a list of verbatim quotes from the passage. Rationale explains your judgment. Output only the JSON object with no additional text.
                            
                            {self_check}
                        """
        exp_prompt = f"""
                        Passage:
                            {{
                                During the winter, Atlanta reported increased COVID 19 transmission, with seventy two new cases identified over two weeks. 
                                The source of these cases remains undetermined. Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission. 
                                The two locations have frequent travel between them, and preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days.
                            }}
                        Question:
                            {{
                                What was the reporting period during which Atlanta identified its new COVID-19 cases?
                            }}
                        Options:
                            {{
                                [
                                    {{
                                        "Index": "0",
                                        "Option": "Two weeks"
                                    }},
                                    {{
                                        "Index": "1",
                                        "Option": "Earlier this month"
                                    }},
                                    {{
                                        "Index": "2",
                                        "Option": "Fourteen days"
                                    }}
                                ]
                            }}
                    """
        exp_output = """
                        {
                            "result": [
                                {
                                    "Index": "0",
                                    "Option": "Two weeks",
                                    "Category": "Correct",
                                    "Evidence": [
                                        "During the winter, Atlanta reported increased COVID 19 transmission, with seventy two new cases identified over two weeks."
                                    ],
                                    "Rationale": "The passage explicitly states that Atlanta's seventy-two new COVID-19 cases were 'identified over two weeks,' directly answering the question about the reporting period for Atlanta's cases."
                                },
                                {
                                    "Index": "1",
                                    "Option": "Earlier this month",
                                    "Category": "Incorrect",
                                    "Evidence": [
                                        "Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission."
                                    ],
                                    "Rationale": "The phrase 'Earlier this month' in the passage refers to when Florida reported its outbreak, not to Atlanta's reporting period. The question specifically asks about Atlanta's reporting period, which is described as 'two weeks,' not 'earlier this month.'"
                                },
                                {
                                    "Index": "2",
                                    "Option": "Fourteen days",
                                    "Category": "Incorrect",
                                    "Evidence": [
                                        "preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days."
                                    ],
                                    "Rationale": "The term 'fourteen days' in the passage refers to the travel history window during which Atlanta cases had visited Florida, not to the reporting period for identifying Atlanta's new cases. Although fourteen days is equivalent to two weeks, the passage uses 'fourteen days' in a different context unrelated to the reporting period."
                                }
                            ]
                            }
                        """
                        
        usr_prompt = f"""
                        Passage:
                            {{
                                {input_para["inputs"]}
                            }}
                        Question:
                            {{
                                {current_question}
                            }}
                        Options:
                            {{
                                {current_choices}
                            }}
                    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exp_prompt},
            {"role": "assistant", "content": exp_output},
            {"role": "user", "content": usr_prompt},
        ]
        prompt_list.append(messages)
    return prompt_list


def evaluation_prompt(input_para_list, questions, selected_options, mode, err=None):     
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_question = questions[current_idx]["Question"]
        current_choices = json.dumps(selected_options[current_idx])
        
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        if mode == "noCOT":
            system_prompt = f"""
                                You are an expert in epidemiology. Answer the retrieval-based question using the provided passage.
                                The question is multi-choice style. There may be one correct answer, multiple correct answers, or no correct answer. Select all options whose answers can be directly found in the passage.
                                
                                Output a JSON object with a single key "results" whose value is a list of index strings for all correct options. If no option is correct, the list should be empty.
                        
                                {self_check}
                            """
        else:
            system_prompt = f"""
                                You are an expert in epidemiology. Answer the question using the provided passage.
                                The question is multi-choice style. There may be one correct answer, multiple correct answers, or no correct answer. You must evaluate each option independently.

                                Follow these steps:
                                Step 1: Read the passage and question carefully.
                                Step 2: For each option, locate information in the passage that is relevant to the option.
                                Step 3: Determine whether the option matches information explicitly stated in the passage and correctly answers the specific question asked.
                                Step 4: Collect the indices of all correct options. If no option is correct, return an empty list.

                                Output a JSON object with a single key "results" whose value is a list of index strings for all correct options. If no option is correct, the list should be empty.

                                {self_check}
                            """
        exp_prompt = f"""
                        Passage:
                            {{
                                During the winter, Atlanta reported increased COVID 19 transmission, with seventy two new cases identified over two weeks. 
                                The source of these cases remains undetermined. Earlier this month, Florida also reported an outbreak of fifty cases linked to community transmission. 
                                The two locations have frequent travel between them, and preliminary interviews suggest that several Atlanta cases had visited Florida within the previous fourteen days.
                            }}
                        Question:
                            {{
                                According to preliminary interviews mentioned in the text, which location had several of the Atlanta cases visited within the previous fourteen days?
                            }}
                        Options:
                            {{
                                [
                                    {{
                                        "Index": "0",
                                        "Option": "Atlanta"
                                    }},
                                    {{
                                        "Index": "1",
                                        "Option": "Florida"
                                    }}
                                ]
                            }}
                    """
        exp_output = f"""
                        {{"results": ["1"]}}
                        """
                        
        usr_prompt = f"""
                        Passage:
                            {{
                                {input_para["inputs"]}
                            }}
                        Question:
                            {{
                                {current_question}
                            }}
                        Options:
                            {{
                                {current_choices}
                            }}
                    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exp_prompt},
            {"role": "assistant", "content": exp_output},
            {"role": "user", "content": usr_prompt},
        ]
        prompt_list.append(messages)
    return prompt_list
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
    for i in range(5):
        #time.sleep(random.uniform(0.05, 0.2))
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
                    response_format=ResponseWrapper
                    #timeout=60
                )
            #print(output)
            return output
        except Exception as err:
            err_info = err
            pass
    
    return err_info



def external_info_generation_prompt(input_para_list, relevant_triples, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        kg = relevant_triples[current_idx]
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to integrate knowledge graph triples from two sources into natural language paragraphs about diseases.
                            You will receive a JSON array containing disease information from two knowledge graphs. The first source is eKG-DONs, which contains real outbreak events extracted from WHO Disease Outbreak News. The second source is iBKH, which contains biomedical relations including symptoms, drugs, pathways, and relationships with other diseases.

                            The structure of each knowledge graph is as follows:
                                {{
                                    "eKG-Dons": {{"Disease Name": {{"Event-id": "description"}}
                                                }},
                                    "ibkh": [{{"Disease Name": {{"Symptoms": {{"relationship": ["object1", "object2"]}},
                                                                "Drugs": {{"relationship": ["object1", "object2"]}},
                                                                "Pathways": {{"relationship": ["object1", "object2"]}},
                                                                "Relationship with Other Disease": {{"relationship": ["object1", "object2"]}}
                                                                }}
                                            }}]
                                }}

                            The explanation of each relationship in iBKH is as follows:
                                {{
                                    "Symptoms": {{"Present": "The symptoms observed in the disease."}},
                                    "Drugs": {{"Associate": "The drug is often observed or studied concurrently with the disease.",
                                                "role in disease pathogenesis": "The drug affects or contributes to the disease mechanisms."}},
                                    "Pathways": {{"Association": "The biological pathways that are linked to the disease."}},
                                    "Relationship with Other Disease": {{"is_a": "The disease is a subtype of or belongs to the category of another disease.",
                                                                        "Resemble": "The disease shares similar characteristics with another disease."}}
                                }}

                                                        
                            Now follow these steps to complete the task.
                            Step 1. Read through the entire JSON input. Identify all disease names that appear across both eKG-DONs and iBKH.
                            Step 2. Normalize the disease names. If you find multiple names referring to the same disease, treat them as one unique disease and prepare to merge their information.
                            Step 3. For each unique disease, gather all related triples from both sources. From eKG-DONs, collect all outbreak event descriptions. From iBKH, collect all relationships and objects across Symptoms, Drugs, Pathways, and Relationship with Other Disease.
                            Step 4. Plan your output. You will write one paragraph for each unique disease. Each paragraph should integrate all information available for that disease from both sources.
                            Step 5. Convert the triples into fluent natural language. Preserve the meaning of every triple. Do not include metadata strings such as event identifiers. Keep the tone factual and neutral.
                            Step 6. When you encounter repeated structures, summarize them appropriately. For example, if multiple outbreak events are present, you may describe patterns across locations or time periods, but do not invent specific counts. If multiple relationships of the same type exist, you may summarize the pattern without losing any objects.
                            Step 7. Review your paragraphs. Make sure you have included all information from the input. Do not add any external facts or speculation. Do not infer relationships beyond what is explicitly provided in the triples.
                            Step 8. Output only the generated paragraphs, one per disease. Do not include any other text.
                    
                            The structure of each knowledge graph is as follows:
                                {{
                                    "eKG-Dons": {{"Disease Name": {{"Event-id": "description"}}
                                                }},
                                    "ibkh": [{{"Disease Name": {{"Symptoms": {{"relationship": ["object1", "object2"]}},
                                                                "Drugs": {{"relationship": ["object1", "object2"]}},
                                                                "Pathways": {{"relationship": ["object1", "object2"]}},
                                                                "Relationship with Other Disease": {{"relationship": ["object1", "object2"]}}
                                                                }}
                                            }}]
                                }}
                            
                            The explanation of each relationship in iBKH is as follows:
                            {{
                                "Symptoms": {{"Present": "The symptoms observed in the disease."}},
                                "Drugs": {{"Associate": "The drug is often observed or studied concurrently with the disease.",
                                            "role in disease pathogenesis": "The drug affects or contributes to the disease mechanisms."}},
                                "Pathways": {{"Association": "The biological pathways that are linked to the disease."}},
                                "Relationship with Other Disease": {{"is_a": "The disease is a subtype of or belongs to the category of another disease.",
                                                                    "Resemble": "The disease shares similar characteristics with another disease."}}
                            }}
                    
                            {self_check}
                        """
        exp_prompt = f"""
                        Knowledge Graph:
                            {{
                                "eKG-Dons": [{{"Coronavirus": {{
                                                            "Event-1": "Outbreak of Coronavirus occurred in Guinea Bissau on 2014-03-17, with confirmed 4 cases.",
                                                            "Event-2": "Outbreak of Coronavirus occurred in Republic of Tajikistan on 2010-04-29, with confirmed 171 cases and 12 deaths."
                                                            }}
                                            }}],
                                "ibkh": [{{"COVID-19": {{"Drugs": {{"Effect": ["Remdesivir"]}},
                                                        "Relationship with Other Disease": {{"is_a": ["coronavirus infectious disease"]}}
                                                        }}
                                        }}]
                            }}
                    """
        exp_ouput = """
                        Documented outbreaks of Coronavirus include an event in the Republic of Tajikistan on April 29, 2010, which resulted in 171 confirmed cases and 12 deaths. Another outbreak was recorded in Guinea Bissau on March 17, 2014, involving 4 confirmed cases.
                        
                        COVID-19 is classified as a coronavirus infectious disease. In terms of pharmaceutical associations, the drug Remdesivir has been observed to have an effect on the condition.
                    """                    
        usr_prompt = f"""    
                    Knowledge Graph:
                        {{
                            {kg}
                        }}
                    """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": exp_prompt},
            {"role": "assistant", "content": exp_ouput},
            {"role": "user", "content": usr_prompt}
        ]
        prompt_list.append(messages)
    return prompt_list






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
                            Your task is to classify an epidemiological passage into the most appropriate class for guiding question generation. The questions to be generated are multi-step inference questions that require combining document evidence with epidemiological principles, rather than simple factual recall from the text.
                            You will be given a list of classes in this format:
                                {{
                                    {question_class}
                                }}

                            Follow these steps to complete the task.
                            Step 1. Read the passage carefully. Identify its main epidemiological focus, including what phenomena it describes, what methods it uses, and what findings it reports.
                            Step 2. Read through all provided classes and their descriptions. Understand what each class is intended to capture.
                            Step 3. Compare the passage content against each class. Consider which class would best support the generation of inference-based questions that require reasoning beyond what is explicitly stated in the passage.
                            Step 4. Select the single most appropriate class. If the passage touches on multiple classes, choose the one that reflects its primary focus and offers the richest opportunity for multi-step reasoning.
                            Step 5. Write your output in the following format. Copy the class index, class name, description exactly as they appear in the provided list, and rationale why you choose it.
                            
                            {self_check}
                        """
        exp_prompt = f"""
                        Passage: 
                            {{
                                Respiratory syncytial virus (RSV) is a leading cause of intensive care unit (ICU) admission and respiratory failure among infants (children aged <1 year) in the United States. In August 2023, CDC’s Advisory Committee on Immunization Practices recommended nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season. Following licensure, nirsevimab effectiveness has been demonstrated against RSV-associated infant hospitalization, but evidence regarding effectiveness against RSV-associated critical illness is limited. In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure (illness requiring continuous positive airway pressure, bilevel positive airway pressure, or invasive mechanical ventilation) after hospital admission was evaluated during December 1, 2024−April 15, 2025. Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset. Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint. These estimates support the recommendation for use of nirsevimab as a prevention strategy to protect infants against severe outcomes from RSV infection.
                            }}
                    """
        exp_output = """ 
                        {
                            "Index": "4",
                            "Class": "Susceptibility & Immunity",
                            "Description": "Describes who is susceptible and why, linking serologic measures to correlates of protection; evaluates effectiveness after vaccination or prior infection and its waning with reinfection, hybrid immunity, and variant escape, including the effects of vaccine dose number and intervals; and assesses severity risk using clinical and contextual prognostic factors.",
                            "Rationale": "This passage centers on evaluating the effectiveness of nirsevimab, a passive immunization strategy, in protecting infants against severe RSV outcomes (ICU admission and acute respiratory failure). The core finding—80-83% effectiveness against critical illness—directly addresses how immunoprophylaxis confers protection to a susceptible population (infants aged <8 months). The study design compares receipt of nirsevimab between cases and controls to estimate protective effectiveness, which is a hallmark of immunity evaluation studies. Additionally, the passage reports the median time from nirsevimab receipt to symptom onset (50-52 days), providing data relevant to duration of protection. This aligns squarely with Class 4's focus on evaluating effectiveness after immunization and assessing severity risk. Multi-step inference questions could explore factors affecting vaccine/prophylaxis effectiveness, the relationship between timing of administration and protection, and comparisons of protection against varying severity endpoints—all requiring integration of immunological principles with the study findings."
                        }                   
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
                            Your task is to classify an epidemiological passage into the most appropriate topic for guiding question generation. The questions to be generated are multi-step inference questions that require combining document evidence with epidemiological principles, rather than simple factual recall from the text.
                            You will be given a list of topics in this format:
                                {{
                                    {question_topic}
                                }}

                            Follow these steps to complete the task.
                            Step 1. Read the passage carefully. Identify its main epidemiological focus, including what phenomena it describes, what methods it uses, and what findings it reports.
                            Step 2. Read through all provided topics and their descriptions. Understand what each topic is intended to capture.
                            Step 3. Compare the passage content against each topic. Consider which topic would best support the generation of inference-based questions that require reasoning beyond what is explicitly stated in the passage.
                            Step 4. Select the single most appropriate topic. If the passage touches on multiple topics, choose the one that reflects its primary focus and offers the richest opportunity for multi-step reasoning.
                            Step 5. Write your output in the following format. Copy the topic index, topic name, description exactly as they appear in the provided list, and rationale why you choose it.

                            {self_check}
                        """
        exp_prompt = f"""
                        Passage: 
                            {{
                                Respiratory syncytial virus (RSV) is a leading cause of intensive care unit (ICU) admission and respiratory failure among infants (children aged <1 year) in the United States. In August 2023, CDC’s Advisory Committee on Immunization Practices recommended nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season. Following licensure, nirsevimab effectiveness has been demonstrated against RSV-associated infant hospitalization, but evidence regarding effectiveness against RSV-associated critical illness is limited. In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure (illness requiring continuous positive airway pressure, bilevel positive airway pressure, or invasive mechanical ventilation) after hospital admission was evaluated during December 1, 2024−April 15, 2025. Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset. Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint. These estimates support the recommendation for use of nirsevimab as a prevention strategy to protect infants against severe outcomes from RSV infection.
                            }}
                    """
        exp_output = """
                        {
                            "Index": "3",
                            "Topic": "Protection effectiveness, waning, reinfection & immune escape",
                            "Description": "Describes protection after vaccination or prior infection, its change over time, risks of reinfection, hybrid immunity, and variant-related escape. Considers how vaccine dose number and dose intervals influence vaccine effectiveness and its waning over time.",
                            "Rationale": "The passage's central objective is to evaluate the protective effectiveness of nirsevimab, a long-acting monoclonal antibody, against severe RSV outcomes in infants. The case-control design directly measures effectiveness by comparing nirsevimab receipt rates between RSV-positive cases admitted to ICU and RSV-negative controls, yielding effectiveness estimates of 80% against ICU admission and 83% against acute respiratory failure. This is a classic vaccine/immunoprophylaxis effectiveness evaluation. The passage also provides timing data (median 50-52 days from nirsevimab receipt to symptom onset), which offers opportunities for multi-step inference questions about duration of protection and potential waning. While the study measures severe endpoints, the research question is fundamentally about whether passive immunization protects against these outcomes—not about identifying prognostic factors that predict severity among infected individuals. Topic 3 best captures this protection effectiveness focus and provides the richest substrate for inference questions about factors influencing prophylaxis performance, durability of protection, and how effectiveness might vary with time since administration."
                        }             
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


def question_generation_prompt(input_para_list, topics, external_info, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_topic = json.dumps({k: topics[current_idx].get(k) for k in ("Topic", "Description")})
        current_external_info = external_info[current_idx]
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to generate a multiple-choice style question that requires multi-step reasoning. The question should be grounded in the passage, guided by the topic, and optionally informed by external domain knowledge.
                            Follow these steps to complete the task.
                            Step 1. Read the passage, the topic description, and the external domain knowledge carefully. Understand what kind of epidemiological reasoning the topic requires.
                            Step 2. Identify a passage-anchored detail that the question must rely on. This should be a specific fact, number, observation, or finding that appears in the passage. The question must be impossible to answer without this anchored detail. Prioritize details that align with the topic, but if the most specific and valuable detail does not perfectly match the topic, you may still select it.
                            Step 3. Select at least two pieces of evidence from the passage that must be combined to answer the question. These pieces of evidence should come from different sentences or different parts of the text. Each piece should contribute a distinct element to the reasoning chain rather than simply restating the same information. Prioritize evidence that supports the type of reasoning the topic requires. In your rationale, explain how your selected evidence relates to the topic.
                            Step 4. Evaluate whether the external domain knowledge is relevant. Read through the external domain knowledge and determine whether it has a substantive connection to both the passage content and the topic. If any meaningful connection exists, you must incorporate relevant information from the external domain knowledge as part of your evidence. Only if you determine that no meaningful connection can be made should you proceed with passage evidence alone. In your rationale, explicitly state your judgment and reasoning about the relevance of external domain knowledge.
                            Step 5. Establish the reasoning chain among your selected evidence. Before writing the question, plan how the evidence pieces connect logically. An intermediate inference is not simply restating or combining facts from the passage. It requires applying an epidemiological principle, drawing a comparison, identifying an implication, or reaching a conclusion that is not explicitly stated in any single evidence piece.
                            Step 6. Before finalizing your question, verify that it truly requires multi-step reasoning. Ask yourself: if a reader sees only one of the selected evidence pieces, can they answer the question? If yes, you must revise your approach to require genuine integration of multiple pieces.
                            Step 7. Verify that the question asks about something the passage does not directly answer. If the passage explicitly provides data, conclusions, or statements that directly address the question, revise the question to target an inference or implication that must be derived rather than extracted. Be cautious of questions that use words like "inference" or "conclude" but actually ask about information the passage already states. The test is whether a reader could answer by locating and paraphrasing a specific sentence, not whether the question sounds like it requires reasoning.                            
                            Step 8. Write one question stem that requires the reasoning chain you planned. The correct answer should be a conclusion that emerges only when the evidence pieces are combined and interpreted together. If the answer is essentially a paraphrase or summary of information stated in any single evidence piece, the question does not meet the requirement.
                            Step 9. Ensure the question leaves room for multiple plausible answer directions. A good question allows for several reasonable-sounding options that require careful reasoning to distinguish. If the question points so directly to one obvious answer that no meaningful distractors could be constructed, broaden or reframe the question.
                            Step 10. Apply the following constraints to your question stem. Keep the question stem concise. Do not embed specific numbers, percentages, or timeframes from the passage. Do not use descriptive phrases that point to key evidence, whether through direct reference, abstract academic language, or paraphrased descriptions of specific findings. Complexity of language does not equal difficulty of reasoning. A concise stem that requires genuine inference is better than an elaborate stem that guides the reader toward specific evidence. The question should set up a reasoning task without indicating which findings are relevant. Avoid using words that directly indicate the type of answer expected, such as limitation, weakness, advantage, implication, or significance. Do not include answers or options. Do not imply how many correct answers there are. Do not explicitly state external domain knowledge in the question stem if that knowledge is part of the reasoning chain. Ensure the question is grammatically correct and logically coherent even without access to the external domain knowledge.
                            Step 11. Use the topic description to guide the reasoning approach, but do not ask the reader to define or identify the topic name directly. The question must focus on the specific epidemiological content in the passage.
                            Step 12. Format your output as a JSON object containing three fields: Question, Evidence, and Rationale. Question is the question stem you generated. Evidence is a list of strings, where each string is an exact verbatim quote from the passage or external domain knowledge that supports the reasoning chain. Each piece of evidence should be a separate string in the list, prefixed with its source. Rationale explains the reasoning chain and how the evidence pieces connect to form the question. Output only the JSON object with no additional text before or after it.
                            
                            {self_check}
                        """
        exp_prompt = f"""
                        Passage:
                            {{
                                Respiratory syncytial virus (RSV) is a leading cause of intensive care unit (ICU) admission and respiratory failure among infants (children aged <1 year) in the United States. In August 2023, CDC’s Advisory Committee on Immunization Practices recommended nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season. Following licensure, nirsevimab effectiveness has been demonstrated against RSV-associated infant hospitalization, but evidence regarding effectiveness against RSV-associated critical illness is limited. In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure (illness requiring continuous positive airway pressure, bilevel positive airway pressure, or invasive mechanical ventilation) after hospital admission was evaluated during December 1, 2024−April 15, 2025. Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset. Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint. These estimates support the recommendation for use of nirsevimab as a prevention strategy to protect infants against severe outcomes from RSV infection.
                            }}
                        External Domain Knowledge:
                            {{
                                Respiratory syncytial virus infectious disease is classified as a viral infectious disease. In terms of pharmaceutical intervention, Palivizumab is identified as a treatment that has an effect on the disease. Ribavirin is also recognized as a treatment and therapy that acts upon the disease and is associated with it. Other treatments and therapies include Dexamethasone, Hydrocortisone, corticosteroids, Vitamin A, Azithromycin, and the compounds rd3 0028 and bms 433771. The disease is further associated with 2',5'-oligoadenylate, Bortezomib, Methylprednisolone hemisuccinate, Resveratrol, sodium tungstate(VI), 5-acetoxyl-2,6,8-trimethylchromone 7-O-beta-D-glucopyranoside, and 5-hydroxy-2,6,8-trimethylchromone 7-O-beta-D-glucopyranoside. Additionally, there are inferred relationships between the disease and a vast array of other entities, encompassing antibiotics, anti-inflammatory agents, hormones, dietary factors, heavy metals, and environmental pollutants such as tobacco smoke and air pollutants.                            }}
                        Topic:
                            {{
                                "Topic": "Protection effectiveness, waning, reinfection & immune escape",
                                "Description": "Describes protection after vaccination or prior infection, its change over time, risks of reinfection, hybrid immunity, and variant-related escape. Considers how vaccine dose number and dose intervals influence vaccine effectiveness and its waning over time."
                            }}
                    """
        exp_output = """
                        {
                            "Question": "In this case-control study of infants, what analytical approach would allow investigators to assess whether protection changes over the course of the season while preserving the validity of the comparison between groups?",
                            "Evidence": [
                                "Passage: 'In August 2023, CDC's Advisory Committee on Immunization Practices recommended nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season.'",
                                "Passage: 'In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.'",
                                "Passage: 'Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset.'"
                            ],
                            "Rationale": "The reasoning chain connects three pieces of evidence to reach a conclusion not stated in the passage. Evidence 1 establishes that nirsevimab is designed as a long-acting agent intended to protect infants throughout their first RSV season. This raises the question of whether protection actually persists across the full season or diminishes over time, which is central to the topic of waning immunity. Evidence 2 provides the study timeframe covering approximately 4.5 months of an RSV season. This contextualizes the window during which protection could potentially be assessed. Evidence 3 describes the case-control structure where RSV-positive infants are cases and RSV-negative infants are controls, with vaccination status compared between groups. To answer this question, one must first understand that the topic concerns whether protection wanes over time. Then one must recognize that this study used a case-control design with a specific comparison structure. Finally, one must apply epidemiological methodology knowledge to identify that stratifying by time since receipt while maintaining the case-control framework would allow assessment of whether the odds ratio changes across time strata. The passage reports overall effectiveness but does not describe any approach for evaluating temporal changes in protection. The answer requires methodological reasoning beyond what is stated. Regarding external domain knowledge, the provided information focuses on therapeutic interventions for RSV such as Palivizumab and corticosteroids. This does not connect substantively to the topic of assessing protection effectiveness over time or study design considerations. Therefore, I proceeded with passage evidence alone."
                        }
                    """
        usr_prompt = f"""
                    Passage: 
                        {{
                            {input_para["inputs"]}
                        }}
                    External Domain Knowledge:
                        {{
                            {current_external_info}
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


def correct_option_generation_prompt(input_para_list, questions, external_info, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_question = json.dumps(questions[current_idx])
        current_external_info = external_info[current_idx]
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to generate correct options for a multiple-choice question that requires multi-step reasoning. The options should be derived conclusions that emerge from integrating the provided evidence, not facts that can be directly retrieved from the passage.
                            You will be given the passage, external domain knowledge, and the output from question generation which includes the question, the evidence, and the rationale explaining the reasoning chain.

                            Follow these steps to complete the task.
                            Step 1. Read the question, the evidence, and the rationale carefully. Understand the reasoning chain that was established during question generation. The correct option must be a valid conclusion of this reasoning chain.
                            Step 2. Identify what the correct answer should capture. Based on the reasoning chain in the rationale, determine what conclusion or implication emerges when the evidence pieces are combined and interpreted together.
                            Step 3. Draft one or more correct options. Each option must satisfy these requirements:
                                - It must be a conclusion that requires integrating at least two pieces of the provided evidence
                                - It must not be a direct paraphrase of any single sentence in the passage
                                - It must not be verifiable by reading only one evidence piece
                                - It must require applying an epidemiological principle or methodological concept to interpret the evidence
                                - It must use different vocabulary from the passage where possible while preserving accuracy
                                - It must be specific enough to be clearly correct given the evidence, not vaguely true
                            Step 4. Keep each option concise. An option should typically be one sentence or a short phrase. Avoid overly technical or textbook-like language. If an option reads like a paragraph from a methods textbook, simplify it while preserving accuracy.
                            Step 5. Follow the same decision about external domain knowledge that was made during question generation. If external domain knowledge was incorporated as part of the evidence in the question generation rationale, the option may reflect conclusions that depend on that knowledge. If external domain knowledge was not used during question generation, do not introduce it here. In either case, do not embed external knowledge as explicit facts in the option text. The option should state a conclusion, and the external knowledge should appear only in your rationale as justification.
                            Step 6. Verify that each option is methodologically sound. If the option proposes an analytical approach or interpretation, ensure it is the most direct and valid way to address the question. Do not include options that are technically possible but weaker or less appropriate than alternatives.
                            Step 7. If external domain knowledge was used in the rationale, the option may reflect conclusions that depend on that knowledge. However, do not embed external knowledge as explicit facts in the option text. The option should state a conclusion, and the external knowledge should appear only in your rationale as justification.
                            Step 8. If generating multiple correct options, ensure they are genuinely distinct. Two options that represent variations of the same method or approach do not count as distinct. For example, stratified analysis and interaction terms in regression are methodologically related approaches to the same question and should not both appear as separate correct options. If you cannot identify multiple genuinely distinct correct answers, generate only one option.
                            Step 9. Check that each option does not contradict the passage or overstate the evidence. The option should be defensible based on what is provided, not based on assumptions beyond the evidence.
                            Step 10. Format each option as a concise statement. It does not need to be a complete sentence, but it must be semantically complete and self-contained.
                            Step 11. For each option, explain in your rationale how the provided evidence pieces and reasoning chain support that specific option.
                            Step 12. Format your output as a JSON object with a single key "result" whose value is a list of option dictionaries. Each dictionary should contain four fields: Category, Option, Evidence, and Rationale. Category should be "Correct Option" for all options generated by this task. Option is the option text itself. Evidence is a list of strings, where each string is an exact verbatim quote from the passage or external domain knowledge that supports this option. Rationale explains how the evidence and reasoning chain lead to this option. Output only the JSON object with no additional text, explanation, or commentary before or after it.
                        
                            {self_check}
                        """
        exp_prompt = f"""
                        Passage:
                            {{
                                Respiratory syncytial virus (RSV) is a leading cause of intensive care unit (ICU) admission and respiratory failure among infants (children aged <1 year) in the United States. In August 2023, CDC’s Advisory Committee on Immunization Practices recommended nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season. Following licensure, nirsevimab effectiveness has been demonstrated against RSV-associated infant hospitalization, but evidence regarding effectiveness against RSV-associated critical illness is limited. In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure (illness requiring continuous positive airway pressure, bilevel positive airway pressure, or invasive mechanical ventilation) after hospital admission was evaluated during December 1, 2024−April 15, 2025. Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset. Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint. These estimates support the recommendation for use of nirsevimab as a prevention strategy to protect infants against severe outcomes from RSV infection.
                            }}
                        External Domain Knowledge:
                            {{
                                Respiratory syncytial virus infectious disease is classified as a viral infectious disease. In terms of pharmaceutical intervention, Palivizumab is identified as a treatment that has an effect on the disease. Ribavirin is also recognized as a treatment and therapy that acts upon the disease and is associated with it. Other treatments and therapies include Dexamethasone, Hydrocortisone, corticosteroids, Vitamin A, Azithromycin, and the compounds rd3 0028 and bms 433771. The disease is further associated with 2',5'-oligoadenylate, Bortezomib, Methylprednisolone hemisuccinate, Resveratrol, sodium tungstate(VI), 5-acetoxyl-2,6,8-trimethylchromone 7-O-beta-D-glucopyranoside, and 5-hydroxy-2,6,8-trimethylchromone 7-O-beta-D-glucopyranoside. Additionally, there are inferred relationships between the disease and a vast array of other entities, encompassing antibiotics, anti-inflammatory agents, hormones, dietary factors, heavy metals, and environmental pollutants such as tobacco smoke and air pollutants.                            }}
                            }}
                        Question:
                            {{
                                "Question": "In this case-control study of infants, what analytical approach would allow investigators to assess whether protection changes over the course of the season while preserving the validity of the comparison between groups?",
                                "Evidence": [
                                    "Passage: 'In August 2023, CDC's Advisory Committee on Immunization Practices recommended nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season.'",
                                    "Passage: 'In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.'",
                                    "Passage: 'Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset.'"
                                ],
                                "Rationale": "The reasoning chain connects three pieces of evidence to reach a conclusion not stated in the passage. Evidence 1 establishes that nirsevimab is designed as a long-acting agent intended to protect infants throughout their first RSV season. This raises the question of whether protection actually persists across the full season or diminishes over time, which is central to the topic of waning immunity. Evidence 2 provides the study timeframe covering approximately 4.5 months of an RSV season. This contextualizes the window during which protection could potentially be assessed. Evidence 3 describes the case-control structure where RSV-positive infants are cases and RSV-negative infants are controls, with vaccination status compared between groups. To answer this question, one must first understand that the topic concerns whether protection wanes over time. Then one must recognize that this study used a case-control design with a specific comparison structure. Finally, one must apply epidemiological methodology knowledge to identify that stratifying by time since receipt while maintaining the case-control framework would allow assessment of whether the odds ratio changes across time strata. The passage reports overall effectiveness but does not describe any approach for evaluating temporal changes in protection. The answer requires methodological reasoning beyond what is stated. Regarding external domain knowledge, the provided information focuses on therapeutic interventions for RSV such as Palivizumab and corticosteroids. This does not connect substantively to the topic of assessing protection effectiveness over time or study design considerations. Therefore, I proceeded with passage evidence alone."
                            }}
                    """
        exp_output = """
                        {
                            "result": [
                                {
                                "Category": "Correct Option",
                                "Option": "Stratify by time since nirsevimab receipt and compare odds ratios across strata to detect any change in protective effect",
                                "Evidence": [
                                    "Passage: 'nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season.'",
                                    "Passage: 'nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.'",
                                    "Passage: 'Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset.'"
                                ],
                                "Rationale": "This option requires integrating multiple evidence pieces through epidemiological reasoning. Evidence 1 establishes nirsevimab as a 'long-acting' agent intended for season-long protection, raising the question of durability. Evidence 2 provides the 4.5-month study window over which protection could be assessed. Evidence 3 describes the case-control structure where exposure status is compared between RSV-positive cases and RSV-negative controls. To assess whether protection wanes over time while preserving the case-control comparison, the appropriate method is stratified analysis—dividing the data by time since receipt and examining whether the odds ratio (the valid measure of association in case-control studies) changes across strata. This approach maintains the internal validity of comparing cases to controls within each time stratum while allowing detection of temporal trends in effectiveness. The passage reports overall effectiveness but does not describe this analytical approach, requiring application of epidemiological methodology to derive the answer."
                                }
                            ]
                        }
                    """
        usr_prompt = f"""
                    Passage: 
                        {{
                            {input_para["inputs"]}
                        }}
                    External Domain Knowledge:
                        {{
                            {current_external_info}
                        }}
                    Question: 
                        {{
                            {current_question}
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


def distractor_generation_prompt(input_para_list, questions, external_info, correct_options, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_question = json.dumps(questions[current_idx])
        #current_correct_options = "\n".join([item["Option"] for item in correct_options[current_idx]])
        current_correct_options = json.dumps([{k: item.get(k) for k in ("Category", "Option")} for item in correct_options[current_idx]])
        current_external_info = external_info[current_idx]
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to generate distractors for a multiple-choice question that requires multi-step reasoning. Distractors must look structurally identical to the correct options but contain a subtle logical flaw that can only be detected through careful reasoning.
                            You will be given the passage, external domain knowledge, the output from question generation which includes the question, evidence, and rationale, and the correct options.

                            Follow these steps to complete the task.
                            Step 1. Read all provided materials carefully. Pay special attention to the reasoning chain described in the question generation rationale. Understand how the evidence pieces connect to reach the correct answer.
                            Step 2. Analyze the correct options. Identify their grammatical structure, length, level of specificity, and the type of conclusion they express. Your distractors must match these characteristics exactly so that no distractor can be eliminated based on style alone.
                            Step 3. Identify multiple vulnerable points in the reasoning chain where a reader might go wrong. Consider these categories of errors:
                                - Confusing related but distinct concepts
                                - Applying a valid method to an incompatible study design
                                - Mixing up the target variable with a superficially similar variable
                                - Using correct terminology but violating underlying assumptions
                                - Drawing conclusions that would require different data than what is available
                            Step 4. Draft distractors that exploit different vulnerable points. Each distractor should target a distinct error type. Use one or more of these strategies:
                                - Use a valid epidemiological technique that sounds relevant but would actually prevent answering the question
                                - Propose an analysis method incompatible with the study design described in the passage
                                - Confuse the variable of interest with a related but different variable
                                - Apply correct domain terminology while violating a methodological constraint
                                - Suggest an approach that abandons a key feature that makes the original comparison valid
                            Step 5. Follow the same decision about external domain knowledge that was made during question generation. If external knowledge was used as evidence, you may use it to construct plausible-sounding but flawed reasoning. If external knowledge was not used, do not introduce it. Do not embed external knowledge as explicit facts in the distractor text.
                            Step 6. Verify that each distractor requires domain expertise to eliminate. A good distractor should tempt someone who has surface-level familiarity with epidemiological methods but does not think through the full implications. If a distractor can be eliminated by anyone who simply read the passage carefully, revise it to require deeper methodological reasoning.
                            Step 7. Verify that each distractor is definitively wrong. The flaw must be clear to an expert who analyzes carefully. Avoid distractors that could be argued as partially correct or context-dependent.
                            Step 8. Verify that distractors exploit genuinely different vulnerabilities. If two distractors are wrong for related reasons, revise one to target a different type of error. Each distractor should test a distinct aspect of methodological understanding.

                            Step 9. Match the style of correct options exactly. Use the same grammatical structure, similar length, and the same level of technical specificity. Each distractor should be indistinguishable from correct options in terms of format and tone.
                            Step 10. For each distractor, document the evidence used and explain in the rationale what specific flaw it contains, which methodological principle it violates, and why someone with incomplete understanding might find it tempting.
                            Step 11. Format your output as a JSON object with a single key "result" whose value is a list of distractor dictionaries. Each dictionary should contain four fields: Category, Option, Evidence, and Rationale. Category should be "Distractor" for all options generated by this task. Option is the distractor text. Evidence is a list of strings, where each string is an exact verbatim quote that the distractor is based on. Rationale explains the specific flaw and why the distractor is plausible but wrong. Output only the JSON object with no additional text before or after it.
                            
                            {self_check}
                    """
        exp_prompt = f"""
                        Passage:
                            {{
                                Respiratory syncytial virus (RSV) is a leading cause of intensive care unit (ICU) admission and respiratory failure among infants (children aged <1 year) in the United States. In August 2023, CDC’s Advisory Committee on Immunization Practices recommended nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season. Following licensure, nirsevimab effectiveness has been demonstrated against RSV-associated infant hospitalization, but evidence regarding effectiveness against RSV-associated critical illness is limited. In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure (illness requiring continuous positive airway pressure, bilevel positive airway pressure, or invasive mechanical ventilation) after hospital admission was evaluated during December 1, 2024−April 15, 2025. Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset. Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint. These estimates support the recommendation for use of nirsevimab as a prevention strategy to protect infants against severe outcomes from RSV infection.
                            }}
                        External Domain Knowledge:
                            {{
                                Respiratory syncytial virus infectious disease is classified as a viral infectious disease. In terms of pharmaceutical intervention, Palivizumab is identified as a treatment that has an effect on the disease. Ribavirin is also recognized as a treatment and therapy that acts upon the disease and is associated with it. Other treatments and therapies include Dexamethasone, Hydrocortisone, corticosteroids, Vitamin A, Azithromycin, and the compounds rd3 0028 and bms 433771. The disease is further associated with 2',5'-oligoadenylate, Bortezomib, Methylprednisolone hemisuccinate, Resveratrol, sodium tungstate(VI), 5-acetoxyl-2,6,8-trimethylchromone 7-O-beta-D-glucopyranoside, and 5-hydroxy-2,6,8-trimethylchromone 7-O-beta-D-glucopyranoside. Additionally, there are inferred relationships between the disease and a vast array of other entities, encompassing antibiotics, anti-inflammatory agents, hormones, dietary factors, heavy metals, and environmental pollutants such as tobacco smoke and air pollutants.                            }}
                            }}
                        Question:
                            {{
                                "Question": "In this case-control study of infants, what analytical approach would allow investigators to assess whether protection changes over the course of the season while preserving the validity of the comparison between groups?",
                                "Evidence": [
                                    "Passage: 'In August 2023, CDC's Advisory Committee on Immunization Practices recommended nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season.'",
                                    "Passage: 'In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.'",
                                    "Passage: 'Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset.'"
                                ],
                                "Rationale": "The reasoning chain connects three pieces of evidence to reach a conclusion not stated in the passage. Evidence 1 establishes that nirsevimab is designed as a long-acting agent intended to protect infants throughout their first RSV season. This raises the question of whether protection actually persists across the full season or diminishes over time, which is central to the topic of waning immunity. Evidence 2 provides the study timeframe covering approximately 4.5 months of an RSV season. This contextualizes the window during which protection could potentially be assessed. Evidence 3 describes the case-control structure where RSV-positive infants are cases and RSV-negative infants are controls, with vaccination status compared between groups. To answer this question, one must first understand that the topic concerns whether protection wanes over time. Then one must recognize that this study used a case-control design with a specific comparison structure. Finally, one must apply epidemiological methodology knowledge to identify that stratifying by time since receipt while maintaining the case-control framework would allow assessment of whether the odds ratio changes across time strata. The passage reports overall effectiveness but does not describe any approach for evaluating temporal changes in protection. The answer requires methodological reasoning beyond what is stated. Regarding external domain knowledge, the provided information focuses on therapeutic interventions for RSV such as Palivizumab and corticosteroids. This does not connect substantively to the topic of assessing protection effectiveness over time or study design considerations. Therefore, I proceeded with passage evidence alone."
                            }}
                        Correct Options:
                            {{
                                [ 
                                    {{ 
                                        "Category": "Correct Option", 
                                        "Option": "Stratify by time since nirsevimab receipt and compare odds ratios across strata to detect any change in protective effect"
                                    }} 
                                ]
                            }}
                    """
        exp_output = """
                        {
                            "result": [
                                    {
                                        "Category": "Distractor",
                                        "Option": "Calculate incidence rates of RSV illness among nirsevimab recipients at successive intervals since receipt to measure changes in protective effect",
                                        "Evidence": [
                                            "In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.",
                                            "Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset."
                                        ],
                                        "Rationale": "This distractor confuses case-control methodology with cohort methodology. Incidence rates require knowing the person-time at risk in a defined population followed prospectively. In a case-control study, participants are sampled based on outcome status, and the ratio of cases to controls is determined by the investigator rather than reflecting natural disease occurrence. Therefore, incidence cannot be calculated from case-control data. This distractor tempts readers who recognize that incidence rates are commonly used in vaccine effectiveness research but do not appreciate the fundamental constraint that case-control designs preclude direct incidence estimation. The flaw requires understanding why sampling on outcome status invalidates incidence calculations."
                                    },
                                    {
                                        "Category": "Distractor",
                                        "Option": "Stratify by calendar month of admission and compare vaccination prevalence between cases and controls to detect temporal variation in protection",
                                        "Evidence": [
                                            "In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.",
                                            "Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset."
                                        ],
                                        "Rationale": "This distractor confuses calendar time with time since vaccination. Stratifying by calendar month of admission would reveal whether vaccine effectiveness varies across the seasonal epidemic curve, reflecting changes in viral circulation intensity or population characteristics over the season. However, this does not address waning immunity, which requires examining how protection changes as a function of time elapsed since an individual received nirsevimab. Two infants admitted in March could have very different durations since receipt. The distractor tempts readers who associate temporal analysis with any time-based stratification but do not distinguish between calendar time and exposure-to-outcome intervals. The methodological flaw is measuring the wrong temporal dimension for the question of waning protection."
                                    },
                                    {
                                        "Category": "Distractor",
                                        "Option": "Match cases and controls on time since nirsevimab receipt to control confounding while comparing RSV outcomes across time intervals",
                                        "Evidence": [
                                            "Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset.",
                                            "Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint."
                                        ],
                                        "Rationale": "This distractor applies a valid epidemiological technique inappropriately. Matching is used to control confounding by ensuring cases and controls are similar on key variables. However, matching on the very variable whose effect you wish to study eliminates the variation needed to assess that effect. If cases and controls are matched on time since nirsevimab receipt, one cannot examine whether protection differs across time strata because variation in timing has been deliberately removed. The distractor tempts readers who recognize matching as a legitimate method for strengthening case-control validity but do not recognize that matching on the analytic variable of interest defeats the study purpose. The flaw is methodologically subtle because it uses correct terminology while violating the principle that you cannot match on a variable you intend to analyze."
                                    },
                                    {
                                        "Category": "Distractor",
                                        "Option": "Apply Kaplan-Meier survival analysis to estimate the probability of remaining RSV-free over time since nirsevimab administration",
                                        "Evidence": [
                                            "In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.",
                                            "Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint."
                                        ],
                                        "Rationale": "This distractor proposes a method incompatible with the study design. Kaplan-Meier survival analysis requires prospective follow-up of a cohort from a defined starting point, tracking time until an event occurs or censoring. Case-control studies identify individuals based on outcome status at a single point rather than following them over time. The design samples cases who already experienced the outcome and controls who did not, precluding the time-to-event framework that survival analysis requires. The distractor tempts readers who recognize survival analysis as an appropriate method for assessing duration of protection but do not appreciate that it requires cohort architecture with prospective ascertainment. The flaw involves applying a valid analytical technique to an incompatible study design."
                                    }
                                ]
                            }
                        """
        usr_prompt = f"""
                    Passage: 
                        {{
                            {input_para["inputs"]}
                        }}
                    External Domain Knowledge:
                        {{
                            {current_external_info}
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
    


def answer_checking_prompt(input_para_list, questions, external_info, options, answer_type=None, err=None):
    example_options = {"Correct Option":
                        [
                            {
                                "Category": "Correct Option",
                                "Option": "Stratify by time since nirsevimab receipt and compare odds ratios across strata to detect any change in protective effect",
                                "Evidence": [
                                    "Passage: 'nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season.'",
                                    "Passage: 'nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.'",
                                    "Passage: 'Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset.'"
                                ],
                                "Rationale": "This option requires integrating multiple evidence pieces through epidemiological reasoning. Evidence 1 establishes nirsevimab as a 'long-acting' agent intended for season-long protection, raising the question of durability. Evidence 2 provides the 4.5-month study window over which protection could be assessed. Evidence 3 describes the case-control structure where exposure status is compared between RSV-positive cases and RSV-negative controls. To assess whether protection wanes over time while preserving the case-control comparison, the appropriate method is stratified analysis—dividing the data by time since receipt and examining whether the odds ratio (the valid measure of association in case-control studies) changes across strata. This approach maintains the internal validity of comparing cases to controls within each time stratum while allowing detection of temporal trends in effectiveness. The passage reports overall effectiveness but does not describe this analytical approach, requiring application of epidemiological methodology to derive the answer."
                            }
                        ],
                        "Distractor": 
                        [
                            {
                                "Category": "Distractor",
                                "Option": "Calculate incidence rates of RSV illness among nirsevimab recipients at successive intervals since receipt to measure changes in protective effect",
                                "Evidence": [
                                    "In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.",
                                    "Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset."
                                ],
                                "Rationale": "This distractor confuses case-control methodology with cohort methodology. Incidence rates require knowing the person-time at risk in a defined population followed prospectively. In a case-control study, participants are sampled based on outcome status, and the ratio of cases to controls is determined by the investigator rather than reflecting natural disease occurrence. Therefore, incidence cannot be calculated from case-control data. This distractor tempts readers who recognize that incidence rates are commonly used in vaccine effectiveness research but do not appreciate the fundamental constraint that case-control designs preclude direct incidence estimation. The flaw requires understanding why sampling on outcome status invalidates incidence calculations."
                            },
                            {
                                "Category": "Distractor",
                                "Option": "Stratify by calendar month of admission and compare vaccination prevalence between cases and controls to detect temporal variation in protection",
                                "Evidence": [
                                    "In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.",
                                    "Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset."
                                ],
                                "Rationale": "This distractor confuses calendar time with time since vaccination. Stratifying by calendar month of admission would reveal whether vaccine effectiveness varies across the seasonal epidemic curve, reflecting changes in viral circulation intensity or population characteristics over the season. However, this does not address waning immunity, which requires examining how protection changes as a function of time elapsed since an individual received nirsevimab. Two infants admitted in March could have very different durations since receipt. The distractor tempts readers who associate temporal analysis with any time-based stratification but do not distinguish between calendar time and exposure-to-outcome intervals. The methodological flaw is measuring the wrong temporal dimension for the question of waning protection."
                            },
                            {
                                "Category": "Distractor",
                                "Option": "Match cases and controls on time since nirsevimab receipt to control confounding while comparing RSV outcomes across time intervals",
                                "Evidence": [
                                    "Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset.",
                                    "Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint."
                                ],
                                "Rationale": "This distractor applies a valid epidemiological technique inappropriately. Matching is used to control confounding by ensuring cases and controls are similar on key variables. However, matching on the very variable whose effect you wish to study eliminates the variation needed to assess that effect. If cases and controls are matched on time since nirsevimab receipt, one cannot examine whether protection differs across time strata because variation in timing has been deliberately removed. The distractor tempts readers who recognize matching as a legitimate method for strengthening case-control validity but do not recognize that matching on the analytic variable of interest defeats the study purpose. The flaw is methodologically subtle because it uses correct terminology while violating the principle that you cannot match on a variable you intend to analyze."
                            },
                            {
                                "Category": "Distractor",
                                "Option": "Apply Kaplan-Meier survival analysis to estimate the probability of remaining RSV-free over time since nirsevimab administration",
                                "Evidence": [
                                    "In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure after hospital admission was evaluated during December 1, 2024−April 15, 2025.",
                                    "Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint."
                                ],
                                "Rationale": "This distractor proposes a method incompatible with the study design. Kaplan-Meier survival analysis requires prospective follow-up of a cohort from a defined starting point, tracking time until an event occurs or censoring. Case-control studies identify individuals based on outcome status at a single point rather than following them over time. The design samples cases who already experienced the outcome and controls who did not, precluding the time-to-event framework that survival analysis requires. The distractor tempts readers who recognize survival analysis as an appropriate method for assessing duration of protection but do not appreciate that it requires cohort architecture with prospective ascertainment. The flaw involves applying a valid analytical technique to an incompatible study design."
                            }
                        ]}
    
    example_outputs = {"Correct Option": 
                       {
                            "result": [
                                {
                                "Category": "Correct Option",
                                "Option": "Stratify by time since nirsevimab receipt and compare odds ratios across strata to detect any change in protective effect",
                                "Coherence": "Yes",
                                "Rationale": "All verification steps pass. Evidence quotes are verbatim from the passage. The option does not embed external-only information. It requires derived reasoning rather than direct retrieval, as the passage reports effectiveness but not analytical methodology. The option directly addresses the question about analytical approaches for assessing temporal changes in protection. The evidence establishes the case-control design, study timeframe, and long-acting nature of the intervention, which together support the need for stratified analysis. The rationale correctly explains that odds ratios are the appropriate measure for case-control studies and that stratification preserves internal validity while allowing temporal trend detection. The proposed approach is epidemiologically sound for assessing waning immunity."
                                }
                            ]
                        },
                       "Distractor": 
                       {
                            "result": [
                                {
                                "Category": "Distractor",
                                "Option": "Calculate incidence rates of RSV illness among nirsevimab recipients at successive intervals since receipt to measure changes in protective effect",
                                "Coherence": "Yes",
                                "Rationale": "All steps pass. The evidence establishes a case-control study design. The option proposes calculating incidence rates, which is methodologically impossible from case-control data since sampling is based on outcome status rather than following a defined population over time. The rationale correctly explains this fundamental constraint. This is a valid distractor with a subtle methodological flaw that would tempt those unfamiliar with case-control limitations."
                                },
                                {
                                "Category": "Distractor",
                                "Option": "Stratify by calendar month of admission and compare vaccination prevalence between cases and controls to detect temporal variation in protection",
                                "Coherence": "Yes",
                                "Rationale": "All steps pass. The evidence establishes the case-control design and study timeframe. The option confuses calendar time with time since vaccination - stratifying by calendar month would assess how effectiveness varies across the epidemic season, not how protection changes as a function of time elapsed since an individual received nirsevimab. The rationale correctly identifies this methodological confusion between two distinct temporal dimensions."
                                },
                                {
                                "Category": "Distractor",
                                "Option": "Match cases and controls on time since nirsevimab receipt to control confounding while comparing RSV outcomes across time intervals",
                                "Coherence": "Yes",
                                "Rationale": "All steps pass. The evidence establishes the case-control framework with timing data. The fundamental flaw is that matching on the variable you wish to study eliminates the variation needed to analyze its effect. If cases and controls have identical time since receipt, one cannot examine whether protection differs across time strata. The rationale correctly explains this subtle methodological error using valid epidemiological terminology."
                                },
                                {
                                "Category": "Distractor",
                                "Option": "Apply Kaplan-Meier survival analysis to estimate the probability of remaining RSV-free over time since nirsevimab administration",
                                "Coherence": "Yes",
                                "Rationale": "All steps pass. The evidence confirms a case-control study design. Kaplan-Meier survival analysis requires prospective follow-up of a cohort from a defined starting point, tracking time until an event occurs. Case-control studies sample based on outcome status at a point in time rather than following individuals prospectively, making survival analysis fundamentally incompatible with this design. The rationale correctly identifies this design-method mismatch."
                                }
                            ]
                        }
                       }


    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_question = questions[current_idx]["Question"]
        current_options = json.dumps(options[current_idx])
        current_external_info = external_info[current_idx]
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to check whether some answers to a question are valid as {answer_type} based on the given passage, external domain knowledge, evidence, and rationale. Evaluate only whether the option is valid under the provided materials, not whether it is factually correct in the real world.

                            Follow these steps exactly:
                            Step 1: Read the passage, the question, all options, the evidence, and the rationales. Read the external domain knowledge only if it is provided.
                            Step 2: For each option, verify evidence source validity. Check whether every evidence quote is copied verbatim from the provided passage or external domain knowledge. Do not require that evidence come from both sources. If no external domain knowledge is provided, all evidence must come from the passage.
                            Step 3: If external domain knowledge is provided, check that the option text does not explicitly embed key information that appears only in the external domain knowledge. The option may rely on domain knowledge implicitly, but must not reveal external-only solving clues in the option text. If no external domain knowledge is provided, skip this step.
                            Step 4: Check that the option is not a direct retrieval. The option must be a derived conclusion requiring reasoning, not a verbatim restatement or simple paraphrase of a single sentence from the passage.
                            Step 5: Check that the option addresses the question. The option must directly respond to what the question asks, not a related but different question.
                            Step 6: Verify that the evidence logically supports the option. Check whether the cited evidence pieces, when combined, actually lead to the conclusion stated in the option. If the evidence is irrelevant or insufficient to support the option, this check fails.
                            Step 7: Verify that the rationale is logically sound. Check whether the reasoning described in the rationale correctly explains how the evidence leads to the option. If the rationale contains logical gaps or errors, this check fails.
                            Step 8: Check whether the option is valid as {answer_type} based on the passage, external domain knowledge, and question. {"Verify that the option is actually correct as an answer to the question from an epidemiological perspective. The option must represent a valid derived conclusion, not just a plausible-sounding statement. Reject if the option contains methodological errors or misinterprets the evidence." if answer_type == "Correct Option" else "Ensure the option is definitively incorrect as an answer to this specific question but constructed using valid elements from the provided content. Verify that the flaw described in the rationale actually exists and is subtle enough to be plausible. Do not reject a distractor only because it describes a different entity than asked about, as this is a valid distractor strategy."}
                            Step 9: Make the final judgment. If Step 2 through Step 8 all pass, return Yes in Coherence. Otherwise return No and explain which step failed in Rationale.
                            Step 10: Format your output as a JSON object with a single key "result" whose value is a list of option dictionaries. Each dictionary should contain four fields: Category, Option, Coherence, and Rationale. Category should be "{answer_type}" for all options. Option is the option text being evaluated. Coherence is either "Yes" or "No". Rationale explains your judgment, including which step failed if Coherence is No. Output only the JSON object with no additional text before or after it.

                            {self_check}
                    """
        exp_prompt = f"""
                        Passage:
                            {{
                                Respiratory syncytial virus (RSV) is a leading cause of intensive care unit (ICU) admission and respiratory failure among infants (children aged <1 year) in the United States. In August 2023, CDC’s Advisory Committee on Immunization Practices recommended nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season. Following licensure, nirsevimab effectiveness has been demonstrated against RSV-associated infant hospitalization, but evidence regarding effectiveness against RSV-associated critical illness is limited. In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure (illness requiring continuous positive airway pressure, bilevel positive airway pressure, or invasive mechanical ventilation) after hospital admission was evaluated during December 1, 2024−April 15, 2025. Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset. Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint. These estimates support the recommendation for use of nirsevimab as a prevention strategy to protect infants against severe outcomes from RSV infection.
                            }}
                        External Domain Knowledge:
                            {{
                                Respiratory syncytial virus infectious disease is classified as a viral infectious disease. In terms of pharmaceutical intervention, Palivizumab is identified as a treatment that has an effect on the disease. Ribavirin is also recognized as a treatment and therapy that acts upon the disease and is associated with it. Other treatments and therapies include Dexamethasone, Hydrocortisone, corticosteroids, Vitamin A, Azithromycin, and the compounds rd3 0028 and bms 433771. The disease is further associated with 2',5'-oligoadenylate, Bortezomib, Methylprednisolone hemisuccinate, Resveratrol, sodium tungstate(VI), 5-acetoxyl-2,6,8-trimethylchromone 7-O-beta-D-glucopyranoside, and 5-hydroxy-2,6,8-trimethylchromone 7-O-beta-D-glucopyranoside. Additionally, there are inferred relationships between the disease and a vast array of other entities, encompassing antibiotics, anti-inflammatory agents, hormones, dietary factors, heavy metals, and environmental pollutants such as tobacco smoke and air pollutants.                            }}
                            }}
                        Question:
                            {{
                                In this case-control study of infants, what analytical approach would allow investigators to assess whether protection changes over the course of the season while preserving the validity of the comparison between groups?
                            }}
                        Options:
                                {json.dumps(example_options[answer_type])}    
                    """
        exp_output = f"""
                        {example_outputs[answer_type]}
                    """
        usr_prompt = f"""
                    Passage: 
                        {{
                            {input_para["inputs"]}
                        }}
                    External Domain Knowledge:
                        {{
                            {current_external_info}
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



def difficulty_judging_prompt(input_para_list, questions, selected_options, err=None, model_name=None):     
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
                            Your task is to read an epidemiology-related passage and answer a multiple-choice style question.

                            Follow these steps exactly:
                            Step 1: Read the passage and the question carefully.
                            Step 2: Read all provided options. The number of correct answers is not predetermined. There may be one correct answer, multiple correct answers, or no correct answer at all. You must evaluate each option independently.
                            Step 3: For each option, reason through whether it correctly answers the question based on the passage and your epidemiological knowledge. Some questions may require applying domain principles to interpret the passage evidence.
                            Step 4: Decide whether each option is Correct or Incorrect. You must output an analysis for every option provided, regardless of whether it is Correct or Incorrect.
                            Step 5: For each option, include evidence and a rationale explaining your reasoning. Evidence should be a list of exact verbatim quotes from the passage that support your judgment. Include all quotes that are necessary to support your reasoning chain. For complex judgments that require integrating multiple facts, this will naturally involve multiple pieces of evidence.
                            Step 6: Format your output as a JSON object with a single key "result" whose value is a list of option dictionaries. Each dictionary should contain five fields: Index, Option, Category, Evidence, and Rationale. Index is the option's original index as a string. Option is the exact option text. Category is either "Correct" or "Incorrect". Evidence is a list of verbatim quotes from the passage. Rationale explains your reasoning. Output only the JSON object with no additional text before or after it.

                            
                            {self_check}
                        """
        exp_prompt = f"""
                        Passage:
                            {{
                                Respiratory syncytial virus (RSV) is a leading cause of intensive care unit (ICU) admission and respiratory failure among infants (children aged <1 year) in the United States. In August 2023, CDC’s Advisory Committee on Immunization Practices recommended nirsevimab, a long-acting monoclonal antibody, to protect against RSV-associated lower respiratory tract infection among all infants aged <8 months born during or entering their first RSV season. Following licensure, nirsevimab effectiveness has been demonstrated against RSV-associated infant hospitalization, but evidence regarding effectiveness against RSV-associated critical illness is limited. In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure (illness requiring continuous positive airway pressure, bilevel positive airway pressure, or invasive mechanical ventilation) after hospital admission was evaluated during December 1, 2024−April 15, 2025. Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset. Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint. These estimates support the recommendation for use of nirsevimab as a prevention strategy to protect infants against severe outcomes from RSV infection.
                            }}
                        Question:
                            {{
                                In this case-control study of infants, what analytical approach would allow investigators to assess whether protection changes over the course of the season while preserving the validity of the comparison between groups?
                            }}
                        Options:
                            {{
                                [
                                    {{
                                        "Index": "0",
                                        "Option": "Apply Kaplan-Meier survival analysis to estimate the probability of remaining RSV-free over time since nirsevimab administration"
                                    }},
                                    {{
                                        "Index": "1",
                                        "Option": "Stratify by time since nirsevimab receipt and compare odds ratios across strata to detect any change in protective effect"
                                    }},
                                    {{
                                        "Index": "2",
                                        "Option": "Stratify by calendar month of admission and compare vaccination prevalence between cases and controls to detect temporal variation in protection"
                                    }},
                                    {{
                                        "Index": "3",
                                        "Option": "Calculate incidence rates of RSV illness among nirsevimab recipients at successive intervals since receipt to measure changes in protective effect"                                    }},
                                    }},
                                    {{
                                        "Index": "4",
                                        "Option": "Match cases and controls on time since nirsevimab receipt to control confounding while comparing RSV outcomes across time intervals"
                                    }}
                                ]
                            }}
                    """
        exp_output = """
                        {
                        "result": [
                                    {
                                        "Index": "0",
                                        "Option": "Apply Kaplan-Meier survival analysis to estimate the probability of remaining RSV-free over time since nirsevimab administration",
                                        "Category": "Incorrect",
                                        "Evidence": [
                                            "In a 27-hospital case-control investigation, nirsevimab effectiveness against both RSV-associated infant ICU admission and acute respiratory failure (illness requiring continuous positive airway pressure, bilevel positive airway pressure, or invasive mechanical ventilation) after hospital admission was evaluated during December 1, 2024−April 15, 2025.",
                                            "Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset."
                                            ],
                                        "Rationale": "Kaplan Meier methods require a cohort with person time and follow up from a defined start point to estimate a survival function. This study is explicitly case control, selecting ICU admitted infants as cases or controls based on RSV test results, which does not provide the population at risk or the time at risk needed to estimate the probability of remaining RSV free."
                                    },
                                    {
                                        "Index": "1",
                                        "Option": "Stratify by time since nirsevimab receipt and compare odds ratios across strata to detect any change in protective effect",
                                        "Category": "Correct",
                                        "Evidence": [
                                            "In a 27-hospital case-control investigation",
                                            "had received nirsevimab ≥7 days before symptom onset.",
                                            "Nirsevimab was 80% effective (95% CI = 70%-86%) against RSV-associated ICU admission and 83% effective (95% CI = 74%-90%) against acute respiratory failure when received a median of 52 days (IQR = 32-89 days) and 50 days (IQR = 32-86 days) before onset for each respective endpoint."
                                            ],
                                        "Rationale": "A standard way to assess whether effectiveness wanes is to treat time since receipt as an effect modifier, for example by stratifying time since nirsevimab receipt and estimating a separate odds ratio within each stratum, then comparing those odds ratios across strata. This preserves the internal validity of the case control comparison within each time since receipt category while directly testing for changing protection over time."
                                    },
                                    {
                                        "Index": "2",
                                        "Option": "Stratify by calendar month of admission and compare vaccination prevalence between cases and controls to detect temporal variation in protection",
                                        "Category": "Incorrect",
                                        "Evidence": [
                                            "during December 1, 2024−April 15, 2025.",
                                            "Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset."
                                            ],
                                        "Rationale": "Stratifying by calendar time can be useful to control or explore seasonal changes in exposure risk and product uptake, but simply comparing nirsevimab prevalence between cases and controls within months is not a valid estimate of protection unless it is translated into an odds ratio based effectiveness measure within each stratum. Also, calendar month stratification primarily addresses time in season, not waning by time since receipt, which is the more direct approach to whether protection changes after administration."
                                    },
                                    {
                                        "Index": "3",
                                        "Option": "Calculate incidence rates of RSV illness among nirsevimab recipients at successive intervals since receipt to measure changes in protective effect",
                                        "Category": "Incorrect",
                                        "Evidence": [
                                            "In a 27-hospital case-control investigation",
                                            "Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms"
                                            ],
                                        "Rationale": "Incidence rates require denominators, specifically person time or counts of individuals at risk in each interval since receipt. This study design samples cases and controls among ICU admissions rather than following a defined cohort over time, so it cannot directly compute incidence rates among recipients in successive intervals."
                                    },
                                    {
                                        "Index": "4",
                                        "Option": "Match cases and controls on time since nirsevimab receipt to control confounding while comparing RSV outcomes across time intervals",
                                        "Category": "Incorrect",
                                        "Evidence": [
                                            "Among 457 case-patients who received a positive RSV test result and 302 control patients who received a negative RSV test result admitted to an ICU with respiratory symptoms, 14% and 45%, respectively, had received nirsevimab ≥7 days before symptom onset."
                                            ],
                                        "Rationale": "Matching on time since nirsevimab receipt is not generally feasible or appropriate here because many infants are unexposed and therefore do not have a time since receipt value to match on. In addition, if you match tightly on time since receipt, you remove the very variation in time since receipt that you need to evaluate whether protection changes across that time scale, leading to overmatching and reduced ability to assess waning."
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
        if model_name == "mistralai/Mistral-7B-Instruct-v0.3":
            messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": usr_prompt},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": exp_prompt},
                {"role": "assistant", "content": exp_output},
                {"role": "user", "content": usr_prompt},
            ]
        
        
        prompt_list.append(messages)
    return prompt_list





def question_entity_query_generation_prompt(input_para_list, questions, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_question = questions[current_idx]["Question"]
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to extract one specific epidemiological entity from the input question. This entity will later be replaced with a descriptive phrase to increase question difficulty.

                            Follow these steps exactly:
                            Step 1: Read the question and identify all entities explicitly mentioned.
                            Step 2: Classify each entity into one of these categories in order of priority:
                                - Priority 1: Specific diseases, pathogens, or syndromes
                                - Priority 2: Specific interventions, vaccines, drugs, or treatments
                                - Priority 3: Vectors, reservoirs, or transmission-related entities
                                - Priority 4: Specific population groups with epidemiological significance
                                - Priority 5: Study designs or epidemiological methods
                            Step 3: Identify which entity is most central to the question's reasoning. This is the entity that the question's logic depends on most heavily.
                            Step 4: Select the entity to extract. Start with the highest-priority entity, but override this choice if a lower-priority entity is substantially more central to the question's reasoning. For example, if the question is primarily about methodology, the study design entity may be more valuable to replace even if higher-priority entities exist.
                            Step 5: Verify the selected entity is specific and searchable. It should be a proper name or technical term that can retrieve a clear definition.
                            Step 6: Output only one entity copied exactly from the question. Include a rationale explaining your choice, especially if you overrode the priority ranking.

                            {self_check}
                        """
        exp_prompt = f"""
                        Question: 
                            {{
                                In this case-control study of infants, what analytical approach would allow investigators to assess whether protection changes over the course of the season while preserving the validity of the comparison between groups?
                            }}
                    """
        exp_output = """
                        {
                            "entity": "case-control study",
                            "rationale": "Although 'infants' (Priority 4) ranks higher than 'case-control study' (Priority 5) by strict priority, the study design is far more central to the question's logic. The question asks about analytical approaches appropriate for assessing temporal changes in protection while maintaining valid group comparisons—this reasoning depends entirely on understanding case-control methodology. Replacing 'case-control study' with a descriptive phrase would significantly increase difficulty, requiring test-takers to first identify the study design before selecting appropriate analytical strategies."
                        }
                    """
                            
        usr_prompt = f""" 
                    Question:  
                        {{ 
                            {current_question} 
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




def question_reconstruction_prompt(searched_snippet_list, questions, replaced_entity, err=None):
    prompt_list = []
    for input_idx, searched_snippet in enumerate(searched_snippet_list):
        current_idx = searched_snippet[0]
        current_entity = searched_snippet[1]["Entity"]
        current_searched_snippet = "\n".join(searched_snippet[1]["Snippet"])
        current_question = questions[current_idx]["Question"]
        
        if current_idx in replaced_entity.keys():
            words_to_avoid = replaced_entity[current_idx]
        else:
            words_to_avoid = None
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            You will be given a question, an entity to replace, and a list of snippets containing information about that entity.
                            Your task is to replace the explicit entity name with a descriptive phrase that requires epidemiological knowledge to interpret. This increases question difficulty by forcing readers to first identify what entity is being described before they can answer the question.

                            Follow these steps exactly:
                            Step 1: Read the question, the entity, and all snippets carefully.
                            Step 2: From the snippets, select the most distinctive and specific details that characterize this entity. Prefer scientific, mechanistic, or epidemiologically significant details over general descriptions. Do not simply summarize all snippets.
                            Step 3: Combine the selected details into one concise descriptive phrase that can replace the entity as a noun phrase. The description should be specific enough to identify the entity for someone with domain knowledge, but should not directly name it.
                            Step 4: Replace the entity in the original question with the descriptive phrase. Do not incorporate other words from the original question into your descriptive phrase. The description should only capture the entity itself, not its modifiers or context from the original question. Do not use the entity's name, its abbreviations, synonyms, or any words that share the same root.{" Also avoid using any words listed in the Words to Avoid section." if words_to_avoid else ""}
                            Step 5: Revise the sentence structure for natural flow. After inserting the descriptive phrase, read the full sentence aloud. If the phrase interrupts the sentence awkwardly or creates a clunky structure, adjust word order, move modifiers, or rephrase for smoother reading. Consider placing longer descriptive phrases at the beginning or end of clauses rather than in the middle. Do not change what the question is asking or add information beyond what is in the snippets.
                            Step 6: Perform a final check. Verify the rewritten question is grammatically correct, the descriptive phrase flows naturally within the sentence without awkward interruptions, and all descriptive content comes from the snippets.
                            Step 7: Output the result with four fields: Replaced_Entity is the entity that was replaced, Snippet_Summary is the descriptive phrase you created, New_Question is the complete rewritten question, and Rationale explains what details you selected from the snippets and why they can uniquely identify the entity.
                            
                            {self_check}
                        """
        exp_prompt = f"""
                        Question: 
                            {{
                                In this case-control study of infants, what analytical approach would allow investigators to assess whether protection changes over the course of the season while preserving the validity of the comparison between groups?
                             }}
                        Entity To Be Replaced:
                            {{
                                case-control study
                            }}
                        Snippet List: 
                            {{
                                A study that compares patients who have a disease or outcome of interest (cases) with patients who do not have the disease or outcome (controls), and looks back retrospectively to compare how frequently the exposure to a risk factor is present in each group to determine the relationship between the risk factor and the ...
                                A case-control study is a type of observational study commonly used to look at factors associated with diseases or outcomes.
                                In a case-control study the prevalence of exposure to a potential risk factor(s) is compared between cases and controls. If the prevalence of exposure is more ...
                                A case-control study is an experimental design that compares a group of participants possessing a condition of interest to a very similar group lacking that ...
                                Case-control study designs are used to estimate the risk for a disease from a specific risk factor. The estimate is the odds ratio, which is a good estimate of ...
                                A study that compares two groups of people: those with the disease or condition under study (cases) and a very similar group of people who do not have the ...
                            }}
                    """
        exp_output = f"""
                        {{ 
                            "Replaced_Entity": "case-control study", 
                            "Snippet_Summary": "retrospective observational design that compares participants by disease status and estimates risk using odds ratios", 
                            "New_Question": "In this retrospective observational design that compares infants by disease status and estimates risk using odds ratios, what analytical approach would allow investigators to assess whether protection changes over the course of the season while preserving the validity of the comparison between groups?",
                            "Rationale": "I selected three key epidemiological characteristics from the snippets that uniquely identify this study design: (1) the retrospective nature ('looks back retrospectively'), (2) the fundamental comparison structure based on disease/outcome status ('compares patients who have a disease...with patients who do not'), and (3) the odds ratio as the measure of association ('the estimate is the odds ratio'). These details are mechanistically specific—cohort studies are prospective and use relative risk, cross-sectional studies examine prevalence at one time point, and randomized trials are interventional. Only this particular observational design combines retrospective temporal direction, selection by outcome status, and odds ratio estimation, making the combination sufficient for an epidemiologist to identify the design without naming it directly."
                        }}
        """
        usr_prompt = f""" 
                    Question:  
                        {{ 
                            {current_question} 
                        }}
                    Entity:
                        {{
                            {current_entity}
                        }}
                    Snippet List: 
                        {{
                            {current_searched_snippet}
                        }}
                    {f"Words to Avoid: {{ {', '.join(words_to_avoid)} }}" if words_to_avoid else ""}
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
                                You are an expert in epidemiology. Answer the question based on the provided passage and your epidemiological knowledge.
                                The question is multi-choice style. There may be one correct answer, multiple correct answers, or no correct answer. Select all correct answers. Some options may require combining passage evidence with epidemiological principles.
                                
                                Output a JSON object with a single key "results" whose value is a list of index strings for all correct options. If no option is correct, the list should be empty.
                                
                                {self_check}
                            """
        else:
            system_prompt = f"""
                                You are an expert in epidemiology. Answer the question based on the provided passage and your epidemiological knowledge.
                                The question is multi-choice style. There may be one correct answer, multiple correct answers, or no correct answer. You must evaluate each option independently.

                                Follow these steps:
                                Step 1: Read the passage and question carefully.
                                Step 2: For each option, reason through whether it correctly answers the question. Some options require multi-step reasoning that combines evidence from the passage with epidemiological principles.
                                Step 3: A correct option must be supported by passage evidence and represent valid epidemiological reasoning. An incorrect option may contain methodological errors, misinterpret the evidence, or require information not available in the passage.
                                Step 4: Decide whether each option is correct or incorrect.
                                Step 5: Collect the indices of all correct options. If no option is correct, return an empty list.

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
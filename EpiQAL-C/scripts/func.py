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
            discussion_string = ""
            for key in content.keys():
                section = content[key].strip()#.replace("\n", "")
                if "discussion" in key.lower() or "conclusion" in key.lower():
                    discussion_string = discussion_string + section + "\n"
                else:
                    content_string = content_string + section + "\n" #"---" + key + "---\n" + section
            
            #print(content_string)
            
            
            #print(content_string)
            
            inputs.append({"idx": str(id), "inputs": content_string, "discussion": discussion_string, "target": target})
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




def correct_option_generation_prompt(input_para_list, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_input = input_para["inputs"]
        current_discussion = input_para["discussion"]
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""    
                            Imagine you are an expert in epidemiology.
                            Your task is to extract one conclusion from the provided Discussion section that will serve as the Correct Option for a reasoning test. Readers will see only the Passage Body and must identify which conclusion can be derived from it.

                            Follow these steps exactly:
                            Step 1: Read the Passage Body and the Discussion section carefully. The Passage Body contains all sections of the paper except the Discussion.
                            Step 2: In the Discussion section, identify candidate sentences that state a conclusion. Good candidates are sentences that interpret findings, draw inferences, or state implications.
                            Step 3: Apply the novelty requirement. The conclusion must not be explicitly stated anywhere in the Passage Body. Reject candidates where the same statement appears in the Results or other sections.
                            Step 4: Apply the derivability requirement. The conclusion must be logically derivable from evidence in the Passage Body by applying general epidemiological principles.
                                Reject conclusions that require:
                                - Results from other studies cited in the Discussion but not described in the Passage Body
                                - Specific facts about diseases, treatments, or populations not mentioned in the Passage Body
                                - Comparisons to external benchmarks or statistics not provided in the Passage Body

                                Allow conclusions that require:
                                - General epidemiological reasoning such as interpreting odds ratios or understanding confounding
                                - Standard methodological knowledge such as recognizing limitations of study designs
                                - Logical inference from the data presented
                            Step 5: Apply the complexity requirement. Prefer conclusions that require integrating multiple pieces of evidence from the Passage Body, require applying epidemiological principles to interpret the data, and represent a key finding rather than a minor observation.
                            Step 6: Apply exclusion criteria. Reject conclusions that are direct numerical summaries already stated in the Results, describe study limitations or future research directions, are speculative statements without clear evidential basis in the Passage Body, or are generic statements applicable to any similar study.
                            Step 7: From the remaining candidates, select the single best conclusion.
                            Step 8: Verify the selected conclusion is complete and self-contained. If it contains pronouns or references requiring prior context, minimally rephrase to make it standalone. Do not change the meaning.
                            Step 9: Identify supporting evidence from the Passage Body. Quote all sentences necessary to derive this conclusion.
                            Step 10: Output your result as a JSON object with a single key "result" whose value is a list containing one dictionary. The dictionary should have five fields. Category is always Correct Option. Option is the conclusion, minimally rephrased if needed. Discussion_Source is the original Discussion sentence quoted exactly verbatim. Evidence is a list of verbatim quotes from the Passage Body. Rationale explains how the conclusion is derived from the evidence. Output only the JSON object with no additional text before or after it.

                            {self_check}
                    """
        exp_prompt = f"""
                        Passage Body:
                            {{
                                Respiratory syncytial virus, or RSV, is a common respiratory virus that infects the nose, throat, and lungs. RSV symptoms make it difficult to distinguish it from the common cold or other respiratory viruses (like the flu or COVID-19). RSV spreads in the fall and winter along with other respiratory viruses. It usually peaks in December and January.
                                RSV does not usually cause severe illness in healthy adults and children. However, some people with RSV infection, especially infants younger than 6 months of age and adults who are older or have certain risk factors, can become very sick and may need to be hospitalized.
                                RSV can also cause more severe illness such as bronchiolitis (inflammation of the small airways in the lungs) and pneumonia (infection of the lungs). It is the most common cause of bronchiolitis and pneumonia in children younger than 1 year of age.
                                In the most severe cases, a person may require:
                                - Additional oxygen,
                                - IV fluids if they can't drink enough to stay hydrated, or
                                - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).
                                In most of these cases, hospitalization lasts only a few days.
                            }}
                        Discussion:
                            {{
                                The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods.
                                
                                Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions.
                                
                                The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units.
                                
                                Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.
                            }}
                    """
        exp_output = """
                        {
                            "result": [
                                {
                                "Category": "Correct Option",
                                "Option": "Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.",
                                "Discussion_Source": "Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.",
                                "Evidence": [
                                    "In the most severe cases, a person may require: - Additional oxygen, - IV fluids if they can't drink enough to stay hydrated, or - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).",
                                    "In most of these cases, hospitalization lasts only a few days."
                                ],
                                "Rationale": "The Passage Body establishes two key facts: (1) the most severe RSV cases may require intensive interventions including mechanical ventilation, and (2) despite this severity, hospitalization typically lasts only a few days. By applying standard clinical epidemiological reasoning, one can infer that if patients requiring the most intensive supportive measures (mechanical ventilation) still have short hospital stays, this indicates that supportive care is effective and most patients achieve favorable short-term outcomes. This conclusion integrates the severity of potential interventions with the unexpectedly brief hospitalization duration to derive a prognostic inference not explicitly stated in the Passage Body."
                                }
                            ]
                        }
                    """
        usr_prompt = f"""
                    Passage Body: 
                        {{
                            {current_input}
                        }}
                    Discussion:
                        {{
                            {current_discussion}
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



def question_generation_prompt(input_para_list, correct_options, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_correct_options = json.dumps(correct_options[current_idx][0])#["Option"]
        current_input = input_para["inputs"]
        current_discussion = input_para["discussion"]
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to generate a question stem for a single-choice reasoning test. The question must be answerable only by the provided Correct Option, which is a conclusion derived from the Passage Body through epidemiological reasoning.
                            You will be given the Passage Body and the Correct Option output from the previous step, which includes the conclusion, its source in the Discussion, the supporting evidence from the Passage Body, and the rationale explaining the reasoning chain.

                            Follow these steps exactly:
                            Step 1: Read the Passage Body and all fields in the Correct Option output carefully. Pay special attention to the Evidence and Rationale fields, which describe what evidence supports the conclusion and how the reasoning chain works.
                            Step 2: Use the provided Evidence as the basis for your question. These are the facts that, when combined with epidemiological reasoning described in the Rationale, lead to the conclusion in the Option field.
                            Step 3: Design a question that requires readers to integrate the evidence pieces and apply the same epidemiological reasoning to arrive at the Correct Option. The question should set up a reasoning task without revealing the answer direction.
                            Step 4: Apply difficulty requirements. A good question should require integrating multiple pieces of evidence rather than relying on a single fact, require applying epidemiological principles to interpret the data, and not be answerable by simply locating a sentence in the Passage Body.
                            Step 5: Apply concealment requirements. The question stem must not use any words or phrases that appear in the Option field, must not use synonyms or paraphrases that directly hint at the conclusion, must not indicate the type of answer expected such as prognosis, risk, or recommendation, must not reveal which evidence pieces are relevant, and must not describe or summarize the content of specific evidence pieces. The question should set up the reasoning task without guiding readers to particular parts of the passage.
                            Step 6: Verify the question is specific to this passage. A reader should not be able to answer correctly using general knowledge alone without reading the Passage Body.
                            Step 7: Output your result with three fields. Question is the question stem without options. Evidence is copied directly from the Correct Option output. Rationale explains why this question requires the reasoning chain to answer correctly.

                            {self_check}
                        """
        exp_prompt = f"""
                        Passage Body:
                            {{
                                Respiratory syncytial virus, or RSV, is a common respiratory virus that infects the nose, throat, and lungs. RSV symptoms make it difficult to distinguish it from the common cold or other respiratory viruses (like the flu or COVID-19). RSV spreads in the fall and winter along with other respiratory viruses. It usually peaks in December and January.
                                RSV does not usually cause severe illness in healthy adults and children. However, some people with RSV infection, especially infants younger than 6 months of age and adults who are older or have certain risk factors, can become very sick and may need to be hospitalized.
                                RSV can also cause more severe illness such as bronchiolitis (inflammation of the small airways in the lungs) and pneumonia (infection of the lungs). It is the most common cause of bronchiolitis and pneumonia in children younger than 1 year of age.
                                In the most severe cases, a person may require:
                                - Additional oxygen,
                                - IV fluids if they can't drink enough to stay hydrated, or
                                - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).
                                In most of these cases, hospitalization lasts only a few days.
                            }}
                        Discussion:
                            {{
                                The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods.
                                
                                Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions.
                                
                                The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units.
                                
                                Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.
                            }}
                        Correct Option:
                            {{
                                "Category": "Correct Option",
                                "Option": "Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.",
                                "Discussion_Source": "Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.",
                                "Evidence": [
                                    "In the most severe cases, a person may require: - Additional oxygen, - IV fluids if they can't drink enough to stay hydrated, or - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).",
                                    "In most of these cases, hospitalization lasts only a few days."
                                ],
                                "Rationale": "The Passage Body establishes two key facts: (1) the most severe RSV cases may require intensive interventions including mechanical ventilation, and (2) despite this severity, hospitalization typically lasts only a few days. By applying standard clinical epidemiological reasoning, one can infer that if patients requiring the most intensive supportive measures (mechanical ventilation) still have short hospital stays, this indicates that supportive care is effective and most patients achieve favorable short-term outcomes. This conclusion integrates the severity of potential interventions with the unexpectedly brief hospitalization duration to derive a prognostic inference not explicitly stated in the Passage Body."
                            }}
                    """
        exp_output = f"""
                        {{
                            "Question": "Based on the clinical information about hospitalized RSV patients presented in the passage, what epidemiological inference can be drawn?",
                            "Evidence": [
                                    "In the most severe cases, a person may require: - Additional oxygen, - IV fluids if they can't drink enough to stay hydrated, or - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).",
                                    "In most of these cases, hospitalization lasts only a few days."
                                ],
                            "Rationale": "This question requires readers to identify and integrate two distinct clinical observations: (1) that the most critical RSV cases may necessitate intensive interventions including mechanical ventilation, and (2) that despite this severity, hospital stays are typically brief. The question prompts epidemiological reasoning by asking what can be inferred—requiring readers to recognize that the juxtaposition of high-intensity intervention with short hospitalization duration implies something about care effectiveness and patient trajectories. The question avoids key terms from the correct option (such as 'favorable,' 'outcomes,' 'supportive care,' or 'short-term') and does not indicate what type of conclusion is expected. It is passage-specific because answering correctly requires engaging with the particular clinical data presented about RSV hospitalization rather than relying on general medical knowledge."
                        }}
                    """
        usr_prompt = f"""
                    Passage Body: 
                        {{
                            {current_input}
                        }}
                    Discussion:
                        {{
                            {current_discussion}
                        }}
                    Correct Option:
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





def distractor_generation_prompt(input_para_list, questions, correct_options, err=None):
    prompt_list = []
    for input_idx, input_para in enumerate(input_para_list):
        current_idx = input_para['idx']
        current_question = json.dumps(questions[current_idx])   #["Question"]
        current_correct_options = correct_options[current_idx][0]["Option"]
        current_input = input_para["inputs"]
        current_discussion = input_para["discussion"]
        
        if err is not None:   
            self_check = f"""
                Your last response contained several issues. Ensure they are not repeated in the next response. 
                {err[input_idx]}
            """
        else:
            self_check = ""

        system_prompt = f"""
                            Imagine you are an expert in epidemiology.
                            Your task is to generate distractors for a reasoning test. The test presents readers with the Passage Body and asks them to identify which conclusion can be derived from it. Distractors should be plausible-sounding conclusions that cannot actually be derived from the Passage Body alone.
                            You will be given the Passage Body, the Discussion, the question, and the correct option.

                            Follow these steps exactly:
                            Step 1: Read all provided materials carefully. Understand why the correct option can be derived from the Passage Body. Pay attention to what aspect of the passage the question focuses on.
                            Step 2: Identify candidate distractor statements from the Discussion section. Do not use the correct option. Good distractors fall into one of these categories:
                                - External dependency: Conclusions that require information from other studies cited in the Discussion but not described in the Passage Body.
                                - Speculation: Statements about future research directions, untested hypotheses, or possibilities using hedging language such as may, might, or could without clear evidential basis in the Passage Body.
                                - Limitations: Statements about study limitations or methodological caveats.
                                - Background only: Statements that merely restate general background knowledge without adding interpretation specific to this study.
                                - Causal reversal: A statement created by reversing or misinterpreting the cause-effect relationship implied in the correct option or the Passage Body evidence.
                            Step 3: Verify each candidate meets two requirements. First, it cannot answer the question. If a candidate could be derived from the Passage Body through valid reasoning, discard it. Second, it should be relevant to what the question asks. Prefer distractors that address similar aspects as the question and the correct option rather than unrelated topics.
                            Step 4: Construct each distractor. For external dependency, speculation, limitations, and background only categories, locate the source sentence in the Discussion and minimally rephrase if needed to be self-contained. For causal reversal, identify the original causal direction, then write a sentence that subtly misinterprets the relationship while keeping the same entities and context. The reversal must not directly contradict explicit facts stated in the Passage Body. A good causal reversal twists the interpretation or inference rather than denying stated facts.
                            Step 5: Ensure each distractor is complete, self-contained, and grammatically parallel to the correct option in style and length.
                            Step 6: Prioritize distractors that are relevant to the question focus. If the Discussion contains limited relevant content, generate multiple causal reversal variants that misinterpret the evidence in different ways. Aim for 3 to 4 distractors total. Ensure they are diverse and do not overlap in their reasoning flaws.
                            Step 7: Output your result as a JSON object with a single key "result" whose value is a list of distractor dictionaries. Each dictionary should have five fields: Category is always Distractor, Option is the distractor sentence, Discussion_Source is the original sentence from the Discussion quoted verbatim or the source used for causal reversal, Evidence is a list of verbatim quotes showing why this cannot be derived from the Passage Body alone, and Rationale explains why this is a valid distractor. Output only the JSON object with no additional text.

                            {self_check}
                    """
        exp_prompt = f"""
                        Passage Body:
                            {{
                                Respiratory syncytial virus, or RSV, is a common respiratory virus that infects the nose, throat, and lungs. RSV symptoms make it difficult to distinguish it from the common cold or other respiratory viruses (like the flu or COVID-19). RSV spreads in the fall and winter along with other respiratory viruses. It usually peaks in December and January.
                                RSV does not usually cause severe illness in healthy adults and children. However, some people with RSV infection, especially infants younger than 6 months of age and adults who are older or have certain risk factors, can become very sick and may need to be hospitalized.
                                RSV can also cause more severe illness such as bronchiolitis (inflammation of the small airways in the lungs) and pneumonia (infection of the lungs). It is the most common cause of bronchiolitis and pneumonia in children younger than 1 year of age.
                                In the most severe cases, a person may require:
                                - Additional oxygen,
                                - IV fluids if they can't drink enough to stay hydrated, or
                                - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).
                                In most of these cases, hospitalization lasts only a few days.
                            }}
                        Discussion:
                            {{
                                The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods.
                                
                                Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions.
                                
                                The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units.
                                
                                Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.
                            }}
                        Question:
                            {{
                                "Question": "Based on the clinical information about hospitalized RSV patients presented in the passage, what epidemiological inference can be drawn?",
                                "Evidence": [
                                        "In the most severe cases, a person may require: - Additional oxygen, - IV fluids if they can't drink enough to stay hydrated, or - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).",
                                        "In most of these cases, hospitalization lasts only a few days."
                                    ],
                                "Rationale": "This question requires readers to identify and integrate two distinct clinical observations: (1) that the most critical RSV cases may necessitate intensive interventions including mechanical ventilation, and (2) that despite this severity, hospital stays are typically brief. The question prompts epidemiological reasoning by asking what can be inferred—requiring readers to recognize that the juxtaposition of high-intensity intervention with short hospitalization duration implies something about care effectiveness and patient trajectories. The question avoids key terms from the correct option (such as 'favorable,' 'outcomes,' 'supportive care,' or 'short-term') and does not indicate what type of conclusion is expected. It is passage-specific because answering correctly requires engaging with the particular clinical data presented about RSV hospitalization rather than relying on general medical knowledge."
                            }}
                        Correct Option:
                            {{
                                Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.
                            }}
                    """
        exp_output = """
                        {
                            "result": [
                                    {
                                        "Category": "Distractor",
                                        "Option": "The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods.",
                                        "Discussion_Source": "The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods.",
                                        "Evidence": [
                                                "RSV symptoms make it difficult to distinguish it from the common cold or other respiratory viruses (like the flu or COVID-19)."
                                            ],
                                        "Rationale": "While the Passage Body mentions that RSV symptoms overlap with other respiratory viruses, the recommendation for laboratory confirmation is a policy inference not derivable from the passage. Additionally, the question specifically asks about hospitalized patients, whereas this statement addresses diagnostic approaches prior to hospitalization."
                                    },
                                    {
                                        "Category": "Distractor",
                                        "Option": "Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions.",
                                        "Discussion_Source": "Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions.",
                                        "Evidence": [
                                                "It is the most common cause of bronchiolitis and pneumonia in children younger than 1 year of age."
                                            ],
                                        "Rationale": "The Passage Body states RSV is the most common cause of these conditions in young children, but the conclusion about priority populations for preventive interventions is a public health policy recommendation that cannot be derived from clinical facts alone. The question asks about epidemiological inferences regarding hospitalized patients, not prevention priorities."
                                    },
                                    {
                                        "Category": "Distractor",
                                        "Option": "The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units.",
                                        "Discussion_Source": "The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units.",
                                        "Evidence": [
                                                "RSV spreads in the fall and winter along with other respiratory viruses.",
                                                "In the most severe cases, a person may require: - Additional oxygen, - IV fluids if they can't drink enough to stay hydrated, or - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe)."
                                            ],
                                        "Rationale": "The Passage Body mentions seasonal timing and severe case interventions but provides no data on healthcare system burden or capacity. The speculation about compounding effects on pediatric ICUs uses hedging language ('may') and requires external epidemiological data not present in the passage."
                                    },
                                    {
                                        "Category": "Distractor",
                                        "Option": "The provision of mechanical ventilation in severe RSV cases indicates that short hospitalization duration is primarily determined by rapid escalation to intensive interventions rather than by patient response to supportive care.",
                                        "Discussion_Source": "Causal reversal of correct option: Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.",
                                            "Evidence": [
                                                "In the most severe cases, a person may require: - Additional oxygen, - IV fluids if they can't drink enough to stay hydrated, or - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).",
                                                "In most of these cases, hospitalization lasts only a few days."
                                            ],
                                        "Rationale": "This distractor reverses the causal inference in the correct option. The correct interpretation links short hospitalization to favorable outcomes from supportive care, whereas this distractor attributes short stays to speed of intervention escalation. The Passage Body provides no information about access patterns, escalation timing, or what determines hospitalization duration, making this inference unsupported."
                                    }
                                ]
                        }
                        """
        usr_prompt = f"""
                    Passage Body: 
                        {{
                            {current_input}
                        }}
                    Discussion:
                        {{
                            {current_discussion}
                        }}
                    Question: 
                        {{
                            {current_question}
                        }}
                    Correct Option:
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
    example_options = {"Correct Option":
                            [
                                {
                                "Category": "Correct Option",
                                "Option": "Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.",
                                "Discussion_Source": "Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.",
                                "Evidence": [
                                    "In the most severe cases, a person may require: - Additional oxygen, - IV fluids if they can't drink enough to stay hydrated, or - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).",
                                    "In most of these cases, hospitalization lasts only a few days."
                                ],
                                "Rationale": "The Passage Body establishes two key facts: (1) the most severe RSV cases may require intensive interventions including mechanical ventilation, and (2) despite this severity, hospitalization typically lasts only a few days. By applying standard clinical epidemiological reasoning, one can infer that if patients requiring the most intensive supportive measures (mechanical ventilation) still have short hospital stays, this indicates that supportive care is effective and most patients achieve favorable short-term outcomes. This conclusion integrates the severity of potential interventions with the unexpectedly brief hospitalization duration to derive a prognostic inference not explicitly stated in the Passage Body."
                                }
                            ],
                        "Distractor": 
                            [
                                {
                                    "Category": "Distractor",
                                    "Option": "The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods.",
                                    "Discussion_Source": "The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods.",
                                    "Evidence": [
                                            "RSV symptoms make it difficult to distinguish it from the common cold or other respiratory viruses (like the flu or COVID-19)."
                                        ],
                                    "Rationale": "While the Passage Body mentions that RSV symptoms overlap with other respiratory viruses, the recommendation for laboratory confirmation is a policy inference not derivable from the passage. Additionally, the question specifically asks about hospitalized patients, whereas this statement addresses diagnostic approaches prior to hospitalization."
                                },
                                {
                                    "Category": "Distractor",
                                    "Option": "Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions.",
                                    "Discussion_Source": "Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions.",
                                    "Evidence": [
                                            "It is the most common cause of bronchiolitis and pneumonia in children younger than 1 year of age."
                                        ],
                                    "Rationale": "The Passage Body states RSV is the most common cause of these conditions in young children, but the conclusion about priority populations for preventive interventions is a public health policy recommendation that cannot be derived from clinical facts alone. The question asks about epidemiological inferences regarding hospitalized patients, not prevention priorities."
                                },
                                {
                                    "Category": "Distractor",
                                    "Option": "The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units.",
                                    "Discussion_Source": "The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units.",
                                    "Evidence": [
                                            "RSV spreads in the fall and winter along with other respiratory viruses.",
                                            "In the most severe cases, a person may require: - Additional oxygen, - IV fluids if they can't drink enough to stay hydrated, or - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe)."
                                        ],
                                    "Rationale": "The Passage Body mentions seasonal timing and severe case interventions but provides no data on healthcare system burden or capacity. The speculation about compounding effects on pediatric ICUs uses hedging language ('may') and requires external epidemiological data not present in the passage."
                                },
                                {
                                    "Category": "Distractor",
                                    "Option": "The provision of mechanical ventilation in severe RSV cases indicates that short hospitalization duration is primarily determined by rapid escalation to intensive interventions rather than by patient response to supportive care.",
                                    "Discussion_Source": "Causal reversal of correct option: Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.",
                                        "Evidence": [
                                            "In the most severe cases, a person may require: - Additional oxygen, - IV fluids if they can't drink enough to stay hydrated, or - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).",
                                            "In most of these cases, hospitalization lasts only a few days."
                                        ],
                                    "Rationale": "This distractor reverses the causal inference in the correct option. The correct interpretation links short hospitalization to favorable outcomes from supportive care, whereas this distractor attributes short stays to speed of intervention escalation. The Passage Body provides no information about access patterns, escalation timing, or what determines hospitalization duration, making this inference unsupported."
                                }
                            ]}
    
    example_outputs = {"Correct Option": 
                       {
                            "result": [
                                {
                                    "Category": "Correct Option",
                                    "Option": "Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.",
                                    "Coherence": "Yes",
                                    "Rationale": "All verification steps pass: (1) Both evidence quotes are verbatim from the Passage Body; (2) The option is not a direct retrieval from the Passage Body but rather a synthesized inference; (3) The option directly addresses the question about epidemiological inferences for hospitalized RSV patients; (4) The evidence pieces are directly relevant to the claims in the option; (5) The rationale logically connects severity of required interventions with short hospitalization duration to derive a prognostic inference; (6) The option can be logically derived from the Passage Body by integrating the facts about severe interventions and brief hospital stays to conclude favorable short-term outcomes with appropriate supportive care."
                                }
                            ]
                        },
                       "Distractor": 
                       {
                            "result": [
                                {
                                    "Category": "Distractor",
                                    "Option": "The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods.",
                                    "Coherence": "Yes",
                                    "Rationale": "Step 2: Evidence is verbatim from Passage Body. Step 3: Option adds policy recommendation not directly stated. Step 4: Option addresses diagnostic approaches rather than hospitalized patients as asked. Step 5: Evidence about symptom overlap is relevant to the premise. Step 6: Rationale correctly identifies the policy inference gap and question mismatch. Step 7: Valid distractor because the laboratory confirmation recommendation is a policy inference requiring external judgment not derivable from clinical facts alone, and it fails to address the specific question about hospitalized patients."
                                },
                                {
                                    "Category": "Distractor",
                                    "Option": "Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions.",
                                    "Coherence": "Yes",
                                    "Rationale": "Step 2: Evidence is verbatim from Passage Body. Step 3: Option adds policy conclusion about priority populations. Step 4: Option addresses prevention priorities rather than hospitalized patients. Step 5: Evidence about disease prevalence is relevant to the premise. Step 6: Rationale correctly explains the policy inference gap. Step 7: Valid distractor because determining priority populations for preventive interventions is a public health policy decision that cannot be derived from clinical prevalence data alone, and the question specifically asks about hospitalized patients."
                                },
                                {
                                    "Category": "Distractor",
                                    "Option": "The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units.",
                                    "Coherence": "Yes",
                                    "Rationale": "Step 2: Both evidence quotes are verbatim from Passage Body. Step 3: Option derives a novel conclusion about healthcare system burden. Step 4: Option addresses system-level implications tangentially related to the question. Step 5: Evidence about seasonality and severe cases is relevant to the topic but insufficient for the conclusion. Step 6: Rationale correctly identifies the lack of healthcare burden data. Step 7: Valid distractor because the inference about compounding healthcare system burden requires external epidemiological data on system capacity and utilization not present in the passage."
                                },
                                {
                                    "Category": "Distractor",
                                    "Option": "The provision of mechanical ventilation in severe RSV cases indicates that short hospitalization duration is primarily determined by rapid escalation to intensive interventions rather than by patient response to supportive care.",
                                    "Coherence": "Yes",
                                    "Rationale": "Step 2: Both evidence quotes are verbatim from Passage Body. Step 3: Option makes a causal inference not directly stated. Step 4: Option addresses hospitalized patients as required. Step 5: Evidence about severe cases and hospitalization duration is topically relevant. Step 6: Rationale correctly explains the causal reversal mechanism. Step 7: Valid distractor because it reverses the logical causal relationship and attributes short hospitalization to intervention speed rather than favorable response to supportive care; the Passage Body provides no information about what determines hospitalization duration, making this a misinterpretation of the evidence."
                                }
                            ]
                        }
                       }


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
                            Your task is to check whether some answers to a question are valid as {answer_type} based on the given Passage Body, evidence, and rationale. The Passage Body contains all sections of the paper except the Discussion. Evaluate only whether the option is valid under the provided materials, not whether it is factually correct in the real world.

                            Follow these steps exactly:
                            Step 1: Read the Passage Body, the question, all options, the evidence, and the rationales carefully.
                            Step 2: For each option, verify source validity. Check whether the Discussion_Source quote exists verbatim in the provided Discussion. Check whether every Evidence quote is copied verbatim from the provided Passage Body. If any quote is fabricated or misquoted, this check fails.
                            Step 3: Check that the option is not a direct retrieval. The option must not be a verbatim restatement or simple paraphrase of a single sentence from the Passage Body. The option should represent a derived conclusion.
                            Step 4: Check that the option addresses the question. The option must directly respond to what the question asks, not a related but different question.
                            Step 5: Verify that the evidence is relevant to the option. Check whether the cited evidence pieces are actually related to the conclusion stated in the option. If the evidence is irrelevant, this check fails.
                            Step 6: Verify that the rationale is logically sound. Check whether the reasoning described in the rationale correctly explains the relationship between the evidence and the option. If the rationale contains logical gaps or errors, this check fails.
                            Step 7: Check whether the option is valid as {answer_type} based on the Passage Body and question. {"Verify that the option can be logically derived from the Passage Body by applying general epidemiological reasoning. The derivation should require integrating evidence pieces and applying domain principles, not just restating facts. The option must not appear verbatim in the Passage Body. Reject if the option cannot be derived from the Passage Body alone or if it appears directly in the Passage Body." if answer_type == "Correct Option" else "Verify that the option cannot be derived from the Passage Body alone. A valid distractor should require external information, make unsupported speculative claims, or misinterpret the causal relationships in the evidence. Verify that the flaw described in the rationale actually exists. Reject if the option could actually be derived from the Passage Body through valid reasoning."}
                            Step 8: Make the final judgment. If Step 2 through Step 7 all pass, return Yes in Coherence. Otherwise return No and explain which step failed in Rationale.
                            Step 9: Format your output as a JSON object with a single key "result" whose value is a list of option dictionaries. Each dictionary should contain four fields: Category, Option, Coherence, and Rationale. Category should be "{answer_type}" for all options. Option is the option text being evaluated. Coherence is either "Yes" or "No". Rationale explains your judgment, including which step failed if Coherence is No. Output only the JSON object with no additional text before or after it.

                            {self_check}
                    """
        exp_prompt = f"""
                        Passage Body:
                            {{
                                Respiratory syncytial virus, or RSV, is a common respiratory virus that infects the nose, throat, and lungs. RSV symptoms make it difficult to distinguish it from the common cold or other respiratory viruses (like the flu or COVID-19). RSV spreads in the fall and winter along with other respiratory viruses. It usually peaks in December and January.
                                RSV does not usually cause severe illness in healthy adults and children. However, some people with RSV infection, especially infants younger than 6 months of age and adults who are older or have certain risk factors, can become very sick and may need to be hospitalized.
                                RSV can also cause more severe illness such as bronchiolitis (inflammation of the small airways in the lungs) and pneumonia (infection of the lungs). It is the most common cause of bronchiolitis and pneumonia in children younger than 1 year of age.
                                In the most severe cases, a person may require:
                                - Additional oxygen,
                                - IV fluids if they can't drink enough to stay hydrated, or
                                - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).
                                In most of these cases, hospitalization lasts only a few days.
                            }}
                        Discussion:
                            {{
                                The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods.
                                
                                Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions.
                                
                                The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units.
                                
                                Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.
                            }}
                        Question:
                            {{
                                Based on the clinical information about hospitalized RSV patients presented in the passage, what epidemiological inference can be drawn?
                            }}
                        Options:
                                {json.dumps(example_options[answer_type])}    
                    """
        exp_output = f"""
                        {json.dumps(example_outputs[answer_type])}
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
                            Your task is to read the Passage Body of an epidemiology paper and answer a multiple-choice style question. The Passage Body contains all sections except the Discussion. You must determine which conclusion can be derived from the Passage Body through epidemiological reasoning.

                            Follow these steps exactly:
                            Step 1: Read the Passage Body and the question carefully.
                            Step 2: Read all provided options. The number of correct answers is not predetermined. There may be one correct answer, multiple correct answers, or no correct answer at all. You must evaluate each option independently.
                            Step 3: For each option, reason through whether it can be logically derived from the Passage Body by applying epidemiological principles. A correct answer must be derivable from the evidence in the Passage Body, not merely plausible-sounding or requiring external information.
                            Step 4: Decide whether each option is Correct or Incorrect. You must output an analysis for every option provided, regardless of whether it is Correct or Incorrect.
                            Step 5: For each option, include evidence and a rationale explaining your reasoning. Evidence should be a list of exact verbatim quotes from the Passage Body that support your judgment. Include all quotes that are necessary to support your reasoning chain. For complex judgments that require integrating multiple facts, this will naturally involve multiple pieces of evidence.
                            Step 6: Format your output as a JSON object with a single key "result" whose value is a list of option dictionaries. Each dictionary should contain five fields: Index, Option, Category, Evidence, and Rationale. Index is the option's original index as a string. Option is the exact option text. Category is either "Correct" or "Incorrect". Evidence is a list of verbatim quotes from the Passage Body. Rationale explains your reasoning. Output only the JSON object with no additional text before or after it.
                            
                            {self_check}
                        """
        exp_prompt = f"""
                        Passage Body:
                            {{
                                Respiratory syncytial virus, or RSV, is a common respiratory virus that infects the nose, throat, and lungs. RSV symptoms make it difficult to distinguish it from the common cold or other respiratory viruses (like the flu or COVID-19). RSV spreads in the fall and winter along with other respiratory viruses. It usually peaks in December and January.
                                RSV does not usually cause severe illness in healthy adults and children. However, some people with RSV infection, especially infants younger than 6 months of age and adults who are older or have certain risk factors, can become very sick and may need to be hospitalized.
                                RSV can also cause more severe illness such as bronchiolitis (inflammation of the small airways in the lungs) and pneumonia (infection of the lungs). It is the most common cause of bronchiolitis and pneumonia in children younger than 1 year of age.
                                In the most severe cases, a person may require:
                                - Additional oxygen,
                                - IV fluids if they can't drink enough to stay hydrated, or
                                - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).
                                In most of these cases, hospitalization lasts only a few days.
                            }}
                        Question:
                            {{
                                Based on the clinical information about hospitalized RSV patients presented in the passage, what epidemiological inference can be drawn?
                            }}
                        Options:
                            {{
                                [
                                    {{
                                        "Index": "0",
                                        "Option": "The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods."
                                    }},
                                    {{
                                        "Index": "1",
                                        "Option": "Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions."
                                    }},
                                    {{
                                        "Index": "2",
                                        "Option": "The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units."
                                    }},
                                    {{
                                        "Index": "3",
                                        "Option": "Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes."
                                    }},
                                    {{
                                        "Index": "4",
                                        "Option": "The provision of mechanical ventilation in severe RSV cases indicates that short hospitalization duration is primarily determined by rapid escalation to intensive interventions rather than by patient response to supportive care."
                                    }}
                                ]
                            }}
                    """
        exp_output = """
                        {
                            "result": [
                                {
                                    "Index": "0",
                                    "Option": "The clinical overlap between RSV and other common respiratory pathogens suggests that symptom-based diagnosis alone is insufficient for accurate case identification, and laboratory confirmation should be considered during peak transmission periods.",
                                    "Category": "Correct",
                                    "Evidence": [
                                            "RSV symptoms make it difficult to distinguish it from the common cold or other respiratory viruses (like the flu or COVID-19).",
                                            "RSV spreads in the fall and winter along with other respiratory viruses. It usually peaks in December and January."
                                        ],
                                    "Rationale": "The passage explicitly states that RSV symptoms make it difficult to distinguish from other respiratory viruses. This clinical overlap directly supports the epidemiological inference that symptom-based diagnosis alone is insufficient for accurate case identification. The mention of peak transmission periods (December-January) and co-circulation with other respiratory viruses further supports the need for laboratory confirmation during these periods to properly attribute cases."
                                },
                                {
                                    "Index": "1",
                                    "Option": "Given that RSV is the leading cause of bronchiolitis and pneumonia in children under one year, this age group represents the highest priority population for preventive interventions.",
                                    "Category": "Correct",
                                    "Evidence": [
                                            "It is the most common cause of bronchiolitis and pneumonia in children younger than 1 year of age.",
                                            "some people with RSV infection, especially infants younger than 6 months of age and adults who are older or have certain risk factors, can become very sick and may need to be hospitalized."
                                        ],
                                    "Rationale": "The passage establishes that RSV is the most common cause of bronchiolitis and pneumonia in children under one year and that infants younger than 6 months are particularly vulnerable to severe illness requiring hospitalization. In epidemiological practice, populations bearing the highest disease burden from a pathogen are prioritized for preventive interventions. This inference follows directly from the evidence presented."
                                },
                                {
                                    "Index": "2",
                                    "Option": "The seasonal co-circulation of RSV with influenza and COVID-19 during winter months may compound healthcare system burden, particularly in pediatric intensive care units.",
                                    "Category": "Incorrect",
                                    "Evidence": [
                                            "RSV spreads in the fall and winter along with other respiratory viruses."
                                        ],
                                    "Rationale": "While the passage confirms seasonal co-circulation of RSV with other respiratory viruses during fall and winter, it does not provide any evidence regarding healthcare system burden or specifically mention pediatric intensive care units. The inference about compounding effects on healthcare systems and particular strain on pediatric ICUs requires external epidemiological knowledge not contained in the passage. This conclusion, while plausible, cannot be derived solely from the passage body."
                                },
                                {
                                    "Index": "3",
                                    "Option": "Although severe cases may require mechanical ventilation, the typically short hospitalization duration suggests that most patients who receive appropriate supportive care achieve favorable short-term outcomes.",
                                    "Category": "Correct",
                                    "Evidence": [
                                            "Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).",
                                            "In most of these cases, hospitalization lasts only a few days."
                                        ],
                                    "Rationale": "The passage explicitly states that even in severe cases requiring interventions including mechanical ventilation, hospitalization typically lasts only a few days. A short hospitalization duration in severe cases is an epidemiological indicator of favorable short-term outcomes with appropriate supportive care. This inference logically follows from the clinical information provided about hospitalization duration."
                                },
                                {
                                    "Index": "4",
                                    "Option": "The provision of mechanical ventilation in severe RSV cases indicates that short hospitalization duration is primarily determined by rapid escalation to intensive interventions rather than by patient response to supportive care.",
                                    "Category": "Incorrect",
                                    "Evidence": [
                                            "In the most severe cases, a person may require: - Additional oxygen, - IV fluids if they can't drink enough to stay hydrated, or - Intubation (have a breathing tube inserted through the mouth and down to the airway) with mechanical ventilation (a machine to help a person breathe).",
                                            "In most of these cases, hospitalization lasts only a few days."
                                        ],
                                    "Rationale": "The passage does not provide any evidence about what determines hospitalization duration. Mechanical ventilation is described as something that may be required in the most severe cases, not as a primary determinant of short hospital stays. The passage lists supportive interventions (oxygen, IV fluids) alongside mechanical ventilation without suggesting that rapid escalation to intensive care is the reason for short hospitalizations. This causal claim cannot be derived from the passage and misrepresents the relationship between interventions and outcomes described therein."
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
                    Entity To Be Replaced:
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
                                You are an expert in epidemiology. Answer the question using the provided Passage Body. The Passage Body contains all sections of the paper except the Discussion.
                                The question is multi-choice style. There may be one correct answer, multiple correct answers, or no correct answer. Select all conclusions that can be logically derived from the Passage Body.
                                
                                Output a JSON object with a single key "results" whose value is a list of index strings for all correct options. If no option is correct, the list should be empty.

                                {self_check}
                            """
        else:
            system_prompt = f"""
                                You are an expert in epidemiology. Answer the question using the provided Passage Body. The Passage Body contains all sections of the paper except the Discussion.
                                The question is multi-choice style. There may be one correct answer, multiple correct answers, or no correct answer. You must evaluate each option independently.

                                Follow these steps:
                                Step 1: Read the Passage Body and question carefully.
                                Step 2: For each option, determine whether it can be logically derived from the evidence in the Passage Body by applying general epidemiological principles.
                                Step 3: A correct option must be derivable from the Passage Body but should not be explicitly stated there verbatim. It represents a valid conclusion or interpretation of the evidence.
                                Step 4: An incorrect option either requires external information not in the Passage Body, makes unsupported speculative claims, or misinterprets the evidence.
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
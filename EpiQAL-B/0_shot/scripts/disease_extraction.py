from .func import *
from .constant import *
import os
import json
import math
from vllm import LLM, SamplingParams
import torch
from gliner import GLiNER
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

def disease_extraction_ner_pipeline(input_para):
    os.makedirs(f"{RESULT_FILE_PATH}/tmp/kg", exist_ok=True)
    
    entities_extracted_ner = {}
    
    model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
            
    for i in tqdm(range(0, len(input_para)), desc=f"{device}: NER Entity Extraction..."):
        text = input_para[i]["inputs"]
        current_idx = input_para[i]['idx']
        chunks = [text[i:i+384] for i in range(0, len(text), 384)]
        all_entities = []
        for chunk in chunks:
            entities = model.predict_entities(chunk, ["disease"], ner_threshold=0.9)
            for entity in entities:
                all_entities.append(entity["text"])
        entities_extracted_ner[current_idx] =  list(set(all_entities))

    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    entities_extracted_ner = sort_dict(entities_extracted_ner)
    with open(f"{RESULT_FILE_PATH}/tmp/kg/entities_extracted.json", "w") as f:
        json.dump(entities_extracted_ner, f, indent=4)
    
    return entities_extracted_ner

if __name__ == "__main__":
    
    train, val, test = get_data()
    print(len(train), len(val), len(test))
    
    input_para = train + val + test
    input_para = input_para[94:102]

    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    
    disease_extraction_ner_pipeline(input_para)
    
    
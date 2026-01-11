from .func import *
from .constant import *
import os
import json
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from collections import defaultdict
import torch
import gc
import random

def entity_alignment_pipeline(input_para, entities_extracted):
    os.makedirs(f"{RESULT_FILE_PATH}/tmp/kg", exist_ok=True)
    random.seed(42)
    
    relevant_triples = defaultdict(dict)
    
    smilarity_encoder = AutoModel.from_pretrained(SIMILARITY_ENCODER_NAME,
                                                    device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(SIMILARITY_ENCODER_NAME)
    
    with open(f"{EKGDONS_FILE_PATH}/entities_kg.json", "r") as f:
        ekg_don_entity_mapping = json.load(f)
    with open(f"{EKGDONS_FILE_PATH}/entities_id.json", "r") as f:
        ekg_don_entity_id = json.load(f)
    ekg_don_embeddings = np.load(f"{EKGDONS_FILE_PATH}/entities_kg_emb.npy")
    
    with open(f"{IBKH_FILE_PATH}/entities_kg.json", "r") as f:
        ibkh_entity_mapping = json.load(f)
    with open(f"{IBKH_FILE_PATH}/entities_id.json", "r") as f:
        ibkh_entity_id = json.load(f)
    ibkh_embeddings = np.load(f"{IBKH_FILE_PATH}/entities_kg_emb.npy")


    faiss.normalize_L2(ekg_don_embeddings) 
    ekg_don_index = faiss.IndexFlatIP(ekg_don_embeddings.shape[1])
    ekg_don_index.add(ekg_don_embeddings)  
    
    faiss.normalize_L2(ibkh_embeddings) 
    ibkh_index = faiss.IndexFlatIP(ibkh_embeddings.shape[1])
    ibkh_index.add(ibkh_embeddings)  
    
    print("#"*30)
    for i in tqdm(range(len(input_para)), desc=f"{smilarity_encoder.device}:Extracting relevant triples..."):
        
        ekg_don_triples = defaultdict(dict)
        ibkh_triples = defaultdict(dict)
        
        for extracting_entity in entities_extracted[input_para[i]["idx"]]:
            toks = tokenizer.batch_encode_plus([extracting_entity], 
                                                padding="max_length", 
                                                max_length=100, 
                                                truncation=True,
                                                return_tensors="pt")
            model_inputs = {}
            for k,v in toks.items():
                model_inputs[k] = v.to(smilarity_encoder.device)
            cls_rep = smilarity_encoder(**model_inputs)[0][:,0,:]
            extracting_entity_emb = cls_rep.cpu().detach().numpy()

            faiss.normalize_L2(extracting_entity_emb)
            #print(extracting_entity_emb.shape)
            #print(extracting_entity)
            
            
            ekg_don_D, ekg_don_I = ekg_don_index.search(extracting_entity_emb, k=SIMILARITY_SEARCH_K)
            ekg_don_mask = ekg_don_D[0] >= KG_SIMILARITY_THRES
            
            for idx in ekg_don_I[0, ekg_don_mask]:
                ekg_don_entity = ekg_don_entity_id[str(idx)]
                #print(idx, ekg_don_entity_mapping[ekg_don_entity]["Disease Name"])
                #flag = 0
                disease_name = ekg_don_entity_mapping[ekg_don_entity]["Disease Name"]
                for key in ekg_don_entity_mapping[ekg_don_entity].keys():
                    if key == "Disease Name":
                        continue
                    if len(ekg_don_entity_mapping[ekg_don_entity][key]) != 0:
                        #flag = 1
                        #ekg_don_triples[disease_name] = ekg_don_entity_mapping[ekg_don_entity][key]
                        #if len(ekg_don_entity_mapping[ekg_don_entity][key]) > MAX_KG_ITEMS_PER_KEY:
                        sampled_keys = random.sample(list(ekg_don_entity_mapping[ekg_don_entity][key].keys()), min(len(ekg_don_entity_mapping[ekg_don_entity][key].keys()), MAX_KG_ITEMS_PER_KEY))
                        ekg_don_triples[disease_name] = sort_dict({k: ekg_don_entity_mapping[ekg_don_entity][key][k] for k in sampled_keys})
                            
                #if flag:
                #    ekg_don_triples.append(ekg_don_entity_mapping[ekg_don_entity])
            #print(ekg_don_triples)
            
            ibkh_D, ibkh_I = ibkh_index.search(extracting_entity_emb, k=SIMILARITY_SEARCH_K)
            ibkh_mask = ibkh_D[0] >= KG_SIMILARITY_THRES
            
            del_key_white_list = ["Associate", "role in disease pathogenesis"]
            for idx in ibkh_I[0, ibkh_mask]:
                ibkh_entity = ibkh_entity_id[str(idx)]
                #print(idx, ibkh_entity_mapping[ibkh_entity]["Disease Name"])

                del_key_list = []
                for del_key in ibkh_entity_mapping[ibkh_entity]["Drugs"].keys():
                    if del_key not in del_key_white_list:
                        del_key_list.append(del_key)
                
                for del_key in del_key_list:
                    del ibkh_entity_mapping[ibkh_entity]["Drugs"][del_key]

                
                #flag = 0
                #new_value = {}
                disease_name = ibkh_entity_mapping[ibkh_entity]["Disease Name"]
                for key in ibkh_entity_mapping[ibkh_entity].keys():
                    if key == "Disease Name":
                        #new_value[key] = ibkh_entity_mapping[ibkh_entity][key]
                        continue
                    if len(ibkh_entity_mapping[ibkh_entity][key]) != 0:
                        #flag = 1
                        #new_value[key] = ibkh_entity_mapping[ibkh_entity][key]
                        #ibkh_triples[disease_name][key] = ibkh_entity_mapping[ibkh_entity][key]
                        
                        selected_dict = {}
                        for ibkh_inside_key in ibkh_entity_mapping[ibkh_entity][key].keys():
                            sampled_list = random.sample(ibkh_entity_mapping[ibkh_entity][key][ibkh_inside_key], min(len(ibkh_entity_mapping[ibkh_entity][key][ibkh_inside_key]), MAX_KG_ITEMS_PER_KEY))
                            selected_dict[ibkh_inside_key] = natsorted(sampled_list)
                        ibkh_triples[disease_name][key] = selected_dict
                #if flag:
                #    ibkh_triples.append(new_value)

                
            #print(ibkh_triples)
            #print("-" * 15)
            
            '''
            data = {"Entities Extracted": extracting_entity}

            if ekg_don_triples:
                data["eKG-Dons"] = ekg_don_triples
            if ibkh_triples:
                data["ibkh"] = ibkh_triples

            if data.keys() != {"Entities Extracted"}:
                relevant_triples[input_para[i]["idx"]].append(data)
            '''
            
        relevant_triples[input_para[i]["idx"]]["eKG-Dons"] = ekg_don_triples
        relevant_triples[input_para[i]["idx"]]["ibkh"] = ibkh_triples
        
    del smilarity_encoder
    del tokenizer
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    relevant_triples = sort_dict(relevant_triples)
    with open(f"{RESULT_FILE_PATH}/tmp/kg/relevant_triples.json", "w") as f:
        json.dump(relevant_triples, f, indent=4)
    
    return relevant_triples


if __name__ == "__main__":
    
    train, val, test = get_data()
    print(len(train), len(val), len(test))
    
    input_para = train + val + test
    input_para = input_para[94:102]
    
    os.makedirs(f"{RESULT_FILE_PATH}/tmp", exist_ok=True)
    
    with open(f"{RESULT_FILE_PATH}/tmp/kg/entities_extracted.json", "r") as f:
        entities_extracted = json.load(f)
        
    entity_alignment_pipeline(input_para, entities_extracted)
    
    
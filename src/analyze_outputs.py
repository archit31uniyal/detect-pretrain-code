import json
import numpy as np
import os
from audit_hp import load_json, save_json

def extract_top_k_indices(data, k=4):
    top_k_idx = np.argsort(data)[-k:]
    return top_k_idx

if __name__ == '__main__':
    json_file_read = "/scratch/deu9yh/llm_privacy/detect-pretrain-code/out/microsoft/Llama2-7b-WhoIsHarryPotter_NousResearch/Llama-2-7b-chat-hf/text/simcse.json"
    json_file_write = "/scratch/deu9yh/llm_privacy/detect-pretrain-code/out/microsoft/Llama2-7b-WhoIsHarryPotter_NousResearch/Llama-2-7b-chat-hf/text/topK_simcse.json"
    data = load_json(json_file_read)

    indices = extract_top_k_indices(data['max_simcse'])

    simcse = data["max_simcse"]
    gen = data['generation']
    gt = data["spurious"]
    cse, gens, g_t = [], [], []

    for idx in indices:
        cse.append(simcse[idx])
        gens.append(gen[idx])
        g_t.append(gt[idx])

    topk = {
        "max_simcse": cse,
        "gt": g_t,
        "generations": gens
    }

    save_json(topk, json_file_write)




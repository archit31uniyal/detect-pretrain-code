import json
import matplotlib.pyplot as plt
import numpy as np

def hist(data, name):
    # Generate the histogram
    cse_score = data['Mean_simcse']
    cse_score = cse_score[:83]
    cse_score_ref = data_ref['Mean_simcse']
    cse_score_ref = cse_score_ref[:83]
    cse_score_prompt = data_prompt['Mean_simcse']
    cse_score_prompt = cse_score_prompt[:83]
    cse_score_prompt_long = data_prompt_long['Mean_simcse']
    cse_score_prompt_long = cse_score_prompt_long[:83]
    plt.title('Histogram of SimCSE scores')
    plt.hist(cse_score, bins=15, alpha=0.6, color='orange', label = 'Unlearned')
    plt.hist(cse_score_ref, bins=15, alpha=0.6, color='blue', label = 'Original')
    plt.hist(cse_score_prompt, bins=15, alpha=0.6, color='green', label = 'Short summary')
    # plt.hist(cse_score_prompt_long, bins=15, alpha=0.6, color='red', label = 'Long summary')
    plt.xlabel('SimCSE score')
    plt.ylabel('Frequency')
    plt.legend(fontsize=6, loc='upper right')
    plt.savefig(f'./{name}')

def scatter(data, name):
    # Generate the scatter plot
    cse_score = data[0]['max_simcse']
    gen = data[0]['generation']
    cse_score = cse_score[:105]
    # cse_score = cse_score[:83]
    cse_score_ref = data[1]['max_simcse']
    gen_ref = data[1]['generation']
    cse_idx = [i for i, (x,y) in enumerate(zip(cse_score, cse_score_ref)) if abs(x-y)<=0.1]
    cse_score = [cse_score[i] for i in cse_idx]
    cse_score_ref = [cse_score_ref[i] for i in cse_idx]
    gen = [gen[i] for i in cse_idx]
    gen_ref = [gen_ref[i] for i in cse_idx]
    simcse = [(x,y) for x,y in zip(cse_score, cse_score_ref)]

    # cse_score_ref = cse_score_ref[:83]
    # cse_score_prompt = data[2]['Max_simcse']
    # cse_score_prompt = cse_score_prompt[:83]
    # cse_score_prompt_long = data[3]['Maxsimcse']
    # cse_score_prompt_long = cse_score_prompt_long[:83]
    plt.title('Scatter plot of SimCSE scores')
    plt.scatter(np.arange(len(cse_score)), cse_score, alpha=0.6, color='orange', label = 'Unlearned')
    plt.scatter(np.arange(len(cse_score_ref)), cse_score_ref, alpha=0.6, color='blue', label = 'Original')
    # plt.scatter(np.arange(len(cse_score_prompt)), cse_score_prompt, alpha=0.6, color='green', label = 'Short summary')
    # plt.scatter(np.arange(len(cse_score_prompt_long)), cse_score_prompt_long, alpha=0.6, color='red', label = 'Long summary')
    plt.xlabel('Index')
    plt.ylabel('SimCSE score')
    plt.legend(fontsize=6, loc='upper right')
    plt.savefig(f'./{name}')
    data = {"unlearn": gen, "original": gen_ref, "simcse": simcse}
    with open('./compare_sentences.json', 'w') as f:
        json.dump(data, f, indent = 4)

if __name__ == "__main__":
    # Load the data
    with open('/scratch/deu9yh/llm_privacy/detect-pretrain-code/out/microsoft/Llama2-7b-WhoIsHarryPotter_NousResearch/Llama-2-7b-chat-hf/text/simcse_gen.json', 'r') as f:
        data = json.load(f)

    with open('/scratch/deu9yh/llm_privacy/detect-pretrain-code/out/microsoft/Llama2-7b-WhoIsHarryPotter_NousResearch/Llama-2-7b-chat-hf/text/simcse_ref_gen.json', 'r') as f:
        data_ref = json.load(f)

    with open("/scratch/deu9yh/llm_privacy/detect-pretrain-code/out/microsoft/Llama2-7b-WhoIsHarryPotter_NousResearch/Llama-2-7b-chat-hf/text/simcse_prompt_gen.json", "r") as f:
        data_prompt = json.load(f)

    with open("/scratch/deu9yh/llm_privacy/detect-pretrain-code/out/microsoft/Llama2-7b-WhoIsHarryPotter_NousResearch/Llama-2-7b-chat-hf/text/simcse_prompt_long_gen.json", "r") as f:
        data_prompt_long = json.load(f)

    data_, data_ref_ = [], []

    data = [data, data_ref]
    # , data_prompt, data_prompt_long

    scatter(data, 'scatter_simcse_max.png')

import os
import re
import json
import random        

import numpy as np
import torch

from tqdm import tqdm

def setup_seed(seed=428):
    os.environ["PYTHONHASHSEED"]=str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

def evaluate(
    model,
    model_name,
    qa_path,
    qa_type,
    video_dir_path,
    output_dir_path,
    seed=42,
):
    setup_seed(seed)
    
    # temp
    output_path = os.path.join(output_dir_path, f"improve_{qa_type}_{model_name}.json")
    if os.path.exists(output_path):
        paired_qas = json.load(open(output_path))
        scores = cal_score(paired_qas)
        return scores
    cache_dir = f"cache_tmp_dir/{model_name}/{qa_type}"
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    paired_qas = json.load(open(qa_path))
    print(f"start eval | model: {model_name} | qa_type: {qa_type}")
    idx = 0
    for qa_dct in tqdm(paired_qas):


        # basic
        basic = qa_dct["basic"]
        basic_question = basic["question"]
        basic_video_path = os.path.join(video_dir_path, basic["video"])
        # caption
        desc_instruction = f"Describe the video: "
        cache_file = os.path.join(cache_dir, f"desc_{idx}.json")
        if os.path.exists(cache_file): description = json.load(open(cache_file))["response"]
        else:
            description = model.generate(
                instruction=desc_instruction,
                video_path=basic_video_path
            )
            json.dump({"response": description}, open(cache_file, "w"))
        basic_instruction = f"Description: {description}\n Please provide a clear response to the question below by watching the video. If necessary, you can also use the accompanying Description to help refine your answer. Your response should be a simple 'yes' or 'no'.\n\n Question: {basic_question}\n Answer the question using 'yes' or 'no': "
        cache_file = os.path.join(cache_dir, f"exp_{idx}.json")
        if os.path.exists(cache_file): basic_predict = json.load(open(cache_file))["response"]
        else:
            basic_predict = model.generate(
                instruction=basic_instruction,
                video_path=basic_video_path
            )
            json.dump({"response": basic_predict}, open(cache_file, "w"))
        basic_instruction = f"Description: {description}\n Please offer a detailed explanation for your answer to the following question. After explaining, verify the accuracy of the information you've used in your explanation. Once you've confirmed the facts, please respond to the question with a simple 'yes' or 'no'.\n\n Question: {basic_question}\n Answer: {basic_predict}\n Answer the question using 'yes' or 'no': "
        cache_file = os.path.join(cache_dir, f"basic_{idx}.json")
        if os.path.exists(cache_file): basic_predict = json.load(open(cache_file))["response"]
        else:
            basic_predict = model.generate(
                instruction=basic_instruction,
                video_path=basic_video_path
            )
            json.dump({"response": basic_predict}, open(cache_file, "w"))
        qa_dct["basic"]["predict"] = basic_predict
        # hallucination
        halluc = qa_dct["hallucination"]
        halluc_question = halluc["question"]
        halluc_video_path = os.path.join(video_dir_path, halluc["video"])
        if halluc_video_path != basic_video_path:
            desc_instruction = f"Describe the video: "
            cache_file = os.path.join(cache_dir, f"hdesc_{idx}.json")
            if os.path.exists(cache_file): description = json.load(open(cache_file))["response"]
            else:
                description = model.generate(
                    instruction=desc_instruction,
                    video_path=halluc_video_path
                )
                json.dump({"response": description}, open(cache_file, "w"))
        halluc_instruction = f"Description: {description}\n Please provide a clear response to the question below by watching the video. If necessary, you can also use the accompanying Description to help refine your answer. Your response should be a simple 'yes' or 'no'.\n\n Question: {halluc_question}\n Answer the question using 'yes' or 'no': "
        cache_file = os.path.join(cache_dir, f"hexp_{idx}.json")
        if os.path.exists(cache_file): halluc_predict = json.load(open(cache_file))["response"]
        else:
            halluc_predict = model.generate(
                instruction=halluc_instruction,
                video_path=halluc_video_path
            )
            json.dump({"response": halluc_predict}, open(cache_file, "w"))
        halluc_instruction = f"Description: {description}\n Please offer a detailed explanation for your answer to the following question. After explaining, verify the accuracy of the information you've used in your explanation. Once you've confirmed the facts, please respond to the question with a simple 'yes' or 'no'.\n\n Question: {halluc_question}\n Answer: {halluc_predict}\n Answer the question using 'yes' or 'no': "
        cache_file = os.path.join(cache_dir, f"halluc_{idx}.json")
        if os.path.exists(cache_file): halluc_predict = json.load(open(cache_file))["response"]
        else:
            halluc_predict = model.generate(
                instruction=halluc_instruction,
                video_path=halluc_video_path
            )
            json.dump({"response": halluc_predict}, open(cache_file, "w"))
        qa_dct["hallucination"]["predict"] = halluc_predict
        idx += 1


        # # basic
        # basic = qa_dct["basic"]
        # basic_question = basic["question"]
        # basic_question = f"{basic_question}\nAnswer the question using 'yes' or 'no'."
        # basic_video_path = os.path.join(video_dir_path, basic["video"])
        # basic_predict = model.generate(
        #     instruction=basic_question,
        #     video_path=basic_video_path
        # )
        # qa_dct["basic"]["predict"] = basic_predict
        # # hallucination
        # halluc = qa_dct["hallucination"]
        # halluc_question = halluc["question"]
        # halluc_question = f"{halluc_question}\nAnswer the question using 'yes' or 'no'."
        # halluc_video_path = os.path.join(video_dir_path, halluc["video"])
        # halluc_predict = model.generate(
        #     instruction=halluc_question,
        #     video_path=halluc_video_path
        # )
        # qa_dct["hallucination"]["predict"] = halluc_predict
    # save predict result
    if not os.path.exists(output_dir_path): os.makedirs(output_dir_path)
    output_path = os.path.join(output_dir_path, f"improve_{qa_type}_{model_name}.json")
    with open(output_path, "w") as jp:
        json.dump(paired_qas, jp, indent=4) 
    
    # output_path = os.path.join(output_dir_path, f"{qa_type}_{model_name}.json")
    # with open(output_path) as jp:
    #     paired_qas = json.load(jp) 

    # evaluate
    scores = cal_score(paired_qas)    

    return scores

def cal_score(results):
    basic_acc = 0
    halluc_acc = 0
    acc = 0
    for result in results:
        
        basic_hit = 0
        halluc_hit = 0
        final_hit = 0

        basic_answer = result["basic"]["answer"]
        basic_predict = result["basic"]["predict"]
        # basic_answer_pattern = f"^{basic_answer}" + r"\b"
        basic_answer_pattern = fr'\b({basic_answer})\b'
        if re.search(basic_answer_pattern, basic_predict, re.IGNORECASE):
            basic_hit = 1

        halluc_answer = result["hallucination"]["answer"]
        halluc_predict = result["hallucination"]["predict"]
        # halluc_answer_pattern = f"^{halluc_answer}" + r"\b"
        halluc_answer_pattern = r'\b('+halluc_answer+ r')\b'
        if re.search(halluc_answer_pattern, halluc_predict, re.IGNORECASE):
            halluc_hit = 1
        
        final_hit = int(basic_hit and halluc_hit)

        basic_acc += basic_hit
        halluc_acc += halluc_hit
        acc += final_hit
    
    scores = {
        "basic_accuracy": basic_acc / len(results),
        "halluc_accuracy": halluc_acc / len(results),
        "accuracy": acc / len(results)
    }

    return scores
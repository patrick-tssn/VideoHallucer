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

    paired_qas = json.load(open(qa_path))
    print(f"start eval | model: {model_name} | qa_type: {qa_type}")
    for qa_dct in tqdm(paired_qas):
        # basic
        basic = qa_dct["basic"]
        basic_question = basic["question"]
        basic_question = f"{basic_question}\nAnswer the question using 'yes' or 'no'."
        basic_video_path = os.path.join(video_dir_path, basic["video"])
        basic_predict = model.generate(
            instruction=basic_question,
            video_path=basic_video_path
        )
        qa_dct["basic"]["predict"] = basic_predict
        # hallucination
        halluc = qa_dct["hallucination"]
        halluc_question = halluc["question"]
        halluc_question = f"{halluc_question}\nAnswer the question using 'yes' or 'no'."
        halluc_video_path = os.path.join(video_dir_path, basic["video"])
        halluc_predict = model.generate(
            instruction=halluc_question,
            video_path=halluc_video_path
        )
        qa_dct["hallucination"]["predict"] = halluc_predict
    # save predict result
    if not os.path.exists(output_dir_path): os.makedirs(output_dir_path)
    output_path = os.path.join(output_dir_path, f"{qa_type}_{model_name}.json")
    with open(output_path, "w") as jp:
        json.dump(paired_qas, jp, indent=4) 
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
        basic_answer_pattern = f"^{basic_answer}" + r"\b"
        if re.match(basic_answer_pattern, basic_predict, re.IGNORECASE):
            basic_hit = 1

        halluc_answer = result["hallucination"]["answer"]
        halluc_predict = result["hallucination"]["predict"]
        halluc_answer_pattern = f"^{halluc_answer}" + r"\b"
        if re.match(halluc_answer_pattern, halluc_predict, re.IGNORECASE):
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

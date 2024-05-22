import os
import re
import json
import random
import ast        

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
        
        # hallucination
        halluc = qa_dct["hallucination"]
        halluc_question = halluc["question"]
        halluc_instruction = f"Please provide an explanation to support your answer to the following question: {halluc_question}\n\n" + "For example, your response should look like this: {'pred': 'yes', 'explanation': ''}."
        halluc_video_path = os.path.join(video_dir_path, halluc["video"])
        halluc_explanation = model.generate(
            instruction=halluc_instruction,
            video_path=halluc_video_path
        )

        # print("question: \n", halluc_question)
        # print("explanation: \n", halluc_explanation)
        try:
            explanation_dct = ast.literal_eval(halluc_explanation)
            halluc_explanation = explanation_dct["explanation"]
            halluc_predict_ori = explanation_dct["pred"]
        except Exception as e:
            # print(e)
            # print(halluc_explanation)
            sents = halluc_explanation.split('.')
            explanation_dct = {"pred": sents[0], "explanation": " ".join(sents[1:])}
            halluc_explanation = " ".join(sents[1:])
            halluc_predict_ori = sents[0]

        halluc_instruction = f"Answer the question using the explanation.\n\n Question: {halluc_question}\n Explanation: {halluc_explanation}\n Answer the question using 'yes' or 'no'."
        halluc_predict_self_exp = model.generate(
            instruction=halluc_instruction,
            video_path=halluc_video_path
        )

        gt_explanation = "The video " + qa_dct["hallucination"]["explanation"]
        halluc_instruction = f"Answer the question using the explanation.\n\n Question: {halluc_question}\n Explanation: {gt_explanation}\n Answer the question using 'yes' or 'no'."
        halluc_predict_gt_exp = model.generate(
            instruction=halluc_instruction,
            video_path=halluc_video_path
        )

        

        qa_dct["hallucination"]["ori_predict"] = halluc_predict_ori
        qa_dct["hallucination"]["self_predict"] = halluc_predict_self_exp
        qa_dct["hallucination"]["gt_predict"] = halluc_predict_gt_exp
        qa_dct["hallucination"]["explanation"] =  explanation_dct
    # save predict result
    if not os.path.exists(output_dir_path): os.makedirs(output_dir_path)
    output_path = os.path.join(output_dir_path, f"exp_{qa_type}_{model_name}.json")
    with open(output_path, "w") as jp:
        json.dump(paired_qas, jp, indent=4) 
    
    # output_path = os.path.join(output_dir_path, f"{qa_type}_{model_name}.json")
    # with open(output_path) as jp:
    #     paired_qas = json.load(jp) 

    # evaluate
    scores = cal_score(paired_qas)    

    return scores

def cal_score(results):
    ori_acc = 0
    self_acc = 0
    gt_acc = 0
    for result in results:
        

        halluc_answer = result["hallucination"]["answer"]
        halluc_predict_ori = result["hallucination"]["ori_predict"]
        halluc_predict_gt_exp = result["hallucination"]["gt_predict"]
        halluc_predict_self_exp = result["hallucination"]["self_predict"]
        
        # halluc_answer_pattern = f"^{halluc_answer}" + r"\b"
        halluc_answer_pattern = r'\b('+halluc_answer+ r')\b'
        if re.match(halluc_answer_pattern, halluc_predict_ori, re.IGNORECASE):
            ori_acc += 1
        if re.match(halluc_answer_pattern, halluc_predict_self_exp, re.IGNORECASE):
            self_acc += 1
        if re.match(halluc_answer_pattern, halluc_predict_gt_exp, re.IGNORECASE):
            gt_acc += 1
        
    
    scores = {
        "ori_accuracy": ori_acc / len(results),
        "self_accuracy": self_acc / len(results),
        "gt_accuracy": gt_acc / len(results),
    }

    return scores
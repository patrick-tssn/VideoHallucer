"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from collections import OrderedDict
import json
import os
import random
import torch
from torch.utils.data import Dataset

from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset  #, VQAEvalDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "direct_answers": "; ".join(ann["direct_answers"]),
                "choices": "; ".join(ann["choices"]),
                "correct_choice": ann["choices"][ann["correct_choice_idx"]],
                "image": sample["image"],
            }
        )


class AOKVQAReasoningDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # self.instruction_pool = [
        #     '{}',
        #     'Question: {}',
        #     '{} A short answer to the question is',
        #     'Q: {} A:',
        #     'Answer the following question based on the image content. Question: {} Short answer:',
        #     # 'Given the image, answer the following question with no more than three words. {}',
        #     'Based on the image, respond to this question with a short answer: {}.',
        #     'Use the provided image to answer the question: {} Provide your answer as short as possible.',
        #     'What is the answer to the following question? "{}"',
        #     'Given this image, answer this question concisely: {} ',
        #     'The question "{}" can be answered using the image. A short answer is'
        # ]
        # self.instruction_pool =[
        #     "[vqa] {}",
        #     "[vqa] Based on the image, respond to this question with a short answer: {}"
        # ]
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.vis_root = vis_root
        self.instruction_pool =[
            "[vqa] {}"
        ]
        annotation = []
        with open(ann_paths, 'r') as f:
            for line in f.readlines():
                json_data = json.loads(line)
                annotation.append(json_data)

        exist_annotation = []
        for ann in annotation:
            image_path = os.path.join(self.vis_root, ann["image_path"].split('/')[-1])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
            else:
                print("does not exists", image_path)
        self.annotation = exist_annotation

    def __len__(self):
        return len(self.annotation)
    
    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_path"].split('/')[-1])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        rationales = ann["analysis"]



        # print("answer key", answer_key)
        # for answer in ann[answer_key]:
        #     print(answer)

        # answer_weight = {}
        # for answer in ann[answer_key]:
        #     if answer in answer_weight.keys():
        #         answer_weight[answer] += 1 / len(ann[answer_key])
        #     else:
        #         answer_weight[answer] = 1 / len(ann[answer_key])

        # answers = list(answer_weight.keys())
        # weights = list(answer_weight.values())

        # answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights
        # choices = ann["choices"]

        # print("question",question)
        # print("answer", rationales)
        return {
            "image": image,
            "question": question,
            # "answer": analysis,
            "reason":rationales,
            # "choice":choices
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        question = self.text_processor(data["question"])
        instruction = random.choice(self.instruction_pool).format(question)

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        
        random_index = random.randint(0,1)
        # reason = random.choice(data["reason"])
        answer = data["reason"]

        analysis = answer.split("\nAnswer:")[0]
        answer = answer.split("\nAnswer:")[-1]

        # answer = data["reaso"]

        if random_index ==0:
            instruction = instruction+analysis+"\nAnswer:"

        elif random_index==1:
            answer = analysis+"\nAnswer:"+answer


        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": answer,
        }


class AOKVQGDataset(AOKVQAReasoningDataset):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.instruction_pool = [
            'Given the image, generate a question whose answer is: {}',
            'Based on the image, provide a question with the answer: {}',
            'Given the visual representation, create a question for which the answer is "{}"',
            'From the image provided, craft a question that leads to the reply: {}',
            'Considering the picture, come up with a question where the answer is: {}',
            'Taking the image into account, generate an question that has the answer: {}'
        ]

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['answer'])
        # instruction = "###Human: <Img><ImageHere></Img> {}###Assistant: ".format(instruction)

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": data['question'],
        }


# class AOKVQAEvalDataset(VQAEvalDataset, __DisplMixin):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         """
#         vis_root (string): Root directory of images (e.g. coco/images/)
#         ann_root (string): directory to store the annotation file
#         """
#
#         self.vis_root = vis_root
#
#         self.annotation = json.load(open(ann_paths[0]))
#
#         answer_list_path = ann_paths[1]
#         if os.path.exists(answer_list_path):
#             self.answer_list = json.load(open(answer_list_path))
#         else:
#             self.answer_list = None
#
#         try:
#             self.coco_fmt_qust_file = ann_paths[2]
#             self.coco_fmt_anno_file = ann_paths[3]
#         except IndexError:
#             self.coco_fmt_qust_file = None
#             self.coco_fmt_anno_file = None
#
#         self.vis_processor = vis_processor
#         self.text_processor = text_processor
#
#         self._add_instance_ids()
#
#     def collater(self, samples):
#         (
#             image_list,
#             question_list,
#             question_id_list,
#             instance_id_list,
#             choices_list,
#             correct_choice_idx_list,
#             direct_answers_list,
#         ) = ([], [], [], [], [], [], [])
#
#         for sample in samples:
#             image_list.append(sample["image"])
#             question_list.append(sample["text_input"])
#             question_id_list.append(sample["question_id"])
#             instance_id_list.append(sample["instance_id"])
#             choices_list.append(sample["choices"])
#             correct_choice_idx_list.append(sample["correct_choice_idx"])
#             direct_answers_list.append(sample["direct_answers"])
#
#         return {
#             "image": torch.stack(image_list, dim=0),
#             "text_input": question_list,
#             "question_id": question_id_list,
#             "instance_id": instance_id_list,
#             "choices": choices_list,
#             "correct_choice_idx": correct_choice_idx_list,
#             "direct_answers": direct_answers_list,
#         }
#
#     def __getitem__(self, index):
#         ann = self.annotation[index]
#
#         image_path = os.path.join(self.vis_root, ann["image"])
#         image = Image.open(image_path).convert("RGB")
#
#         image = self.vis_processor(image)
#         question = self.text_processor(ann["question"])
#
#         choices = ann["choices"]
#         if "correct_choice_idx" in ann:
#             correct_choice_idx = ann["correct_choice_idx"]
#         else:
#             correct_choice_idx = None
#
#         if "direct_answers" in ann:
#             direct_answers = ann["direct_answers"]
#         else:
#             direct_answers = None
#
#         return {
#             "image": image,
#             "text_input": question,
#             "question_id": ann["question_id"],
#             "instance_id": ann["instance_id"],
#             "choices": choices,
#             "correct_choice_idx": correct_choice_idx,
#             "direct_answers": direct_answers,
#         }

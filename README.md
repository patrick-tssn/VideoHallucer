<div align="center">

# VideoHallucer: Evaluating Intrinsic and Extrinsic Hallucinations in Large Video-Language Models

[![videohallucer-page](https://img.shields.io/badge/videohallucer-page-green)](https://videohallucer.github.io/)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-<COLOR>.svg)](https://arxiv.org/abs/<INDEX>)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://<CONFERENCE>) -->

</div>


![image](assets/teaser.png)

**Table of Contents**

- [VideoHallucer](#videohallucer)
    - [Introduction](#introduction)
    - [Statistics](#statistics)
    - [Data](#data)
- [VideoHallucerKit](#videohallucerkit)
    - [Installation](#installation)
    - [Usage](#usage)
- [Leaderboard](#leaderboard)



## VideoHallucer


### Introduction

Recent advancements in Multimodal Large Language Models (MLLMs) have extended their capabilities to video understanding. Yet, these models are often plagued by "hallucinations", where irrelevant or nonsensical content is generated, deviating from the actual video context. This work introduces VideoHallucer, the **first comprehensive benchmark for hallucination detection in large video-language models (LVLMs)**. VideoHallucer categorizes hallucinations into two main types: intrinsic and extrinsic, offering further subcategories for detailed analysis, including object-relation, temporal, semantic detail, extrinsic factual, and extrinsic non-factual hallucinations. We adopt an adversarial binary VideoQA method for comprehensive evaluation, where pairs of basic and hallucinated questions are crafted strategically. By evaluating eleven LVLMs on VideoHallucer, we reveal that (i) the majority of current models exhibit significant issues with hallucinations; (ii) while scaling datasets and parameters improves models' ability to detect basic visual cues and counterfactuals, it provides limited benefit for detecting extrinsic factual hallucinations; (iii) existing models are more adept at detecting facts than identifying hallucinations. As a byproduct, these analyses further instruct the development of our self-PEP framework, achieving an average of 5.38\% improvement in hallucination resistance across all model architectures.




### Statistics

| | Object-Relation Hallucination | Temporal Hallucination | Semantic Detail Hallucination | External Factual Hallucination | External Nonfactual Hallucination |
| ---- | ---- | ---- | ---- | ---- | ---- |
|Questions | 400 | 400 | 400 | 400 | 400 |
|Videos | 183 | 165 | 400| 200 | 200 |

The Extrinsic Factual Hallucination and Extrinsic Non-factual Hallucination share same videos and basic questions

### Data

You can download the videohallucer from [huggingface](), containing both json and videos.

```
videohallucer_datasets                    
    ├── object_relation
        ├── object_relation.json
        └── videos
    ├── temporal
        ├── temporal.json
        └── videos
    ├── semantic_detail
        ├── semantic_detail.json
        └── videos
    ├── external_factual
        ├── external_factual.json
        └── videos
    └── external_nonfactual
        ├── external_nonfactual.json
        └── videos
```


We offer a selection of case examples from our dataset for further elucidation:

```
[
    {
        "basic": {
            "video": "1052_6143391925_916_970.mp4",
            "question": "Is there a baby in the video?",
            "answer": "yes"
        },
        "hallucination": {
            "video": "1052_6143391925_916_970.mp4",
            "question": "Is there a doll in the video?",
            "answer": "no"
        },
        "type": "subject"
    },
...
]
```



## VideoHallucerKit 

### Installation


**Available Baselines**

- VideoChatGPT-7B
- Valley2-7B
- Video-LLaMA-2-7B/13B
- VideoChat2-7B
- VideoLLaVA-7B
- LLaMA-VID-7B/13B
- VideoLaVIT-7B
- MiniGPT4-Video-7B
- PLLaVA-7B/13B/34B
- LLaVA-NeXT-Video-DPO-DPO-7B/34B
- Gemini-1.5-pro

- LLaVA
- GPT4V-Azure

For detailed instructions on installation and checkpoints, please consult the [INSTALLATION](INSTALLATION.md) guide.



### Usage

inference
```bash
cd baselines
python ../model_testing_zoo.py --model_name Gemini-1.5-pro # ["VideoChatGPT", "Valley", "Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "Gemini-1.5-pro"])
```

evaluate on VideoHallucer
```bash
cd baselines
python ../evaluations/evaluation.py  --model_name Gemini-1.5-pro --eval_obj # [--eval_obj_rel, --eval_temporal, --eval_semantic, --eval_fact, --eva_nonfact]
```



## Leaderboard

more detailed results see `baselines/results`


|  Model    |  Object-Relation    |  Temporal     |  Semantic Detail | Extrinsic Fact | Extrinsic Non-fact | Overall |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|  PLLaVA-34B    |     59 |  47    |  60  | 5.5  | 53.5 | 45 |
|  PLLaVA-13B    |     57.5 |  35.5    |  65  | 5  | 43 | 41.2 |
|  PLLaVA    |     60 |  23.5    |  57  | 9.5  | 40.5 | 38.1 |
|  Gemini-1.5-pro    |     52 |  18.5    |  53.5  | 16.5  | 48.5 | 37.8 |
|  LLaVA-NeXT-Video-DPO-34B    |     50.5 |  30    |  40  | 7  | 34 | 32.3 |
|  LLaVA-NeXT-Video-DPO    |     51.5 |  28    |  38  | 14  | 28.5 | 32.0 |
|  LLaMA-VID    |   44.5   |  27    |  25.5   | 12.5 | 36.5 | 29.2 |
|  MiniGPT4-Video    |    27.5  |  18    |  23.5    | 12 | 30.5 | 22.3 |
|  LLaMA-VID    |   43.5   |  21    |  17   | 2.5 | 21 | 21 |
|  VideoLaVIT    |  35.5    |  25.5    | 10.5     | 4 | 19 | 18.9 |
|  VideoLLaVA    |   34.5   |  13.5    | 12    | 3 | 26 | 17.8 |
|  Video-LLaMA-2    | 18    | 7.5     | 1     | 6.5 | 17 | 10 |
|  VideoChat2    |  10.5    | 7.5     | 9     | 7 | 0.5 | 7.8 |
|  VideoChatGPT    |  6    |  0    | 2     | 7 | 17  | 6.4|
|  Video-LLaMA-2    | 8.5    | 0     | 7.5     | 0 | 0.5 | 3.3 |
|  Valley2    |   4.5   |  3    | 2.5     | 0.5 | 3.5 | 2.8 |


## Acknowledgement


- We thank [vllm-safety-benchmark](https://github.com/UCSC-VLAA/vllm-safety-benchmark) for inspiring the framework of VideoHallucerKit.
- We thank Center for AI Safety for supporting our computing needs. 
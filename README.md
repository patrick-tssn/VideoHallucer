<div align="center">

# HaVeBench: A Comprehensive Hallucination Benchmark for Video-Language Models

[![havebench-page](https://img.shields.io/badge/havebench-page-green)](https://havebench.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-<COLOR>.svg)](https://arxiv.org/abs/<INDEX>)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://<CONFERENCE>)

</div>


![image](assets/teaser.png)

**Table of Contents**

- [HaVeBench](#havebench)
    - [Introduction](#introduction)
    - [Statistics](#statistics)
    - [Data](#data)
- [HaVeKit](#havekit)
    - [Installation](#installation)
    - [Usage](#usage)
- [Leaderboard](#leaderboard)



## HaVeBench


### Introduction

We introduce HaVeBench, the first comprehensive benchmark designed to assess hallucination in video-language models. 
Within HaVeBench, we establish a clear taxonomy of hallucinations, distinguishing between two primary categories: intrinsic and extrinsic. Specifically, Intrinsic hallucinations involve generated content that directly contradicts information present in the source video, and can be categorised into three subtypes: object-relation, temporal, and semantic detail hallucinations. While extrinsic hallucinations involve content that cannot be verified against the source, and can be classified as either extrinsic factual, aligning with general knowledge but not present in the source video, or extrinsic non-factual, which includes all the others. 


### Statistics

| | Object-Relation Hallucination | Temporal Hallucination | Semantic Detail Hallucination | External Factual Hallucination | External Nonfactual Hallucination |
| ---- | ---- | ---- | ---- | ---- | ---- |
|Questions | 400 | 400 | 400 | 400 | 400 |
|Videos | 183 | 165 | 400| 200 | 200 |

The Extrinsic Factual Hallucination and Extrinsic Non-factual Hallucination share same videos and basic questions

### Data

You can download the havebench [here](), containing both json and videos.

```
havebench_datasets                    
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



## HaVeKit 

### Installation


**Available Baselines**

- VideoChatGPT
- Valley2
- Video-LLaMA-2
- VideoChat2
- VideoLLaVA
- LLaMA-VID
- VideoLaVIT
- MiniGPT4-Video
- PLLaVA
- LLaVA-NeXT-Video
- Gemini-1.5-pro

- LLaVA
- GPT4V

For detailed instructions on installation and checkpoints, please consult the [INSTALLATION](INSTALLATION.md) guide.



### Usage

inference
```bash
cd baselines
python ../model_testing_zoo.py --model_name Gemini-1.5-pro # ["VideoChatGPT", "Valley", "Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "Gemini-1.5-pro"])
```

evaluate on HaVeBench
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
|  LLaVA-NeXT-Video-34B    |     50.5 |  30    |  40  | 7  | 34 | 32.3 |
|  LLaVA-NeXT-Video    |     51.5 |  28    |  38  | 14  | 28.5 | 32.0 |
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

# HaVeBench: A Comprehensive Hallucination Benchmark for Video-Language Models

**Table of Contents**

- [HaVeBench](#havebench)
    - [Introduction](#introduction)
    - [Statistics](#statistics)
- [HaVeKit](#havekit)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Leaderboard](#leaderboard)

1. HaVeBench: comprehensive hallucination benchmark for video-language models
2. HaVeKit: one-stop evaluation tookit

## HaVeBench


### Introduction


### Statistics



examples:



object relation hallucination: 200 QA-pair

`havebench_datasets/object_relation/object_relation.json`

temporal hallucination: 200 QA-pair

`havebench_datasets/temporal/temporal.json`

semantic hallucination: 200 QA-pair

`havebench_datasets/semantic_detail/semantic_detail.json`

external factual hallucination: 200 QA-pair

`havebench_datasets/external_factual/external_factual.json`

external nonfactual hallucination: 200 QA-pair

`havebench_datasets/external_nonfactual/external_nonfactual.json`


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
- Gemini-1.5-pro

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
python ../evaluations/evaluation.py  --model_name Gemini-1.5-pro --eval_obj # [--eval_]
```



### Leaderboard

more results see `baselines/results`



Overall

|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|  VideoChatGPT    |      |      |      |
|  Valley2    |      |      |      |
|  Video-LLaMA-2    |      |      |      |
|  VideoChat2    |      |      |      |
|  VideoLLaVA    |      |      |      |
|  LLaMA-VID    |      |      |      |
|  VideoLaVIT    |      |      |      |
|  Gemini-1.5-pro    |      |      |      |



Object-Relation
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|  VideoChatGPT    | 0.955     |  0.07    |  0.06    |
|  Valley2    |   0.765   | 0.08     |  0.035    |
|  Video-LLaMA-2    |  0.885    | 0.18     | 0.155     |
|  VideoChat2    |    0.21  |   0.205   | 0.02     |
|  VideoLLaVA    |  0.95    | 0.38     | 0.345     |
|  LLaMA-VID    |   0.785   |  0.59    |  0.435    |
|  VideoLaVIT    |    0.945  |  0.39    | 0.355     |
|  MiniGPT4-Video    |  0.79    |  0.2    | 0.165     |
|  Gemini-1.5-pro    |  0.845    |  0.56    | 0.52     |


Temporal
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|  VideoChatGPT    |   1.0   |  0.0    | 0.0     |
|  Valley2    |   0.17   |  0.04    |  0.0    |
|  Video-LLaMA-2    |   0.91   |  0.08    |  0.07    |
|  VideoChat2    |   0.09   |  0.125    | 0.015     |
|  VideoLLaVA    |  0.975    |  0.135    |  0.135    |
|  LLaMA-VID    |    0.86  |  0.25    | 0.21     |
|  VideoLaVIT    |  0.885    | 0.255     |  0.27    |
|  MiniGPT4-Video    |  0.655    |  0.165    | 0.09     |
|  Gemini-1.5-pro    |    0.805  | 0.18     |  0.175    |



Semantic Detail
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|  VideoChatGPT    |  0.955    |  0.05    |  0.025    |
|  Valley2    |    0.82  |  0.015    |  0.005    |
|  Video-LLaMA-2    |  0.885    | 0.07     | 0.045     |
|  VideoChat2    |  0.185    |  0.105    | 0.01     |
|  VideoLLaVA    |  0.97    |  0.03    |   0.01   |
|  LLaMA-VID    |   0.885   |  0.095    |  0.015    |
|  VideoLaVIT    |     0.96 |   0.04   |   0.0   |
|  MiniGPT4-Video    |  0.765    |  0.085    | 0.055     |
|  Gemini-1.5-pro    | 0.87     |   0.09   |  0.005    |


External Factual
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|  VideoChatGPT    |  0.865    |  0.135    |  0.07    |
|  Valley2    |    0.09   |   0.065  |   0.0   | 
|  Video-LLaMA-2    |  0.75    |  0.09    |  0.065    |
|  VideoChat2    |  0.115    | 0.055     |  0.015    |
|  VideoLLaVA    |  0.93    |   0.045   | 0.03     |
|  LLaMA-VID    |  0.98    |  0.025    | 0.025     |
|  VideoLaVIT    |  0.975    | 0.06     |  0.04    |
|  MiniGPT4-Video    |  0.83    |  0.07    | 0.035     |
|  Gemini-1.5-pro    |      |      |      |


External Nonfactual
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|  VideoChatGPT    | 0.855     | 0.275     |  0.17    |
|  Valley2    |   0.1   |  0.085    |  0.02    |
|  Video-LLaMA-2    |  0.685    |   0.235   |  0.14    |
|  VideoChat2    |   0.105   | 0.065     | 0.005     |
|  VideoLLaVA    |   0.93   |  0.315    | 0.26     |
|  LLaMA-VID    |   0.98   | 0.225      | 0.21     |
|  VideoLaVIT    |    0.975  |  0.215    | 0.19     |
|  MiniGPT4-Video    |   0.815   |  0.25    |  0.195    |
|  Gemini-1.5-pro    |      |      |      |


Fact Detect
|  Model    |  hallucination-det    |  fact-det     |  delta |
| ---- | ---- | ---- | ---- |
|  VideoChatGPT    |   0.205   | 0.275     |  0.17    |
|  Valley2    |      |  0.075    |  0.02    |
|  Video-LLaMA-2    |  0.1625    |   0.235   |  0.14    |
|  VideoChat2    |   0.06   | 0.065     | 0.005     |
|  VideoLLaVA    |   0.18   |  0.315    | 0.26     |
|  LLaMA-VID    |   0.125  | 0.225      | 0.21     |
|  VideoLaVIT    |    0.1375  |  0.215    | 0.19     |
|  MiniGPT4-Video    |   0.16   |  0.25    |  0.195    |
|  Gemini-1.5-pro    |      |      |      |

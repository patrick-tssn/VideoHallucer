# HaVeBench

examples:



object relation hallucination: 200 QA

`havebench_datasets/object_relation/object_relation.json`

temporal hallucination: 200 QA

`havebench_datasets/temporal/temporal.json`

semantic hallucination: 200 QA

`havebench_datasets/semantic_detail/semantic_detail.json`


external factual hallucination: 200 QA

`havebench_datasets/external_factual/external_factual.json`

external nonfactual hallucination: 200 QA

`havebench_datasets/external_nonfactual/external_nonfactual.json`



possible baselines

## Installation

Follow existing repositories to install the environments:
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)
- [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)
- [Video Chat2](https://github.com/OpenGVLab/Ask-Anything)
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)
- [InternVideo](https://github.com/OpenGVLab/InternVideo)
- [Valley](https://github.com/RupertLuo/Valley)

## Datasets

The full dataset should look like this.
```
├── havebench_datasets/
    ├── object_relation
        ├── object_relation.json
        └── videos/*.mp4
    ├── temporal
        ├── temporal.json
        └── videos/*.mp4
    ├── semantic_detail
        ├── semantic_detail.json
        └── videos/*.mp4
    ├── fact
        ├── fact.json
        └── videos/*.mp4
```
The QA data should look like this
```json
[
    {
        "basic": {
            "video": "ILSVRC2015_train_00265005_0_30.mp4",
            "question": "Is there a bird in the video?",
            "answer": "yes"
        },
        "hallucination": {
            "video": "ILSVRC2015_train_00265005_0_30.mp4",
            "question": "Is there a crow in the video?",
            "answer": "no"
        },
        "type": "subject"
    },
    ...
]
```

## Leaderboard

object-relation
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|  Gemini-1.5-pro    |  0.71    |  0.24    | 0.19     |
|  VideoLLaVA    |  0.88    | 0.53     | 0.44     |

semantic detail
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|   Gemini-1.5-pro   |   0.64   |    0.37  |   0.05   |
|  VideoLLaVA    |  0.81    | 0.19     | 0.02     |


temporal
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|   Gemini-1.5-pro   |      |      |      |
|  VideoLLaVA    |      |      |      |


fact
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|   Gemini-1.5-pro   |      |      |      |
|  VideoLLaVA    |      |      |      |


overall
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|   Gemini-1.5-pro   |      |      |      |
|  VideoLLaVA    |      |      |      |



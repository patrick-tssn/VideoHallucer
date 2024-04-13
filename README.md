# HaVeBench

examples:

object relation hallucination: 200 QA

`obj_rel_hallucination/obj_rel_hallucination.json`
```json
{
        "video_id": "ILSVRC2015_val_00072000_165_195.mp4",
        "positive": {
            "question": "Does a person stand behind a dog in the video?",
            "answer": "yes"
        },
        "negative": {
            "question": "Does a person stand infront a dog in the video?",
            "answer": "no"
        },
        "type": "relation"
    }
```

semantic hallucination: 200 QA

`semantic_hallucination/semantic_hallucination.json`
```json
{
        "video_id": "kMJMR68Dz4s.mp4",
        "positive": {
            "video_id": "kMJMR68Dz4s_a.mp4",
            "question": "Is there only one pen on the notebook in the video?",
            "answer": "yes"
        },
        "negative": {
            "video_id": "kMJMR68Dz4s_b.mp4",
            "question": "Is there only one pen on the notebook in the video?",
            "answer": "no"
        }
    }
```


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



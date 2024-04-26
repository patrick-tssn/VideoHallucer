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
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), 
    - [installation](https://github.com/PKU-YuanGroup/Video-LLaVA?tab=readme-ov-file#%EF%B8%8F-requirements-and-installation)
    - [ckpt](https://huggingface.co/LanguageBind/Video-LLaVA-7B)
- [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA), 
    - [installation](https://github.com/DAMO-NLP-SG/Video-LLaMA?tab=readme-ov-file#usage)
    - [ckpt](https://github.com/DAMO-NLP-SG/Video-LLaMA?tab=readme-ov-file#pre-trained--fine-tuned-checkpoints)
- [Video Chat2](https://github.com/OpenGVLab/Ask-Anything), 
    - [installation](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2#usage)
    - [ckpt: llama-7b](https://github.com/OpenGVLab/Ask-Anything/issues/150), [ckpt: UMT, stage2, stage3, vicuna-delta + script](https://github.com/OpenGVLab/Ask-Anything/issues/130)
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT), 
    - [installation](https://github.com/mbzuai-oryx/Video-ChatGPT?tab=readme-ov-file#installation-wrench)
    - [ckpt: Video-ChatGPT-7B, LLaVA-Lightening-7B-v1-1](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/docs/offline_demo.md#download-video-chatgpt-weights)
- [Valley](https://github.com/RupertLuo/Valley)
    - [installation](https://github.com/RupertLuo/Valley?tab=readme-ov-file#install)
    - [ckpt](https://huggingface.co/luoruipu1/Valley2-7b)


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
    ├── external_factual
        ├── external_factual.json
        └── videos/*.mp4
    ├── external_nonfactual
        ├── external_nonfactual.json
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


external_factual
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|   Gemini-1.5-pro   |      |      |      |
|  VideoLLaVA    |      |      |      |


external_nonfactual
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|   Gemini-1.5-pro   |      |      |      |
|  VideoLLaVA    |      |      |      |



overall
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|   Gemini-1.5-pro   |      |      |      |
|  VideoLLaVA    |      |      |      |



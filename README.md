# HaVeBench

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



possible baselines

## Installation

Follow existing repositories to install the environments:

- [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA), 
    - [installation](https://github.com/DAMO-NLP-SG/Video-LLaMA?tab=readme-ov-file#usage) (torchaudio error: OSError: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory --> reinstall torchaudio)
    - [checkpoint: Video-LLaMA-2-7B-Finetuned](https://github.com/DAMO-NLP-SG/Video-LLaMA?tab=readme-ov-file#pre-trained--fine-tuned-checkpoints), [checkpoint: VIT](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), [checkpoint: qformer](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth)
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT), 
    - [installation](https://github.com/mbzuai-oryx/Video-ChatGPT?tab=readme-ov-file#installation-wrench)
    - [checkpoints: Video-ChatGPT-7B, LLaVA-Lightening-7B-v1-1](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/docs/offline_demo.md#download-video-chatgpt-weights), [checkpoints: clip-vit](https://huggingface.co/openai/clip-vit-large-patch14)
- [Valley2](https://github.com/RupertLuo/Valley)
    - [installation](https://github.com/RupertLuo/Valley?tab=readme-ov-file#install)
    - [checkpoint: Valley2-7b](https://huggingface.co/luoruipu1/Valley2-7b)
- [Video Chat2](https://github.com/OpenGVLab/Ask-Anything), 
    - [installation](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2#usage) (ERROR: Could not find a version that satisfies the requirement torch==1.13.1+cu117 --> pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117) (pip install packaging) (inference doesn't need flash-attention)
    - [checkpoint: llama-7b](https://github.com/OpenGVLab/Ask-Anything/issues/150), [checkpoints: UMT-L-Qformer, VideoChat2_7B_stage2, VideoChat2_7B_stage3, Vicuna-7B-delta + script](https://github.com/OpenGVLab/Ask-Anything/issues/130)
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), 
    - [installation](https://github.com/PKU-YuanGroup/Video-LLaVA?tab=readme-ov-file#%EF%B8%8F-requirements-and-installation)
    - [checkpoint: Video-LLaVA-7B](https://huggingface.co/LanguageBind/Video-LLaVA-7B)
- [VideoLaVIT](https://github.com/jy0205/LaVIT/tree/main/VideoLaVIT)
    - [installation](https://github.com/jy0205/LaVIT/tree/main/VideoLaVIT#requirements) (note: package motion-vector-extractor not supported on CentOS) (pip install accelerate)
    - [checkpoint: Video-LaVIT-v1](https://huggingface.co/rain1011/Video-LaVIT-v1/tree/main/language_model_sft)
- [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID)
    - [installation](https://github.com/dvlab-research/LLaMA-VID?tab=readme-ov-file#install)
    - [checkpoint: llama-vid-7b-full-224-video-fps-1](https://huggingface.co/YanweiLi/llama-vid-7b-full-224-video-fps-1), [checkpoint: eva_vit_g](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), [checkpoint: bert](https://huggingface.co/openai/bert-base-uncased)
- [Gemini API](https://github.com/google-gemini/cookbook)

note: if decord --> raise DECORDError(err_str) --> conda install ffmpeg




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


## Usage

inference
```bash
cd baselines
python ../model_testing_zoo.py --model_name Gemini-1.5-pro # ["VideoChatGPT", "Valley", "Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "Gemini-1.5-pro"])
```

inference on HaVeBench
```bash
cd baselines
python ../evaluations/evaluation.py  --model_name Gemini-1.5-pro --eval_obj # [--eval_]
```



## Leaderboard

object-relation
|  Model    |  Basic    |  Halluciantion     |  Overall |
| ---- | ---- | ---- | ---- |
|  Gemini-1.5-pro    |  0.86    |  0.56    | 0.52     |
|  VideoLLaVA    |  0.95    | 0.38     | 0.345     |
|  Video-LLaMA-2    |  0.885    | 0.18     | 0.155     |

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



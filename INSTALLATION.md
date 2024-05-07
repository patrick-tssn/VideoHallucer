

To set up the environments, follow the instructions in the existing repositories and download the necessary checkpoints. Additionally, we offer guidance for this step, which addresses potential issues such as package version conflicts and system-related problems.


*(optional) checkpoints allow for manual downloading; otherwise, the model will download automatically if the Internet works fine.*


- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)
    - Installation: [Instruction](https://github.com/mbzuai-oryx/Video-ChatGPT?tab=readme-ov-file#installation-wrench) 
    - Checkpoints:
        - Source: [Video-ChatGPT-7B, LLaVA-Lightening-7B-v1-1](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/docs/offline_demo.md#download-video-chatgpt-weights), [clip-vit (optional)](https://huggingface.co/openai/clip-vit-large-patch14)
        - Structure:
            ``` 
                ├── checkpoints/Video-ChatGPT-7B
                    ├── LLaVA-7B-Lightening-v1-1
                    ├── Video-ChatGPT-7B
                    └── clip-vit-large-patch14 (optional)
            ```

- [Valley2](https://github.com/RupertLuo/Valley)
    - Installation: [Instruction](https://github.com/RupertLuo/Valley?tab=readme-ov-file#install)
    - Checkpoints:
        - Source: [Valley2-7b](https://huggingface.co/luoruipu1/Valley2-7b)
        - Structure:
            ``` 
                ├── checkpoints/Valley2-7b
            ```

- [Video-LLaMA-2](https://github.com/DAMO-NLP-SG/Video-LLaMA) 
    - Installation: [Instruction](https://github.com/DAMO-NLP-SG/Video-LLaMA?tab=readme-ov-file#usage) 
        - Possible Issues: `torchaudio error: OSError: libtorch_cuda_cpp.so: cannot open shared object file: No such file or directory`  --> Solution: reinstall torchaudio
    - Checkpoints: 
        - Source: [Video-LLaMA-2-7B-Finetuned](https://github.com/DAMO-NLP-SG/Video-LLaMA?tab=readme-ov-file#pre-trained--fine-tuned-checkpoints), [VIT (optional)](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), [qformer (optional)](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth)
        - Structure: 
            ``` 
                ├── checkpoints/Video-LLaMA-2-7B-Finetuned
                    ├── AL_LLaMA_2_7B_Finetuned.pth
                    ├── imagebind_huge.pth
                    ├── llama-2-7b-chat-hf
                    ├── VL_LLaMA_2_7B_Finetuned.pth
                    ├── blip2_pretrained_flant5xxl.pth (optional)
                    └── eva_vit_g.pth (optional)
            ```


- [VideoChat2](https://github.com/OpenGVLab/Ask-Anything)
    - Installation: [Instruction](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2#usage)
        - Possible Issue 1: `ERROR: Could not find a version that satisfies the requirement torch==1.13.1+cu117` --> Solution: `pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`
        - Possible Issue 2: `flash-attention error` --> Solution: inference doesn't need flash-attention
    - Checkpoints: 
        - Source: [llama-7b](https://github.com/OpenGVLab/Ask-Anything/issues/150), [UMT-L-Qformer, VideoChat2_7B_stage2, VideoChat2_7B_stage3, Vicuna-7B-delta + script](https://github.com/OpenGVLab/Ask-Anything/issues/130)
        - Structure: 
            ``` 
                ├── checkpoints/VideoChat2
                    ├── umt_l16_qformer.pth
                    ├── videochat2_7b_stage2.pth
                    ├── videochat2_7b_stage3.pth
                    └── vicuna-7b-v0
            ```
    
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), 
    - Installation: [Instruction](https://github.com/PKU-YuanGroup/Video-LLaVA?tab=readme-ov-file#%EF%B8%8F-requirements-and-installation)
    - Checkpoints:
        - Source: [Video-LLaVA-7B](https://huggingface.co/LanguageBind/Video-LLaVA-7B), [LanguageBind_Video (optional)](https://huggingface.co/LanguageBind/LanguageBind_Video_merge), [LanguageBind_Image (optional)](https://huggingface.co/LanguageBind/LanguageBind_Image)
        - Structure: 
            ``` 
                ├── checkpoints/VideoLLaVA
                    ├── Video-LLaVA-7B
                    ├── LanguageBind_Video_merge (optional)
                    └── LanguageBind_Image (optional)
            ```


- [VideoLaVIT](https://github.com/jy0205/LaVIT/tree/main/VideoLaVIT)
    - Installation: [Instruction](https://github.com/jy0205/LaVIT/tree/main/VideoLaVIT#requirements)
        - Possible Issue: `package motion-vector-extractor not supported on CentOS` --> No Alternative on CentOS
        - Possible Issue: `missing accelerate, apex`
    - Checkpoints: 
        - Source: [Video-LaVIT-v1](https://huggingface.co/rain1011/Video-LaVIT-v1/tree/main/language_model_sft)
        - Structure: 
            ``` 
                ├── checkpoints/Video-LaVIT-v1/language_model_sft 
            ```

- [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID)
    - Installation: [Instruction](https://github.com/dvlab-research/LLaMA-VID?tab=readme-ov-file#install)
    - Checkpoints:
        - Source: [llama-vid-7b-full-224-video-fps-1](https://huggingface.co/YanweiLi/llama-vid-7b-full-224-video-fps-1), [eva_vit_g](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), [bert (optional)](https://huggingface.co/openai/bert-base-uncased)
        - Structure: 
            ``` 
                ├── checkpoints/VideoChat2
                    ├── llama-vid-7b-full-224-video-fps-1
                    ├── LAVIS/eva_vit_g.pth
                    └── bert-base-uncased (optional)
            ```

- [MiniGPT4-video](https://github.com/Vision-CAIR/MiniGPT4-video)
    - Installation: [Instruction](https://github.com/Vision-CAIR/MiniGPT4-video?tab=readme-ov-file#rocket-demo)
    - Checkpoints:
        - Source: [video_mistral_checkpoint_last](https://huggingface.co/Vision-CAIR/MiniGPT4-Video/blob/main/checkpoints/video_mistral_checkpoint_last.pth), [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [vit (optional)](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)
        - Structure: 
            ``` 
                ├── checkpoints/MiniGPT4-Video
                    ├── checkpoints/video_mistral_checkpoint_last.pth
                    ├── Mistral-7B-Instruct-v0.2
                    └── eva_vit_g.pth (optional)
            ```


- [Gemini API](https://github.com/google-gemini/cookbook)


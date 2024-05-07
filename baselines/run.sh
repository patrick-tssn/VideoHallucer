# conda activate video_chatgpt
# python ../evaluations/evaluation.py --model_name VideoChatGPT --detect_fact
# conda activate valley
# python ../evaluations/evaluation.py --model_name Valley2 --detect_fact
# conda activate videollama
# python ../evaluations/evaluation.py --model_name Video-LLaMA-2 --detect_fact
# conda activate videochat2
# python ../evaluations/evaluation.py --model_name VideoChat2 --detect_fact
# conda activate videollava
# python ../evaluations/evaluation.py --model_name VideoLLaVA --detect_fact
# conda activate llamavid
# python ../evaluations/evaluation.py --model_name LLaMA-VID --detect_fact
# conda activate minigpt4_video
# python ../evaluations/evaluation.py --model_name MiniGPT4-Video --detect_fact


python ../evaluations/evaluation.py --model_name GPT4V --eval_obj_rel
python ../evaluations/evaluation.py --model_name GPT4V --eval_semantic
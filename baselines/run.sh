conda activate video_chatgpt
python ../evaluations/evaluation.py --model_name VideoChatGPT --eval_obj_rel --eval_semantic --eval_temporal --eval_fact --eval_nonfact
conda activate valley
python ../evaluations/evaluation.py --model_name Valley2 --eval_obj_rel --eval_semantic --eval_temporal --eval_fact --eval_nonfact
conda activate videollama
python ../evaluations/evaluation.py --model_name Video-LLaMA-2 --eval_obj_rel --eval_semantic --eval_temporal --eval_fact --eval_nonfact
conda activate videochat2
python ../evaluations/evaluation.py --model_name VideoChat2 --eval_obj_rel --eval_semantic --eval_temporal --eval_fact --eval_nonfact
conda activate videollava
python ../evaluations/evaluation.py --model_name VideoLLaVA --eval_obj_rel --eval_semantic --eval_temporal --eval_fact --eval_nonfact
conda activate llamavid
python ../evaluations/evaluation.py --model_name LLaMA-VID --eval_obj_rel --eval_semantic --eval_temporal --eval_fact --eval_nonfact
conda activate minigpt4_video
python ../evaluations/evaluation.py --model_name MiniGPT4-Video --eval_obj_rel --eval_semantic --eval_temporal --eval_fact --eval_nonfact



# conda activate video_chatgpt
# python ../evaluations/evaluation_exp.py --model_name VideoChatGPT 
# conda activate valley
# python ../evaluations/evaluation_exp.py --model_name Valley2 
# conda activate videollama
# python ../evaluations/evaluation_exp.py --model_name Video-LLaMA-2 
# conda activate videochat2
# python ../evaluations/evaluation_exp.py --model_name VideoChat2 
# conda activate videollava
# python ../evaluations/evaluation_exp.py --model_name VideoLLaVA 
# conda activate llamavid
# python ../evaluations/evaluation_exp.py --model_name LLaMA-VID 
# conda activate minigpt4_video
# python ../evaluations/evaluation_exp.py --model_name MiniGPT4-Video 



# conda activate video_chatgpt
# python ../evaluations/evaluation.py --model_name VideoChatGPT --eval_semantic
# conda activate valley
# python ../evaluations/evaluation.py --model_name Valley2 --eval_semantic
# conda activate videollama
# python ../evaluations/evaluation.py --model_name Video-LLaMA-2 --eval_semantic
# conda activate videochat2
# python ../evaluations/evaluation.py --model_name VideoChat2 --eval_semantic
# conda activate videollava
# python ../evaluations/evaluation.py --model_name VideoLLaVA --eval_semantic
# conda activate llamavid
# python ../evaluations/evaluation.py --model_name LLaMA-VID --eval_semantic
# conda activate minigpt4_video
# python ../evaluations/evaluation.py --model_name MiniGPT4-Video --eval_semantic


# python ../evaluations/evaluation.py --model_name GPT4V --eval_obj_rel
# python ../evaluations/evaluation.py --model_name GPT4V --eval_semantic
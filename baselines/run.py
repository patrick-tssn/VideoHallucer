models = ["VideoChatGPT", "Valley2", "Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "MiniGPT4-Video"]
envs = ["video_chatgpt", "valley", "videollama", "videochat2", "videollava", "llamavid", "videolavit", "minigpt4_video"]

models = ["MiniGPT4-Video"]
envs = ["minigpt4_video"]
tasks = ["obj_rel", "temporal", "semantic", "fact", "nonfact"]
runs = []
for i in range(len(models)):
    for task in tasks:
        model = models[i]
        env = envs[i]
        head = f"""#!/bin/bash
#SBATCH --job-name={env}_{task}
#SBATCH --partition=DGX
##SBATCH --exclude=hgx-hyperplane[02]
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_logs/{model}_{task}.out
#SBATCH --error=./slurm_logs/{model}_{task}.error.out"""
        env_cmd = f"source ~/scratch/anaconda3/bin/activate\nconda activate {env}"
        # run_cmd = f"python ../evaluations/evaluation.py --model_name {model} --eval_obj_rel --eval_semantic --eval_temporal --eval_fact --eval_nonfact"
        run_cmd = f"python ../evaluations/evaluation.py --model_name {model} --eval_{task}"

        with open(f"run_{env}_{task}.slurm", "w") as fp:
            fp.writelines(head+"\n\n")
            fp.writelines(env_cmd+"\n")
            fp.writelines(run_cmd)
        runs.append(f"run_{env}_{task}.slurm")


with open("run_batch.sh", "w") as fp:
    for r in runs:
        fp.writelines("sbatch " + r+"\n")

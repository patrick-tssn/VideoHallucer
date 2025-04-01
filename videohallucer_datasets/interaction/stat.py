import os
import json
from moviepy.editor import VideoFileClip
from tqdm import tqdm


inter = json.load(open("interaction.json"))

vid_dur = 0
vid_cnt = 0
q_dur = 0
q_cnt = 0

for dct in tqdm(inter):
    vid = "videos/" + dct["basic"]["video"]
    q1 = len(dct["basic"]["question"].split())
    q2 = len(dct["hallucination"]["question"].split())
    q_dur += (q1 + q2)
    q_cnt += 2
    video = VideoFileClip(vid)
    vid_dur += video.duration
    vid_cnt += 1
    
print(vid_dur / vid_cnt)
print(q_dur / q_cnt)
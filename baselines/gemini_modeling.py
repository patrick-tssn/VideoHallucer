import os
import time
import random
random.seed(43)
import numpy as np
import google.generativeai as genai

from gemini.extract_frames import extract_frame_from_video
from gemini.upload import File, get_timestamp, make_request

from base import ViLLMBaseModel

GOOGLE_API_KEY_POOL = [
    "AIzaSyAWR09KnE9ICCpTjqvz7fgm5h6kEUSIS70",
    "AIzaSyAouD5zRi9SNI1OKhkZMMkk1vdgvEygDbg",
    "AIzaSyBXjPXXi8QkDDSMUR9fazcyz7pT9w_GUBQ",
    "AIzaSyD8OVqYxBncBPt-pyMfANbJ9pK0ctHYYUM",
    "AIzaSyACfh3WrxFNkg3twPRi-HPPW4VdkiPHlmg"
]

class Gemini(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args["model_path"], model_args["device"])
        assert (
            "model_path" in model_args
            and "device" in model_args
        )

        # self.frame_extraction_directory = model_args["frame_path"]\
        self.frame_extraction_directory = "./cache_dir/gemini10"
        self.frame_prefix = "_frame"

    def generate(self, instruction, video_path):
        # assert len(videos) == 1
        # video_path = videos[0]
        # instruction = instruction[0] if type(instruction)==list else instruction

        # 1. extract frame: default 1 fps
        extract_frame_from_video(video_path, self.frame_extraction_directory, self.frame_prefix)

        # 2. upload frame to gemini
        files = os.listdir(self.frame_extraction_directory)
        files = sorted(files)
        files_to_upload = []
        max_frame_length = 128
        if len(files) > max_frame_length:
            sample_idx = np.linspace(0, len(files)-1, max_frame_length, dtype=int).tolist() # FIXME: the API is easily broken up
            for idx in sample_idx:
                files_to_upload.append(
                    File(file_path=os.path.join(self.frame_extraction_directory, files[idx]), frame_prefix=self.frame_prefix))
        else:
            for file in files:
                files_to_upload.append(
                    File(file_path=os.path.join(self.frame_extraction_directory, file), frame_prefix=self.frame_prefix))

        # for file in files:
        #         files_to_upload.append(
        #             File(file_path=os.path.join(self.frame_extraction_directory, file), frame_prefix=self.frame_prefix))



        while 1:
            try:
                GOOGLE_API_KEY = random.choice(GOOGLE_API_KEY_POOL)
                genai.configure(api_key=GOOGLE_API_KEY)
                # Upload the files to the API
                # Only upload a 10 second slice of files to reduce upload time.
                # Change full_video to True to upload the whole video.
                full_video = True
                uploaded_files = []
                # print(f'Uploading {len(files_to_upload) if full_video else 10} files. This might take a bit...')
                for file in files_to_upload if full_video else files_to_upload[:10]:
                    # print(f'Uploading: {file.file_path}...')
                    response = genai.upload_file(path=file.file_path)
                    file.set_file_response(response)
                    uploaded_files.append(file)
                # print(f"Completed file uploads!\n\nUploaded: {len(uploaded_files)} files")

                # 3. generate
                # Create the prompt.
                prompt = instruction
                # Set the model to Gemini 1.5 Pro.
                model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
                request = make_request(prompt, uploaded_files)
                # print(request)
                response = model.generate_content(request,
                                                request_options={"timeout": 600})

                # delete uploaded files to save quota
                # print(f'Deleting {len(uploaded_files)} images. This might take a bit...')
                for file in uploaded_files:
                    genai.delete_file(file.response.name)
                    # print(f'Deleted {file.file_path} at URI {file.response.uri}')
                # print(f"Completed deleting files!\n\nDeleted: {len(uploaded_files)} files")
                response = response.text
                time.sleep(random.randint(5, 10))
                break
            except Exception as e:
                print(e)
                if "blocked" in str(e):
                    response = ""
                    break
                time.sleep(random.randint(0, 10))
        return response
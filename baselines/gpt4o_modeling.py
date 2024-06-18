import os, shutil, cv2
from gpt4o.api_wrap import OpenAIAPIWrapper

from base import ViLLMBaseModel


class GPT4O(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args["model_path"], model_args["device"])
        assert (
            "model_path" in model_args
            and "device" in model_args
        )

        self.model = OpenAIAPIWrapper()
        self.model_name = 'GPT4O'

    def generate(self, instruction, video_path):
        
        response, num_tokens = self.model.get_completion(instruction, video_path=video_path)
        response = response.strip()

        return response

def create_frame_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)


import os, shutil, cv2
from gpt4v.api_wrap import OpenAIAPIWrapper

from base import ViLLMBaseModel


class GPT4V(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args["model_path"], model_args["device"])
        assert (
            "model_path" in model_args
            and "device" in model_args
        )

        self.model = OpenAIAPIWrapper()

    def generate(self, instruction, video_path):
        
        image_path = extract_frame_from_video(video_path)
        response, num_tokens = self.model.get_completion(instruction, image_path=image_path)
        response = response.strip()

        return response

def create_frame_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

def extract_frame_from_video(video_file_path, FPS=1):
    # print(f"Extracting {video_file_path} at 1 frame per second. This might take a bit...")
    # print(video_file_path)
    FRAME_EXTRACTION_DIRECTORY = "./cache_dir/gpt4v"
    FRAME_PREFIX = "gpt4v_"
    create_frame_output_dir(FRAME_EXTRACTION_DIRECTORY)
    vidcap = cv2.VideoCapture(video_file_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps  # Time interval between frames (in seconds)
    output_file_prefix = os.path.basename(video_file_path).replace('.', '_')
    frame_count = 0
    count = 0
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success: # End of video
            break
        if int(count / fps) == frame_count: # Extract a frame every second
            min = frame_count // 60
            sec = frame_count % 60
            time_string = f"{min:02d}:{sec:02d}"
            image_name = f"{output_file_prefix}{FRAME_PREFIX}{time_string}.jpg"
            output_filename = os.path.join(FRAME_EXTRACTION_DIRECTORY, image_name)
            cv2.imwrite(output_filename, frame)
            frame_count += 1
        count += 1
    vidcap.release() # Release the capture object\n",
    files = os.listdir(FRAME_EXTRACTION_DIRECTORY)
    files = sorted(files)
    selected_file = files[len(files)//2]
    file_path = os.path.join(FRAME_EXTRACTION_DIRECTORY, selected_file)
    return file_path
    # print(f"Completed video frame extraction!\n\nExtracted: {frame_count} frames")


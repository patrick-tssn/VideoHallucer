import os
# import google.generativeai as genai

class File:
    def __init__(self, file_path: str, frame_prefix: str, display_name: str = None):
        self.file_path = file_path
        if display_name:
            self.display_name = display_name
        self.timestamp = get_timestamp(file_path, frame_prefix)

    def set_file_response(self, response):
        self.response = response

def get_timestamp(filename, FRAME_PREFIX):
    """Extracts the frame count (as an integer) from a filename with the format
        'output_file_prefix_frame00:00.jpg'.
    """
    parts = filename.split(FRAME_PREFIX)
    # print(parts)
    if len(parts) != 2:
        return None  # Indicates the filename might be incorrectly formatted
    return parts[1].split('.')[0] 

def make_request(prompt, files):
    request = [prompt]
    for file in files:
        request.append(file.timestamp)
        request.append(file.response)
    return request

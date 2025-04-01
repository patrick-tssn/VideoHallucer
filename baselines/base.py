import torch

class ViLLMBaseModel(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device=device
        self.model_path=model_path

    def forward(self, instruction, videos):
        return self.generate(instruction, videos)
    
    def generate(self, instruction, videos):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        raise NotImplementedError
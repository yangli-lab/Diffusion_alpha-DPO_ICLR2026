
# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
from torch import nn
from glob import glob
from tqdm import tqdm
import json

# load model

class PickScore(nn.Module):
    def __init__(self, processor_name_or_path=None, model_pretrained_name_or_path=None):
        super().__init__()
        if processor_name_or_path == None:
            processor_name_or_path = "./CLIP-ViT-H-14-laion2B-s32B-b79K"
        if model_pretrained_name_or_path == None:
            model_pretrained_name_or_path = "./PickScore_v1"
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to("cuda")

    """
        return: list []
    """
    def calc_probs(self, prompt: str, images: list, relative=False):
        
        # preprocess
        images = [Image.open(image).convert("RGB") for image in images]
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to("cuda")
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            # score
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            # get probabilities if you have multiple images to choose from
            if relative:
                probs = torch.softmax(scores, dim=-1)
            else:
                probs = scores
        return probs.cpu().tolist()

if __name__ == "__main__":
    pickscore = PickScore()
    test_image = ["test.png"]
    prompt = "test_prompt"
    score = pickscore.calc_probs(prompt, test_image)
    print(score)

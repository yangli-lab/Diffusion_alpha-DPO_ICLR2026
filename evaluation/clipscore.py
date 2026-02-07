from evaluation.clip_model import CLIPEvaluator
from torchvision import transforms
import cv2
from PIL import Image
from glob import glob
from tqdm.auto import tqdm
from torch import nn
   
"""
    return: list []
"""
class CLIP_Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.clipmodel = CLIPEvaluator('cuda')
    def process_clip(self, files, prompt):
        clip_scores = []
        for file in files:
            com_tensor = transforms.ToTensor()
            img = com_tensor(Image.open(file).convert("RGB"))
            text_image_sim = self.clipmodel.txt_to_img_similarity([prompt], img.unsqueeze(0)).cpu().tolist()
            clip_scores.append(text_image_sim)
        return clip_scores

if __name__ == "__main__":
    clipscore = CLIP_Score()
    test_image = ["test.png"]
    prompt = "test prompt"
    clip_scores = clipscore.process_clip(test_image, prompt)
    print(clip_scores)

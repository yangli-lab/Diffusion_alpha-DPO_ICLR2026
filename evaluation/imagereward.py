import ImageReward as RM
from glob import glob
from tqdm.auto import tqdm
import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

"""
    return: list []
"""
model = RM.load("ImageReward-v1.0")
# model = RM.load("/root/.cache/ImageReward")

if __name__ == "__main__":
    test_image = ["test.png"] * 2
    prompt = "test prompt"
    imagereward = model.score(prompt, test_image)
    print(imagereward)

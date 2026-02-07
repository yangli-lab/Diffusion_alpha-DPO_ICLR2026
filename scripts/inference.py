import os
import torch.distributed as dist
import torch


import json
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.utils import make_image_grid
import random
import numpy as np
import sys

def seed_everything(random_seed):
    np.random.seed(int(random_seed))
    torch.manual_seed(int(random_seed))
    torch.cuda.manual_seed(int(random_seed))
    generator = torch.manual_seed(random_seed)

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    seed_everything(42)

    dataset = sys.argv[1]

    if dataset == 'pick2pic_v2_test':
        with open('./evaluation/prompts/pick2pic_v2_test_unique.txt', 'r') as f:
            lines = f.readlines()
    elif dataset == 'pick2pic_v2_validation':
        with open('./evaluation/prompts/pick2pic_v2_validation_unique.txt', 'r') as f:
            lines = f.readlines()
    elif dataset == 'hpdv2':
        with open('./evaluation/prompts/scores/hpdv2_prompt/hpdv2_prompt.json', 'r') as f:
            lines = json.load(f)
    elif dataset == 'partiprompt':
        with open('./evaluation/prompts/PartiPrompts.txt', 'r') as f:
            lines = f.readlines()
    else:
        with open('./evaluation/prompts/pick2pic_v2_test_unique.txt', 'r') as f:
            lines = f.readlines()[:100]
    # load pipeline
    inference_dtype = torch.float16

    lora_weights_path = {
            'alpha_dpo_official': '/path/to/ckpt'
        }

    for k, lora_weights_path in lora_weights_path.items():
        if k == "dpo_sdxl_official":
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "/path",
                torch_dtype=inference_dtype,
            ).to("cuda")
            pipe.scheduler = DDIMScheduler.from_config(
                pipe.scheduler.config
            )
            pipe.vae = AutoencoderKL.from_pretrained(
                'path',
                torch_dtype=inference_dtype,
            ).to("cuda")
            unet_id = "/path"
            pipe.unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "path",
                torch_dtype=inference_dtype,
            ).to("cuda")
            pipe.scheduler = DDIMScheduler.from_config(
                pipe.scheduler.config
            )
            pipe.vae = AutoencoderKL.from_pretrained(
                'path',
                torch_dtype=inference_dtype,
            ).to("cuda")
            unet_id = lora_weights_path
            pipe.unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
        output_base_dir = None
            for li, line in enumerate(lines):
                if li % world_size != rank:
                    continue
                    
                prompt = line
                # prompt = "" # uncondition
                generator=torch.Generator(device='cuda').manual_seed(42)
                images = pipe(
                    prompt=prompt,
                    generator=generator,
                    guidance_scale=5,
                    output_type='pil',
                    num_images_per_prompt=4,
                    num_inference_steps=50,
                ).images
                grid_image = make_image_grid(images, 2,2)
                output_dir = os.path.join(output_base_dir, str(li).zfill(5))
                os.makedirs(output_dir, exist_ok=True)
                for ii, image in enumerate(images):
                    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
                    image.save(os.path.join(output_dir, 'samples', f"{str(ii).zfill(4)}.png"))
                grid_image.save(os.path.join(output_dir,  'grid.png'))
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import os
from dotenv import load_dotenv

load_dotenv(".env")

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
YOUR_TOKEN = os.environ.get("YOUR_TOKEN")
 
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)
pipe.to(DEVICE)

with autocast(DEVICE):
    prompt = "kawaii cat"
    while prompt != "q": 
        prompt = input("set prompt [36m>[m ")
        if prompt == "q":
            break
        image = pipe(prompt, guidance_scale=7.5)["sample"][0]
        save_path = os.environ.get("USERPROFILE") + "\\Documents\\stable-diffusion\\" + prompt + ".png"
        image.save(save_path)
        print("saved : ", save_path)


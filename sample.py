import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
import os
from dotenv import load_dotenv

load_dotenv(".env")

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
YOUR_TOKEN = os.environ.get("YOUR_TOKEN")
MAX_SAME_PROMPT = 100
 
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)
pipe.to(DEVICE)

def get_save_path(prompt, count):
    return os.environ.get("USERPROFILE") + "\\Documents\\stable-diffusion\\" + prompt + "_" + str(count).zfill(3) + ".png"

with autocast(DEVICE):
    prompt = "kawaii cat"
    while prompt != "q": 
        buf = input("set prompt (input \"q\" to exit) [36m>[m ")
        if buf == "q":
            break
        if buf != "":
           prompt = buf
        image = pipe(prompt, guidance_scale=7.5)["sample"][0]
        count = 0
        save_path = get_save_path(prompt, count)
        while count <= MAX_SAME_PROMPT:
            if os.path.exists(save_path):
                save_path = get_save_path(prompt, count)
                count += 1
            else:
                image.save(save_path)
                print("saved : ", save_path)
                break


# Read the License and agree with its terms
# https://huggingface.co/CompVis/stable-diffusion-v1-4
#
# If you not have YOUR_TOKEN, then create your account.
# https://huggingface.co/join
#
# Create YOUR_TOKEN on this site
# https://huggingface.co/settings/tokens
# 
# Create `.env` referring to `sample.env`
# Create `$env:USERPROFILE\Documents\stable-diffusion`
# pip install ftfy
# pip install python-dotenv
# 
# Finally, run `python ./sample.py`

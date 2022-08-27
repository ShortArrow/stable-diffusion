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

def ANSIEC(num: int) -> str:
    return "[{code}m".format(code = num)

def get_max_path_length() -> str:
    return 260

def make_save_path(prompt: str, count: int) -> str:
    return (os.environ.get("USERPROFILE")
            + "\\Documents\\stable-diffusion\\"
            + prompt + "_" + str(count).zfill(3) + ".png")

def get_save_path(prompt: str, count: int) -> str:
    buf = make_save_path(prompt, count)
    # make file name length shorten to os can handle
    if len(buf) > get_max_path_length():
        num = (len(prompt)-(len(buf)-get_max_path_length()))
        prompt = prompt[0:num]
        buf = make_save_path(prompt, count)
    return buf

def get_exit_command() -> str:
    return "q"

def get_input_prompt() -> str:
    return ("set prompt (input \""
            + get_exit_command()
            + "\" to exit) "
            + ANSIEC(36) + ">" + ANSIEC(0))

with autocast(DEVICE):
    prompt = "kawaii cat"
    while prompt != get_exit_command(): 
        buf = input(get_input_prompt())
        if buf == get_exit_command():
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

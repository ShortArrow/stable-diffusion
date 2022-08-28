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
    if len(repr(buf)) > get_max_path_length():
        num = (len(prompt)-(len(repr(buf))-get_max_path_length()))
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
                print("prompt : {", prompt, "}")
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
# wait a minute, appear set prompt>
#
# prompt making like this.
# https://gchq.github.io/CyberChef/#recipe=Find_/_Replace(%7B'option':'Regex','string':'%5C%5Cn'%7D,',%20',true,false,true,false)&input=YXR0cmFjdGl2ZSBmZW1hbGUKd2VsbC1ncm9vbWVkIGZhY2UKaGlnaGx5IGRldGFpbGVkIGZhY2UKZXhwcmVzc2l2ZSBkZXRhaWxlZCBmYWNlIGFuZCBleWVzCmV4dHJlbWVseSBoaWdoIHF1YWxpdHkgYW5pbWUgaW4gSmFwYW4KZWxlZ2FudCB5b3VuZyBmZW1hbGUKY3V0ZSBmZW1hbGUKcHJldHR5IGZlbWFsZQpsb3ZlbHkgZmVtYWxlCmluIGNpdHlzY2FwZQpoZXIgaXMgc3RhcmluZyBhdCBtZSB3aXRoIGV5ZQp3cml0dGVuIGJ5IGphcGFuZXNlCmZ1bGwgYm9keQptYWtvdG9zaGlua2FpCmtvbm9zdWJhCnRyYWNlcgpqYXBhbmVzZSBhbmltZQ
# keyword are pick from this
# https://lexica.art/

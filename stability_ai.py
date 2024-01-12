#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install pynvml

#Install diffusers
#pip install diffusers transformers accelerate scipy safetensors

#pip install diffusers --upgrade #top upgrade

import gc
import torch
import time

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def report_gpu():
    gc.collect()
    torch.cuda.empty_cache()

report_gpu() #Flush cache

#model_id = "stabilityai/stable-diffusion-2-1"  #low quality images
model_id = "runwayml/stable-diffusion-v1-5"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
torch.cuda.empty_cache() #empty cache to free up memory

prompt = "cute realistic dog playing with a ball"
height = 600  #set lower height / width for lower gpu
width = 600
torch.cuda.empty_cache()
image = pipe(prompt,height,width).images[0]

ts = str(int(time.time())) #timestamp as string

imageName = prompt.replace(" ","-")+"-"+ts+ ".png"

image.save(f'./text-to-image/'+imageName) #change path to save
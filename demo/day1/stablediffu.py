from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("mps")

prompt = "a futuristic city in space"
image = pipe(prompt).images[0]
image.save("test_output.png")
from diffusers import AutoPipelineForText2Image
import torch

device = torch.device("mps")

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)

# Prompt Engineering
prompt = "a muscular person drinking protein shake"
# The higher the guidance_scale, the closer it is to the Prompt, 
# but too high a value can harm the naturalness (recommended 2~7).
# The higher the num_inference_steps, the more detailed the image, 
# but too high a value can harm the naturalness (recommended 4~20).
image = pipe(prompt, guidance_scale=2.0, num_inference_steps=4).images[0]
image.save("protein_shake.png")

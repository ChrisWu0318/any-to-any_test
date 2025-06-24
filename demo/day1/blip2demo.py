from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("mps")

image = Image.open("test_output.png").convert("RGB")
inputs = processor(images=image, return_tensors="pt").to("mps")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print(caption)

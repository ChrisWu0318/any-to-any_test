import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("mps")

# Initialize text-to-image pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# Initialize image-to-text model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def text_to_image(prompt):
    image = pipe(prompt).images[0]
    return image

def image_to_text(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

with gr.Blocks() as demo:
    gr.Markdown("## 🌟 Any-to-Any多模态系统（Mac M4 MPS）")

    with gr.Tab("Text → Image"):
        prompt_input = gr.Textbox(label="输入文本描述")
        image_output = gr.Image()
        btn_img = gr.Button("生成图像")
        btn_img.click(text_to_image, prompt_input, image_output)

    with gr.Tab("Image → Text"):
        image_input = gr.Image()
        text_output = gr.Textbox(label="图片内容描述")
        btn_txt = gr.Button("生成描述")
        btn_txt.click(image_to_text, image_input, text_output)

demo.launch()

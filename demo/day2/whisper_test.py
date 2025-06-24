import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import numpy as np

device = torch.device("mps")  # Mac M4 使用 MPS 后端

# === Text → Image ===
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to(device)

# === Image → Text ===
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# === Audio → Text ===
asr_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to("mps")


# Text → Image
def text_to_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Image → Text
def image_to_text(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

# Audio → Text
def audio_to_text(audio):
    if audio is None:
        return "❌ 没有音频输入"
    sr, waveform = audio  # 注意顺序！

    print("waveform:", type(waveform), waveform.shape if hasattr(waveform, 'shape') else None, "sr:", sr, type(sr))

    # 采样率判定
    try:
        sr_val = int(sr)
    except Exception as e:
        return f"❌ 无法识别音频采样率: {e}"

    if not isinstance(sr_val, int) or sr_val <= 0:
        return "❌ 音频采样率无效"

    # 处理立体声（(2, n)），转单声道（(n,)）
    if isinstance(waveform, np.ndarray) and len(waveform.shape) == 2:
        waveform = waveform.mean(axis=0)

    # 保证 float32
    waveform = waveform.astype(np.float32)

    # whisper 要求采样率 16000
    if sr_val != 16000:
        waveform_torch = torch.tensor(waveform, dtype=torch.float32)
        waveform_torch = waveform_torch.unsqueeze(0) if len(waveform_torch.shape) == 1 else waveform_torch
        waveform_resampled = torchaudio.functional.resample(waveform_torch, sr_val, 16000)
        waveform = waveform_resampled.squeeze().cpu().numpy()

    # 直接传 numpy 数组给 asr_processor
    inputs = asr_processor(waveform, sampling_rate=16000, return_tensors="pt").to(device)
    # 强制英文识别
    forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language="en", task="transcribe")
    predicted_ids = asr_model.generate(inputs["input_features"], forced_decoder_ids=forced_decoder_ids)
    text = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text


# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🌟 Any-to-Any 多模态系统（支持 Text/Image/Audio 互转）")

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

    with gr.Tab("Audio → Text"):
        audio_input = gr.Audio(type="numpy", label="上传音频或录音")
        audio_text_output = gr.Textbox(label="语音识别结果")
        btn_asr = gr.Button("转为文字")
        btn_asr.click(audio_to_text, audio_input, audio_text_output)

demo.launch()

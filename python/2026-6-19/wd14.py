import os
import numpy as np
import csv
import base64
import mimetypes
import sys

from PIL import Image
from pathlib import Path
from onnxruntime import InferenceSession

from ollama import chat
from ollama import ChatResponse
from ollama import Client
from google import genai
from google.genai import types

models_dir = Path.cwd()

defaults = {
    "model": "wd-v1-4-moat-tagger-v2",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "ortProviders": ["CPUExecutionProvider"],
    "HF_ENDPOINT": "https://huggingface.co"
}

def tag(image, model_name, threshold=0.35, character_threshold=0.85, exclude_tags="", replace_underscore=True, trailing_comma=False, client_id=None, node=None):
    name = os.path.join(models_dir, model_name + ".onnx")
    model = InferenceSession(name, providers=defaults["ortProviders"])

    input = model.get_inputs()[0]
    height = input.shape[1]

    # 缩小至最大尺寸，并用白色垫子垫好
    ratio = float(height)/max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    # 读取 CSV 文件中的所有标签，并找到每个类别的起始位置。
    tags = []
    general_index = None
    character_index = None
    with open(os.path.join(models_dir, model_name + ".csv")) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if general_index is None and row[2] == "0":
                general_index = reader.line_num - 2
            elif character_index is None and row[2] == "4":
                character_index = reader.line_num - 2
            if replace_underscore:
                tags.append(row[1].replace("_", " "))
            else:
                tags.append(row[1])

    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    result = list(zip(tags, probs[0]))

    # rating = max(result[:general_index], key=lambda x: x[1])
    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]

    all = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag[0] not in remove]

    res = ("" if trailing_comma else ", ").join((item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for item in all))

    print(res)
    return res


def main(img, client):
    # img = r"G:\img_WD14\Lora07.jpg"

    print("WD14 输出:")
    current_tags = tag(Image.open(img), "wd-eva02-large-tagger-v3")

    
    prompt = f"""
    You are an expert anime image tagging assistant specializing in dataset preparation for training Illustrious and Pony-based LoRA models.
    Your task is to refine, correct, and restructure the raw tags provided by the WD14 tagger into a highly standardized, structured single-line format.

    【WD14 Raw Tags Analysis】
    {current_tags}

    【Strict Output Formatting Requirements】
    1. The output MUST be a single line of raw tags, entirely in lowercase, separated ONLY by a comma and a space (`, `).
    2. Do NOT include any introductory phrases, conversational text, quotes, or Markdown formatting (no ``` or bold text).
    3. Do NOT invent new tags unless absolutely necessary; prioritize standard Danbooru-style tags from the provided WD14 data.
    4. Follow this strict structural ordering for the final tag string:
    [Art Style] -> [Color Palette] -> [Composition/Framing] -> [Lighting] -> [Character Core ("1girl, solo" if single girl)] -> [Facial/Physical Features] -> [Clothing/Outfit] -> [Action/Pose] -> [Background Objects & Setting]

    Example Output Format:
    masterpiece, anime art, vibrant color, full body, cinematic lighting, 1girl, solo, long hair, blue eyes, school uniform, pleated skirt, standing, classroom, window, desk
    """ 

    print(prompt) 

    with open(img, 'rb') as f:
        image_bytes = f.read()


    """
    client = Client(
        host='http://localhost:11434',
        headers={'x-some-header': 'some-value'}
        )
    
    response = client.chat(model='huihui_ai/qwen3.5-abliterated:4B', messages=[{
            'role': 'user',
            'content': f"{prompt}",
            'images': [f"{img}"]
        }],
    options={
            'temperature': 0,
            'num_ctx': 64 * 1024,
            'repeat_penalty': 1.2,
            'seed': 365,
            'top_k': 20,
            'top_p': 0.5
            })

    print("ollama输出:")    
    print(response.message.content)
    """
    
    # 获取文件的真实 MIME 类型
    mime_type, _ = mimetypes.guess_type(img)
    if not mime_type:
        sys.exit(1)
    print(f"图片 MIME 类型：{mime_type}")

    response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt
            ]
        )

    print(response.text)
    return response.text


if __name__ == "__main__":

    # 必须在导入 google-genai 或其他网络请求库之前设置
    proxy_server = "http://192.168.0.236:1090"  # 替换为你的本地代理端口

    os.environ["HTTP_PROXY"] = proxy_server
    os.environ["HTTPS_PROXY"] = proxy_server

    IMAGE_DIR = r"E:\AI\012"
    client = genai.Client()

    if not os.path.exists(IMAGE_DIR):
        print(f"错误找不到目录: {IMAGE_DIR}")
    else:
        success_count = 0

        for file_name in os.listdir(IMAGE_DIR):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue

            # base_name = os.path.split(file_name)[0]
            base_name = Path(file_name).stem
            txt_name = f"{base_name}.txt"

            img_path = os.path.join(IMAGE_DIR, file_name)
            txt_path = os.path.join(IMAGE_DIR, txt_name)

            print(f"✅ 图片: {img_path}开始进行打标")
            txt_tag = main(img_path, client)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(txt_tag)

            print(f"✅ 已成功创建对应的标签文件: {txt_path}")
            success_count += 1

        print(f"🎉 处理完成！共为 {success_count} 张图片创建了对应的 txt 文件。")
    
    client.close()
        
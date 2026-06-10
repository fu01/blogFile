import os
import numpy as np
import csv

from PIL import Image
from pathlib import Path
from onnxruntime import InferenceSession

from my_ollama import chat
from my_ollama import ChatResponse
from my_ollama import Client

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


def main(img):
    # img = r"G:\img_WD14\Lora07.jpg"

    print("WD14 输出:")
    current_tags = tag(Image.open(img), "wd-eva02-large-tagger-v3")

    prompt = f"""
    你是一个专门为 Stable Diffusion LoRA 训练优化文本描述（Caption）的 AI 专家。
    请结合为你提供的图像，以及 WD14 提取的标签，生成一份最适合训练的、精简的【纯英文】自然语言描述。

    【WD14 提取的核心标签】：
    {current_tags}

    【硬性执行规则】：
    1. 输出语言：必须完全使用【纯英文】输出。绝对不能包含任何汉字、标点符号解释或开场白（例如：绝对不要写 "The image shows...", "This is a...", 必须直接以特征单词或短语开始）。
    2. 权重排序：最核心、最重要的特征必须排在最前面（如角色名、发型、核心服装），然后再顺畅地过渡到姿态、表情、画面构图、背景和艺术风格。
    3. 完美融合：把 WD14 提供的精准词组与你的视觉理解整合在一起，拆散并融合成由【名词短语或形容词词组】组成的集合，用逗号 , 分隔。绝对不要出现完整的叙述性句子（主谓宾结构）。例如：应该写 long pink hair, silver helmet, steel plate armor，绝对不要写 she has long pink hair and she wears a silver helmet.
    4. 彻底去污：完全忽略图片中可能出现的任何文字、水印、画师签名或版权提示（例如 "Saiyukako"、"禁止转载" 等）。绝对不能把这些文字写进英文描述中，防止污染 LoRA 模型。
    5. 字数控制：最终生成的纯英文描述必须紧凑、高效，总字数严格控制在 80 个单词以内。
    6. 禁忌词汇：整篇描述中绝对不允许出现以下连接词、代词和谓语动词：is, are, has, wears, clad in, adorned with, showing, with, there is。只能包含描述特征的实体词和修饰词！
    7. 人数与性别必带：整篇描述的【最开头】，必须首先明确给出画面中的人数与性别标签（例如：1girl, solo, 或 1girl, 1boy, 或 2girls, ）。哪怕背景有模糊的配角或路人，也必须根据画面主体，死死锁住最核心的人数状态，绝对不能遗漏！

    请直接输出最终优化后的纯英文描述文本：
    """

    client = Client(
        host='http://localhost:11434',
        headers={'x-some-header': 'some-value'}
        )
    
    response = client.chat(model='huihui_ai/qwen3.5-abliterated:9b', messages=[
        {
            'role': 'user',
            'content': f"{prompt}",
            'images': [f"{img}"]
        },
    ])
    # print(response['message']['content'])
    # or access fields directly from the response object

    print("ollama输出:")
    print(response.message.content)

    return response.message.content

if __name__ == "__main__":
    IMAGE_DIR = r"E:\AI\04"

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
            txt_tag = main(img_path)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(txt_tag)

            print(f"✅ 已成功创建对应的标签文件: {txt_path}")
            success_count += 1

        print(f"🎉 处理完成！共为 {success_count} 张图片创建了对应的 txt 文件。")

        
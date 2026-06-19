# 优化打标的标签
from llama_cpp import Llama
import json
import os
import sys
import shutil

if __name__ == "__main__":
    llm = Llama(
        model_path=r"G:\img_WD14\GGUF\Huihui-Qwen3.5-9B-abliterated-Q4_K_M.gguf",
        n_ctx=16*1024,
        n_gpu_layers=-1,
        flash_attn=True, #  Use flash attention.
        use_mmap=True, #  Use mmap if possible.
        offload_kqv=True, # Offload K, Q, V to GPU.
        chat_format="chatml",
        verbose=False
        )
    
    TXT_DIR = r"E:\AI\010"
    output_folder = r"E:\AI\011"

    if not os.path.exists(TXT_DIR):
        print(f"路径错误: {TXT_DIR}")
        sys.exit(1)

    for file_name in os.listdir(TXT_DIR):
        if not file_name.lower().endswith(('.txt', )):
            continue
            
        txt_path = os.path.join(TXT_DIR, file_name)

        print(f"读取文件: {txt_path}")

        tags_temp = ""
        with open(txt_path, "r", encoding="utf-8") as f:
            tags_temp = f.read()
            print(f"原tags: {tags_temp}")

        content = f"""
        示例输出格式：
        {{
        "think": "思考过程（用中文简述）",
        "final_answer": "最终精简后的标签，格式为英文逗号分隔的单行文本"
        }}
        

        精简以下标签，并按照相同的格式输出精简后的标签(不要换行):
        {tags_temp}

        ## 要求
        1. 属性合并：若有子集标签，则保留子集，删除父集标签
        2. 同义词处理：若标签含义重合，保留更精确的一个
        3. 1girl, solo, 服装, 人物视角, 人物姿态等标签优先保留
        4. 要保留“主体、特征、构图”这三个核心维度
        5. 剔除不重要的标签，只保留重要的标签（最多20个）
        """
        print(content)  

        user_json = {
            "think": {
                "type": "string",
                "description": "思考过程（用中文简述）"
            },
            "final_answer": {
                "type": "string",
                "description": "最终精简后的标签，格式为英文逗号分隔的单行文本）"
            }
        }

        output = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的标签清洗专家。请输出一个 JSON 对象。",
                },
                {"role": "user", "content": content},
            ],
            response_format={
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": user_json,
                    "required": ["think", "final_answer"]
                },
            },
            temperature=0.4,
            top_k=20,
            min_p=0.05,
            top_p=0.5,
            max_tokens=1000
        )

        data = json.loads(output['choices'][0]['message']['content'])
        tags_temp = data['final_answer']


        print('*'*60)
        print(f'优化结果:{tags_temp}')

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(tags_temp)
        print('*'*60)

        shutil.move(txt_path, output_folder)
        print(f"🚚已移动文件: {file_name}")
        print('*'*60)
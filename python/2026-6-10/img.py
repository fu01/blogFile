import os
from collections import Counter
from PIL import Image

def count_image_resolutions(folder_path):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')

    resolution_list = []

    error_count = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)

                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        resolution_list.append((width, height))
                except Exception as e:
                    print(f"无法读取文件 {file}: {e}")
                    error_count += 1

    resolution_counts = Counter(resolution_list)

    print("\n" + "="*30)
    print(f"📊 统计结果 (目标文件夹: {folder_path})")

    print("="*30)
    print(f"总计成功读取图片数: {len(resolution_list)} 张")

    if error_count > 0:
        print(f"无法读取的文件数: {error_count} 张")

    for res, count in resolution_counts.most_common():
        res_str = f"{res[0]} * {res[1]}"
        print(f"{res_str} | {count}张")
    print("="*30)

if __name__ == "__main__":
    target_folder = r"E:\AI\03"

    if os.path.exists(target_folder):
        count_image_resolutions(target_folder)
    else:
        print("图片目录错误!")
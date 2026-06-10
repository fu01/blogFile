import os
import shutil
from PIL import Image

def isolate_specific_images(folder_path, target_size=(512, 512)):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')

    output_folder = os.path.join(folder_path, f"_found_{target_size[0]}x{target_size[1]}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0

    for root, dirs, files in os.walk(folder_path):
        if output_folder in root:
            continue
    
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)

                is_match = False
                try:
                    with Image.open(file_path) as img:
                        if img.size == target_size:
                            is_match = True

                    if is_match:
                        shutil.move(file_path, output_folder)
                        print(f"🚚已移动文件: {file}")
                        count += 1

                except Exception as e:
                    print(f"处理文件 {file} 时出错: {e}")

        print(f"\n🎉 搞定！已将 {count} 张图片复制到：\n{output_folder}")

if __name__ == "__main__":
    target_folder = r"E:\AI\03"
    isolate_specific_images(target_folder, target_size=(769, 512))

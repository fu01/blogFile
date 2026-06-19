import os

from PIL import Image

def center_crop_to_512_768(input_dir, output_dir=None):
    # 支持的图片后缀
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    # 获取文件夹内所有文件
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not files:
        print("未在指定文件夹中找到有效的图片文件。")
        return

    print(f"开始处理，共找到 {len(files)} 张图片...")
    

    for file_name in files:
        img_path = os.path.join(input_dir, file_name)

        # 如果没有指定输出目录，则覆盖原图
        output_path = os.path.join(output_dir, file_name) if output_dir else img_path


        with Image.open(img_path) as img:
            w, h = img.size

            # 确认一下是不是 513*768
            if w == 513 and h == 768:
                # 定义裁剪边界：(左, 上, 右, 下)
                # 左边砍掉 1 像素，剩下 512 像素
                box = (1, 0, 513, 768) 
                cropped_img = img.crop(box)
                cropped_img.save(output_path, quality=95)
                print(f"已完美裁剪: {img_path}")

if __name__ == "__main__":
    input_dir = r"E:\AI\008"
    output_dir = r"E:\AI\008"
    center_crop_to_512_768(input_dir)
import os

from PIL import Image

def resize_images(input_dir, output_dir=None, target_size=1024, mode="max"):
    """
    批量缩放图片尺寸，保持宽高比不变。
    
    :param input_dir: 输入图片文件夹路径
    :param output_dir: 输出图片文件夹路径（如果为 None，则直接覆盖原图）
    :param target_size: 缩放的目标尺寸（像素）
    :param mode: 缩放模式:
                 'max' - 将长边缩小到 target_size
                 'width' - 将宽度统一缩小到 target_size
                 'height' - 将高度统一缩小到 target_size
    """

    # 如果指定了输出目录且不存在，则创建
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    # 支持的图片后缀
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    # 获取文件夹内所有文件
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print("未在指定文件夹中找到有效的图片文件。")
        return
    
    print(f"开始处理，共找到 {len(files)} 张图片...")
    success_count = 0

    for file_name in files:
        input_path = os.path.join(input_dir, file_name)

        # 如果没有指定输出目录，则覆盖原图
        output_path = os.path.join(output_dir, file_name) if output_dir else input_path

        try:
            with Image.open(input_path) as img:
                # 获取原图宽高
                orig_w, orig_h = img.size

                # 根据不同模式计算缩放后的新宽高
                # 哪边长缩哪边
                if mode == "max":
                    if orig_w > orig_h:
                        new_w = target_size
                        new_h = int(orig_h * (target_size / orig_w))
                    else:
                        new_h = target_size
                        new_w = int(orig_w * (target_size / orig_h))
                elif mode == "width":
                    # 固定宽度
                    new_w = target_size
                    new_h = int(orig_h * (target_size / orig_w))
                elif mode == "height":
                    # 固定高度
                    new_h = target_size
                    new_w = int(orig_w * (target_size / orig_h))
                else:
                    print(f"未知的模式: {mode}，跳过此图片。")
                    continue

                # 如果原图已经小于等于目标尺寸，可以根据需要选择是否跳过（这里默认总是调整）
                # 使用高质量的 Resampling.LANCZOS 滤镜进行缩放
                resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                # 保存图片，保持原格式
                resized_img.save(output_path, quality=95)
                print(f"已处理: {file_name} -> {new_w}x{new_h}")
                success_count += 1

        except Exception as e:
            print(f"处理文件失败 {file_name}: {e}")

        print(f"\n处理完成！成功缩放 {success_count}/{len(files)} 张图片。")

if __name__ == "__main__":
    # 填入你的图片文件夹路径
    INPUT_FOLDER = r"E:\AI\008" 
    
    # 填入输出文件夹路径（如果想直接修改原图，这里写 None）
    OUTPUT_FOLDER = None

    # 运行缩放：把长边统一限制在 1024 像素（宽高比不变）
    # mode 可选: "max" (限制长边), "width" (固定宽), "height" (固定高)
    resize_images(
        input_dir=INPUT_FOLDER, 
        output_dir=OUTPUT_FOLDER, 
        target_size=768, 
        mode="max"
    )
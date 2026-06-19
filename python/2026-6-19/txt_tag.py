# tag 去重复
import os
import sys

def tag_one(tag_str):
    tags_string = tag_str

    clean_tags_list = [tag.strip() for tag in tags_string.split(',')]

    # print(clean_tags_list)

    # dict.fromkeys 会创建一个以列表元素为键的字典，键是唯一的，且保持插入顺序
    unique_tags = list(dict.fromkeys(clean_tags_list))
    # print(unique_tags)

    # 使用逗号加空格进行拼接
    result_string = ", ".join(unique_tags)

    # print(result_string)
    return result_string


if __name__ == "__main__":
    TXT_DIR = r"E:\AI\009"

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
            tags = f.read()
            print(f"原tags: {tags}")
            tags_temp = tag_one(tags)

        print('*'*60)
        print(f'去重结果:{tags_temp}')

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(tags_temp)
        print('*'*60)
    
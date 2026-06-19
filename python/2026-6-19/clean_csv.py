import os
import csv


CSV_FILE = r"G:\img_WD14\e621_sfw.csv"
tags = set()

# 🎯 设置词频阈值：只有在原站出现过 500 次以上的词才保留
# 如果你觉得 Token 还是多，可以把 500 改成 1000
MIN_FREQUENCY = 500

try:
    # 使用 Python 自带的 csv 库，它能完美自动处理双引号引发的切分问题
    with open(CSV_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            
            # 核心操作：只要第一列的核心主标签
            main_tag = row[0].strip().lower()

            # 🌟 核心修改：只保留 Category 0 (General Tags)
            tag_category = row[1].strip()
            if tag_category != '0':
                continue

            # 提取第三列的词频数量
            try:
                frequency = int(row[2].strip())
            except ValueError:
                # 如果第三列不是数字（比如脏数据），直接跳过
                continue

            # 过滤掉无意义的空值、单字母和纯数字
            if main_tag and not main_tag.isdigit() and len(main_tag) > 1:
                if frequency >= MIN_FREQUENCY:
                    # 兼容处理圆括号转义，比如把 (clothing) 换成 \(clothing\)
                    if '(' in main_tag and '\\(' not in main_tag:
                        main_tag = main_tag.replace('(', '\\(').replace(')', '\\)')
                    
                    tags.add(main_tag)

    # 排序输出
    tag_list = sorted(list(tags))
    total_tags = len(tag_list)
    estimated_tokens = int(total_tags * 1.3) # 估算 Token

    print(f"✨ 别名脱水大成功！")
    print(f"  - 剥离别名后，最终的核心主标签仅剩: {total_tags} 个")
    print(f"  - 塞入本地大模型预计仅消耗: 约 {estimated_tokens} Tokens")
    print("-" * 50)

    # 导出为你 Ollama Prompt 直接能读的 txt
    with open("wai_pure_main_tags.txt", "w", encoding="utf-8") as out_f:
        out_f.write(", ".join(tag_list))
    print("💾 已成功导出极致纯净字典: wai_pure_main_tags.txt")

except Exception as e:
    print(f"洗表失败: {e}")
    exit()
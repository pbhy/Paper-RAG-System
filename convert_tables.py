import os
import pandas as pd
from bs4 import BeautifulSoup

def convert_html_tables_in_markdown(input_path, output_path):
    # 读取 Markdown 内容
    with open(input_path, "r", encoding="utf-8") as file:
        markdown_text = file.read()

    soup = BeautifulSoup(markdown_text, "html.parser")
    tables = soup.find_all("html")

    print(f"发现 {len(tables)} 个 HTML 表格，正在转换...")

    # 用于替换原始 HTML 表格的文本内容
    replacements = [\## ]

    for i, table in enumerate(tables):
        try:
            # 解析为 DataFrame
            df = pd.read_html(str(table))[0]
            # 转为 Markdown 表格文本
            markdown_table = df.to_markdown(index=False)
            # 保存替换目标
            replacements.append((str(table), markdown_table))
        except Exception as e:
            print(f"⚠️ 表格 {i+1} 解析失败，跳过：{e}")
            continue

    # 替换原始 HTML 表格
    for original, replacement in replacements:
        markdown_text = markdown_text.replace(original, f"\n\n{replacement}\n\n")

    # 写入新的 Markdown 文件
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(markdown_text)

    print(f"✅ 表格转换完成，保存为：{output_path}")

# 使用方法（修改文件路径为你的实际路径）
if __name__ == "__main__":
    input_file = "data/VideoRAG.md"             # 原始 Markdown 文件
    output_file = "data/VideoRAG_cleaned.md"     # 输出的新 Markdown 文件
    convert_html_tables_in_markdown(input_file, output_file)

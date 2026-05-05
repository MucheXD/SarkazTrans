#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反向查找：对于每个 encode 后的字母，找出哪些字符会编码成它
生成一个 26 x (chars_size + 1) 的矩阵到 map.txt
"""

import re
from encode import encode_text

# 26 个字母
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


def load_chars(chars_file):
    """读取字表文件，返回字符列表"""
    chars = []
    with open(chars_file, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.rstrip('\n\r')
            if char:  # 只添加非空行
                chars.append(char)
    return chars


def encode_char(char):
    """对单个字符进行编码"""
    return encode_text(char, recognize_marks=False)


def generate_reverse_map(chars_file, output_file):
    """
    生成反向映射矩阵
    对于每个字母，遍历字表中的所有字符，
    如果该字符的 encode 结果等于该字母，则记为 1，否则为 0
    """
    print("读取字表...")
    chars = load_chars(chars_file)
    print(f"字表大小: {len(chars)}")
    
    print("生成反向映射矩阵...")
    results = []
    
    for letter in ALPHABET:
        print(f"  处理字母 '{letter}'...", end='', flush=True)
        row = [letter]  # 行首是字母标识
        
        for char in chars:
            # 特殊处理：如果字表项是 [UNK]，总是设置为 1
            if char == '[UNK]':
                row.append('1')
                continue

            encoded = encode_char(char)
            if len(encoded) != 1:
                # 说明是特殊字符 肯定是要排除的
                row.append('0')
                continue
            # 如果编码结果等于该字母，记为 1；否则为 0
            if encoded == letter:
                row.append('1')
            else:
                row.append('0')
        
        results.append(' '.join(row))
        print(f" 完成 ({len(chars)} 个字符)")
    
    print("\n输出到文件...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
    
    print(f"完成! 输出文件: {output_file}")
    print(f"矩阵大小: {len(ALPHABET)} x {len(chars) + 1}")


if __name__ == "__main__":
    chars_file = 'collect/data/vocab.txt'
    output_file = 'collect/data/map.txt'
    generate_reverse_map(chars_file, output_file)

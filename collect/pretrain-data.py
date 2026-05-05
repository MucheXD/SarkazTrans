#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert chinese-short-sentences.csv to pretrain.jsonl

Processing pipeline:
1. Cleaning: Remove invalid texts (6 rules)
2. Classification & Extraction: t=0/1 + quote extraction
3. Encoding & Validation: encode_text() + length check
4. Output: JSONL format
"""

import re
import json
import sys
from pathlib import Path

# Load encode_text function
sys.path.insert(0, str(Path(__file__).parent))
from encode import encode_text


# ============================================================================
# Helper Functions
# ============================================================================

def is_brackets_balanced(text):
    """
    Check if brackets/quotes are balanced (flat checking, not nested).
    
    Pairs to check:
    - Single quotes: ' ' 
    - Double quotes: " " (half-width) and " " (full-width)
    - Parentheses: () （）
    - Angle brackets: 《》
    - Square brackets: [] 【】
    - Curly braces: {} 
    
    Returns True if all bracket pairs are balanced.
    """
    # Define bracket pairs as list of tuples to avoid key collision
    bracket_pairs = [
        ("'", "'"),
        ('"', '"'),
        ('"', '"'),
        ('(', ')'),
        ('（', '）'),
        ('《', '》'),
        ('[', ']'),
        ('【', '】'),
        ('{', '}'),
    ]
    
    # Create maps
    opening_to_closing = {op: cl for op, cl in bracket_pairs}
    closing_chars = set(cl for op, cl in bracket_pairs)
    
    stack = []
    for ch in text:
        if ch in opening_to_closing:
            stack.append(ch)
        elif ch in closing_chars:
            if not stack:
                return False
            last_opening = stack[-1]
            if opening_to_closing[last_opening] != ch:
                return False
            stack.pop()
    
    return len(stack) == 0


def has_consecutive_chars(text, threshold=4):
    """
    Check if text has same character/symbol appearing consecutively >= threshold times.
    """
    for i in range(len(text) - threshold + 1):
        if len(set(text[i:i+threshold])) == 1:
            return True
    return False


def is_text_clean(text):
    """
    Apply 5 cleaning rules:
    1. Brackets balanced
    2. Length <= 120
    3. Length > 4
    4. No error symbol '�'
    5. No consecutive chars >= 4 times
    
    Returns True if text passes all rules.
    """
    # Rule 1: Brackets balanced
    if not is_brackets_balanced(text):
        return False
    
    # Rule 2: Not too long
    if len(text) > 120:
        return False
    
    # Rule 3: Not too short
    if len(text) <= 4:
        return False
    
    # Rule 4: No error symbol
    if '�' in text:
        return False
    
    # Rule 5: No consecutive chars
    if has_consecutive_chars(text, threshold=4):
        return False
    
    return True


def classify_text(text):
    """
    Classify text based on length.
    t=1 if len <= 15, else t=0
    """
    return 1 if len(text) <= 15 else 0


def extract_quoted_blocks(text):
    """
    Extract quoted blocks from text using greedy matching.
    
    Rules:
    - Find all opening quotes (" U+201C or " U+0022) and match with closing quote
    - Strip whitespace from extracted content
    - Discard if stripped content ends with : or ：
    - Only keep if length is in [5, 15]
    
    Returns list of extracted texts.
    """
    extracted = []
    i = 0
    
    # Use Unicode escapes to ensure correct characters
    ldq = '\u201c'  # " Left Double Quotation Mark
    rdq = '\u201d'  # " Right Double Quotation Mark
    std_quote = '"'  # " Quotation Mark (half-width)
    
    while i < len(text):
        ch = text[i]
        closing_char = None
        
        # Determine if this is an opening quote and what the closing char is
        if ch == ldq:
            closing_char = rdq
        elif ch == std_quote:
            closing_char = std_quote
        
        if closing_char:
            # Try to find closing quote
            closing_idx = text.find(closing_char, i + 1)
            if closing_idx != -1:
                # Found closing quote
                content = text[i + 1 : closing_idx].strip()
                
                # Check if it ends with : or ：
                if content and content[-1] not in (':', '：'):
                    # Check length [5, 15]
                    if 5 <= len(content) <= 15:
                        extracted.append(content)
                
                i = closing_idx + 1
            else:
                i += 1
        else:
            i += 1
    
    return extracted


def process_line(line):
    """
    Process a single line of text.
    
    Returns list of dicts: [{"t": int, "i": str, "o": str}, ...]
    """
    # Remove all whitespace characters (not just leading/trailing)
    text = re.sub(r"\s+", "", line)
    # Also remove the characters: [ ] |
    text = re.sub(r"[\[\]\|]", "", text)
    if not text:
        return []

    # 统一大小写，避免训练阶段 lower() 引发 i/o 漂移
    text = text.lower()
    
    # Cleaning check
    if not is_text_clean(text):
        return []
    
    results = []
    
    # Classify original text
    t_original = classify_text(text)
    
    # Encode original text
    encoded_original = encode_text(text)
    
    # Final check: length match
    if len(text) != len(encoded_original):
        return []
    
    # Add original text
    results.append({
        "t": t_original,
        "i": encoded_original,
        "o": text
    })
    
    # Extract quoted blocks
    extracted_blocks = extract_quoted_blocks(text)
    for block in extracted_blocks:
        block = block.lower()
        # Encode extracted block
        encoded_block = encode_text(block)
        
        # Final check: length match
        if len(block) != len(encoded_block):
            continue
        
        # Add extracted block with t=1
        results.append({
            "t": 1,
            "i": encoded_block,
            "o": block
        })
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    input_file = Path(__file__).parent / "data" / "chinese-short-sentences.csv"
    output_file = Path(__file__).parent / "data" / "pretrain.jsonl"
    
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    total_lines = 0
    valid_records = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            total_lines += 1
            records = process_line(line)
            for record in records:
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                valid_records += 1
            
            if total_lines % 10000 == 0:
                print(f"Processed {total_lines} lines, {valid_records} valid records so far...")
    
    print(f"\n✓ Done!")
    print(f"Total input lines: {total_lines}")
    print(f"Total output records: {valid_records}")


if __name__ == "__main__":
    main()

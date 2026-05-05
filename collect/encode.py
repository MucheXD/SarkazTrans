# 余数到字母的映射表，按图片内容手动录入
mod56_map = {
    0: 'g', 1: 'k', 2: 'a', 3: 'm', 4: 'z', 5: 't', 6: 'l', 7: 'b',
    8: 'd', 9: 'q', 10: 'i', 11: 'y', 12: 'f', 13: 'u', 14: 'c', 15: 'x',
    16: 'b', 17: 'h', 18: 's', 19: 'j', 20: 'o', 21: 'p', 22: 'r', 23: 'n',
    24: 'w', 25: 'e', 26: 'y', 27: 'g', 28: 't', 29: 'j', 30: 'm', 31: 'e',
    32: 'v', 33: 'c', 34: 'h', 35: 'd', 36: 'x', 37: 's', 38: 'a', 39: 'n',
    40: 'q', 41: 'o', 42: 'l', 43: 'k', 44: 'r', 45: 'v', 46: 'w', 47: 'i',
    48: 'y', 49: 'p', 50: 'j', 51: 'z', 52: 'q', 53: 'u', 54: 'h', 55: 'e'
}

import re

def encode_text(text, recognize_marks=True):
	"""
	输入文本，按字符unicode取模56，用映射表输出编码结果
	
	参数：
	- text: 输入文本
	- recognize_marks: 是否识别特殊标记（默认True）
	  - <@...> 转换为 [
	  - </> 转换为 ]
	  - <...HTML标签...> 转换为 |
	
	例如：
	  "负责<@qu.key>训练</>的教官" 转换为 "by[jg]rhq"
	  '您只要用<image="icon" scale=2>调度券来兑换' 转换为 "yhcv|maysos"
	"""
	if recognize_marks:
		# 处理特殊标记，顺序很重要
		text = re.sub(r'<@[^>]*>', '[', text)  # 匹配 <@...任意字符...>
		text = re.sub(r'</>', ']', text)  # 匹配 </>
		text = re.sub(r'<[^>]*>', '|', text)  # 匹配其他的 <...HTML标签...>
	
	result = []
	for ch in text:
		if ch in ('[', ']', '|'):
			# 特殊标记直接添加到结果中
			result.append(ch)
		else:
			code = ord(ch)
			mod = code % 56
			mapped = mod56_map.get(mod, '?')
			result.append(mapped)
	return ''.join(result)

if __name__ == "__main__":
	s = input("请输入文本：")
	print("编码结果：", encode_text(s))

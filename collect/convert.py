import json
import re
from pathlib import Path
from encode import encode_text

ROLE_MARKER_RE = re.compile(r'\{[A-Za-z]+\}')


def preprocess_dialog_text(text):
	"""
	在正式处理前做基础归一化
	移除空格、换行和换行转义（\n）
	"""
	text = text.replace('\\n', '')
	text = text.replace('\r', '')
	text = text.replace('\n', '')
	text = text.replace(' ', '')
	return text

def clean_dialog_text(text):
	"""
	从dialogText中去除标记符号
	移除：<@...>, </>, <HTML标签>
	"""
	# 移除 <@...> 标记
	text = re.sub(r'<@[^>]*>', '', text)
	# 移除 </> 标记
	text = re.sub(r'</>', '', text)
	# 移除 HTML标签 <...>
	text = re.sub(r'<[^>]*>', '', text)
	return text


def split_dialog_text_by_role_markers(text):
	"""
	如果文本中存在角色标识符，则拆成多个独立片段
	例如：{F}A{M}B -> [A, B]
	"""
	segments = []
	current_segment = []
	has_role_marker = False

	for chunk in re.split(r'(\{[A-Za-z]+\})', text):
		if not chunk:
			continue

		if ROLE_MARKER_RE.fullmatch(chunk):
			has_role_marker = True
			if current_segment:
				segment = ''.join(current_segment)
				if segment:
					segments.append(segment)
				current_segment = []
		else:
			current_segment.append(chunk)

	if current_segment:
		segment = ''.join(current_segment)
		if segment:
			segments.append(segment)

	if has_role_marker:
		return segments

	return [text]

def convert_type_to_t(dialog_type):
	"""
	将 type 字段转换为 t 值
	"""
	if dialog_type in ('dialog', 'summary'):
		return 0
	elif dialog_type == 'option':
		return 1
	else:
		raise ValueError(f"未知的dialog type: {dialog_type}")

def process_json_file(json_path):
	"""
	处理单个JSON文件，返回训练数据列表
	"""
	training_data = []
	
	try:
		with open(json_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
	except json.JSONDecodeError as e:
		print(f"❌ JSON解析失败: {json_path}")
		print(f"   错误: {e}")
		exit(1)
	except Exception as e:
		print(f"❌ 读取文件失败: {json_path}")
		print(f"   错误: {e}")
		exit(1)
	
	# 尝试获取 data.dialog 数组
	if 'data' not in data or 'dialog' not in data['data']:
		print(f"⚠️  跳过: {json_path} - 没有 data.dialog 字段")
		return training_data
	
	dialogs = data['data']['dialog']
	
	# 断言 dialog 是数组
	assert isinstance(dialogs, list), \
		f"{json_path}: data.dialog 应该是数组，实际类型: {type(dialogs)}"
	
	file_name = Path(json_path).name
	
	for idx, dialog in enumerate(dialogs):
		# 断言 dialog 是字典
		assert isinstance(dialog, dict), \
			f"{file_name}[{idx}]: dialog 项应该是对象，实际类型: {type(dialog)}"
		
		# 检查必要字段
		if 'type' not in dialog:
			print(f"❌ {file_name}[{idx}]: 缺少 'type' 字段")
			exit(1)
		
		dialog_type = dialog['type']
		
		# 根据 type 确定使用的文本字段
		if dialog_type == 'dialog':
			text_field = 'dialogText'
		elif dialog_type == 'option':
			text_field = 'optionText'
		elif dialog_type == 'summary':
			text_field = 'summaryText'
		else:
			print(f"❌ {file_name}[{idx}]: 未知的dialog type: {dialog_type}")
			exit(1)
		
		# 检查对应的文本字段
		if text_field not in dialog:
			print(f"❌ {file_name}[{idx}]: 缺少 '{text_field}' 字段 (type={dialog_type})")
			exit(1)
		
		dialog_text = dialog[text_field]
		
		# 断言字段类型
		assert isinstance(dialog_type, str), \
			f"{file_name}[{idx}]: type 应该是字符串，实际: {type(dialog_type)}"
		
		assert isinstance(dialog_text, str), \
			f"{file_name}[{idx}]: {text_field} 应该是字符串，实际: {type(dialog_text)}"
		
		try:
			# 转换type为t
			t = convert_type_to_t(dialog_type)
		
			# 基础预处理
			dialog_text = preprocess_dialog_text(dialog_text)
			dialog_text = dialog_text.lower()
		
			# 如果存在角色标识符，则拆成多个训练样本
			dialog_text_segments = split_dialog_text_by_role_markers(dialog_text)
			
			for dialog_text_segment in dialog_text_segments:
				dialog_text_segment = dialog_text_segment.lower()
				# 清理文本（去除标记符号）
				o = clean_dialog_text(dialog_text_segment)
			
				# 编码文本
				i = encode_text(dialog_text_segment, recognize_marks=True)
			
				# 如果文本内容为空，则跳过此条目
				if not i or not o:
					continue
			
				# 构建训练数据
				training_item = {
					"t": t,
					"i": i,
					"o": o
				}
			
				training_data.append(training_item)
			
		except ValueError as e:
			print(f"❌ {file_name}[{idx}]: {e}")
			print(f"   type 值: {dialog_type}")
			exit(1)
		except AssertionError as e:
			print(f"❌ {file_name}[{idx}]: 断言失败")
			print(f"   {e}")
			exit(1)
		except Exception as e:
			print(f"❌ {file_name}[{idx}]: 处理异常")
			print(f"   {e}")
			exit(1)
	
	return training_data

def main(save_path='data/train.jsonl'):
	# 获取脚本所在目录作为基础路径
	base_dir = Path(__file__).resolve().parent
	json_dir = base_dir / 'data' / 'json'
	save_path = base_dir / save_path
	
	if not json_dir.exists():
		print(f"❌ 目录不存在: {json_dir}")
		exit(1)
	
	# 获取所有JSON文件
	json_files = sorted(json_dir.glob('*.json'))
	
	if not json_files:
		print(f"❌ 在 {json_dir} 中未找到JSON文件")
		exit(1)
	
	print(f"📂 找到 {len(json_files)} 个JSON文件")
	print(f"📝 输出文件: {save_path}")
	print()
	
	total_items = 0
	
	# 确保输出目录存在
	save_path.parent.mkdir(parents=True, exist_ok=True)
	
	try:
		with open(save_path, 'w', encoding='utf-8') as f:
			# 处理每个JSON文件
			for json_file in json_files:
				training_data = process_json_file(str(json_file))
				
				# 输出训练数据到文件
				for item in training_data:
					f.write(json.dumps(item, ensure_ascii=False) + '\n')
					total_items += 1
	except IOError as e:
		print(f"❌ 文件写入失败: {save_path}")
		print(f"   错误: {e}")
		exit(1)
	
	print(f"✅ 完成! 共生成 {total_items} 条训练数据")
	print(f"📄 已保存到: {save_path}")

if __name__ == '__main__':
	main()

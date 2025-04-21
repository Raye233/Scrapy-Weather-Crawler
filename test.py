import re

# 示例字符串
text = "这里有一些数字 123, -456.789, 0.123, -0.456, 和一些非数字 字符串abc123"

# 正则表达式模式
pattern = r'[-+]?\d*\.?\d+'

# 查找所有匹配的数字
numbers = re.findall(pattern, text)

# 打印结果
print(numbers)
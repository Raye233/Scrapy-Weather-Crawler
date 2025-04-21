<<<<<<< HEAD
import re

# 示例字符串
text = "这里有一些数字 123, -456.789, 0.123, -0.456, 和一些非数字 字符串abc123"

# 正则表达式模式
pattern = r'[-+]?\d*\.?\d+'

# 查找所有匹配的数字
numbers = re.findall(pattern, text)

# 打印结果
print(numbers)
=======
import pandas as pd
import numpy as np


filepath = r'year3_weather_data.csv'
features = pd.read_csv(filepath, encoding='gbk')
features['时间'] = pd.to_datetime(features['时间'], format='%Y年%m月%d日').dt.strftime('%Y-%m-%d')
column_name = '空气质量'

# 替换0值为1到10之间的随机数
features[column_name] = features[column_name].apply(lambda x: np.random.randint(1, 11) if x == 0 else x)
features.to_csv(filepath, index=False)
>>>>>>> 0f0260a (first commit)

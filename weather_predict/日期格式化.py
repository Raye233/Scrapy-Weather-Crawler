import re
import pandas as pd
import numpy as np


filepath = r'F:\Rayedata2\weather_crawl\combined_file.csv'
features = pd.read_csv(filepath, encoding='gbk')
features['时间'] = pd.to_datetime(features['时间'], format='%Y年%m月%d日').dt.strftime('%Y-%m-%d')
column_name = '空气质量'

# 替换0值为1到10之间的随机数
features[column_name] = features[column_name].apply(lambda x: np.random.randint(1, 11) if x == 0 else x)
features.to_csv(filepath, index=False)


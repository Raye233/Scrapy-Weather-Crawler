import re
import pandas as pd
import numpy as np
from tools import charset_detect


def format_date(filepath):
    encoding = charset_detect.detect_encoding(filepath)
    features = pd.read_csv(filepath, encoding=encoding)
    try:
        features['时间'] = pd.to_datetime(features['时间'], format='%Y年%m月%d日').dt.strftime('%Y-%m-%d')
    except:
        pass
    column_name = '空气质量'
    # 替换0值为1到10之间的随机数
    features[column_name] = features[column_name].apply(lambda x: np.random.randint(1, 11) if x == 0 else x)
    features.to_csv(filepath, index=False)


if __name__ == '__main__':
    filepath = None
    format_date(filepath)

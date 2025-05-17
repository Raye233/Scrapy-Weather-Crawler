import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import matplotlib
import chardet

rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}

path = r'../combined_file.csv'
# 1. 数据准备与清洗
# =================================================================
# 读取数据（假设文件名为 weather_data.csv）

def detect_csv_encoding(file_path):
    with open(file_path, 'rb') as f:
        res = chardet.detect(f.read())
        return res['encoding']


df = pd.read_csv(path,
                 encoding=detect_csv_encoding(path),
                 parse_dates=['时间'],
                 date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'),
                 sep=',',
                 )

# 数据清洗示例
df = df.dropna()  # 删除缺失值
df = df[(df['湿度'] >= 0) & (df['湿度'] <= 100)]  # 过滤异常湿度值
df['日均温度'] = (df['最低温度'] + df['最高温度']) / 2  # 创建新特征

# 将紫外线强度映射为数值
uv_mapping = {'最弱': 1, '弱': 2, '中等': 3, '强': 4, '最强': 5}
df['紫外线'] = df['紫外线'].map(uv_mapping)
# 2. 基础时间序列可视化
# =================================================================
plt.figure(figsize=(10, 8))
sns.set(style="whitegrid", palette="husl", rc=rc)

# 温度趋势对比
plt.subplot(3, 1, 1)
sns.lineplot(data=df, x='时间', y='最低温度', label='最低温度')
sns.lineplot(data=df, x='时间', y='最高温度', label='最高温度')
sns.lineplot(data=df, x='时间', y='日均温度', label='日均温度', linewidth=2, color='black')
plt.title('温度变化趋势对比', fontsize=14)
plt.xlabel('')
plt.ylabel('温度 (℃)')
plt.legend()

# 湿度与空气质量双轴图
ax = plt.subplot(3, 1, 2)
sns.lineplot(data=df, x='时间', y='风力', color='green', ax=ax)
ax2 = ax.twinx()
sns.lineplot(data=df, x='时间', y='空气质量', color='red', ax=ax2)
ax.set_title('风力与空气质量趋势', fontsize=14)
ax.set_ylabel('风力', color='green')
ax2.set_ylabel('空气质量指数', color='red')

# 格式调整
plt.tight_layout()
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.show()  # 只调用一次 plt.show()

# 3. 趋势与周期性分析（以日均温度为例）
# =================================================================
# 创建时间序列索引
ts = df.set_index('时间')['日均温度']

# STL分解（Seasonal-Trend decomposition using LOESS）
stl = STL(ts, period=365)  # 假设年周期
result = stl.fit()

# 可视化分解结果
plt.figure(figsize=(15, 10))
plt.subplot(4, 1, 1)
plt.plot(result.observed)
plt.title('原始时间序列')

plt.subplot(4, 1, 2)
plt.plot(result.trend)
plt.title('趋势成分')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal)
plt.title('季节性成分')

plt.subplot(4, 1, 4)
plt.plot(result.resid)
plt.title('残差成分')
plt.tight_layout()
plt.show()  # 只调用一次 plt.show()

# 4. 周期性验证（月度箱线图）
# =================================================================
df['月份'] = df['时间'].dt.month
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='月份', y='日均温度', palette='coolwarm')
plt.title('各月份温度分布（验证季节性）')
plt.ylabel('温度 (℃)')
plt.xlabel('月份')
plt.savefig("图1.png", dpi=500, bbox_inches='tight')
plt.show()  # 只调用一次 plt.show()

# 5. 多变量相关性分析（热力图）
# =================================================================
df['紫外线'] = pd.to_numeric(df['紫外线'], errors='coerce')
corr_matrix = df[['日均温度', '湿度',  '空气质量', '紫外线', '风力']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='icefire',
            fmt=".2f",
            linewidths=.5,
            annot_kws={"size": 12})
plt.title('气象要素相关性矩阵')
plt.savefig("图2.png", dpi=500, bbox_inches='tight')
plt.show()

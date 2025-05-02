import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime
import matplotlib
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy

matplotlib.use('Qt5Agg')  # 必须设置Qt兼容后端
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 优先使用系统字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# plt.rcParams['interactive'] = False

def create_blank_charts(parent):
    container = QWidget(parent)
    layout = QVBoxLayout(container)
    # 创建图表
    fig, axs = plt.subplots(7, 1, figsize=(12, 80), constrained_layout=True)

    # ==1== 一日的最低温
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Low_Temperature')
    axs[0].set_title('最低温度')

    # ==2== 一日的最高温
    axs[1].set_xlabel('')
    axs[1].set_ylabel('High_Temperature')
    axs[1].set_title('最高温度')

    # ==3== 湿度
    axs[2].set_xlabel('')
    axs[2].set_ylabel('Humidity')
    axs[2].set_title('湿度')

    # ==4== 风向柱状图实例
    axs[3].set_xlabel('风向')
    axs[3].set_ylabel('天数')
    axs[3].set_title('风向分布')

    # ==5== 风力柱状图
    axs[4].set_xlabel('风力等级')
    axs[4].set_ylabel('天数')
    axs[4].set_title('风力分布')

    # ==6== 紫外线柱状图
    axs[5].set_xlabel('紫外线强度')
    axs[5].set_ylabel('天数')
    axs[5].set_title('紫外线分布')

    # ==7== 空气质量
    axs[6].set_xlabel('')
    axs[6].set_ylabel('Air_Quality')
    axs[6].set_title('空气质量')

    plt.subplots_adjust(
        left=0.08,  # 左边距
        right=0.95,  # 右边距
        hspace=0.8  # 子图垂直间距
    )

    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)

    return container


def create_weather_charts(parent, filepath):
    # 读取数据
    # filepath = r'F:\Rayedata2\weather_crawl\weather_predict\year3_weather_data.csv'
    features = pd.read_csv(filepath, encoding='gbk')

    # 将日期列转换为datetime类型
    features['时间'] = pd.to_datetime(features['时间'], format='%Y/%m/%d')

    # 筛选指定时间范围的数据
    start_date = '2023-01-01'
    end_date = '2025-01-01'
    mask = (features['时间'] >= start_date) & (features['时间'] <= end_date)
    filtered_features = features.loc[mask]

    # 创建日期列表
    times = filtered_features['时间'].tolist()

    container = QWidget(parent)
    layout = QVBoxLayout(container)
    # 创建图表
    fig, axs = plt.subplots(7, 1, figsize=(12, 80), constrained_layout=True)

    # ==1== 一日的最低温
    axs[0].plot(times, filtered_features['最低温度'])
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Low_Temperature')
    axs[0].set_title('最低温度')

    # ==2== 一日的最高温
    axs[1].plot(times, filtered_features['最高温度'])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('High_Temperature')
    axs[1].set_title('最高温度')

    # ==3== 湿度
    axs[2].plot(times, filtered_features['湿度'])
    axs[2].set_xlabel('')
    axs[2].set_ylabel('Humidity')
    axs[2].set_title('湿度')

    # ==4== 风向柱状图实例
    wind_direction_counts = filtered_features['风向'].value_counts().sort_index()
    axs[3].bar(wind_direction_counts.index, wind_direction_counts.values, color='skyblue')
    for i, v in enumerate(wind_direction_counts.values):
        axs[3].text(i, v + 1, str(v), ha='center', va='bottom')
    axs[3].set_xlabel('风向')
    axs[3].set_ylabel('天数')
    axs[3].set_title('风向分布')

    # ==5== 风力柱状图
    wind_level_counts = filtered_features['风力'].value_counts().sort_index()
    axs[4].bar(wind_level_counts.index, wind_level_counts.values, color='lightgreen')
    for i, v in enumerate(wind_level_counts.values):
        axs[4].text(i, v + 1, str(v), ha='center', va='bottom')
    axs[4].set_xlabel('风力等级')
    axs[4].set_ylabel('天数')
    axs[4].set_title('风力分布')

    # ==6== 紫外线柱状图
    uv_counts = filtered_features['紫外线'].value_counts().sort_index()
    axs[5].bar(uv_counts.index, uv_counts.values, color='orange')
    for i, v in enumerate(uv_counts.values):
        axs[5].text(i, v + 1, str(v), ha='center', va='bottom')
    axs[5].set_xlabel('紫外线强度')
    axs[5].set_ylabel('天数')
    axs[5].set_title('紫外线分布')

    # ==7== 空气质量
    axs[6].plot(times, filtered_features['空气质量'])
    axs[6].set_xlabel('')
    axs[6].set_ylabel('Air_Quality')
    axs[6].set_title('空气质量')

    plt.subplots_adjust(
        left=0.08,    # 左边距
        right=0.95,   # 右边距
        hspace=0.8    # 子图垂直间距
    )

    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)

    return container
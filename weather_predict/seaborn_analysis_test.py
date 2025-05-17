import chardet
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib import gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import matplotlib

# 全局样式设置
matplotlib.rcParams.update({
    'font.sans-serif': 'SimHei',
    'axes.unicode_minus': False,
    'figure.autolayout': True  # 自动调整布局
})


def detect_csv_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read())['encoding']


def create_blank_charts_(parent):
    """创建空白占位图表"""
    container = QWidget(parent)
    layout = QVBoxLayout(container)

    fig = plt.figure()
    plt.text(0.32, 0.9, "图表将在此处显示",
             fontsize=24, ha='center', va='center', color='black')
    plt.axis('off')

    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)
    return container


def create_weather_charts_(parent, filepath):
    """创建包含四个分析图表的Qt组件"""
    container = QWidget(parent)
    layout = QVBoxLayout(container)

    try:
        # 数据加载与预处理
        features = load_and_process_data(filepath)

        # 创建包含四个子图的画布
        fig = plt.figure(figsize=(14, 35), dpi=100)  # 增大画布尺寸
        gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.6)  # 增加子图间距

        # === 图表1：温度趋势 ===
        ax1 = fig.add_subplot(gs[0])
        plot_temperature_trend(ax1, features)

        # === 图表2：STL分解 ===
        ax2 = fig.add_subplot(gs[1])
        plot_stl_decomposition(ax2, features)

        # === 图表3：月度分布 ===
        ax3 = fig.add_subplot(gs[2])
        plot_monthly_distribution(ax3, features)

        # === 图表4：相关性分析 ===
        ax4 = fig.add_subplot(gs[3])
        plot_correlation_heatmap(ax4, features)

        fig.tight_layout(pad=4.0)  # 增加整体边距
        fig.subplots_adjust(left=0.08, right=0.92)  # 控制左右边距

        # 集成到Qt
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

    except Exception as e:
        return create_error_widget(parent, str(e))

    return container


def load_and_process_data(filepath):
    """数据加载与预处理"""
    # 读取数据
    df = pd.read_csv(
        filepath,
        encoding=detect_csv_encoding(filepath),
        parse_dates=['时间'],
        date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'),
        sep=','
    )

    # 特征工程
    df['日均温度'] = (df['最低温度'] + df['最高温度']) / 2
    uv_mapping = {'最弱': 1, '弱': 2, '中等': 3, '强': 4, '最强': 5}
    df['紫外线'] = df['紫外线'].map(uv_mapping)

    return df


def plot_temperature_trend(ax, df):
    """绘制温度趋势图表"""
    sns.lineplot(data=df, x='时间', y='最低温度', ax=ax, label='最低温度')
    sns.lineplot(data=df, x='时间', y='最高温度', ax=ax, label='最高温度')
    sns.lineplot(data=df, x='时间', y='日均温度', ax=ax,
                 label='日均温度', color='black', linewidth=1.5)

    ax.set_title('温度变化趋势对比', fontsize=12)
    ax.set_ylabel('温度 (℃)')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(loc='upper left')


def plot_stl_decomposition(ax, df):
    """独立STL组件绘图函数"""
    ts = df.set_index('时间')['日均温度']
    result = STL(ts, period=365).fit()

    # === 统一绘图参数 ===
    line_config = {
        '原始序列': {'data': result.observed, 'color': '#1f77b4', 'ls': '-'},
        '趋势成分': {'data': result.trend, 'color': '#ff7f0e', 'ls': '-'},
        '季节成分': {'data': result.seasonal, 'color': '#2ca02c', 'ls': '-'},
        '残差': {'data': result.resid, 'color': '#d62728', 'ls': '-'}
    }

    # === 绘制所有组件 ===
    for label, config in line_config.items():
        config['data'].plot(
            ax=ax,
            label=label,
            linewidth=0.8,
            color=config['color'],
            linestyle=config['ls']
        )

    # === 格式设置 ===
    ax.set_title('STL分解组件', fontsize=12, pad=12)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.3), frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # === Y轴优化 ===
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # 限制Y轴刻度数量


def plot_monthly_distribution(ax, df):
    """月度箱线图"""
    df['月份'] = df['时间'].dt.month
    sns.boxplot(data=df, x='月份', y='日均温度',
                ax=ax, palette='coolwarm',
                showmeans=True,
                meanprops={'marker': 'o',
                           'markerfacecolor': 'white',
                           'markeredgecolor': 'black'})

    ax.set_title('各月份温度分布', fontsize=12)
    ax.set_xlabel('月份')
    ax.set_ylabel('温度 (℃)')


def plot_correlation_heatmap(ax, df):
    """相关性热力图"""
    corr_matrix = df[['日均温度', '湿度', '空气质量', '紫外线', '风力']].corr()
    sns.heatmap(corr_matrix, ax=ax, annot=True,
                cmap='icefire', fmt=".2f",
                linewidths=.5, annot_kws={"size": 10})

    ax.set_title('气象要素相关性矩阵', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)


def create_error_widget(parent, message):
    """错误提示组件"""
    container = QWidget(parent)
    layout = QVBoxLayout(container)

    fig = plt.figure()
    plt.text(0.32, 0.9, f"数据加载失败：\n{message}",
             ha='center', va='center', color='red')
    plt.axis('off')

    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)
    return container